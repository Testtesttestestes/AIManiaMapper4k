"""
Automatic batch size: the largest batch whose training step fits the GPU memory (``train_hybrid.py --batch_size auto``).

Two probe steps (forward, every loss of the run, backward - on real crops of the dataset, with the run's AMP and
techniques) at small batches give the peak memory as a line ``base + per_item * batch``; the batch that fills
``fraction`` of the card (minus what other programs hold, plus the optimizer's Adam moments and DDP's gradient
buckets that the probe does not allocate) is then verified with a real step and lowered by 15 % while it runs out of
memory. The batch is even (pairs of difficulties need it); with several GPUs the smallest card decides.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch


def _peak(step: Callable[[int], None], bs: int, dev: torch.device) -> Optional[int]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    try:
        step(bs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    torch.cuda.synchronize(dev)
    return int(torch.cuda.max_memory_allocated(dev))


def find_batch_size(step: Callable[[int], None], model: torch.nn.Module, dev: torch.device, fraction: float = 0.9,
                    max_bs: int = 512, distributed: bool = False, log=print) -> Optional[int]:
    """``step(bs)``: one full training step without the optimizer update. None when not on a CUDA device."""
    if dev.type != "cuda":
        return None
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(dev)
    other = max(0, total - free - torch.cuda.memory_reserved(dev))          # other programs on the card
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    extra = 2 * param_bytes + (param_bytes if distributed else 0)          # Adam moments (+ DDP buckets)
    budget = fraction * total - other - extra
    p1 = _peak(step, 2, dev)
    p2 = _peak(step, 6, dev) if p1 is not None else None
    if p1 is None or p2 is None:
        log("[batch] even a batch of 2 does not fit the card; use a smaller --preset or --crop_sec")
        return 2
    per = max(1.0, (p2 - p1) / 4.0)
    base = p1 - 2 * per
    bs = int((budget - base) / per)
    bs = max(2, min(max_bs, bs - bs % 2))
    while bs > 2:
        pk = _peak(step, bs, dev)
        if pk is not None and pk <= budget * 1.02:
            break
        bs = max(2, int(bs * 0.85))
        bs -= bs % 2
    log(f"[batch] auto: {bs} per GPU (card {total / 2 ** 30:.1f} GB, other programs {other / 2 ** 30:.1f} GB, "
        f"~{per / 2 ** 20:.0f} MB per crop, aiming at {100 * fraction:.0f} % of the card)")
    if distributed:
        import torch.distributed as dist
        t = torch.tensor([bs], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        bs = int(t.item())
    torch.cuda.empty_cache()
    return bs


def repeat_batch(items, bs: int, collate) -> Dict[str, torch.Tensor]:
    """A batch of ``bs`` from a few cached items (probing does not need distinct crops, only their shapes)."""
    return collate([items[i % len(items)] for i in range(bs)])
