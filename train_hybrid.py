#!/usr/bin/env python3
"""
Training of the hybrid network (neural parameter predictor for the universal map algorithm).

    python train_hybrid.py --data dataset_hybrid --preset base --epochs 40
    python train_hybrid.py --data dataset_hybrid --resume            # continue exactly where it stopped
    torchrun --nproc_per_node=4 train_hybrid.py --data dataset_hybrid  # several GPUs (DistributedDataParallel)

Design decisions that fix the problems of the old Model 3 training:
  * ONE objective, ONE checkpoint namespace (checkpoints/hybrid/). There is no second training mode
    that rewrites the base weights, so resuming never "resets" the loss.
  * Every loss term is a normalized per-element mean (all ~O(1)); each term is printed separately.
  * The real quality signal is the per-millisecond comparison of the RENDERED map with the reference
    map on held-out songs (event F1 at 0..64 ms, timing MAE, ...). best.pt is chosen by it.
  * Validation songs are disjoint from training songs (split by audio, not by chunk).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from osu_mapper.hybrid.dataset import HybridCropDataset, HybridIndex, collate_hybrid
from osu_mapper.hybrid.inference import CHECKPOINT_FORMAT, HybridPredictor, SongInputs, resolve_device, star_rating_of
from osu_mapper.hybrid.losses import LossWeights, compute_losses, crop_metrics
from osu_mapper.hybrid.metrics import per_ms_report, summarize_reports
from osu_mapper.hybrid.model import HybridModelConfig, HybridParamNet, PRESETS, build_config, count_parameters
from osu_mapper.hybrid.params import PARAM_VERSION
from osu_mapper.hybrid.renderer import DecodeKnobs, decode_outputs, render_track
from osu_mapper.hybrid.audio_features import beat_features


def init_distributed(requested_device: str) -> Tuple[bool, int, int, torch.device]:
    """Sets up DistributedDataParallel when launched with torchrun (WORLD_SIZE > 1)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return False, 0, 1, resolve_device(requested_device)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and requested_device != "cpu"
    if use_cuda:
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
    else:
        dev = torch.device("cpu")
    # NCCL is the fast GPU backend but does not exist on Windows; gloo works everywhere
    backend = "nccl" if (use_cuda and dist.is_nccl_available() and sys.platform != "win32") else "gloo"
    dist.init_process_group(backend=backend)
    return True, rank, world, dev


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_at(step: int, total: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))


def sigma_at(step: int, total: int, start: float, end: float, frac: float = 0.3) -> float:
    p = min(1.0, step / max(1, frac * total))
    return start + (end - start) * p


def to_device(batch: Dict[str, torch.Tensor], dev: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


def forward_batch(model: HybridParamNet, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return model(
        batch["feats"], batch["beat"], batch["sr"], batch["style"], batch["style_mask"],
        batch["coarse"], batch["coarse_mask"],
    )


@torch.no_grad()
def update_ema(ema: HybridParamNet, model: HybridParamNet, decay: float) -> None:
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1.0 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


@torch.no_grad()
def validate_crops(model: HybridParamNet, loader: DataLoader, dev: torch.device, weights: LossWeights, frame_ms: int, sigma: float) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = to_device(batch, dev)
        out = forward_batch(model, batch)
        _, logs = compute_losses(out, batch, weights, frame_ms, sigma_ms=sigma)
        logs.update(crop_metrics(out, batch))
        for k, v in logs.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
    return {k: v / max(1, n) for k, v in sums.items()}


@torch.no_grad()
def validate_full_maps(
    model: HybridParamNet,
    index: HybridIndex,
    map_indices: List[int],
    crop_frames: int,
    dev: torch.device,
    thresholds: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
    ln_thresholds: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6),
    style_known: bool = False,
) -> Tuple[Dict[str, float], float, float]:
    """Renders complete validation maps and compares them with the references per millisecond.

    Returns (metrics at the best event threshold, best event threshold, best LN threshold).
    """
    model.eval()
    predictor = HybridPredictor(model, index.feature_config, crop_frames, dev)
    per_threshold: Dict[float, List[Dict[str, float]]] = {t: [] for t in thresholds}
    cache = []
    for mi in map_indices:
        entry = index.maps[mi]
        track = index.track(mi)
        feats = np.asarray(index.features(entry["song_id"], 0), dtype=np.float32)
        F = feats.shape[0]
        beat = beat_features(track.timing_points, 0, F, track.frame_ms)
        inputs = SongInputs(feats=feats, coarse=index.coarse(entry["song_id"]), beat=beat, duration_ms=F * track.frame_ms)
        style = np.asarray(entry["style"], np.float32)
        smask = np.asarray(entry["style_mask"], np.float32) if style_known else np.zeros_like(style)
        out = predictor.predict(inputs, float(entry.get("sr", 0.0)), style, smask)
        ref = render_track(track)
        cache.append((out, ref, track.timing_points, float(entry.get("sr", 0.0))))
        for th in thresholds:
            notes = decode_outputs(out, DecodeKnobs(onset_threshold=th), track.timing_points)
            per_threshold[th].append(per_ms_report(notes, ref))
    best_th, best_score, best_summary = thresholds[0], -1.0, {}
    for th, reps in per_threshold.items():
        summ = summarize_reports(reps)
        score = summ.get("event_f1@16", 0.0)
        if score > best_score:
            best_th, best_score, best_summary = th, score, summ
    best_ln, best_ln_f1 = 0.5, -1.0
    for lt in ln_thresholds:
        f1 = float(np.mean([per_ms_report(decode_outputs(o, DecodeKnobs(onset_threshold=best_th, ln_threshold=lt), tps), r)["ln_f1"] for o, r, tps, _ in cache]))
        if f1 > best_ln_f1:
            best_ln, best_ln_f1 = lt, f1
    best_summary["ln_f1_at_best_ln_threshold"] = best_ln_f1
    # does the network itself produce the requested difficulty? (no density calibration at generation)
    sr_err = []
    for o, _r, tps, target in cache:
        if target > 0:
            notes = decode_outputs(o, DecodeKnobs(onset_threshold=best_th, ln_threshold=best_ln), tps)
            sr_err.append(star_rating_of(notes, tps) - target)
    if sr_err:
        best_summary["sr_mae"] = float(np.mean(np.abs(sr_err)))
        best_summary["sr_bias"] = float(np.mean(sr_err))
    return best_summary, best_th, best_ln


def save_checkpoint(path: Path, **payload: Any) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the hybrid neural->parametric osu!mania 4K mapper")
    ap.add_argument("--data", default="dataset_hybrid", help="dataset folder from prepare_hybrid_dataset.py")
    ap.add_argument("--checkpoint_dir", default="checkpoints/hybrid")
    ap.add_argument("--preset", default="base", choices=sorted(PRESETS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps_per_epoch", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--crop_sec", type=float, default=20.48, help="training window length (also the inference window)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.02)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay of evaluation weights (0 disables)")
    ap.add_argument("--sigma_start", type=float, default=3.0, help="ms spread of timing targets at the start")
    ap.add_argument("--sigma_end", type=float, default=1.0, help="ms spread of timing targets after warm-up")
    ap.add_argument("--event_weight", type=float, default=4.0)
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--val_crops", type=int, default=256)
    ap.add_argument("--val_maps", type=int, default=24, help="complete validation maps rendered per epoch (per-ms metrics)")
    ap.add_argument("--val_style_known", action="store_true", help="give the reference style to the model during validation")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="auto", choices=["auto", "off", "bf16", "fp16"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", nargs="?", const="auto", default=None, help="resume from hybrid_last.pt (or a given path)")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    distributed, rank, world, dev = init_distributed(args.device)
    is_main = rank == 0
    if not is_main:  # only rank 0 prints and writes checkpoints
        import builtins
        builtins.print = lambda *a, **k: None
    set_seed(args.seed)
    index = HybridIndex(args.data)
    fcfg = index.feature_config
    crop_frames = int(round(args.crop_sec * 1000.0 / fcfg.frame_ms))
    crop_frames -= crop_frames % 4
    train_idx, val_idx = index.split_by_song(args.val_fraction, args.seed)
    if not train_idx:
        sys.exit("No training maps (did every map fail the roundtrip check?)")
    print(f"[data] {len(index.maps)} maps, {len(index.songs)} songs -> train {len(train_idx)} maps / val {len(val_idx)} maps (split by song)")

    ckpt_dir = Path(args.checkpoint_dir)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = ckpt_dir / "hybrid_last.pt", ckpt_dir / "hybrid_best.pt"

    resume_ckpt: Optional[Dict[str, Any]] = None
    if args.resume:
        rp = last_path if args.resume == "auto" else Path(args.resume)
        if not rp.exists():
            sys.exit(f"--resume: '{rp}' not found")
        resume_ckpt = torch.load(rp, map_location="cpu", weights_only=False)
        if resume_ckpt.get("format") != CHECKPOINT_FORMAT:
            sys.exit(f"--resume: '{rp}' is not a hybrid checkpoint")
        if resume_ckpt.get("param_version") != PARAM_VERSION:
            sys.exit("--resume: parameter space version changed, the old checkpoint cannot be continued")
        if resume_ckpt.get("feature_config") != fcfg.to_dict():
            sys.exit("--resume: the dataset was prepared with a different feature config than the checkpoint")
        cfg = HybridModelConfig.from_dict(resume_ckpt["model_config"])
        crop_frames = int(resume_ckpt.get("crop_frames", crop_frames))
        print(f"[resume] {rp} (epoch {resume_ckpt.get('epoch')}, step {resume_ckpt.get('step')})")
    else:
        cfg = build_config(fcfg.dims(), fcfg.total_dim(), args.preset)

    model = HybridParamNet(cfg).to(dev)
    ema = copy.deepcopy(model).eval() if (args.ema > 0 and is_main) else None
    if ema is not None:
        for p in ema.parameters():
            p.requires_grad_(False)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if (p.ndim < 2 or "norm" in name or "missing" in name or "null" in name or "gate" in name) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98))

    use_amp = args.amp != "off" and dev.type == "cuda"
    amp_dtype = torch.bfloat16 if (args.amp == "bf16" or (args.amp == "auto" and dev.type == "cuda" and torch.cuda.is_bf16_supported())) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    step, start_epoch, best_score, history = 0, 0, -float("inf"), []
    decode_threshold = 0.5
    ln_threshold = 0.5
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state"])
        if ema is not None and resume_ckpt.get("ema_state"):
            ema.load_state_dict(resume_ckpt["ema_state"])
        if resume_ckpt.get("optimizer_state"):
            opt.load_state_dict(resume_ckpt["optimizer_state"])
        step = int(resume_ckpt.get("step", 0))
        start_epoch = int(resume_ckpt.get("epoch", 0))
        best_score = float(resume_ckpt.get("best_score", -float("inf")))
        history = list(resume_ckpt.get("history", []))
        decode_threshold = float(resume_ckpt.get("decode_threshold", 0.5))
        ln_threshold = float(resume_ckpt.get("ln_threshold", 0.5))

    # wrap after loading the checkpoint; all ranks start from identical weights (DDP broadcasts rank 0)
    train_model = DistributedDataParallel(model, device_ids=[dev.index] if dev.type == "cuda" else None) if distributed else model

    weights = LossWeights()
    total_steps = (start_epoch + args.epochs) * args.steps_per_epoch if resume_ckpt is None else max(int(resume_ckpt.get("total_steps", 0)), (start_epoch + args.epochs) * args.steps_per_epoch)
    print(f"[model] preset={args.preset if resume_ckpt is None else 'from checkpoint'} params={count_parameters(model):,} device={dev} amp={'on ' + str(amp_dtype) if use_amp else 'off'}")
    print(f"[train] window={crop_frames} frames ({crop_frames * fcfg.frame_ms / 1000:.2f} s), batch={args.batch_size} x {world} GPU(s) = {args.batch_size * world}, "
          f"steps/epoch={args.steps_per_epoch}, total steps={total_steps}")

    # every rank draws its own random crops (different seed), so no sampler is needed
    train_ds = HybridCropDataset(index, train_idx, crop_frames, args.steps_per_epoch * args.batch_size, train=True, seed=args.seed + 1000 * rank)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_hybrid, drop_last=True)
    val_loader = None
    if val_idx:
        val_ds = HybridCropDataset(index, val_idx, crop_frames, args.val_crops, train=False, seed=args.seed)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_hybrid)
    rng = random.Random(args.seed)
    val_full = sorted(rng.sample(val_idx, min(args.val_maps, len(val_idx)))) if val_idx else []

    def payload(epoch: int, summary: Dict[str, float]) -> Dict[str, Any]:
        return {
            "format": CHECKPOINT_FORMAT,
            "param_version": PARAM_VERSION,
            "model_config": cfg.to_dict(),
            "feature_config": fcfg.to_dict(),
            "crop_frames": crop_frames,
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict() if ema is not None else None,
            "optimizer_state": opt.state_dict(),
            "epoch": epoch,
            "step": step,
            "total_steps": total_steps,
            "best_score": best_score,
            "decode_threshold": decode_threshold,
            "ln_threshold": ln_threshold,
            "val_summary": summary,
            "history": history,
            "args": vars(args),
        }

    epoch = start_epoch
    try:
        for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
            train_ds.set_epoch(epoch)
            train_model.train()
            t0 = time.time()
            run: Dict[str, float] = {}
            n_run = 0
            for batch in train_loader:
                batch = to_device(batch, dev)
                lr = lr_at(step, total_steps, args.warmup_steps, args.lr, args.min_lr)
                for g in opt.param_groups:
                    g["lr"] = lr
                sigma = sigma_at(step, total_steps, args.sigma_start, args.sigma_end)
                with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                    out = forward_batch(train_model, batch)
                loss, logs = compute_losses(out, batch, weights, cfg.frame_ms, sigma_ms=sigma, event_weight=args.event_weight)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if ema is not None:
                    update_ema(ema, model, min(args.ema, (1 + step) / (10 + step)))
                step += 1
                for k, v in logs.items():
                    run[k] = run.get(k, 0.0) + v
                n_run += 1
                if step % args.log_every == 0:
                    avg = {k: v / n_run for k, v in run.items()}
                    print(f"  step {step:>7} | lr {lr:.2e} | sigma {sigma:.2f}ms | total {avg['total']:.4f} | time {avg['time']:.4f} lane {avg['lane']:.3f} chord {avg['chord']:.3f} "
                          f"stagger {avg['stagger']:.3f} ln {avg['ln_flag']:.3f}/{avg['ln_len']:.3f} style {avg['style']:.3f}", flush=True)
                    run, n_run = {}, 0

            if not is_main:  # validation and checkpoints happen on rank 0; the others wait
                dist.barrier()
                continue
            eval_model = ema if ema is not None else model
            record: Dict[str, Any] = {"epoch": epoch, "step": step, "train_sec": round(time.time() - t0, 1)}
            summary: Dict[str, float] = {}
            if val_loader is not None:
                vc = validate_crops(eval_model, val_loader, dev, weights, cfg.frame_ms, args.sigma_end)
                record.update({f"val_{k}": v for k, v in vc.items()})
                print(f"[epoch {epoch}] val crops: total {vc['total']:.4f} | time {vc['time']:.4f} | lane {vc['lane']:.3f} | frame F1 {vc.get('frame_f1', 0):.3f} "
                      f"| offset exact {vc.get('offset_exact', 0):.3f} (+-1ms {vc.get('offset_within1ms', 0):.3f}) | lane acc {vc.get('lane_acc', 0):.3f} | chord acc {vc.get('chord_acc', 0):.3f}")
            if val_full:
                t1 = time.time()
                summary, decode_threshold, ln_threshold = validate_full_maps(eval_model, index, val_full, crop_frames, dev, style_known=args.val_style_known)
                record.update({f"map_{k}": v for k, v in summary.items()})
                record["decode_threshold"] = decode_threshold
                record["ln_threshold"] = ln_threshold
                print(f"[epoch {epoch}] RENDERED MAPS vs reference ({len(val_full)} songs, thresholds event {decode_threshold} / LN {ln_threshold}, {time.time() - t1:.1f}s): "
                      f"per-ms loss {summary.get('per_ms_loss', float('nan')):.4f} | event F1 @0ms {summary.get('event_f1@0', 0):.3f} @2ms {summary.get('event_f1@2', 0):.3f} "
                      f"@16ms {summary.get('event_f1@16', 0):.3f} @64ms {summary.get('event_f1@64', 0):.3f} | timing MAE {summary.get('timing_mae_ms', float('nan')):.2f}ms "
                      f"| chord acc {summary.get('chord_size_acc', 0):.3f} | LN F1 {summary.get('ln_f1_at_best_ln_threshold', 0):.3f} | density x{summary.get('density_ratio', float('nan')):.2f} "
                      f"| SR error {summary.get('sr_mae', float('nan')):.2f}* (bias {summary.get('sr_bias', float('nan')):+.2f})")
            history.append(record)
            if summary:
                score = float(summary.get("event_f1@16", 0.0))   # rendered-map quality on held-out songs
            elif "val_total" in record:
                score = -float(record["val_total"])
            else:
                score = float(step)                              # no validation data: newest is best
            save_checkpoint(last_path, **payload(epoch, summary))
            if score > best_score:
                best_score = score
                save_checkpoint(best_path, **payload(epoch, summary))
                print(f"  * new best ({score:.4f}) -> {best_path}")
            with open(ckpt_dir / "history.json", "w", encoding="utf-8") as fh:
                json.dump(history, fh, indent=1)
            if distributed:
                dist.barrier()
    except KeyboardInterrupt:
        if is_main:
            print(f"\n[paused] saving {last_path} (resume with: python train_hybrid.py --data {args.data} --resume)")
            save_checkpoint(last_path, **payload(max(start_epoch, epoch - 1), {}))
        return
    print(f"[done] best score {best_score:.4f}; checkpoints in {ckpt_dir}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
