#!/usr/bin/env python3
"""
Training of the v5 network (osu_mapper/v5): sound-source selector + encoder + autoregressive event decoder.

    python train_v5.py --data dataset_hybrid --preset base --batch_size auto
    torchrun --standalone --nproc_per_node=3 train_v5.py --data dataset_hybrid --preset base --batch_size auto
    python train_v5.py --data dataset_hybrid --resume                # continue (same schedule)
    python train_v5.py --data dataset_hybrid --resume --finetune     # new short schedule at a low learning rate

The dataset is the hybrid one (prepare_hybrid_dataset.py); with --external demucs_stems (and mert95m) the network
hears the stems separately and learns which source every note follows. Only real maps are used (no thinned or
self-generated ones). Every epoch: token loss on fixed validation windows, then complete maps of validation songs
are generated (sampled) and compared with the human ones: F1@16 + --expressive_weight x expressiveness (bursts,
rhythm, density per bar) decides the best checkpoint (v5_best.pt). Maps with a high loss are drawn more often in the
next epoch (--hard_share, "learn where it errs"). Ctrl+C saves v5_last.pt; the file <checkpoint_dir>/STOP ends
training after the current epoch; --patience stops when the score stops improving.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from osu_mapper.hybrid.audio_features import beat_features
from osu_mapper.hybrid.dataset import HybridIndex
from osu_mapper.hybrid.metrics import expressive_score, per_ms_report, summarize_reports
from osu_mapper.hybrid.renderer import render_track
from osu_mapper.v5.data import SongCache, V5Dataset, collate, group_columns, song_inputs
from osu_mapper.v5.generate import generate_song
from osu_mapper.v5.model import PRESETS, V5Config, V5Net, count_parameters, v5_losses
from train_hybrid import init_distributed, lr_at


def save(path: Path, **payload: Any) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def to_dev(b: Dict[str, torch.Tensor], dev) -> Dict[str, torch.Tensor]:
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def val_tokens(model: V5Net, loader, dev) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    n = 0
    for b in loader:
        b = to_dev(b, dev)
        out = model(b)
        _, logs = v5_losses(out, b)
        for k, v in logs.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
    return {k: v / max(1, n) for k, v in sums.items()}


@torch.no_grad()
def val_maps(model: V5Net, index: HybridIndex, cache: SongCache, map_idx: List[int], dev, temperature: float,
             cfg_scale: float, max_sec: float) -> Dict[str, float]:
    reps = []
    for i in map_idx:
        e = index.maps[i]
        tr = index.track(i)
        ref = render_track(tr)
        fa = song_inputs(np.asarray(index.features(e["song_id"], 0)), cache.slices, tr.timing_points,
                         sections="section" in model.cfg.groups, highlight=model.cfg.highlight)
        F = min(fa.shape[0], int(max_sec * 100)) if max_sec > 0 else fa.shape[0]
        fa = fa[:F]
        ba = beat_features(tr.timing_points, 0, F, 10)
        notes = generate_song(model, fa, ba, tr.timing_points, float(e.get("sr", 0.0)), temperature=temperature,
                              cfg_scale=cfg_scale, device=dev)
        ref = [n for n in ref if n.start < F * 10]
        rep = per_ms_report(notes, ref, timing_points=tr.timing_points)
        rep["expressive"] = expressive_score(rep)
        reps.append(rep)
    return summarize_reports(reps)


def fit_climaxes(index: HybridIndex, train_idx: List[int], val_idx: List[int], slices, cache_path: Path) -> List[float]:
    """Climax model of the song parts from the mappers' kiai time (osu_mapper/v5/sections.py), kept only when it beats
    "the loud parts are the climaxes" on the validation maps; [] = the loudness rule."""
    import hashlib
    from osu_mapper.v5.sections import fit_highlight, highlight_samples

    key = hashlib.sha1(json.dumps(sorted(index.maps[i]["track"] for i in train_idx)).encode()).hexdigest()
    try:
        d = json.loads(cache_path.read_text(encoding="utf-8"))
        if d.get("key") == key:
            print(f"[sections] climax model from cache: {d['note']}")
            return list(d["weights"])
    except (OSError, ValueError, KeyError):
        pass

    def samples(idx):
        out, done = [], set()
        for i in idx:
            e = index.maps[i]
            tr = index.track(i)
            out += highlight_samples(np.asarray(index.features(e["song_id"], 0)), slices, tr.timing_points)
        return out
    t0 = time.time()
    tr_s, va_s = samples(train_idx), samples(val_idx)
    weights: List[float] = []
    note = "no kiai in the maps: climaxes by loudness"
    if len(tr_s) >= 20:
        w = fit_highlight(tr_s)
        if va_s:
            X = np.stack([x for x, _ in va_s])
            y = np.array([t for _, t in va_s]) >= 0.5
            acc_m = float(((1 / (1 + np.exp(-X @ np.array(w)))) >= 0.5).__eq__(y).mean())
            acc_l = float(((X[:, 0] > 0.5) == y).mean())
            keep = acc_m > acc_l
            note = (f"climaxes learned from kiai: {acc_m:.2f} of validation parts right, loudness rule {acc_l:.2f} -> "
                    + ("learned model" if keep else "loudness rule"))
            weights = w if keep else []
        else:
            weights, note = w, "climaxes learned from kiai (no validation maps with kiai)"
    print(f"[sections] {note} ({len(tr_s)} parts with kiai, {time.time() - t0:.0f}s)")
    try:
        tmp = cache_path.with_name(cache_path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"key": key, "weights": weights, "note": note}), encoding="utf-8")
        os.replace(tmp, cache_path)
    except OSError:
        pass
    return weights


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset_hybrid")
    ap.add_argument("--checkpoint_dir", default="checkpoints/v5")
    ap.add_argument("--preset", default="base", choices=sorted(PRESETS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps_per_epoch", type=int, default=1000)
    ap.add_argument("--batch_size", default="16", help="per GPU, or 'auto' (the largest that fits --mem_fraction)")
    ap.add_argument("--mem_fraction", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--cause_weight", type=float, default=0.2, help="loss of the cause head (which source a note follows)")
    ap.add_argument("--sr_balance", type=float, default=0.5)
    ap.add_argument("--hard_tilt", type=float, default=1.0, help="draw harder maps more often: x (1 + SR)^tilt")
    ap.add_argument("--hard_share", type=float, default=0.3,
                    help="share of the draws given to maps by their loss (hard example mining, updated every epoch)")
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--val_windows", type=int, default=4, help="fixed validation windows per validation map")
    ap.add_argument("--val_maps", type=int, default=8, help="complete validation maps generated per epoch")
    ap.add_argument("--val_max_sec", type=float, default=120.0, help="first N seconds of each validation map (0 = all)")
    ap.add_argument("--expressive_weight", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--cfg_scale", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", nargs="?", const="auto", default=None)
    ap.add_argument("--finetune", action="store_true",
                    help="with --resume: a new short schedule (warmup 200 steps, cosine over --epochs) at --lr (default "
                         "here 3e-5), instead of continuing the old one")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--regimes", type=int, default=4,
                    help="latent regimes of the song parts at the start (learned from the maps, no names; 0 = off)")
    ap.add_argument("--regimes_max", type=int, default=16)
    ap.add_argument("--grow_every", type=int, default=3,
                    help="every N epochs the regime with the highest loss is split in two (its copy specialises): "
                         "'grow where it errs' as in MarI/O / NEAT; 0 = never")
    ap.add_argument("--no_sections", action="store_true",
                    help="do not give the network the parts of the song (intro / verse / chorus / drop / bridge / solo / "
                         "arpeggio ..., osu_mapper/v5/sections.py)")
    args = ap.parse_args()

    distributed, rank, world, dev = init_distributed(args.device)
    is_main = rank == 0
    if not is_main:
        sys.stdout = open(os.devnull, "w")
    torch.manual_seed(args.seed + rank)
    ck_dir = Path(args.checkpoint_dir)
    ck_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = ck_dir / "v5_last.pt", ck_dir / "v5_best.pt"

    index = HybridIndex(args.data)
    slices = index.data.get("feature_slices") or index.feature_config.slices()
    if "demucs_stems" not in slices:
        print("[warn] the dataset has no demucs stems: the source selector hears only the mix and the cause labels are all "
              "'grid'. Prepare it with: prepare_hybrid_dataset.py ... --external demucs_stems mert95m")
    train_idx, val_idx = index.split(args.val_fraction, args.seed)
    real = lambda i: not index.maps[i].get("derived") and not index.maps[i].get("self")
    train_idx = [i for i in train_idx if real(i)]
    val_idx = [i for i in val_idx if real(i)]
    print(f"[data] {len(train_idx)} training maps / {len(val_idx)} validation maps (real maps only, split by piece)")

    resume = None
    if args.resume:
        p = last_path if args.resume == "auto" else Path(args.resume)
        resume = torch.load(p, map_location="cpu", weights_only=False)
        cfg = V5Config.from_dict(resume["config"])
        print(f"[resume] {p} (epoch {resume['epoch']}, step {resume['step']})")
    else:
        cfg = V5Config(groups=group_columns(slices, sections=not args.no_sections), dropout=args.dropout,
                       regimes=args.regimes_max if args.regimes > 0 else 0, regimes_active=max(1, args.regimes),
                       **PRESETS[args.preset])
        if not args.no_sections:
            cfg.highlight = fit_climaxes(index, train_idx, val_idx, slices, Path(args.data) / "v5_climaxes.json")
    if group_columns(slices, sections="section" in cfg.groups) != cfg.groups:
        sys.exit("the dataset's features differ from the checkpoint's (e.g. other --external): use the same dataset")
    model = V5Net(cfg).to(dev)
    if resume:
        model.load_state_dict(resume["model"])
    else:
        gaps = []
        for i in train_idx[:2000]:
            t = np.unique([n.start // cfg.frame_ms for n in render_track(index.track(i))])
            gaps.append(np.diff(t))
        model.init_distance_prior(np.concatenate(gaps) if gaps else [10])
    ema = copy.deepcopy(model).eval() if args.ema > 0 else None
    if resume and ema is not None and resume.get("ema"):
        ema.load_state_dict(resume["ema"])
    print(f"[model] v5 {args.preset if not resume else 'from checkpoint'}: {count_parameters(model):,} parameters, "
          f"{len(cfg.groups) + 1} input groups, device {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    if resume and resume.get("opt") and not args.finetune:
        opt.load_state_dict(resume["opt"])
    use_amp = dev.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    vocab = cfg.vocab()
    W = cfg.window_frames
    auto = str(args.batch_size).lower() == "auto"
    bs = 8 if auto else int(args.batch_size)
    train_ds = V5Dataset(index, train_idx, vocab, W, args.steps_per_epoch * 512, seed=args.seed + 1000 * rank,
                         sr_balance=args.sr_balance, hard_tilt=args.hard_tilt, hard_share=args.hard_share,
                         max_tokens=cfg.max_tokens, sections="section" in cfg.groups, highlight=cfg.highlight)
    if auto:
        from osu_mapper.hybrid.autobatch import find_batch_size, repeat_batch
        probe = [train_ds[i] for i in range(8)]

        def probe_step(n: int) -> None:
            b = to_dev(repeat_batch(probe, n, collate), dev)
            model.train()
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                out = model(b)
                loss, _ = v5_losses(out, b, args.cause_weight)
            loss.backward()
            model.zero_grad(set_to_none=True)
        found = find_batch_size(probe_step, model, dev, args.mem_fraction, distributed=distributed, log=print)
        bs = found or 8
    train_ds.samples = args.steps_per_epoch * bs
    loader = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=args.workers, collate_fn=collate,
                        drop_last=True, persistent_workers=False)
    val_loader = None
    if val_idx and is_main:
        fixed = V5Dataset.fixed_windows(index, val_idx, W, args.val_windows, seed=args.seed)
        val_ds = V5Dataset(index, val_idx, vocab, W, 0, train=False, fixed=fixed, sr_balance=0, max_tokens=cfg.max_tokens,
                           sections="section" in cfg.groups, highlight=cfg.highlight)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate)
    val_full = sorted(val_idx, key=lambda i: float(index.maps[i].get("sr", 0.0)))
    if val_full and args.val_maps > 0:
        pick = np.unique(np.linspace(0, len(val_full) - 1, min(args.val_maps, len(val_full))).round().astype(int))
        val_full = [val_full[p] for p in pick]
    cache = SongCache(index, sections="section" in cfg.groups, highlight=cfg.highlight)

    train_model = model
    if distributed:
        train_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index] if dev.type == "cuda" else None)
    step = int(resume["step"]) if resume else 0
    start_epoch = int(resume["epoch"]) if resume else 0
    best = float(resume.get("best", -1e9)) if resume and not args.finetune else -1e9
    history = list(resume.get("history", [])) if resume else []
    if args.finetune:
        sched0, total, warm = step, args.epochs * args.steps_per_epoch, 200
        lr_base = args.lr if args.lr != 3e-4 else 3e-5
    else:
        sched0, total, warm = 0, max(int(resume.get("total", 0)) if resume else 0, (start_epoch + args.epochs) * args.steps_per_epoch), args.warmup_steps
        lr_base = args.lr
    print(f"[train] window {W * cfg.frame_ms / 1000:.1f} s, batch {bs} x {world} GPU(s), {args.steps_per_epoch} steps/epoch, "
          f"lr {lr_base:g}" + (" (new fine-tuning schedule)" if args.finetune else ""))
    per_map_loss = np.full(len(train_idx), np.nan)
    reg_loss = np.zeros(max(1, cfg.regimes))
    reg_n = np.zeros(max(1, cfg.regimes))

    got = {"n": 0}

    def on_sig(signum, frame):
        got["n"] += 1
        if got["n"] == 1:
            raise KeyboardInterrupt
    for sg in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sg is not None:
            try:
                signal.signal(sg, on_sig)
            except (ValueError, OSError):
                pass
    stop_file = ck_dir / "STOP"
    if is_main:
        stop_file.unlink(missing_ok=True)
        print(f"[train] to stop after the current epoch, create the file {stop_file}")

    def payload(epoch: int, summary: Dict[str, float]) -> Dict[str, Any]:
        return {"format": "v5-checkpoint-1", "config": cfg.to_dict(), "model": model.state_dict(),
                "ema": ema.state_dict() if ema is not None else None, "opt": opt.state_dict(), "epoch": epoch,
                "step": step, "total": total, "best": best, "history": history, "summary": summary,
                "feature_config": index.feature_config.to_dict(), "args": vars(args)}

    best_epoch = start_epoch
    epoch = start_epoch
    try:
        for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
            train_ds.set_epoch(epoch)
            model.train()
            t0 = time.time()
            run: Dict[str, float] = {}
            nrun = 0
            for b in loader:
                b = to_dev(b, dev)
                lr = lr_at(step - sched0, total, warm, lr_base, args.min_lr if not args.finetune else lr_base / 10)
                for g in opt.param_groups:
                    g["lr"] = lr
                with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                    out = train_model(b)
                    loss, logs = v5_losses(out, b, args.cause_weight)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if ema is not None:
                    with torch.no_grad():
                        d = min(args.ema, (1 + step) / (10 + step))
                        for pe, pm in zip(ema.parameters(), model.parameters()):
                            pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                # learn where it errs: running loss of every map (this rank's draws)
                ks = b["map_k"].view(-1).cpu().numpy()
                li = out["_per_item"].float().cpu().numpy()
                if out.get("regime") is not None:
                    rg = torch.mode(out["regime"], dim=1).values.cpu().numpy()       # main regime of every window
                    for r_, l_ in zip(rg, li):
                        reg_loss[r_] += l_
                        reg_n[r_] += 1
                for k_, l_ in zip(ks, li):
                    per_map_loss[k_] = l_ if not np.isfinite(per_map_loss[k_]) else 0.8 * per_map_loss[k_] + 0.2 * l_
                step += 1
                for k, v in logs.items():
                    run[k] = run.get(k, 0.0) + v
                nrun += 1
                if step % args.log_every == 0:
                    a = {k: v / nrun for k, v in run.items()}
                    print(f"  step {step:>7} | lr {lr:.2e} | tokens {a['tokens']:.4f} (time {a.get('time_tokens', float('nan')):.3f}) "
                          f"| cause {a.get('cause', float('nan')):.3f} acc {a.get('cause_acc', float('nan')):.2f}", flush=True)
                    run, nrun = {}, 0
            train_ds.set_hard_weights(per_map_loss)
            # grow where it errs: the active regime with the highest mean loss (and real use) is split in two
            if model.regime is not None and args.grow_every > 0 and epoch % args.grow_every == 0:
                act = int(model.regime.active)
                use = reg_n[:act] / max(1.0, reg_n[:act].sum())
                mean = np.where(reg_n[:act] > 0, reg_loss[:act] / np.maximum(reg_n[:act], 1), -1.0)
                src = int(np.argmax(np.where(use >= 0.05, mean, -1.0)))
                if distributed:
                    t_ = torch.tensor([src], device=dev)
                    dist.broadcast(t_, 0)
                    src = int(t_.item())
                new_k = model.regime.grow(src)
                if new_k is not None:
                    if ema is not None:
                        ema.regime.load_state_dict(model.regime.state_dict())
                    cfg.regimes_active = new_k + 1
                    print(f"[grow] regime {src} (mean loss {mean[src]:.3f}, {100 * use[src]:.0f}% of windows) split -> "
                          f"{new_k + 1} regimes; use per regime: " + " ".join(f"{100 * u:.0f}%" for u in use))
            reg_loss[:] = 0.0
            reg_n[:] = 0.0
            stop = False
            if is_main:
                em = ema if ema is not None else model
                rec: Dict[str, Any] = {"epoch": epoch, "step": step, "train_sec": round(time.time() - t0, 1)}
                summary: Dict[str, float] = {}
                if val_loader is not None:
                    vt = val_tokens(em, val_loader, dev)
                    rec.update({f"val_{k}": v for k, v in vt.items()})
                    print(f"[epoch {epoch}] val windows: tokens {vt.get('tokens', float('nan')):.4f} (time {vt.get('time_tokens', float('nan')):.3f}) "
                          f"| cause acc {vt.get('cause_acc', float('nan')):.3f}")
                if val_full:
                    t1 = time.time()
                    summary = val_maps(em, index, cache, val_full, dev, args.temperature, args.cfg_scale, args.val_max_sec)
                    rec.update({f"map_{k}": v for k, v in summary.items()})
                    print(f"[epoch {epoch}] GENERATED MAPS vs human ({len(val_full)} songs, {time.time() - t1:.0f}s): event F1 "
                          f"@16ms {summary.get('event_f1@16', 0):.3f} @64ms {summary.get('event_f1@64', 0):.3f} | density "
                          f"x{summary.get('density_ratio', float('nan')):.2f} | expressiveness {summary.get('expressive', 0):.3f}: "
                          f"bursts F1 {summary.get('burst_f1', 0):.3f} (length x{summary.get('burst_ms_ratio', float('nan')):.2f}) "
                          f"| rhythm distance {summary.get('subdiv_tv', float('nan')):.3f} | 1/4 and finer "
                          f"{100 * summary.get('fine_share_pred', 0):.0f}% vs {100 * summary.get('fine_share_ref', 0):.0f}% "
                          f"| density per bar r={summary.get('bar_density_corr', 0):.2f}")
                    score = summary.get("event_f1@16", 0.0) + args.expressive_weight * summary.get("expressive", 0.0)
                else:
                    score = -rec.get("val_tokens", float(step))
                history.append(rec)
                save(last_path, **payload(epoch, summary))
                if score > best:
                    best, best_epoch = score, epoch
                    save(best_path, **payload(epoch, summary))
                    print(f"  * new best ({score:.4f}) -> {best_path}")
                elif args.patience > 0 and epoch - best_epoch >= args.patience:
                    print(f"[early stop] no improvement for {epoch - best_epoch} epochs (best {best:.4f} at epoch {best_epoch})")
                    stop = True
                if stop_file.exists():
                    print(f"[stop] {stop_file} found")
                    stop_file.unlink(missing_ok=True)
                    stop = True
                (ck_dir / "history_v5.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
            if distributed:
                f = torch.tensor([1 if stop else 0], device=dev)
                dist.broadcast(f, 0)
                stop = bool(f.item())
            if stop:
                break
    except (KeyboardInterrupt, RuntimeError) as exc:
        if isinstance(exc, RuntimeError):
            if not distributed:
                raise
            try:
                for _ in range(50):
                    if got["n"]:
                        break
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            if not got["n"]:
                raise
        if is_main:
            print(f"\n[paused] saving {last_path}", flush=True)
            save(last_path, **payload(max(start_epoch, epoch - 1), {}))
            print("[paused] saved", flush=True)
        if distributed:
            sys.stdout.flush()
            os._exit(0)
        return
    print(f"[done] best score {best:.4f} (epoch {best_epoch}); checkpoints in {ck_dir}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
