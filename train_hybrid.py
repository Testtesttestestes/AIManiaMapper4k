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
from osu_mapper.hybrid.calibration import calibrate
from osu_mapper.hybrid.losses import LossWeights, compute_losses, crop_metrics
from osu_mapper.hybrid.metrics import per_ms_report, summarize_reports
from osu_mapper.hybrid.model import HybridModelConfig, HybridParamNet, PRESETS, build_config, count_parameters
from osu_mapper.hybrid.params import PARAM_VERSION
from osu_mapper.hybrid.renderer import DecodeKnobs, decode_outputs, render_track
from osu_mapper.hybrid.sr_response import format_response, sr_response
from osu_mapper.hybrid.audio_features import FeatureConfig, beat_features


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
    # long timeout: before the first collective every rank builds the dataset caches (motif targets, star ratings at
    # every rate), which can take far longer than the default 10 minutes on a big dataset
    import datetime
    # the heartbeat monitor only dumps debug info on hangs; when torchrun exits first (Ctrl+C) it floods the console
    # with "TCPStore ... Broken pipe" every second. Hangs are still caught by the collective timeout below.
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(hours=3))
    return True, rank, world, dev


class Tee:
    """Writes everything printed to the console into a log file as well (checkpoint_dir/train.log)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for st in self.streams:
            st.write(data)
            st.flush()
        return len(data)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


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
    canvas = (batch["canvas_state"], batch["canvas_ms"], batch["canvas_ln"]) if "canvas_state" in batch else None
    plan = (batch["plan_in_family"], batch["plan_in_copy"]) if "plan_in_family" in batch else None
    return model(
        batch["feats"], batch["beat"], batch["sr"], batch["style"], batch["style_mask"],
        batch["coarse"], batch["coarse_mask"], canvas=canvas, plan=plan, augmented=batch.get("augmented"),
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
    thresholds: Tuple[float, ...] = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7),
    ln_thresholds: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6),
    style_known: bool = False,
    min_maps_per_bin: int = 2,
) -> Tuple[Dict[str, float], float, float, Dict[int, float]]:
    """Renders complete validation maps and compares them with the references per millisecond.

    Returns (metrics at the best event threshold, best event threshold, best LN threshold,
    best event threshold per 1-star bin of the reference maps - dense maps need lower thresholds).
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
    by_bin: Dict[int, List[int]] = {}
    for j, (_o, _r, _t, target) in enumerate(cache):
        if target > 0:
            by_bin.setdefault(int(min(target, 9.99)), []).append(j)
    threshold_by_sr: Dict[int, float] = {}
    for b, js in sorted(by_bin.items()):
        if len(js) >= min_maps_per_bin:
            threshold_by_sr[b] = max(thresholds, key=lambda th: float(np.mean([per_threshold[th][j]["event_f1@16"] for j in js])))
    best_ln, best_ln_f1 = 0.5, -1.0
    for lt in ln_thresholds:
        f1 = float(np.mean([per_ms_report(decode_outputs(o, DecodeKnobs(onset_threshold=best_th, ln_threshold=lt), tps), r)["ln_f1"] for o, r, tps, _ in cache]))
        if f1 > best_ln_f1:
            best_ln, best_ln_f1 = lt, f1
    best_summary["ln_f1_at_best_ln_threshold"] = best_ln_f1
    best_summary.update(calibrate(((o, r, tps) for o, r, tps, _ in cache), best_th, best_ln))
    # does the network itself produce the requested difficulty? (no density calibration at generation)
    sr_err = []
    for o, _r, tps, target in cache:
        if target > 0:
            notes = decode_outputs(o, DecodeKnobs(onset_threshold=best_th, ln_threshold=best_ln), tps)
            sr_err.append(star_rating_of(notes, tps) - target)
    if sr_err:
        best_summary["sr_mae"] = float(np.mean(np.abs(sr_err)))
        best_summary["sr_bias"] = float(np.mean(sr_err))
    return best_summary, best_th, best_ln, threshold_by_sr


@torch.no_grad()
def measure_sr_response(model, index: HybridIndex, map_indices: List[int], crop_frames: int, dev: torch.device, n_maps: int,
                        threshold: float, ln_threshold: float, threshold_by_sr: Dict[int, float]) -> Dict[str, float]:
    """SR obedience on a few validation maps spread over the star-rating range (see sr_response.py)."""
    cand = sorted((i for i in map_indices if float(index.maps[i].get("sr", 0.0)) > 0), key=lambda i: float(index.maps[i]["sr"]))
    if not cand:
        return {}
    pick = [cand[p] for p in sorted(set(np.linspace(0, len(cand) - 1, min(n_maps, len(cand))).round().astype(int).tolist()))]
    model.eval()
    predictor = HybridPredictor(model, index.feature_config, crop_frames, dev, threshold, ln_threshold)
    predictor.threshold_by_sr = dict(threshold_by_sr)
    items = []
    for mi in pick:
        entry, track = index.maps[mi], index.track(mi)
        feats = np.asarray(index.features(entry["song_id"], 0), dtype=np.float32)
        F = feats.shape[0]
        inputs = SongInputs(feats=feats, coarse=index.coarse(entry["song_id"]), beat=beat_features(track.timing_points, 0, F, track.frame_ms), duration_ms=F * track.frame_ms)
        items.append((inputs, render_track(track), list(track.timing_points), float(entry["sr"])))
    return sr_response(predictor, items)


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
    ap.add_argument("--batch_size", default="16",
                    help="crops per GPU and step, or 'auto': the largest batch whose training step fits --mem_fraction of the "
                         "card (probed with real crops and every loss of the run)")
    ap.add_argument("--mem_fraction", type=float, default=0.9,
                    help="--batch_size auto: share of the GPU memory to fill (lower it if other programs need the card)")
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
    ap.add_argument("--split", default=None, choices=["piece", "audio"],
                    help="train/val split groups: 'piece' = all audio files / rate versions (x1.1, x1.25) of one song together "
                         "(default for new trainings), 'audio' = one group per audio file (older checkpoints; kept automatically on --resume/--eval_only)")
    ap.add_argument("--rep_dropout", nargs="*", default=None, metavar="NAME=P",
                    help="per-representation modality dropout during training, e.g. mert95m=0.5 demucs_stems=0.3 "
                         "(default 0.15 for all); high values stop the model from recognising performers by voice")
    ap.add_argument("--density_weight", type=float, default=0.3, help="loss: expected events/notes per window vs the reference")
    ap.add_argument("--pair_weight", type=float, default=0.3, help="loss: density ratio of the same window at two difficulties")
    ap.add_argument("--ladder_weight", type=float, default=0.3,
                    help="loss: difficulty ladder - which moments of a song its maps of different star ratings all map "
                         "(adds the importance head; 0 = off; an older checkpoint gets the head on --resume)")
    ap.add_argument("--jack_weight", type=float, default=0.2,
                    help="loss: at fast transitions (< 160 ms) P(the lane code repeats a lane of the previous chord) vs the map "
                         "(unseen maps: network 0.27, human 0.09); 0 = off")
    ap.add_argument("--pattern_weight", type=float, default=0.2,
                    help="loss: pattern knowledge heads - rhythm / shape / hand / flow / copies of every note, computed over the "
                         "whole map by the motif analysis (osu_mapper/hybrid/motif_targets.py); 0 = off")
    ap.add_argument("--prior_weight", type=float, default=0.2,
                    help="loss: distillation of the motif library - how 1606 real maps of the same star rating continue the "
                         "last 6 chords (needs --motifs); 0 = off")
    ap.add_argument("--motifs", default=None, help="motif library (mine_patterns.py; default checkpoints/hybrid/motifs.npz, "
                                                   "else test_NN/motifs.npz)")
    ap.add_argument("--dpo_weight", type=float, default=0.1,
                    help="preference optimization (DPO) on the pairs of make_self_maps.py (human map > the network's map, "
                         "best > worst candidate by the objective reward); used only when the dataset has pairs; 0 = off")
    ap.add_argument("--dpo_beta", type=float, default=0.1, help="DPO: strength of the pull to the frozen reference network")
    ap.add_argument("--dpo_every", type=int, default=4, help="DPO: one pair batch every N training steps")
    ap.add_argument("--plan_prob", type=float, default=0.5,
                    help="think first, then map: share of crops that get the per-bar technique plan (family + copy kind, "
                         "from the motif analysis) as input; the plan head always learns to propose it (needs --pattern_weight "
                         "> 0, which computes the plan targets); 0 = off")
    ap.add_argument("--plan_weight", type=float, default=0.2, help="loss: plan head")
    ap.add_argument("--corrupt_prob", type=float, default=0.3,
                    help="break and fix: share of canvas crops where almost the whole map is shown but partly broken (notes "
                         "removed / added / moved, wrong lanes, jacks, wrong chords, LN flips); the network predicts every frame "
                         "and its error head learns which shown frames are wrong (the self-correction critic); 0 = off")
    ap.add_argument("--error_weight", type=float, default=0.3, help="loss: error head of break-and-fix crops")
    ap.add_argument("--canvas_prob", type=float, default=0.5,
                    help="masked discrete diffusion: share of crops that get part of their own map as input (the rest are "
                         "fully masked = the plain network); adds the canvas input; 0 = off (see osu_mapper/hybrid/canvas.py)")
    ap.add_argument("--ladder_k", type=int, default=4, help="difficulty ladder: maps of the same song per training crop")
    ap.add_argument("--pair_prob", type=float, default=0.5, help="share of odd batch items drawn as a sibling difficulty of the previous item (needs an even --batch_size)")
    ap.add_argument("--derived_weight", type=float, default=0.3, help="share of training draws given to derived (thinned) maps, see make_derived_maps.py")
    ap.add_argument("--rate_aug", type=float, default=0.3,
                    help="share of training crops sped up x1.1..--rate_max together with their music (features resampled, star "
                         "rating measured by rosu-pp at that clock rate); rates that land in rare star bins are preferred. "
                         "The network saturated around 5* before (sr_diagnose.py); 0 = off")
    ap.add_argument("--rate_max", type=float, default=1.3,
                    help="fastest rate of --rate_aug (1.1 .. 1.5): beyond x1.3 a sped-up map drifts from how mappers snap at "
                         "that tempo; the hole in the hard bins is still filled")
    ap.add_argument("--sr_rare", type=float, default=0.5,
                    help="loss weight of a crop = (real maps in its 1-star bin)^-sr_rare (mean 1 per batch): rare, hard "
                         "difficulties weigh more; 0 = off")
    ap.add_argument("--finetune_min_sr", type=float, default=0.0,
                    help="train only on maps of at least this star rating (or that reach it sped up with --rate_aug): a short "
                         "second stage after the full training, e.g. --resume --finetune_min_sr 5 --epochs 3 --lr 5e-5")
    ap.add_argument("--sr_balance", type=float, default=0.5,
                    help="draw maps of rare star ratings more often: 0 = by length (default), 0.5 = sqrt-balanced, 1 = every 1-star bin equally")
    ap.add_argument("--hard_tilt", type=float, default=0.0,
                    help="draw harder maps more often on top of --sr_balance: weight x (1 + SR)^hard_tilt (1 = an 8* map ~3x as "
                         "often as a 2* map, 2 = ~9x); 0 = off")
    ap.add_argument("--real_only", action="store_true",
                    help="train only on the real maps of the dataset: derived (thinned, make_derived_maps.py) and self-generated "
                         "(make_self_maps.py) maps are left out, and so is DPO (its pairs come from network maps)")
    ap.add_argument("--pitch_shift", type=float, default=0.5,
                    help="share of training crops transposed by +-1..2 semitones in feature space (stops the network from "
                         "recognizing songs; 0 = off)")
    ap.add_argument("--global_dropout", type=float, default=None,
                    help="probability to hide the whole-song summary during training (default 0.2 from the config). The summary "
                         "identifies the song; 0.5 makes memorizing harder")
    ap.add_argument("--style_dropout", type=float, default=None, help="probability to hide the whole style vector (default 0.3)")
    ap.add_argument("--dropout", type=float, default=None, help="dropout inside the network (default 0.1; 0.2 for small datasets)")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop when the validation score did not improve for N epochs (0 = train all epochs); also with "
                         "several GPUs. hybrid_best.pt keeps the best epoch either way. Creating the file <checkpoint_dir>/STOP "
                         "ends training after the current epoch")
    ap.add_argument("--val_crops", type=int, default=256)
    ap.add_argument("--val_maps", type=int, default=24, help="complete validation maps rendered per epoch (per-ms metrics)")
    ap.add_argument("--val_sr_response", type=int, default=8,
                    help="validation maps generated at -1/0/+1 star around their SR to measure whether the model obeys the request (0 = off)")
    ap.add_argument("--response_weight", type=float, default=0.0,
                    help="best checkpoint score = F1@16 + this x SR-response slope (clipped to 0..1.5); e.g. 0.1")
    ap.add_argument("--val_style_known", action="store_true", help="give the reference style to the model during validation")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="auto", choices=["auto", "off", "bf16", "fp16"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", nargs="?", const="auto", default=None, help="resume from hybrid_last.pt (or a given path)")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_only", nargs="?", const="best", default=None, metavar="CHECKPOINT",
                    help="no training: run the full validation of a checkpoint (default hybrid_best.pt, or 'last', or a path)")
    ap.add_argument("--update_checkpoint", action="store_true",
                    help="with --eval_only: store the re-measured thresholds and calibration in the checkpoint")
    args = ap.parse_args()
    if args.eval_only:
        ck_dir = Path(args.checkpoint_dir)
        args.resume = str(ck_dir / "hybrid_best.pt") if args.eval_only == "best" else (
            str(ck_dir / "hybrid_last.pt") if args.eval_only == "last" else args.eval_only)

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
    if args.split is None:  # a resumed / evaluated checkpoint keeps the split it was trained with
        args.split = "piece"
        if args.resume:
            peek_path = (Path(args.checkpoint_dir) / "hybrid_last.pt") if args.resume == "auto" else Path(args.resume)
            if peek_path.exists():
                args.split = torch.load(peek_path, map_location="cpu", weights_only=False).get("args", {}).get("split") or "audio"
    train_idx, val_idx = index.split(args.val_fraction, args.seed, args.split)
    if not train_idx:
        sys.exit("No training maps (did every map fail the roundtrip check?)")
    val_idx = [i for i in val_idx if not index.maps[i].get("derived")]  # validation = real maps only
    if args.real_only:
        before = len(train_idx)
        train_idx = [i for i in train_idx if not index.maps[i].get("derived") and not index.maps[i].get("self")]
        print(f"[data] real maps only: {len(train_idx)} of {before} training maps (derived / self-generated left out)")
        if args.dpo_weight > 0:
            print("[dpo] off with --real_only (the pairs compare human maps with network maps)")
            args.dpo_weight = 0.0
        if not train_idx:
            sys.exit("--real_only: no real training maps")
    if args.finetune_min_sr > 0:
        reach = {}
        if args.rate_aug > 0:
            from osu_mapper.hybrid.dataset import rate_star_ratings
            reach = rate_star_ratings(index, train_idx)
        before = len(train_idx)
        train_idx = [i for i in train_idx if float(index.maps[i].get("sr", 0.0)) >= args.finetune_min_sr
                     or max(reach.get(index.maps[i]["track"], {}).values() or [0.0]) >= args.finetune_min_sr]
        print(f"[data] fine-tuning on the hard end: {len(train_idx)} of {before} training maps reach {args.finetune_min_sr:g}*")
        if not train_idx:
            sys.exit("--finetune_min_sr: no training map reaches that star rating")
    n_derived = sum(1 for i in train_idx if index.maps[i].get("derived"))
    print(f"[data] {len(index.maps)} maps, {len(index.songs)} songs -> train {len(train_idx)} maps"
          + (f" ({n_derived} derived, {100 * args.derived_weight:.0f}% of draws)" if n_derived else "")
          + f" / val {len(val_idx)} maps (split by {args.split})")
    sr_hist = np.bincount([int(np.clip(float(index.maps[i].get("sr", 0.0)), 0, 9.99)) for i in train_idx], minlength=10)
    print("[data] train maps per star rating: " + ", ".join(f"{b}-{b + 1}*: {int(c)}" for b, c in enumerate(sr_hist) if c)
          + (f" | sr_balance {args.sr_balance:g}" if args.sr_balance > 0 else "")
          + (f" | hard_tilt {args.hard_tilt:g}" if args.hard_tilt > 0 else ""))

    ckpt_dir = Path(args.checkpoint_dir)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_name = "eval.log" if args.eval_only else "train.log"
        sys.stdout = Tee(sys.stdout, open(ckpt_dir / log_name, "a", encoding="utf-8"))
        print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} | {' '.join(sys.argv)} =====")
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
        if FeatureConfig.from_dict(resume_ckpt.get("feature_config", {})).to_dict() != fcfg.to_dict():
            sys.exit("--resume/--eval_only: the dataset was prepared with a different feature config than the checkpoint "
                     "(e.g. with/without --external); use the dataset the model was trained on")
        cfg = HybridModelConfig.from_dict(resume_ckpt["model_config"])
        crop_frames = int(resume_ckpt.get("crop_frames", crop_frames))
        print(f"[resume] {rp} (epoch {resume_ckpt.get('epoch')}, step {resume_ckpt.get('step')})")
    else:
        cfg = build_config(fcfg.dims(), fcfg.total_dim(), args.preset)
    new_head = False
    if args.ladder_weight > 0 and not cfg.importance_head and not args.eval_only:
        cfg.importance_head = True
        new_head = resume_ckpt is not None
    if args.canvas_prob > 0 and not cfg.canvas and not args.eval_only:
        cfg.canvas = True
        new_head = new_head or resume_ckpt is not None
    if args.corrupt_prob > 0 and args.error_weight > 0 and args.canvas_prob > 0 and not cfg.error_head and not args.eval_only:
        cfg.error_head = True
        new_head = new_head or resume_ckpt is not None
    if args.rate_aug > 0 and not cfg.aug_tag and not args.eval_only:
        cfg.aug_tag = True          # sped-up crops are tagged; generation runs untagged = "like a real map"
        new_head = new_head or resume_ckpt is not None
    if args.plan_prob > 0 and args.pattern_weight > 0 and not cfg.plan and not args.eval_only:
        cfg.plan = True
        new_head = new_head or resume_ckpt is not None
    if args.pattern_weight > 0 and not cfg.pattern_heads and not args.eval_only:
        cfg.pattern_heads = True
        new_head = new_head or resume_ckpt is not None
    motif_lib = None
    if (args.pattern_weight > 0 or args.prior_weight > 0) and not args.eval_only:
        from osu_mapper.hybrid.motifs import load_default_motifs
        motif_lib = load_default_motifs(args.motifs, log=print, levels=("lanes",))
        if motif_lib is None and args.prior_weight > 0:
            print("[patterns] no motif library: the library prior (--prior_weight) is off, the pattern heads still train")
            args.prior_weight = 0.0

    if args.rep_dropout is not None:  # training-only setting: may be changed when resuming
        overrides = {}
        for item in args.rep_dropout:
            name, _, val = item.partition("=")
            if name not in cfg.rep_dims or not val:
                sys.exit(f"--rep_dropout: expected NAME=P with NAME in {list(cfg.rep_dims)}, got '{item}'")
            overrides[name] = float(val)
        cfg.rep_dropout_by_name = overrides
    for name in ("global_dropout", "style_dropout", "dropout"):  # training-only regularization, may change on --resume
        if getattr(args, name) is not None:
            setattr(cfg, name, float(getattr(args, name)))
    print(f"[model] regularization: dropout {cfg.dropout:g}, song summary dropout {cfg.global_dropout:g}, style dropout {cfg.style_dropout:g}, "
          f"pitch-shift augmentation {args.pitch_shift:g}")
    if cfg.rep_dropout_by_name:
        print(f"[model] modality dropout: {cfg.rep_dropout:g} default, " + ", ".join(f"{k}={v:g}" for k, v in cfg.rep_dropout_by_name.items()))
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
    calib = {"onset_shift_ms": 0, "ln_length_scale": 1.0}
    threshold_by_sr: Dict[int, float] = {}
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state"], strict=not new_head)
        if ema is not None and resume_ckpt.get("ema_state"):
            ema.load_state_dict(resume_ckpt["ema_state"], strict=not new_head)
        if new_head:
            # the parameter list changed: Adam moments of the old checkpoint do not line up any more
            print("[resume] added new parts (difficulty-ladder head / map canvas / pattern heads); optimizer state starts fresh (warm restart)")
        elif resume_ckpt.get("optimizer_state"):
            opt.load_state_dict(resume_ckpt["optimizer_state"])
        step = int(resume_ckpt.get("step", 0))
        start_epoch = int(resume_ckpt.get("epoch", 0))
        best_score = float(resume_ckpt.get("best_score", -float("inf")))
        history = list(resume_ckpt.get("history", []))
        decode_threshold = float(resume_ckpt.get("decode_threshold", 0.5))
        ln_threshold = float(resume_ckpt.get("ln_threshold", 0.5))
        calib = {"onset_shift_ms": int(resume_ckpt.get("onset_shift_ms", 0)), "ln_length_scale": float(resume_ckpt.get("ln_length_scale", 1.0))}
        threshold_by_sr = {int(k): float(v) for k, v in (resume_ckpt.get("decode_threshold_by_sr") or {}).items()}

    # wrap after loading the checkpoint; all ranks start from identical weights (DDP broadcasts rank 0)
    train_model = DistributedDataParallel(model, device_ids=[dev.index] if dev.type == "cuda" else None) if distributed else model

    weights = LossWeights(density=args.density_weight, pair=args.pair_weight, ladder=args.ladder_weight, jack=args.jack_weight,
                          patterns=args.pattern_weight, prior=args.prior_weight,
                          error=args.error_weight if args.corrupt_prob > 0 else 0.0,
                          plan=args.plan_weight if (args.plan_prob > 0 and args.pattern_weight > 0) else 0.0)
    auto_batch = str(args.batch_size).lower() == "auto"
    args.batch_size = 16 if auto_batch else int(args.batch_size)
    if args.pair_prob > 0 and args.batch_size % 2 and not auto_batch:
        print("[warn] --batch_size is odd: pairs disabled (use an even batch size)")
        args.pair_prob = 0.0
    total_steps = (start_epoch + args.epochs) * args.steps_per_epoch if resume_ckpt is None else max(int(resume_ckpt.get("total_steps", 0)), (start_epoch + args.epochs) * args.steps_per_epoch)
    print(f"[model] preset={args.preset if resume_ckpt is None else 'from checkpoint'} params={count_parameters(model):,} device={dev} amp={'on ' + str(amp_dtype) if use_amp else 'off'}")
    print(f"[train] window={crop_frames} frames ({crop_frames * fcfg.frame_ms / 1000:.2f} s), batch={args.batch_size} x {world} GPU(s) = {args.batch_size * world}, "
          f"steps/epoch={args.steps_per_epoch}, total steps={total_steps}")

    # every rank draws its own random crops (different seed), so no sampler is needed
    train_ds = HybridCropDataset(index, train_idx, crop_frames, args.steps_per_epoch * args.batch_size, train=True, seed=args.seed + 1000 * rank,
                                 sr_balance=args.sr_balance, derived_weight=args.derived_weight, pair_prob=args.pair_prob if args.pair_weight > 0 else 0.0,
                                 pitch_shift_prob=args.pitch_shift, ladder_k=args.ladder_k if args.ladder_weight > 0 else 0,
                                 canvas_prob=args.canvas_prob,
                                 pattern_targets=(args.pattern_weight > 0 or args.prior_weight > 0), motifs=motif_lib,
                                 rate_aug=args.rate_aug, sr_rare=args.sr_rare, rate_max=args.rate_max,
                                 corrupt_prob=args.corrupt_prob if args.canvas_prob > 0 else 0.0,
                                 plan_prob=args.plan_prob if args.pattern_weight > 0 else 0.0, hard_tilt=args.hard_tilt)
    draw_bins = np.array([int(np.clip(float(e.get("sr", 0.0)), 0, 9.99)) for e in train_ds.entries])
    draw_share = np.bincount(draw_bins, weights=train_ds.weights, minlength=10)
    print("[data] share of training draws per star rating: "
          + ", ".join(f"{b}-{b + 1}*: {100 * w:.0f}%" for b, w in enumerate(draw_share) if sr_hist[b] or w > 0))
    if auto_batch:
        from osu_mapper.hybrid.autobatch import find_batch_size, repeat_batch
        probe_items = [train_ds[i] for i in range(12)]
        frac = args.mem_fraction * (0.75 if (args.dpo_weight > 0 and index.data.get("dpo_pairs") and not distributed) else 1.0)

        def probe_step(bs: int) -> None:
            b = to_device(repeat_batch(probe_items, bs, collate_hybrid), dev)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                o = forward_batch(model, b)
            l, _ = compute_losses(o, b, weights, cfg.frame_ms, sigma_ms=args.sigma_start, event_weight=args.event_weight)
            l.backward()
            model.zero_grad(set_to_none=True)
            del o, l, b

        found_bs = find_batch_size(probe_step, model, dev, frac, distributed=distributed, log=print)
        if found_bs is None:
            print("[batch] auto needs a CUDA GPU: using 16")
        else:
            args.batch_size = found_bs
            train_ds.samples_per_epoch = args.steps_per_epoch * args.batch_size
        print(f"[train] batch={args.batch_size} x {world} GPU(s) = {args.batch_size * world}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_hybrid, drop_last=True)
    val_loader = None
    if val_idx:
        val_ds = HybridCropDataset(index, val_idx, crop_frames, args.val_crops, train=False, seed=args.seed,
                                   pattern_targets=(args.pattern_weight > 0 or args.prior_weight > 0), motifs=motif_lib)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_hybrid)
    # preference pairs (make_self_maps.py): DPO against a frozen copy of the network at the start of this run
    dpo_iter = ref_model = None
    dpo_pairs = [p for p in index.data.get("dpo_pairs", []) if p.get("song_id") in {index.maps[i]["song_id"] for i in train_idx}]
    if args.dpo_weight > 0 and dpo_pairs and distributed:
        # a second forward pass per step (the pair batch) does not fit DistributedDataParallel's one-pass gradient sync
        print("[dpo] off with several GPUs: run the DPO stage on one GPU (python train_hybrid.py --resume ... --dpo_weight 0.1)")
    elif args.dpo_weight > 0 and dpo_pairs and not args.eval_only:
        from osu_mapper.hybrid.dpo import DPOPairDataset
        dpo_ds = DPOPairDataset(index, dpo_pairs, crop_frames, 10 ** 9, seed=args.seed + 31 * rank)
        dpo_iter = iter(DataLoader(dpo_ds, batch_size=max(2, args.batch_size // 2), shuffle=False, num_workers=0,
                                   collate_fn=collate_hybrid))
        ref_model = copy.deepcopy(model).eval()
        for p_ in ref_model.parameters():
            p_.requires_grad_(False)
        print(f"[dpo] {len(dpo_pairs)} preference pairs, beta {args.dpo_beta:g}, one batch every {args.dpo_every} steps")
    rng = random.Random(args.seed)
    val_full = sorted(rng.sample(val_idx, min(args.val_maps, len(val_idx)))) if val_idx else []

    train_srs = [float(index.maps[i].get("sr", 0.0)) for i in train_idx if float(index.maps[i].get("sr", 0.0)) > 0]
    sr_range = [round(min(train_srs), 2), round(float(np.percentile(train_srs, 99)), 2), round(max(train_srs), 2)] if train_srs else None

    def payload(epoch: int, summary: Dict[str, float]) -> Dict[str, Any]:
        return {
            "format": CHECKPOINT_FORMAT,
            "param_version": PARAM_VERSION,
            "model_config": cfg.to_dict(),
            "feature_config": fcfg.to_dict(),
            "crop_frames": crop_frames,
            "sr_range": sr_range,
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict() if ema is not None else None,
            "optimizer_state": opt.state_dict(),
            "epoch": epoch,
            "step": step,
            "total_steps": total_steps,
            "best_score": best_score,
            "decode_threshold": decode_threshold,
            "decode_threshold_by_sr": threshold_by_sr,
            "ln_threshold": ln_threshold,
            "onset_shift_ms": calib["onset_shift_ms"],
            "ln_length_scale": calib["ln_length_scale"],
            "val_summary": summary,
            "history": history,
            "args": vars(args),
        }

    def evaluate(tag: str, epoch: int, train_sec: float) -> Tuple[Dict[str, Any], Dict[str, float]]:
        nonlocal decode_threshold, ln_threshold, calib, threshold_by_sr
        eval_model = ema if ema is not None else model
        record: Dict[str, Any] = {"epoch": epoch, "step": step, "train_sec": round(train_sec, 1)}
        summary: Dict[str, float] = {}
        if val_loader is not None:
            vc = validate_crops(eval_model, val_loader, dev, weights, cfg.frame_ms, args.sigma_end)
            record.update({f"val_{k}": v for k, v in vc.items()})
            print(f"[{tag}] val crops: total {vc['total']:.4f} | time {vc['time']:.4f} | lane {vc['lane']:.3f} | frame F1 {vc.get('frame_f1', 0):.3f} "
                  f"| offset exact {vc.get('offset_exact', 0):.3f} (+-1ms {vc.get('offset_within1ms', 0):.3f}) | lane acc {vc.get('lane_acc', 0):.3f} | chord acc {vc.get('chord_acc', 0):.3f}")
        if val_full:
            t1 = time.time()
            summary, decode_threshold, ln_threshold, threshold_by_sr = validate_full_maps(eval_model, index, val_full, crop_frames, dev, style_known=args.val_style_known)
            record.update({f"map_{k}": v for k, v in summary.items()})
            record["decode_threshold"] = decode_threshold
            record["ln_threshold"] = ln_threshold
            calib = {"onset_shift_ms": int(summary.get("onset_shift_ms", 0)), "ln_length_scale": float(summary.get("ln_length_scale", 1.0))}
            print(f"[{tag}] RENDERED MAPS vs reference ({len(val_full)} songs, thresholds event {decode_threshold} / LN {ln_threshold}, {time.time() - t1:.1f}s): "
                  f"per-ms loss {summary.get('per_ms_loss', float('nan')):.4f} | event F1 @0ms {summary.get('event_f1@0', 0):.3f} @2ms {summary.get('event_f1@2', 0):.3f} "
                  f"@16ms {summary.get('event_f1@16', 0):.3f} @64ms {summary.get('event_f1@64', 0):.3f} | timing MAE {summary.get('timing_mae_ms', float('nan')):.2f}ms "
                  f"| chord acc {summary.get('chord_size_acc', 0):.3f} | LN F1 {summary.get('ln_f1_at_best_ln_threshold', 0):.3f} | density x{summary.get('density_ratio', float('nan')):.2f} "
                  f"| SR error {summary.get('sr_mae', float('nan')):.2f}* (bias {summary.get('sr_bias', float('nan')):+.2f}) "
                  f"| start error median {summary.get('start_error_median_ms', float('nan')):+.1f}ms, LN length x{summary.get('ln_length_ratio_median', float('nan')):.2f} "
                  f"-> calibration shift {calib['onset_shift_ms']:+d}ms, LN scale {calib['ln_length_scale']:.2f}")
            print(f"[{tag}] surplus vs reference notes: duplicate events (<=30 ms from a hit) {summary.get('surplus_duplicate_events_pct', float('nan')):.2f}% "
                  f"| isolated extra events {summary.get('surplus_isolated_events_pct', float('nan')):.2f}% | extra chord notes {summary.get('surplus_chord_notes_pct', float('nan')):.2f}% "
                  f"|| missing: events {summary.get('missed_events_pct', float('nan')):.2f}%, chord notes {summary.get('missing_chord_notes_pct', float('nan')):.2f}%")
            if threshold_by_sr:
                print(f"[{tag}] event threshold by star rating: " + " ".join(f"{b}*={t:g}" for b, t in sorted(threshold_by_sr.items())))
            if args.val_sr_response > 0:
                t2 = time.time()
                resp = measure_sr_response(eval_model, index, val_full, crop_frames, dev, args.val_sr_response, decode_threshold, ln_threshold, threshold_by_sr)
                if resp:
                    summary.update({f"resp_{k}": v for k, v in resp.items()})
                    record.update({f"resp_{k}": v for k, v in resp.items()})
                    print(f"[{tag}] SR response ({time.time() - t2:.0f}s): {format_response(resp)}")
        return record, summary

    if args.eval_only:
        if not (val_loader or val_full):
            sys.exit("no validation songs in this dataset (check --val_fraction)")
        print(f"[eval] checkpoint {args.resume} (epoch {start_epoch}, step {step}), {len(val_full)} complete validation maps")
        record, summary = evaluate("eval", start_epoch, 0.0)
        out_json = ckpt_dir / f"eval_{Path(args.resume).stem}.json"
        out_json.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"[eval] metrics -> {out_json}")
        if args.update_checkpoint:
            ck = torch.load(args.resume, map_location="cpu", weights_only=False)
            ck.update({"decode_threshold": decode_threshold, "decode_threshold_by_sr": threshold_by_sr, "ln_threshold": ln_threshold,
                       "onset_shift_ms": calib["onset_shift_ms"], "ln_length_scale": calib["ln_length_scale"], "val_summary": summary})
            save_checkpoint(Path(args.resume), **ck)
            print(f"[eval] thresholds and calibration written into {args.resume}")
        return

    epoch = start_epoch
    best_epoch = start_epoch
    # Ctrl+C (SIGINT, also forwarded by torchrun) and SIGTERM: the first one interrupts training, later ones are ignored
    # so that the checkpoint is saved in one piece
    got_signal = {"n": 0}

    def _on_signal(signum, frame):
        got_signal["n"] += 1
        if got_signal["n"] == 1:
            raise KeyboardInterrupt
    import signal as _signal
    for _sig in (_signal.SIGINT, getattr(_signal, "SIGTERM", None)):
        if _sig is not None:
            try:
                _signal.signal(_sig, _on_signal)
            except (ValueError, OSError):
                pass
    stop_file = ckpt_dir / "STOP"
    if is_main:
        stop_file.unlink(missing_ok=True)
        print(f"[train] to stop after the current epoch (and go on with the next steps), create the file {stop_file}"
              + (f"; stops by itself after {args.patience} epochs without improvement" if args.patience > 0 else ""))
    try:
        for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
            train_ds.set_epoch(epoch)
            train_model.train()
            t0 = time.time()
            run: Dict[str, float] = {}
            n_run = 0
            run_n: Dict[str, int] = {}
            for batch in train_loader:
                batch = to_device(batch, dev)
                lr = lr_at(step, total_steps, args.warmup_steps, args.lr, args.min_lr)
                for g in opt.param_groups:
                    g["lr"] = lr
                sigma = sigma_at(step, total_steps, args.sigma_start, args.sigma_end)
                with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                    out = forward_batch(train_model, batch)
                loss, logs = compute_losses(out, batch, weights, cfg.frame_ms, sigma_ms=sigma, event_weight=args.event_weight)
                if dpo_iter is not None and step % max(1, args.dpo_every) == 0:
                    from osu_mapper.hybrid.dpo import dpo_loss, map_logprob
                    db = to_device(next(dpo_iter), dev)
                    nb = db["feats"].shape[0]
                    zst = torch.zeros(nb, cfg.style_dim, device=dev)
                    with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                        po = train_model(db["feats"], db["beat"], db["sr"], zst, zst, db["coarse"], db["coarse_mask"])
                        with torch.no_grad():
                            ro = ref_model(db["feats"], db["beat"], db["sr"], zst, zst, db["coarse"], db["coarse_mask"])
                    tc = {k[2:]: v for k, v in db.items() if k.startswith("c_")}
                    tr = {k[2:]: v for k, v in db.items() if k.startswith("r_")}
                    ld, acc = dpo_loss(map_logprob(po, tc, db["valid"]), map_logprob(po, tr, db["valid"]),
                                       map_logprob(ro, tc, db["valid"]), map_logprob(ro, tr, db["valid"]), args.dpo_beta)
                    loss = loss + args.dpo_weight * ld
                    logs["dpo"], logs["dpo_acc"] = float(ld.detach()), acc
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
                    run_n[k] = run_n.get(k, 0) + 1          # some terms (DPO) are not computed every step
                n_run += 1
                if step % args.log_every == 0:
                    avg = {k: v / max(1, run_n.get(k, n_run)) for k, v in run.items()}
                    print(f"  step {step:>7} | lr {lr:.2e} | sigma {sigma:.2f}ms | total {avg['total']:.4f} | time {avg['time']:.4f} lane {avg['lane']:.3f} chord {avg['chord']:.3f} "
                          f"stagger {avg['stagger']:.3f} ln {avg['ln_flag']:.3f}/{avg['ln_len']:.3f} style {avg['style']:.3f} "
                          f"density {avg['density']:.3f} pair {avg['pair']:.3f} ({avg['pairs']:.1f}/batch)"
                          + (f" ladder {avg['ladder']:.3f}" if "ladder" in avg else "")
                          + (f" jack {avg['jack']:.3f}" if "jack" in avg else "")
                          + (f" error {avg['error']:.3f}" if "error" in avg else "")
                          + (f" patterns {avg['patterns']:.3f}" if "patterns" in avg else "")
                          + (f" prior {avg['prior']:.3f}" if "prior" in avg else "")
                          + (f" plan {avg['plan']:.3f}" if "plan" in avg else "")
                          + (f" dpo {avg['dpo']:.3f} (acc {avg['dpo_acc']:.2f})" if "dpo" in avg else ""), flush=True)
                    run, n_run, run_n = {}, 0, {}

            stop = False
            if is_main:     # validation and checkpoints happen on rank 0; the others wait for its decision below
                record, summary = evaluate(f"epoch {epoch}", epoch, time.time() - t0)
                history.append(record)
                if summary:
                    score = float(summary.get("event_f1@16", 0.0))   # rendered-map quality on held-out songs
                    if args.response_weight > 0 and "resp_slope" in summary:
                        score += args.response_weight * float(np.clip(summary["resp_slope"], 0.0, 1.5))
                elif "val_total" in record:
                    score = -float(record["val_total"])
                else:
                    score = float(step)                              # no validation data: newest is best
                save_checkpoint(last_path, **payload(epoch, summary))
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    save_checkpoint(best_path, **payload(epoch, summary))
                    print(f"  * new best ({score:.4f}) -> {best_path}")
                elif args.patience > 0 and epoch - best_epoch >= args.patience:
                    print(f"[early stop] no improvement for {epoch - best_epoch} epochs (best {best_score:.4f} at epoch {best_epoch})")
                    stop = True
                if stop_file.exists():
                    print(f"[stop] {stop_file} found: training ends here (best {best_score:.4f} at epoch {best_epoch})")
                    stop_file.unlink(missing_ok=True)
                    stop = True
                with open(ckpt_dir / "history.json", "w", encoding="utf-8") as fh:
                    json.dump(history, fh, indent=1)
            if distributed:     # rank 0 decides for everyone (also acts as the barrier after validation)
                flag = torch.tensor([1 if stop else 0], device=dev)
                dist.broadcast(flag, 0)
                stop = bool(flag.item())
            if stop:
                break
    except (KeyboardInterrupt, RuntimeError) as exc:
        if isinstance(exc, RuntimeError):
            # with several GPUs another rank may leave on Ctrl+C first: this rank's collective then fails with
            # "connection closed" before its own signal arrives. Wait a moment for it; any other error is real.
            if not distributed:
                raise
            try:
                for _ in range(50):
                    if got_signal["n"]:
                        break
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            if not got_signal["n"]:
                raise
        if is_main:
            print(f"\n[paused] saving {last_path} (resume with: python train_hybrid.py --data {args.data} --resume); "
                  f"the best epoch so far is in {best_path}", flush=True)
            save_checkpoint(last_path, **payload(max(start_epoch, epoch - 1), {}))
            print("[paused] saved", flush=True)
        if distributed:
            # torchrun stops on Ctrl+C too, and with it the store the ranks talk through: a normal NCCL teardown would
            # wait for the other ranks and its monitor threads would print connection errors and keep the process
            # alive. Everything worth keeping is saved, so leave at once.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        return
    print(f"[done] best score {best_score:.4f}; checkpoints in {ckpt_dir}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
