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
import random
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
from osu_mapper.v5.infer import generate_map
from osu_mapper.v5.model import PRESETS, V5Config, V5Net, canvas_losses, count_parameters, v5_losses
from train_hybrid import init_distributed, lr_at


def save(path: Path, **payload: Any) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def to_dev(b: Dict[str, torch.Tensor], dev) -> Dict[str, torch.Tensor]:
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def losses_of(model_cfg, out, b, args):
    if model_cfg.decoder == "canvas":
        return canvas_losses(out, b, args.cause_weight, event_weight=args.event_weight, error_weight=args.error_weight,
                             regime_balance=args.regime_balance, seam_weight=args.seam_weight)
    return v5_losses(out, b, args.cause_weight, window_ms=model_cfg.window_ms, regime_balance=args.regime_balance)


@torch.no_grad()
def own_error_batch(model, b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Learn from the network's own errors (step-unrolled denoising, SUNDAE - Savinov et al. 2021): the network first
    restores the hidden frames of the batch itself; its own version becomes the visible canvas, the target stays the
    human map, every frame counts, and the error head is told exactly which frames the network got wrong - the real
    errors of inference, not only the synthetic damage. ``model``: the network itself, not the DDP wrapper (a pass of
    the wrapper syncs buffers between the GPUs, and only some steps make this extra pass)."""
    from osu_mapper.v5 import frames as fr

    out = model(b)["canvas"]
    hid = b["cv_hidden"].bool()                                                  # [B, T]
    st = torch.distributions.Categorical(logits=out["state"].float()).sample()   # [B, T, 4]
    a = out["start"].float().argmax(-1)
    e = out["end"].float().argmax(-1)
    no = torch.full_like(st, fr.NO_OFF)
    st_new = torch.where(hid[..., None], st, b["cv_state"])
    a_new = torch.where(hid[..., None], torch.where((st == fr.TAP) | (st == fr.HEAD), a, no), b["cv_start"])
    e_new = torch.where(hid[..., None], torch.where(st == fr.TAIL, e, no), b["cv_end"])
    tgt_a = torch.where(b["cv_tgt_start"] >= 0, b["cv_tgt_start"], torch.full_like(b["cv_tgt_start"], fr.NO_OFF))
    tgt_e = torch.where(b["cv_tgt_end"] >= 0, b["cv_tgt_end"], torch.full_like(b["cv_tgt_end"], fr.NO_OFF))
    wrong = ((st_new != b["cv_tgt_state"]) | (a_new != tgt_a) | (e_new != tgt_e)).any(-1)
    nb = dict(b)
    nb.update(cv_state=st_new, cv_start=a_new, cv_end=e_new, cv_hidden=torch.zeros_like(b["cv_hidden"]),
              cv_loss=torch.ones_like(b["cv_loss"]), cv_err=wrong.float(), cv_err_mask=torch.ones_like(b["cv_err_mask"]))
    return nb


@torch.no_grad()
def seam_batch(model, b: Dict[str, torch.Tensor], steps: int = 4, seed: int = 0):
    """Learn the overlap zones of generation (the seam head). Generation windows overlap by half: the earlier window
    places the end of its new half without seeing what follows; the later one proposes its own version of that zone
    with the continuation in view, and where they differ the seam head decides. Here both happen on a human map, as in
    generation (eval mode, the zone revealed in ``steps`` rounds): the earlier window [-W/2, W/2) (first half the human
    map, second half revealed) gives version A of [0, W/2); the later window [0, W) (its first quarter = A, as locked
    in generation, the rest revealed) gives version B of [W/4, W). The example: A on [0, W/4), the zone [W/4, W/2) = A
    or B (half the items each), B after it; the target is the human map, the error labels are the real errors, and
    the seam head learns them on the zone. ``model``: the network itself (no DDP pass). Returns (batch, stats)."""
    from osu_mapper.v5 import frames as fr
    from osu_mapper.v5.canvas_gen import restore_window

    was = model.training
    model.eval()
    try:
        B, W = b["cv_hidden"].shape
        Z, H2 = W // 4, W // 2
        dev = b["feats"].device
        rng = np.random.default_rng(seed)
        cond = model.condition(b["sr"], b["style"], b["style_mask"], None, b.get("aug"))
        npi = lambda t: t.cpu().numpy().astype(np.int8)

        def restore(feats, beat, st, so, eo, hidden):
            mem = model.encode(feats, beat)["memory"]
            return restore_window(model, mem, cond, cond, st, so, eo, hidden, steps=steps, temperature=1.0, cfg_scale=1.0,
                                  rng=rng)

        # 1. the earlier window: its second half revealed without anything after it
        hid = np.zeros((B, W), dtype=bool)
        hid[:, H2:] = True
        sa, oa, ea = restore(torch.cat([b["seam_feats"], b["feats"][:, :H2]], 1),
                             torch.cat([b["seam_beat"], b["beat"][:, :H2]], 1),
                             npi(b["seam_state"]), npi(b["seam_start"]), npi(b["seam_end"]), hid)
        A = (sa[:, H2:], oa[:, H2:], ea[:, H2:])                                  # frames [0, W/2) of this window
        # 2. the later window (this one): its first quarter is A, the zone and the rest revealed with all in view
        gt = (npi(b["cv_tgt_state"]), npi(b["cv_tgt_start"]), npi(b["cv_tgt_end"]))
        later = [g.copy() for g in gt]
        for x, a in zip(later, A):
            x[:, :Z] = a[:, :Z]
        hid = np.zeros((B, W), dtype=bool)
        hid[:, Z:] = True
        sb, ob, eb = restore(b["feats"], b["beat"], *later, hid)
        # 3. the example: A before the zone, A or B on the zone, B after it
        x = [sb.copy(), ob.copy(), eb.copy()]
        use_a = rng.random(B) < 0.5
        for xi, a in zip(x, A):
            xi[use_a, Z:H2] = a[use_a, Z:H2]
        differs = lambda p, q, sl: ((p[0][:, sl] != q[0][:, sl]) | (p[1][:, sl] != q[1][:, sl])
                                    | (p[2][:, sl] != q[2][:, sl])).any(-1)
        zone_sl = slice(Z, H2)
        wrong = differs(x, gt, slice(None))
        stats = {"seam_differ": float(differs(A, (sb, ob, eb), zone_sl).mean()),
                 "seam_wrong_a": float(differs(A, gt, zone_sl).mean()),
                 "seam_wrong_b": float(differs((sb, ob, eb), gt, zone_sl).mean())}
        none = np.zeros(W, dtype=bool)
        s_in, a_in, e_in = (np.stack(v) for v in zip(*[fr.inputs(x[0][r], x[1][r], x[2][r], none) for r in range(B)]))
        zone = np.zeros((B, W), dtype=np.float32)
        zone[:, Z:H2] = 1.0
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
        nb = {k: v for k, v in b.items() if not k.startswith("seam_")}
        nb.update(cv_state=T(s_in), cv_start=T(a_in), cv_end=T(e_in), cv_hidden=torch.zeros_like(b["cv_hidden"]),
                  cv_loss=torch.ones_like(b["cv_loss"]), cv_err=T(wrong.astype(np.float32)),
                  cv_err_mask=torch.ones_like(b["cv_err_mask"]), cv_seam_mask=T(zone))
        return nb, stats
    finally:
        model.train(was)


@torch.no_grad()
def val_tokens(model: V5Net, loader, dev, args=None) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    n = 0
    for b in loader:
        b = to_dev(b, dev)
        out = model(b)
        if model.cfg.decoder == "canvas":
            _, logs = canvas_losses(out, b)
        else:
            _, logs = v5_losses(out, b, window_ms=model.cfg.window_ms)
        for k, v in logs.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
    return {k: v / max(1, n) for k, v in sums.items()}


@torch.no_grad()
def val_style(entry: Dict[str, Any], keys: str):
    """Style given to the network for a validation map: 'ln' = only the share of long notes of the human map (a
    mapper's choice that the audio does not decide), 'all' = the whole style, 'none' = nothing."""
    from osu_mapper.hybrid.solver import STYLE_KEYS
    st = np.asarray(entry["style"], dtype=np.float32)
    known = np.asarray(entry.get("style_mask", [1.0] * len(st)), dtype=np.float32)
    m = np.zeros_like(st)
    if keys == "all":
        m = known
    elif keys == "ln" and "ln_ratio" in STYLE_KEYS:
        m[STYLE_KEYS.index("ln_ratio")] = known[STYLE_KEYS.index("ln_ratio")]
    return st, m


def val_maps(model: V5Net, index: HybridIndex, cache: SongCache, map_idx: List[int], dev, temperature: float,
             cfg_scale: float, max_sec: float, style_keys: str = "ln", passes: int = 1, canvas_steps: int = 8,
             refine: int = 1) -> Dict[str, float]:
    reps = []
    seams: Dict[str, int] = {}
    for i in map_idx:
        e = index.maps[i]
        tr = index.track(i)
        ref = render_track(tr)
        fa = song_inputs(np.asarray(index.features(e["song_id"], 0)), cache.slices, tr.timing_points,
                         sections="section" in model.cfg.groups, highlight=model.cfg.highlight)
        F = min(fa.shape[0], int(max_sec * 100)) if max_sec > 0 else fa.shape[0]
        fa = fa[:F]
        ba = beat_features(tr.timing_points, 0, F, 10)
        st, sm = val_style(e, style_keys)
        notes = generate_map(model, fa, ba, tr.timing_points, float(e.get("sr", 0.0)), passes=passes, style=st,
                             style_mask=sm, temperature=temperature, cfg_scale=cfg_scale, device=dev,
                             steps=canvas_steps, refine_passes=refine, stats=seams)
        ref = [n for n in ref if n.start < F * 10]
        rep = per_ms_report(notes, ref, timing_points=tr.timing_points)
        rep["expressive"] = expressive_score(rep)
        reps.append(rep)
    out = summarize_reports(reps)
    # ratios (burst length, density) of single songs can be huge (a song with almost no human rice and one long
    # stream of the network): the median over the songs is what is printed, not the mean
    for k in ("burst_ms_ratio", "density_ratio"):
        vals = np.array([r.get(k, np.nan) for r in reps], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[k] = float(np.median(vals))
    out.update({"overlap_" + k: float(v) for k, v in seams.items()})
    return out


def fit_climaxes(index: HybridIndex, train_idx: List[int], val_idx: List[int], slices, cache_path: Path) -> List[float]:
    """Climax model of the song parts from the mappers' kiai time (osu_mapper/v5/sections.py), kept only when it beats
    "the loud parts are the climaxes" on the validation maps; [] = the loudness rule."""
    import hashlib
    from osu_mapper.v5.sections import fit_highlight

    key = hashlib.sha1(json.dumps(sorted(index.maps[i]["track"] for i in train_idx)).encode()).hexdigest()
    try:
        d = json.loads(cache_path.read_text(encoding="utf-8"))
        if d.get("key") == key:
            print(f"[sections] climax model from cache: {d['note']}")
            return list(d["weights"])
    except (OSError, ValueError, KeyError):
        pass

    def samples(idx):
        """Parts are found once per song (red lines of its first map); every map with kiai labels them."""
        from osu_mapper.v5.sections import descriptor, find_sections, kiai_frames
        by_song: Dict[str, List[int]] = {}
        for i in idx:
            by_song.setdefault(index.maps[i]["song_id"], []).append(i)
        out = []
        for sid, ids in by_song.items():
            tps = [list(index.track(i).timing_points) for i in ids]
            kiais = [any(int(getattr(tp, "effects", 0)) & 1 for tp in t) for t in tps]
            if not any(kiais):
                continue
            feats = np.asarray(index.features(sid, 0))
            secs = find_sections(feats, slices, tps[0])
            for t, has in zip(tps, kiais):
                if has:
                    k = kiai_frames(t, feats.shape[0])
                    out += [(descriptor(s_), float(k[s_.start:s_.end].mean())) for s_ in secs]
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


def fit_ln_share(index: HybridIndex, train_idx: List[int], val_idx: List[int], slices, cache_path: Path) -> List[float]:
    """LN share of a map from its music and star rating (osu_mapper/v5/lnstyle.py), kept only when it beats the
    dataset's average share on the validation maps; [] = leave the share to the network when none is given."""
    import hashlib
    from osu_mapper.hybrid.solver import STYLE_KEYS
    from osu_mapper.v5.lnstyle import fit, predict, song_descriptor

    key = hashlib.sha1(json.dumps(sorted(index.maps[i]["track"] for i in train_idx)).encode()).hexdigest()
    try:
        d = json.loads(cache_path.read_text(encoding="utf-8"))
        if d.get("key") == key:
            print(f"[style] LN share model from cache: {d['note']}")
            return list(d["weights"])
    except (OSError, ValueError, KeyError):
        pass
    j = STYLE_KEYS.index("ln_ratio")

    def samples(idx):
        by_song: Dict[str, List[int]] = {}
        for i in idx:
            by_song.setdefault(index.maps[i]["song_id"], []).append(i)
        out = []
        for sid, ids in by_song.items():
            feats = np.asarray(index.features(sid, 0))
            for i in ids:
                e = index.maps[i]
                out.append((song_descriptor(feats, slices, index.track(i).timing_points, float(e.get("sr", 0.0))),
                            float(e["style"][j])))
        return out
    t0 = time.time()
    tr_s, va_s = samples(train_idx), samples(val_idx)
    weights: List[float] = []
    note = "too few maps"
    if len(tr_s) >= 20:
        w = fit(tr_s)
        mean = float(np.mean([y for _, y in tr_s]))
        if va_s:
            e_m = float(np.mean([abs(predict(w, x) - y) for x, y in va_s]))
            e_b = float(np.mean([abs(mean - y) for _, y in va_s]))
            keep = e_m < 0.95 * e_b
            note = (f"LN share from the audio: error {e_m:.3f} on validation maps, dataset average {e_b:.3f} -> "
                    + ("model used when no --style ln_ratio is given" if keep else "not used (the network decides)"))
            weights = w if keep else []
        else:
            weights, note = w, "LN share from the audio (no validation maps)"
    print(f"[style] {note} ({len(tr_s)} maps, {time.time() - t0:.0f}s)")
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
    ap.add_argument("--mem_fraction", type=float, default=0.85,
                    help="share of the card the auto batch size aims at (windows differ in length: keep a margin)")
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
    ap.add_argument("--val_maps", type=int, default=16,
                    help="complete validation maps generated per epoch (more = steadier numbers, slower validation)")
    ap.add_argument("--val_max_sec", type=float, default=120.0, help="first N seconds of each validation map (0 = all)")
    ap.add_argument("--expressive_weight", type=float, default=0.3)
    ap.add_argument("--val_style", default="ln", choices=["ln", "all", "none"],
                    help="what the network is told about each validation map: ln = its share of long notes (an LN map or "
                         "a rice map of the same song is the mapper's choice, the audio does not decide it)")
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
    ap.add_argument("--decoder", default="canvas", choices=["canvas", "ar"],
                    help="canvas (default): the map as frames x lanes, hidden parts restored with everything else in view "
                         "(as the v4 canvas); ar: the older autoregressive event decoder")
    ap.add_argument("--mirror", type=float, default=0.5, help="canvas: share of windows with mirrored lanes (1234 -> 4321)")
    ap.add_argument("--rotate", type=float, default=0.25,
                    help="canvas: share of windows with rotated lanes (1234 -> 4123 / 3412 / 2341), tagged in the condition "
                         "(generation never rotates); star ratings are the same")
    ap.add_argument("--shift", type=float, default=0.5,
                    help="canvas: share of windows moved by 1-9 ms inside the frame (features interpolated, notes moved)")
    ap.add_argument("--self_correct", type=float, default=0.25,
                    help="canvas: share of steps that learn from the network's own errors: it restores the hidden frames "
                         "itself first, then fixes its own version (the error head gets the real errors)")
    ap.add_argument("--seam", type=float, default=0.1,
                    help="canvas: share of steps that learn the overlap zones of generation windows: the network maps a "
                         "window's end without seeing what follows and again with it in view; the seam head learns which "
                         "version is wrong where (generation: it decides differing overlaps). Slower steps; 0 = off")
    ap.add_argument("--seam_steps", type=int, default=4, help="canvas: rounds of revealing in the seam steps")
    ap.add_argument("--seam_weight", type=float, default=0.3, help="canvas: loss of the seam head")
    ap.add_argument("--event_weight", type=float, default=4.0, help="canvas: weight of frames with a note in the loss")
    ap.add_argument("--error_weight", type=float, default=0.3, help="canvas: loss of the error head")
    ap.add_argument("--canvas_steps", type=int, default=8, help="canvas: rounds of revealing a window (validation maps)")
    ap.add_argument("--refine", type=int, default=1, help="canvas: self-correction passes of the validation maps")
    ap.add_argument("--draft_prob", type=float, default=0.5,
                    help="share of training windows where the encoder also sees a damaged draft of the map (bursts made "
                         "1/2, notes moved / missing / extra, long notes broken, spans cut out) and maps it again: the "
                         "re-mapping passes of generation (--passes); 0 = off. Added to an older checkpoint on --resume")
    ap.add_argument("--passes", type=int, default=2,
                    help="validation maps: 1 = one pass, 2+ = draft, then mapped again with the whole draft in view")
    ap.add_argument("--regime_balance", type=float, default=0.01,
                    help="load-balancing loss of the regimes (Switch Transformer): keeps one regime from taking every part")
    ap.add_argument("--no_ln_model", action="store_true",
                    help="do not fit the model of the LN share from the audio (osu_mapper/v5/lnstyle.py)")
    ap.add_argument("--no_sections", action="store_true",
                    help="do not give the network the parts of the song (intro / verse / chorus / drop / bridge / solo / "
                         "arpeggio ..., osu_mapper/v5/sections.py)")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # windows of very different lengths
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
                       decoder=args.decoder, **PRESETS[args.preset])
        if not args.no_sections:
            # one process fits the climax model (minutes of CPU work), the others wait and read its cached result
            cache_file = Path(args.data) / "v5_climaxes.json"
            if is_main:
                cfg.highlight = fit_climaxes(index, train_idx, val_idx, slices, cache_file)
            if distributed:
                dist.barrier()
            if not is_main:
                cfg.highlight = fit_climaxes(index, train_idx, val_idx, slices, cache_file)
    if not cfg.ln_model and not args.no_ln_model:
        # LN share from the audio, for generation without --style ln_ratio (also on --resume: the weights are not touched)
        ln_cache = Path(args.data) / "v5_ln_share.json"
        if is_main:
            cfg.ln_model = fit_ln_share(index, train_idx, val_idx, slices, ln_cache)
        if distributed:
            dist.barrier()
        if not is_main:
            cfg.ln_model = fit_ln_share(index, train_idx, val_idx, slices, ln_cache)
    if group_columns(slices, sections="section" in cfg.groups) != cfg.groups:
        sys.exit("the dataset's features differ from the checkpoint's (e.g. other --external): use the same dataset")
    new_parts = False
    if args.draft_prob > 0 and not cfg.draft and cfg.decoder == "ar":
        cfg.draft = True                     # re-mapping input added to a checkpoint made without it (starts at zero)
        new_parts = resume is not None
    model = V5Net(cfg).to(dev)
    if resume:
        missing, unexpected = model.load_state_dict(resume["model"], strict=False)
        if unexpected or any(not k.startswith("draft_in.") for k in missing):
            sys.exit(f"the checkpoint does not fit this network: missing {missing[:5]}, unexpected {unexpected[:5]}")
        if missing:
            print("[resume] added the draft input for the re-mapping passes (starts at zero: the network maps as before "
                  "and learns to use the draft)")
    elif cfg.decoder == "ar":
        gaps = []
        for i in train_idx[:2000]:
            t = np.unique([n.start // cfg.frame_ms for n in render_track(index.track(i))])
            gaps.append(np.diff(t))
        model.init_distance_prior(np.concatenate(gaps) if gaps else [10])
    ema = copy.deepcopy(model).eval() if args.ema > 0 else None
    if resume and ema is not None and resume.get("ema"):
        ema.load_state_dict(resume["ema"], strict=False)
    print(f"[model] v5 {args.preset if not resume else 'from checkpoint'}: {count_parameters(model):,} parameters, "
          f"{len(cfg.groups) + 1} input groups, device {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    if resume and resume.get("opt") and not args.finetune and not new_parts:
        opt.load_state_dict(resume["opt"])
    elif resume and resume.get("opt") and not args.finetune and new_parts:
        # a part was added: the Adam statistics of every old weight are carried over by name (a fresh optimizer at the
        # full learning rate would throw a trained network back), only the new part starts from zero
        old_names = [n for n, _ in model.named_parameters() if not n.startswith("draft_in.")]
        by_name = dict(model.named_parameters())
        old_state = resume["opt"].get("state", {})
        moved = 0
        for i, n in enumerate(old_names):
            st_ = old_state.get(i)
            if st_ is not None and n in by_name:
                p_ = by_name[n]
                opt.state[p_] = {k: (v.to(p_.device) if torch.is_tensor(v) and v.dim() > 0 else v) for k, v in st_.items()}
                moved += 1
        print(f"[resume] optimizer state of {moved} weights carried over")
    use_amp = dev.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    vocab = cfg.vocab()
    W = cfg.window_frames
    auto = str(args.batch_size).lower() == "auto"
    bs = 8 if auto else int(args.batch_size)
    train_ds = V5Dataset(index, train_idx, vocab, W, args.steps_per_epoch * 512, seed=args.seed + 1000 * rank,
                         sr_balance=args.sr_balance, hard_tilt=args.hard_tilt, hard_share=args.hard_share,
                         max_tokens=cfg.max_tokens, sections="section" in cfg.groups, highlight=cfg.highlight,
                         draft_prob=args.draft_prob if cfg.draft else 0.0, decoder=cfg.decoder,
                         mirror=args.mirror, rotate=args.rotate, shift=args.shift,
                         seam=args.seam if cfg.decoder == "canvas" and getattr(cfg, "seam_head", False) else 0.0,
                         sync_seed=args.seed)
    if auto:
        from osu_mapper.hybrid.autobatch import find_batch_size, repeat_batch
        # the memory of a step grows with the token count: probe with the longest of 64 random windows
        cand = [train_ds[i] for i in range(128 if cfg.decoder == "ar" else 8)]
        probe = sorted(cand, key=lambda it: -int(it["tokens"].shape[0]))[:4] if cfg.decoder == "ar" else cand[:4]

        def probe_step(n: int) -> None:
            b = to_dev(repeat_batch(probe, n, collate), dev)
            model.train()
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                out = model(b)
                loss, _ = losses_of(cfg, out, b, args)
            loss.backward()
            model.zero_grad(set_to_none=True)
        ema_bytes = sum(p.numel() * p.element_size() for p in ema.parameters()) if ema is not None else 0
        found = find_batch_size(probe_step, model, dev, args.mem_fraction, distributed=distributed, log=print,
                                extra_bytes=ema_bytes)
        bs = found or 8
    train_ds.samples = args.steps_per_epoch * bs
    train_ds.batch_size = bs                       # seam batches are decided per batch (the same on every GPU)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=args.workers, collate_fn=collate,
                        drop_last=True, persistent_workers=False)
    val_loader = None
    if val_idx and is_main:
        fixed = V5Dataset.fixed_windows(index, val_idx, W, args.val_windows, seed=args.seed)
        val_ds = V5Dataset(index, val_idx, vocab, W, 0, train=False, fixed=fixed, sr_balance=0, max_tokens=cfg.max_tokens,
                           sections="section" in cfg.groups, highlight=cfg.highlight, decoder=cfg.decoder)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate)
    val_full = sorted(val_idx, key=lambda i: float(index.maps[i].get("sr", 0.0)))
    if val_full and args.val_maps > 0:
        pick = np.unique(np.linspace(0, len(val_full) - 1, min(args.val_maps, len(val_full))).round().astype(int))
        val_full = [val_full[p] for p in pick]
    cache = SongCache(index, sections="section" in cfg.groups, highlight=cfg.highlight)

    sync_rng = random.Random(args.seed * 7 + 3)          # the same on every GPU
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
            cnt: Dict[str, int] = {}
            nrun = 0
            for b in loader:
                b = to_dev(b, dev)
                lr = lr_at(step - sched0, total, warm, lr_base, args.min_lr if not args.finetune else lr_base / 10)
                for g in opt.param_groups:
                    g["lr"] = lr
                # the same decision on every GPU (a seed they share): the slower steps happen together
                seam = "seam_feats" in b
                own = cfg.decoder == "canvas" and sync_rng.random() < args.self_correct and not seam
                sstats = None
                with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                    if seam:
                        b, sstats = seam_batch(model, b, steps=args.seam_steps, seed=args.seed * 7919 + step)
                    elif own:
                        b = own_error_batch(model, b)
                    out = train_model(b)
                    loss, logs = losses_of(cfg, out, b, args)
                if own or seam:
                    pre = "own_" if own else "seamstep_"
                    logs = {(pre + k if k in ("state", "error", "note_f1") else k): v for k, v in logs.items()}
                if sstats:
                    logs.update(sstats)
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
                    cnt[k] = cnt.get(k, 0) + 1                  # own / seam numbers come only from their own steps
                nrun += 1
                if step % args.log_every == 0:
                    a = {k: v / max(1, cnt.get(k, 0)) for k, v in run.items()}
                    if cfg.decoder == "canvas":
                        print(f"  step {step:>7} | lr {lr:.2e} | hidden frames: state {a.get('state', float('nan')):.4f} "
                              f"notes F1 {a.get('note_f1', float('nan')):.3f} | offsets {a.get('start', float('nan')):.3f}/"
                              f"{a.get('end', float('nan')):.3f} | error head {a.get('error', float('nan')):.3f} | own errors: "
                              f"state {a.get('own_state', float('nan')):.4f} error head {a.get('own_error', float('nan')):.3f}"
                              + (f" | seams: versions differ on {100 * a['seam_differ']:.0f}% of the zone, wrong: earlier "
                                 f"{100 * a['seam_wrong_a']:.0f}% later {100 * a['seam_wrong_b']:.0f}%, seam head "
                                 f"{a.get('seam', float('nan')):.3f} (constant guess {a.get('seam_base', float('nan')):.3f})"
                                 if "seam_differ" in a else ""), flush=True)
                    else:
                        print(f"  step {step:>7} | lr {lr:.2e} | tokens {a['tokens']:.4f} (time {a.get('time_tokens', float('nan')):.3f}) "
                              f"| cause {a.get('cause', float('nan')):.3f} acc {a.get('cause_acc', float('nan')):.2f}", flush=True)
                    run, cnt, nrun = {}, {}, 0
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
                    if cfg.decoder == "canvas":
                        print(f"[epoch {epoch}] val windows: hidden frames state {vt.get('state', float('nan')):.4f}, notes F1 "
                              f"{vt.get('note_f1', float('nan')):.3f} | error head {vt.get('error', float('nan')):.3f}")
                    else:
                        print(f"[epoch {epoch}] val windows: tokens {vt.get('tokens', float('nan')):.4f} (time {vt.get('time_tokens', float('nan')):.3f}) "
                              f"| cause acc {vt.get('cause_acc', float('nan')):.3f}")
                if val_full:
                    t1 = time.time()
                    summary = val_maps(em, index, cache, val_full, dev, args.temperature, args.cfg_scale, args.val_max_sec,
                                       args.val_style, args.passes if cfg.draft else 1, args.canvas_steps, args.refine)
                    rec.update({f"map_{k}": v for k, v in summary.items()})
                    print(f"[epoch {epoch}] GENERATED MAPS vs human ({len(val_full)} songs, {time.time() - t1:.0f}s): event F1 "
                          f"@16ms {summary.get('event_f1@16', 0):.3f} @64ms {summary.get('event_f1@64', 0):.3f} | density "
                          f"x{summary.get('density_ratio', float('nan')):.2f} | expressiveness {summary.get('expressive', 0):.3f}: "
                          f"bursts F1 {summary.get('burst_f1', 0):.3f} (length x{summary.get('burst_ms_ratio', float('nan')):.2f}) "
                          f"| rhythm distance {summary.get('subdiv_tv', float('nan')):.3f} | 1/4 and finer "
                          f"{100 * summary.get('fine_share_pred', 0):.0f}% vs {100 * summary.get('fine_share_ref', 0):.0f}% "
                          f"| density per bar r={summary.get('bar_density_corr', 0):.2f}")
                    print(f"[epoch {epoch}] long notes: {100 * summary.get('ln_share_pred', 0):.0f}% of notes vs "
                          f"{100 * summary.get('ln_share_ref', 0):.0f}% | LN starts F1 {summary.get('ln_start_f1', 0):.3f} "
                          f"| LN lengths distance {summary.get('ln_len_tv', float('nan')):.3f} | LN F1 {summary.get('ln_f1', 0):.3f}")
                    if summary.get("overlap_windows", 0) > 0:
                        judge = "seam head" if getattr(em, "seam_head", None) is not None else "error head"
                        print(f"[epoch {epoch}] overlaps of windows: {summary['overlap_differed']:.0f} of "
                              f"{summary['overlap_windows']:.0f} differed; differing stretches decided by the {judge}: kept "
                              f"the earlier {summary['overlap_kept_earlier']:.0f}, took the later {summary['overlap_took_later']:.0f}")
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
