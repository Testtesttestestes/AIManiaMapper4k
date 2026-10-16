#!/usr/bin/env python3
"""
Measures how well a checkpoint obeys the requested star rating (no retraining).

    python hybrid_sr_response.py --checkpoint checkpoints/hybrid/hybrid_best.pt --data dataset_hybrid
    python hybrid_sr_response.py --guidance 1 2 3 --deltas -2 -1 0 1 2 --maps 30

Validation songs (same split as training) are generated at their reference star rating and at shifted
requests; the output is a few short rows of numbers:

    g=1  slope=0.31 mono=40% sr@-1=2.71 sr@+0=3.05 sr@+1=3.30 ref=3.40 f1=0.612 n=30

slope 1.0 = every requested star becomes one real star; 0 = the request is ignored. mono = share of maps
whose result gets harder with every step of the request. f1 = event F1 @16 ms at the reference SR.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from osu_mapper.hybrid.audio_features import beat_features
from osu_mapper.hybrid.dataset import HybridIndex
from osu_mapper.hybrid.inference import HybridPredictor, SongInputs
from osu_mapper.hybrid.renderer import render_track
from osu_mapper.hybrid.sr_response import format_response, sr_response


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/hybrid/hybrid_best.pt")
    ap.add_argument("--data", default="dataset_hybrid")
    ap.add_argument("--maps", type=int, default=30, help="validation maps (spread evenly over the star-rating range)")
    ap.add_argument("--deltas", type=float, nargs="+", default=[-1.0, 0.0, 1.0], help="requested SR = reference SR + delta")
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.0, 2.0], help="SR guidance values to compare")
    ap.add_argument("--threshold", type=float, default=None, help="fixed event threshold (default: checkpoint's, per star rating if calibrated)")
    ap.add_argument("--disable_reps", nargs="*", default=[], help="e.g. mert95m")
    ap.add_argument("--by_bin", action="store_true", help="also one row per 1-star bin of the references (first guidance value)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    targs = ck.get("args", {})
    predictor = HybridPredictor.from_checkpoint(args.checkpoint, device=args.device, disable_reps=args.disable_reps)
    index = HybridIndex(args.data)
    _, val_idx = index.split_by_song(float(targs.get("val_fraction", 0.1)), int(targs.get("seed", 42)))
    val_idx = [i for i in val_idx if not index.maps[i].get("derived") and float(index.maps[i].get("sr", 0.0)) > 0]
    if not val_idx:
        raise SystemExit("no validation maps with a star rating in this dataset")
    val_idx.sort(key=lambda i: float(index.maps[i]["sr"]))
    pick = sorted(set(np.linspace(0, len(val_idx) - 1, min(args.maps, len(val_idx))).round().astype(int).tolist()))
    chosen = [val_idx[p] for p in pick]

    items = []
    for mi in chosen:
        entry, track = index.maps[mi], index.track(mi)
        feats = np.asarray(index.features(entry["song_id"], 0), dtype=np.float32)
        F = feats.shape[0]
        inputs = SongInputs(feats, index.coarse(entry["song_id"]), beat_features(track.timing_points, 0, F, track.frame_ms), F * track.frame_ms)
        items.append((inputs, render_track(track), list(track.timing_points), float(entry["sr"])))
    srs = [it[3] for it in items]
    print(f"{len(items)} validation maps, reference SR {min(srs):.2f}..{max(srs):.2f}, deltas {args.deltas}"
          + (f", reps off: {' '.join(args.disable_reps)}" if args.disable_reps else ""))
    for g in args.guidance:
        res = sr_response(predictor, items, args.deltas, g, args.threshold,
                          progress=lambda a, b: print(f"  g={g:g}: {a}/{b}", end="\r", flush=True))
        print(" " * 40, end="\r")
        print(format_response(res, f"g={g:g}  "))
    if args.by_bin:
        g = args.guidance[0]
        bins = {}
        for it in items:
            bins.setdefault(int(it[3]), []).append(it)
        for b, its in sorted(bins.items()):
            print(format_response(sr_response(predictor, its, args.deltas, g, args.threshold), f"g={g:g} {b}*  "))


if __name__ == "__main__":
    main()
