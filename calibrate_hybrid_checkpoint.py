#!/usr/bin/env python3
"""
Measures and stores systematic timing offsets of an already trained hybrid checkpoint (no retraining).

    python calibrate_hybrid_checkpoint.py --checkpoint checkpoints/hybrid/hybrid_best.pt --data dataset_hybrid

Runs the model on the validation songs (same split as training), renders the maps and compares them with the
references: median start error of notes (positive = late, negative = early) and median ratio of long-note lengths
(> 1 = releases too late). Writes `onset_shift_ms` and `ln_length_scale` into the checkpoint (a .bak copy is kept);
generate_hybrid.py applies them automatically. Use --dry_run to only print the measurement.
"""

from __future__ import annotations

import argparse
import random
import shutil

import numpy as np
import torch

from osu_mapper.hybrid.audio_features import beat_features
from osu_mapper.hybrid.calibration import calibrate
from osu_mapper.hybrid.dataset import HybridIndex
from osu_mapper.hybrid.inference import HybridPredictor, SongInputs
from osu_mapper.hybrid.renderer import render_track


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/hybrid/hybrid_best.pt")
    ap.add_argument("--data", default="dataset_hybrid")
    ap.add_argument("--maps", type=int, default=40, help="validation maps to use")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    targs = ck.get("args", {})
    predictor = HybridPredictor.from_checkpoint(args.checkpoint, device=args.device)
    index = HybridIndex(args.data)
    _, val_idx = index.split(float(targs.get("val_fraction", 0.1)), int(targs.get("seed", 42)), targs.get("split") or "audio")
    if not val_idx:
        raise SystemExit("no validation songs in this dataset")
    chosen = sorted(random.Random(0).sample(val_idx, min(args.maps, len(val_idx))))
    items = []
    for i, mi in enumerate(chosen, 1):
        entry, track = index.maps[mi], index.track(mi)
        feats = np.asarray(index.features(entry["song_id"], 0), dtype=np.float32)
        F = feats.shape[0]
        inputs = SongInputs(feats, index.coarse(entry["song_id"]), beat_features(track.timing_points, 0, F, track.frame_ms), F * track.frame_ms)
        items.append((predictor.predict(inputs, float(entry.get("sr", 0.0))), render_track(track), track.timing_points))
        print(f"  [{i}/{len(chosen)}] {entry.get('version', '')}", flush=True)
    res = calibrate(items, predictor.decode_threshold, predictor.ln_threshold)
    print(f"\nmatched events {res['calib_events']}, matched long notes {res['calib_long_notes']}")
    print(f"note start error, median : {res.get('start_error_median_ms', float('nan')):+.1f} ms (negative = notes too early)")
    print(f"LN length ratio, median  : x{res.get('ln_length_ratio_median', float('nan')):.2f} (>1 = releases too late)")
    print(f"-> onset_shift_ms = {res['onset_shift_ms']:+d}, ln_length_scale = {res['ln_length_scale']:.3f}")
    if args.dry_run:
        return
    shutil.copy2(args.checkpoint, args.checkpoint + ".bak")
    ck["onset_shift_ms"] = int(res["onset_shift_ms"])
    ck["ln_length_scale"] = float(res["ln_length_scale"])
    ck["calibration"] = res
    torch.save(ck, args.checkpoint)
    print(f"written into {args.checkpoint} (backup: {args.checkpoint}.bak)")


if __name__ == "__main__":
    main()
