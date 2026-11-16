#!/usr/bin/env python3
"""
Calibrates how the v5 canvas reveals long notes, on the validation songs of the training split.

When a window is revealed in rounds, a frame of a lane the network thinks may be held (HOLD / TAIL likely) but whose
long note has not started yet could be committed empty early - and then the long note can no longer pass through it,
nor start before it: long notes get cut and mostly vanish (validation showed 2-3 % long notes against 21 % in the
human maps). With ``hold_wait`` such a lane waits for the frame before it. How likely is "may be held" depends on the
network, so this script tries several values on the validation songs, prints long-note share, F1 and expressiveness
for each, and stores the best (F1@16 + 0.3 x expressiveness, as the training picks checkpoints) in the checkpoint:
generate_v5.py then uses it (--hold_wait overrides).

    python calibrate_v5.py --data dataset_hybrid --checkpoint checkpoints/v5/v5_best.pt
    python calibrate_v5.py ... --values 0,0.5,0.3,0.2,0.1 --val_maps 16 --val_max_sec 60 --no_write

Run it when training is over (or on a copy: the training writes v5_best.pt again when it finds a better epoch).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from osu_mapper.hybrid.dataset import HybridIndex
from osu_mapper.hybrid.inference import resolve_device
from osu_mapper.v5.data import SongCache
from osu_mapper.v5.model import V5Config, V5Net
from train_v5 import val_maps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset_hybrid")
    ap.add_argument("--checkpoint", default="checkpoints/v5/v5_best.pt")
    ap.add_argument("--values", default="0,0.5,0.35,0.25,0.15,0.1", help="hold_wait values to try (0 = off)")
    ap.add_argument("--val_maps", type=int, default=16)
    ap.add_argument("--val_max_sec", type=float, default=60.0)
    ap.add_argument("--val_fraction", type=float, default=0.1, help="as in training")
    ap.add_argument("--seed", type=int, default=42, help="as in training")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_write", action="store_true", help="only print, do not store the best value")
    args = ap.parse_args()

    dev = resolve_device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = V5Config.from_dict(ck["config"])
    if getattr(cfg, "decoder", "ar") != "canvas":
        raise SystemExit("this checkpoint is not a canvas network: nothing to calibrate")
    model = V5Net(cfg)
    model.load_state_dict(ck.get("ema") or ck["model"])
    model.to(dev).eval()
    index = HybridIndex(args.data)
    _, val_idx = index.split(args.val_fraction, args.seed)
    val_idx = [i for i in val_idx if not index.maps[i].get("derived") and not index.maps[i].get("self")]
    val = sorted(val_idx, key=lambda i: float(index.maps[i].get("sr", 0.0)))
    pick = np.unique(np.linspace(0, len(val) - 1, min(args.val_maps, len(val))).round().astype(int))
    val = [val[p] for p in pick]
    cache = SongCache(index, sections="section" in cfg.groups, highlight=cfg.highlight)
    print(f"[calibrate] {len(val)} validation maps, first {args.val_max_sec:.0f} s each, device {dev}")
    best, best_v = -1e9, 0.0
    for v in [float(x) for x in args.values.split(",")]:
        t0 = time.time()
        with torch.no_grad():
            s = val_maps(model, index, cache, val, dev, 1.0, 1.0, args.val_max_sec, "ln", 1, args.steps, args.refine,
                         hold_wait=v if v > 0 else None)
        score = s.get("event_f1@16", 0.0) + 0.3 * s.get("expressive", 0.0)
        print(f"  hold_wait {('off' if v <= 0 else f'{v:.2f}'):>5}: long notes {100 * s.get('ln_share_pred', 0):.0f}% vs "
              f"{100 * s.get('ln_share_ref', 0):.0f}% | LN F1 {s.get('ln_f1', 0):.3f} | F1@16 {s.get('event_f1@16', 0):.3f} "
              f"| density x{s.get('density_ratio', float('nan')):.2f} | expressiveness {s.get('expressive', 0):.3f} "
              f"| score {score:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best, best_v = score, v
    print(f"[calibrate] best: hold_wait {'off' if best_v <= 0 else f'{best_v:.2f}'} (score {best:.4f})")
    if not args.no_write:
        ck["hold_wait"] = best_v if best_v > 0 else None
        path = Path(args.checkpoint)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(ck, tmp)
        os.replace(tmp, path)
        print(f"[calibrate] stored in {path}: generate_v5.py uses it (--hold_wait overrides)")


if __name__ == "__main__":
    main()
