#!/usr/bin/env python3
"""
Calibrates how the v5 canvas reveals long notes, on the validation songs of the training split.

When a window is revealed in rounds, a frame of a lane the network thinks may be held (HOLD / TAIL likely) but whose
long note has not started yet could be committed empty early - and then the long note can no longer pass through it,
nor start before it: long notes get cut and mostly vanish (validation showed 2-3 % long notes against 21 % in the
human maps). With ``hold_wait`` such a lane waits for the frame before it. How likely is "may be held" depends on the
network, so this script tries several values on the validation songs, prints long-note share, F1 and expressiveness
for each, and stores the best (F1@16 + 0.3 x expressiveness, as the training picks checkpoints) in the checkpoint -
separately for maps with few, some and many long notes (one value for all helped long-note maps and cost rice maps):
generate_v5.py picks it by the share of long notes it is asked for (--hold_wait overrides).

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
    ap.add_argument("--val_maps", type=int, default=40)
    ap.add_argument("--groups", default="0.1,0.3", help="edges of the groups by the human share of long notes")
    ap.add_argument("--min_maps", type=int, default=4, help="fewer maps in a group: no choice there (off)")
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
    from osu_mapper.hybrid.renderer import render_track

    def ln_share(i):
        n = render_track(index.track(i))
        return sum(x.ln for x in n) / max(1, len(n))

    edges = [float(x) for x in args.groups.split(",")]
    groups = [[] for _ in range(len(edges) + 1)]
    for i in val:
        groups[int(np.searchsorted(edges, ln_share(i), side="right"))].append(i)
    names = [f"long notes < {100 * edges[0]:.0f}%"] + [f"{100 * a:.0f}-{100 * b:.0f}%" for a, b in zip(edges, edges[1:])] \
        + [f">= {100 * edges[-1]:.0f}%"]
    print(f"[calibrate] {len(val)} validation maps, first {args.val_max_sec:.0f} s each, device {dev}; groups by the human "
          "share of long notes: " + ", ".join(f"{n_}: {len(g)}" for n_, g in zip(names, groups)))
    values = [float(x) for x in args.values.split(",")]
    by_group = []
    for name, g in zip(names, groups):
        if len(g) < args.min_maps:
            print(f"[calibrate] {name}: {len(g)} maps, too few to choose (--min_maps {args.min_maps}): off")
            by_group.append(0.0)
            continue
        print(f"[calibrate] {name} ({len(g)} maps):")
        best, best_v = -1e9, 0.0
        for v in values:
            t0 = time.time()
            with torch.no_grad():
                s = val_maps(model, index, cache, g, dev, 1.0, 1.0, args.val_max_sec, "ln", 1, args.steps, args.refine,
                             hold_wait=v if v > 0 else None)
            score = s.get("event_f1@16", 0.0) + 0.3 * s.get("expressive", 0.0)
            print(f"  hold_wait {('off' if v <= 0 else f'{v:.2f}'):>5}: long notes {100 * s.get('ln_share_pred', 0):.0f}% vs "
                  f"{100 * s.get('ln_share_ref', 0):.0f}% | LN F1 {s.get('ln_f1', 0):.3f} | F1@16 {s.get('event_f1@16', 0):.3f} "
                  f"| density x{s.get('density_ratio', float('nan')):.2f} | expressiveness {s.get('expressive', 0):.3f} "
                  f"| score {score:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if score > best:
                best, best_v = score, v
        print(f"  -> {'off' if best_v <= 0 else f'{best_v:.2f}'}")
        by_group.append(best_v)
    table = {"edges": edges, "values": [v if v > 0 else None for v in by_group]}
    print("[calibrate] chosen: " + ", ".join(f"{n_} -> {'off' if not v else f'{v:.2f}'}" for n_, v in zip(names, table["values"])))
    if not args.no_write:
        ck["hold_wait"] = None
        ck["hold_wait_by_ln"] = table
        path = Path(args.checkpoint)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(ck, tmp)
        os.replace(tmp, path)
        print(f"[calibrate] stored in {path}: generate_v5.py picks the value by the share of long notes it is asked for "
              "(--style ln_ratio=..., or the one the LN model guesses from the music); --hold_wait overrides")


if __name__ == "__main__":
    main()
