#!/usr/bin/env python3
"""
Star-rating statistics of 4K mania maps: per 1-star bin how many maps, LN share, notes per second.

    python beatmap_stats.py --input_dir "sample beatmaps"          # raw .osz / folders (computes SR with rosu-pp)
    python beatmap_stats.py --data dataset_hybrid                   # prepared dataset (instant, uses stored SR)
    python beatmap_stats.py --input_dir "new maps" --list --min_sr 6   # also list the maps, hardest first

With --data, real and derived (thinned) maps are counted separately and the train/val split is shown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def table(rows: List[Dict], title: str) -> None:
    if not rows:
        print(f"{title}: no maps")
        return
    srs = np.array([r["sr"] for r in rows])
    print(f"\n{title}: {len(rows)} maps, SR {srs.min():.2f}..{srs.max():.2f} (median {np.median(srs):.2f})")
    print(f"{'stars':>7} {'maps':>6} {'share':>7} {'LN share':>9} {'NPS':>6}")
    for b in range(0, int(min(srs.max(), 15)) + 1):
        sel = [r for r in rows if b <= r["sr"] < b + 1 or (b == 15 and r["sr"] >= 15)]
        if not sel:
            continue
        ln = np.median([r["ln"] for r in sel if r["ln"] is not None]) if any(r["ln"] is not None for r in sel) else float("nan")
        nps = np.median([r["nps"] for r in sel if r["nps"] is not None]) if any(r["nps"] is not None for r in sel) else float("nan")
        print(f"{b:>3}-{b + 1:<3} {len(sel):>6} {100 * len(sel) / len(rows):>6.1f}% {100 * ln:>8.0f}% {nps:>6.1f}")


def from_folder(input_dir: str) -> List[Dict]:
    from osu_mapper.hybrid.sources import iter_song_groups
    from prepare_hybrid_dataset import star_rating

    rows = []
    for group in iter_song_groups(input_dir):
        for osu_name, osu_text, m in group.maps:
            if not m.notes:
                continue
            sr = star_rating(osu_text, m)
            span = max(1.0, (max(n.start for n in m.notes) - min(n.start for n in m.notes)) / 1000.0)
            rows.append({"sr": sr, "ln": sum(n.ln for n in m.notes) / len(m.notes), "nps": len(m.notes) / span,
                         "name": f"{Path(group.source).name} [{m.metadata.get('Version', osu_name)}]", "derived": False})
        print(f"  {len(rows)} maps...", end="\r", flush=True)
    if rows and all(r["sr"] <= 0 for r in rows):
        raise SystemExit("star ratings are 0: rosu-pp is not installed (python install_rosu.py)")
    return rows


def from_dataset(data: str) -> List[Dict]:
    from osu_mapper.hybrid.dataset import split_key

    idx = json.loads((Path(data) / "index.json").read_text(encoding="utf-8"))
    rows = []
    for m in idx["maps"]:
        if not m.get("roundtrip_exact", True) or float(m.get("sr", 0)) <= 0:
            continue
        style, mask = m.get("style") or [], m.get("style_mask") or []
        ln = float(style[2]) if len(style) > 2 and (len(mask) < 3 or mask[2] or m.get("derived")) else None
        nps = float(np.expm1(style[1] * np.log1p(30.0))) if len(style) > 1 else None
        rows.append({"sr": float(m["sr"]), "ln": ln, "nps": nps, "name": f"{Path(m.get('source', '')).name} [{m.get('version', '')}]",
                     "derived": bool(m.get("derived")), "song": split_key(m)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--input_dir", "-i", default=None, help="folder with .osz files / song folders")
    src.add_argument("--data", default=None, help="prepared dataset folder (prepare_hybrid_dataset.py)")
    ap.add_argument("--list", action="store_true", help="also list maps, hardest first")
    ap.add_argument("--min_sr", type=float, default=0.0, help="with --list: only maps at least this hard")
    ap.add_argument("--val_fraction", type=float, default=0.1, help="with --data: split used by training")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.data:
        rows = from_dataset(args.data)
        real = [r for r in rows if not r["derived"]]
        table(real, "real maps")
        derived = [r for r in rows if r["derived"]]
        if derived:
            table(derived, "derived (thinned) maps")
        import random
        songs = sorted({r["song"] for r in rows})
        random.Random(args.seed).shuffle(songs)
        val = set(songs[:max(1, int(round(len(songs) * args.val_fraction)))])
        table([r for r in real if r["song"] in val], f"validation split for new trainings (val_fraction {args.val_fraction:g}, rate versions grouped)")
    else:
        rows = from_folder(args.input_dir or "sample beatmaps")
        table(rows, f"maps in '{args.input_dir or 'sample beatmaps'}'")
    if args.list:
        print()
        for r in sorted((r for r in rows if r["sr"] >= args.min_sr), key=lambda r: -r["sr"]):
            print(f"{r['sr']:6.2f}  {'(derived) ' if r['derived'] else ''}{r['name']}")


if __name__ == "__main__":
    main()
