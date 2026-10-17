#!/usr/bin/env python3
"""
Algorithmic lane polishing of an existing 4K map (any .osu / .osz, generated or not).

    python polish_map.py output_beatmaps/hybrid_map.osz --strength 0.5
    python polish_map.py my_map.osu --strength 0.7 -o my_map_polished.osu

Note times, chord sizes and long-note lengths stay exactly; only lanes are re-assigned against fast jacks,
one-hand passages and anchors (osu_mapper/hybrid/ergonomics.py). For an .osz every 4K difficulty is
polished and written into a new .osz next to the input (suffix _polished) with " (polished)" in its name.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

from osu_mapper.hybrid.beatmap import parse_osu_mania
from osu_mapper.hybrid.ergonomics import ergonomics_report, format_report, improve_ergonomics


def polish_text(text: str, strength: float, tag: str) -> str:
    m = parse_osu_mania(text)
    before = ergonomics_report(m.notes)
    notes = improve_ergonomics(m.notes, m.timing_points, strength)
    print(f"  [{m.metadata.get('Version', '')}] {format_report(before)}  ->  {format_report(ergonomics_report(notes))}")
    head, _, _ = text.partition("[HitObjects]")
    head = re.sub(r"(?m)^(Version:.*?)\s*$", lambda g: g.group(1) + tag, head, count=1)
    head = re.sub(r"(?m)^BeatmapID:.*$", "BeatmapID:0", head, count=1)
    lines = []
    for n in sorted(notes):
        x = int(n.col * 512 / 4 + 256 / 4)
        lines.append(f"{x},192,{n.start},128,0,{n.end}:0:0:0:0:" if n.ln else f"{x},192,{n.start},1,0,0:0:0:0:")
    return head + "[HitObjects]\n" + "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()
    src = Path(args.input)
    tag = f" (polished {args.strength:g})"
    if src.suffix.lower() == ".osz":
        dst = Path(args.output) if args.output else src.with_name(src.stem + "_polished.osz")
        with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename.lower().endswith(".osu"):
                    try:
                        data = polish_text(data.decode("utf-8", errors="replace"), args.strength, tag).encode("utf-8")
                        item.filename = item.filename[:-4] + " polished.osu"
                    except Exception as err:
                        print(f"  [skip] {item.filename}: {err}")
                zout.writestr(item, data)
    else:
        dst = Path(args.output) if args.output else src.with_name(src.stem + " polished.osu")
        dst.write_text(polish_text(src.read_text(encoding="utf-8", errors="replace"), args.strength, tag), encoding="utf-8")
    print(f"[done] {dst}")


if __name__ == "__main__":
    main()
