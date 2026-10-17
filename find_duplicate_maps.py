#!/usr/bin/env python3
"""
Finds duplicate beatmaps by the CONTENT of their .osu files only (archive names, audio and images are ignored).

Two .osu files are duplicates if they are byte-identical after stripping the BOM / line-ending / trailing-space
differences, or (for 4K mania) if every hit object and timing point is identical (only comments, storyboard or
metadata lines differ), or if every hit object is identical even though timing points differ (re-uploads with
other volumes / kiai / green lines; --strict_timing turns this last rule off). An .osz archive is redundant when every one of its .osu files is a duplicate of a
difficulty kept in another archive; the archive whose name starts with a beatmap-set ID is preferred.

    python find_duplicate_maps.py                         # report only
    python find_duplicate_maps.py --delete                # delete redundant archives
    python find_duplicate_maps.py --dir "F:\\mania4k dataset" --delete
"""

from __future__ import annotations

import argparse
import hashlib
import re
import zipfile
from pathlib import Path
from typing import Dict, List

from osu_mapper.hybrid.beatmap import parse_osu_mania
from osu_mapper.hybrid.sources import LFS_HINT, is_lfs_pointer


def osu_keys(raw: bytes, notes_only: bool = True) -> List[str]:
    text = raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    norm = "\n".join(line.rstrip() for line in text.split("\n")).strip()
    keys = ["text:" + hashlib.sha1(norm.encode("utf-8")).hexdigest()]
    m = parse_osu_mania(text)
    if m is not None:
        payload = repr((m.canonical_notes(), [tp.as_tuple() for tp in m.timing_points]))
        keys.append("game:" + hashlib.sha1(payload.encode("utf-8")).hexdigest())
        if notes_only and len(m.notes) >= 20:  # the same notes = the same chart, whatever the green lines say
            keys.append("notes:" + hashlib.sha1(repr(m.canonical_notes()).encode("utf-8")).hexdigest())
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="sample beatmaps")
    ap.add_argument("--delete", action="store_true", help="delete archives whose every difficulty is a duplicate")
    ap.add_argument("--strict_timing", action="store_true", help="identical notes with different timing points are NOT duplicates")
    args = ap.parse_args()

    archives = sorted(Path(args.dir).glob("*.osz"), key=lambda p: (re.match(r"^\d+ ", p.name) is None, p.name))
    pointers = [p for p in archives if is_lfs_pointer(p)]
    if pointers:
        raise SystemExit(f"{len(pointers)} of {len(archives)} .osz files are not downloaded yet.\n{LFS_HINT}")
    owner: Dict[str, str] = {}
    redundant: List[Path] = []
    dup_osu = 0
    for osz in archives:
        with zipfile.ZipFile(osz) as z:
            files = [n for n in z.namelist() if n.lower().endswith(".osu")]
            keysets = [osu_keys(z.read(n), not args.strict_timing) for n in files]
        matches = [next((owner[k] for k in ks if k in owner), None) for ks in keysets]
        dup_osu += sum(m is not None for m in matches)
        if files and all(matches):
            redundant.append(osz)
            print(f"[duplicate] {osz.name}  ->  same content as {sorted(set(matches))}")
            continue
        for n, ks, m in zip(files, keysets, matches):
            if m is not None:
                print(f"[partial]   {osz.name} :: {n} duplicates {m}")
            for k in ks:
                owner.setdefault(k, osz.name)
    print(f"\n{len(archives)} archives, {dup_osu} duplicate .osu files, {len(redundant)} fully redundant archives")
    if args.delete:
        for p in redundant:
            p.unlink()
            print(f"deleted {p}")


if __name__ == "__main__":
    main()
