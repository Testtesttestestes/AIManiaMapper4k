#!/usr/bin/env python3
"""
Mines every recurring pattern of real maps into a motif library (osu_mapper/hybrid/motifs.py).

    python mine_patterns.py --input_dir "sample beatmaps"                  # .osz archives (only the .osu text is read)
    python mine_patterns.py --osu_dir extracted_osu --min_sets 3 --max_len 32
    python mine_patterns.py --input_dir "sample beatmaps" --report motif_report.md

Writes checkpoints/hybrid/motifs.npz (the library) and a Markdown report with every top motif drawn on four
columns, per level (exact / lanes / rhythm / shape / hands / flow / ln) and per star-rating band, plus how maps
repeat themselves (exact, mirrored and varied copies of earlier passages). CPU only, no audio needed.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np

from osu_mapper.hybrid.beatmap import parse_osu_mania
from osu_mapper.hybrid.motifs import (COPY_DIST_BEATS, COPY_KINDS, COPY_LEN_BEATS, LEVELS, SR_BINS, Corpus,
                                      MotifLibrary, build_library, draw, name_motif)


def star_rating(text: str, m) -> float:
    try:
        import rosu_pp_py

        return float(rosu_pp_py.Difficulty().calculate(rosu_pp_py.Beatmap(content=text)).stars)
    except Exception:
        from osu_mapper.hybrid.inference import star_rating_of

        return star_rating_of(m.notes, m.timing_points, m.overall_difficulty)


def iter_texts(args):
    def set_id(name: str, fallback: int) -> int:
        mt = re.match(r"\s*(\d+)", name)
        return int(mt.group(1)) if mt else fallback

    k = 0
    if args.input_dir:
        from osu_mapper.hybrid.sources import _decode_text, is_lfs_pointer

        for osz in sorted(Path(args.input_dir).rglob("*.osz")):
            k += 1
            if is_lfs_pointer(osz):
                continue
            try:
                with zipfile.ZipFile(osz) as z:
                    for e in z.namelist():
                        if e.lower().endswith(".osu"):
                            yield set_id(osz.name, 10 ** 7 + k), f"{osz.name}::{e}", _decode_text(z.read(e))
            except (zipfile.BadZipFile, OSError) as err:
                print(f"  [error] {osz}: {err}")
    if args.osu_dir:
        dirs = {}
        for p in sorted(Path(args.osu_dir).rglob("*.osu")):
            dirs.setdefault(p.parent, []).append(p)
        for d, files in dirs.items():
            k += 1
            numeric = re.match(r"\s*\d+", d.name) is not None
            for p in files:
                text = p.read_bytes().decode("utf-8", errors="replace")
                if numeric:
                    sid = set_id(d.name, 0)
                else:  # a folder without a set id may hold several sets: one set per artist + title
                    meta = tuple(re.findall(r"^(?:Artist|Title):(.*)$", text, flags=re.M)[:2])
                    sid = 10 ** 8 + zlib.crc32("|".join((d.name,) + meta).encode()) % 10 ** 8
                yield sid, str(p), text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", default=None, help="folder with .osz archives")
    ap.add_argument("--osu_dir", default=None, help="folder with .osu files (one sub-folder per beatmap set)")
    ap.add_argument("--out", default="checkpoints/hybrid/motifs.npz")
    ap.add_argument("--report", default="motif_report.md")
    ap.add_argument("--min_sets", type=int, default=3, help="a motif must occur in at least this many beatmap sets")
    ap.add_argument("--max_len", type=int, default=32, help="longest motif in chords")
    ap.add_argument("--levels", default=",".join(LEVELS))
    args = ap.parse_args()
    if not args.input_dir and not args.osu_dir:
        sys.exit('give --input_dir "sample beatmaps" or --osu_dir <folder>')
    t0 = time.time()
    corpus = Corpus()
    seen = set()
    n_text = 0
    for sid, name, text in iter_texts(args):
        n_text += 1
        m = parse_osu_mania(text)
        if m is None or len(m.notes) < 20:
            continue
        key = hash(tuple(sorted(m.canonical_notes())))
        if key in seen:
            continue
        seen.add(key)
        corpus.add(name, sid, m.notes, m.timing_points, star_rating(text, m))
        if len(corpus.maps) % 200 == 0:
            print(f"  {len(corpus.maps)} 4K maps read ({time.time() - t0:.0f}s)", flush=True)
    print(f"[mine] {len(corpus.maps)} 4K maps of {len({m.set_id for m in corpus.maps})} sets "
          f"({n_text} difficulties read, {sum(m.n_chords for m in corpus.maps)} chords)", flush=True)
    if not corpus.maps:
        sys.exit("no 4K maps found")
    lib = build_library(corpus, min_sets=args.min_sets, max_len=args.max_len, levels=args.levels.split(","))
    lib.save(args.out)
    print(lib.summary())
    if args.report:
        Path(args.report).write_text(report(lib), encoding="utf-8")
        print(f"[mine] report -> {args.report}")
    print(f"[done] {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")


BANDS = ((0, 3, "1-3*"), (3, 5, "3-5*"), (5, 7, "5-7*"), (7, 12, "7*+"))


def report(lib: MotifLibrary) -> str:
    out = ["# Motif library report", "", "```", lib.summary(), "```", ""]
    out += ["## How maps repeat themselves (LZ77 parse of every map)", "",
            "| stars | maps | exact copy | mirrored copy | variation (<= 50 % lanes changed) | same rhythm only | copy length (median beats) |", "|---|---|---|---|---|---|---|"]
    f = lib.form
    for a, b, name in BANDS:
        ch = f.chords[a:b].sum()
        if ch <= 0:
            continue
        cov = f.covered[a:b].sum(axis=0) / ch
        lh = f.length[a:b].sum(axis=(0, 1))
        med = int(np.searchsorted(np.cumsum(lh) / max(1, lh.sum()), 0.5))
        lab = f"<= {COPY_LEN_BEATS[med]}" if med < COPY_LEN_BEATS.size else f"> {COPY_LEN_BEATS[-1]}"
        out.append(f"| {name} | {int(f.maps[a:b].sum())} | {cov[0]:.0%} | {cov[1]:.0%} | {cov[2]:.0%} | {cov[3]:.0%} | {lab} |")
    ph = f.start_phase.sum(axis=0)
    out += ["", f"Copies start on a downbeat {ph[0] / max(1, ph.sum()):.0%}, on a beat {ph[1] / max(1, ph.sum()):.0%}, "
            f"on a 1/2 {ph[2] / max(1, ph.sum()):.0%}, elsewhere {ph[3] / max(1, ph.sum()):.0%}.", ""]
    out += ["## Coverage: share of chords inside a known motif", "", "| level | stars | >= 4 chords | >= 8 chords |", "|---|---|---|---|"]
    for lv, c in lib.coverage.items():
        for a, b, name in BANDS:
            tot = c[a:b, 0].sum()
            if tot > 0:
                out.append(f"| {lv} | {name} | {c[a:b, 1].sum() / tot:.0%} | {c[a:b, 2].sum() / tot:.0%} |")
    for lv, t in lib.tables.items():
        out += ["", f"## Level `{lv}`", ""]
        for a, b, name in BANDS:
            for lo, hi, what in ((4, 7, "4-7 chords"), (8, 15, "8-15 chords"), (16, 999, "16+ chords")):
                top = lib.top(lv, sr=(a + b) / 2, min_len=lo, max_len=hi, k=8)
                if not top:
                    continue
                out += [f"### {name}, {what}", ""]
                for k in top:
                    tk = t.tokens(k)
                    w = lib.sr_weight(lv, (a + b) / 2)[k]
                    nm = name_motif(lv, tk)
                    out.append(f"**{nm or 'motif'}** - {len(tk)} chords, ~{t.beats[k]:.1f} beats, {t.count[k]} times in "
                               f"{t.sets[k]} sets ({w:.0%} of maps at {name}); starts on downbeat "
                               f"{t.phase_hist[k][0] / max(1, t.phase_hist[k].sum()):.0%}")
                    out += ["```", draw(lv, tk), "```", ""]
    return "\n".join(out)


if __name__ == "__main__":
    main()
