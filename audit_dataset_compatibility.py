#!/usr/bin/env python3
"""
Dataset check for the hybrid mapper: what the network will actually learn from your maps.

    python audit_dataset_compatibility.py                       # "sample beatmaps"
    python audit_dataset_compatibility.py --dir "D:/osu/Songs" --workers 8   # also folders of .osu (osu!'s Songs)
    python audit_dataset_compatibility.py --map "sample beatmaps/mymap.osz"
    python audit_dataset_compatibility.py --fast                # skip the exact-roundtrip check (much faster)

Per 4K mania difficulty (the same parser and the same parameter solver as prepare_hybrid_dataset.py):

1. Exact reproduction: the map is turned into the network's parameters and rendered back; anything that does not
   come back note for note (or needs the overflow list) is real data loss. The network keeps every note to the
   millisecond, so a note between grid ticks is NOT lost.
2. Rhythm grid, per red line in effect at each note (3 ms tolerance): 1/1, 1/2, 1/4, 1/8, triplets (1/3, 1/6),
   1/12-1/16 and off-grid (unsnapped, 1/5, 1/7) - how much of the map the beat-grid features describe.
3. Tempo: real tempo changes (red lines with another BPM) separate from SV (green lines), and the meters (4/4,
   3/4, 5/4, 7/8 ...) including meter changes.
4. Star ratings (rosu-pp) per 1-star bin: where the dataset is thin (train_hybrid.py --rate_aug / --sr_rare /
   make_self_maps.py fill it).
5. Long notes: share, zero-gap chains (release -> press on the same lane <= 5 ms), notes inside a held LN.
6. Duplicates: difficulties with identical notes (the cleanup removes them: --clean).

Cleanup (keeps only what training reads: 4K mania difficulties + the audio file they name):
    python audit_dataset_compatibility.py --clean                         # report only, nothing is changed
    python audit_dataset_compatibility.py --clean --apply                 # remove videos, bg, hitsounds, clones...
    python audit_dataset_compatibility.py --clean --apply --out_dir clean # cleaned copies, originals untouched
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from osu_mapper.dataset.source_resolver import resolve_beatmap_sources

GRID_NAMES = ("1/1", "1/2", "1/4", "1/3-1/6", "1/8", "1/12-1/16", "off-grid")


@dataclass
class MapAuditResult:
    map_name: str
    osz_name: str
    star_rating: float
    main_bpm: float
    red_lines: int
    tempo_changes: int                 # red lines whose BPM differs from the previous one
    sv_lines: int                      # green lines (scroll speed only)
    meters: List[int]
    total_notes: int
    ln_ratio_pct: float
    roundtrip_exact: Optional[bool]    # None = not checked (--fast)
    overflow_notes: int
    grid_pct: Dict[str, float]         # share of notes per grid class
    zero_gap_ln_count: int
    overlap_note_count: int
    lane_distribution: List[int]
    notes_key: str = ""
    problems: List[str] = field(default_factory=list)
    notes_info: List[str] = field(default_factory=list)


def audit_beatmap_content(content: str, map_name: str, osz_name: str, roundtrip: bool = True) -> Optional[MapAuditResult]:
    from osu_mapper.hybrid.beatmap import parse_osu_mania
    from osu_mapper.hybrid.patterns import div_class, main_bpm

    m = parse_osu_mania(content)
    if m is None or len(m.notes) < 20:
        return None
    notes = sorted(m.notes, key=lambda n: (n.start, n.col))
    reds = sorted((tp for tp in m.timing_points if tp.uninherited and tp.beat_len > 0), key=lambda r: r.time)
    greens = [tp for tp in m.timing_points if not tp.uninherited]
    bpms = [round(60000.0 / r.beat_len, 1) for r in reds]
    tempo_changes = sum(1 for a, b in zip(bpms, bpms[1:]) if abs(a - b) > 0.05)
    starts = sorted({n.start for n in notes})
    cls = Counter(div_class(float(t), reds) for t in starts) if reds else Counter({len(GRID_NAMES) - 1: len(starts)})
    grid = {GRID_NAMES[k]: round(100.0 * cls.get(k, 0) / max(1, len(starts)), 1) for k in range(len(GRID_NAMES))}
    # long notes: zero-gap chains and notes inside a held LN, per lane
    zero_gap = overlaps = 0
    by_lane: Dict[int, list] = {}
    for n in notes:
        by_lane.setdefault(n.col, []).append(n)
    for lane in by_lane.values():
        for a, b in zip(lane, lane[1:]):
            if a.ln and a.end > a.start:
                if b.start < a.end:
                    overlaps += 1
                elif b.start - a.end <= 5:
                    zero_gap += 1
    exact, overflow = None, 0
    if roundtrip:
        from osu_mapper.hybrid.renderer import render_track
        from osu_mapper.hybrid.solver import SolveStats, solve_map

        stats = SolveStats()
        track = solve_map(m, stats=stats)
        exact = render_track(track) == sorted(m.notes)
        overflow = int(stats.overflow)
    try:
        import rosu_pp_py

        sr = float(rosu_pp_py.Difficulty().calculate(rosu_pp_py.Beatmap(content=content)).stars)
    except Exception:
        sr = 0.0
    lanes = [0, 0, 0, 0]
    for n in notes:
        if 0 <= n.col < 4:
            lanes[n.col] += 1
    r = MapAuditResult(
        map_name=m.metadata.get("Version", map_name), osz_name=osz_name, star_rating=round(sr, 2),
        main_bpm=round(main_bpm(m.timing_points, notes[-1].start) or 0.0, 1), red_lines=len(reds), tempo_changes=tempo_changes,
        sv_lines=len(greens), meters=sorted({max(1, int(r_.meter or 4)) for r_ in reds}) or [4],
        total_notes=len(notes), ln_ratio_pct=round(100.0 * sum(n.ln for n in notes) / len(notes), 1),
        roundtrip_exact=exact, overflow_notes=overflow, grid_pct=grid, zero_gap_ln_count=zero_gap,
        overlap_note_count=overlaps, lane_distribution=lanes,
        notes_key=hashlib.sha1(repr(m.canonical_notes()).encode("utf-8")).hexdigest())
    if exact is False:
        r.problems.append("does not reproduce exactly (prepare_hybrid_dataset.py skips it)")
    if overflow:
        r.problems.append(f"{overflow} notes only in the overflow list")
    if not reds:
        r.problems.append("no red line: no beat grid")
    if grid["off-grid"] > 25.0:
        r.notes_info.append(f"{grid['off-grid']:.0f}% notes off the grid (unsnapped / 1/5 / 1/7)")
    if tempo_changes:
        r.notes_info.append(f"{tempo_changes} tempo changes")
    if any(x != 4 for x in r.meters):
        r.notes_info.append("meter " + "/".join(f"{x}/4" for x in r.meters))
    if overlaps:
        r.problems.append(f"{overlaps} notes inside a held long note")
    return r


def _audit_text(job):
    text, name, osz, roundtrip = job
    try:
        return audit_beatmap_content(text, name, osz, roundtrip)
    except Exception as e:  # one broken difficulty must not stop the check
        print(f"[Warn] {osz} :: {name}: {e}")
        return None


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


def collect_texts(source: Path) -> List[Tuple[str, str, str]]:
    from osu_mapper.hybrid.sources import is_lfs_pointer

    if source.is_file():
        if source.suffix.lower() == ".osu":
            return [(_decode(source.read_bytes()), source.name, source.parent.name)]
        osz_files, folders = [source], []
    else:
        osz_files, folders = resolve_beatmap_sources(source)
        # song folders right inside the given folder (osu!'s Songs folder, unpacked sets)
        known = {Path(f).resolve() for f in folders}
        for sub in [source] + sorted(p for p in source.iterdir() if p.is_dir()):
            if sub.resolve() not in known and any(sub.glob("*.osu")):
                folders.append(sub)
    out: List[Tuple[str, str, str]] = []
    skipped = 0
    for osz in osz_files:
        if is_lfs_pointer(osz):
            skipped += 1
            continue
        try:
            with zipfile.ZipFile(osz, "r") as zf:
                out += [(_decode(zf.read(n)), n, Path(osz).name) for n in zf.namelist() if n.lower().endswith(".osu")]
        except Exception as e:
            print(f"[Warn] Failed reading {Path(osz).name}: {e}")
    for folder in folders:
        out += [(_decode(p.read_bytes()), p.name, Path(folder).name) for p in Path(folder).glob("*.osu")]
    if skipped:
        print(f"[Warn] {skipped} archives are Git LFS pointers (not downloaded): git lfs pull")
    return out


def audit_all_dataset(source_dir: str | Path, roundtrip: bool = True, workers: int = 1) -> List[MapAuditResult]:
    texts = collect_texts(Path(source_dir))
    print(f"[Dataset Audit] {len(texts)} difficulties to check" + ("" if roundtrip else " (no roundtrip check)"))
    jobs = [(t, n, o, roundtrip) for t, n, o in texts]
    if workers > 1 and len(jobs) > 50:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            res = list(pool.map(_audit_text, jobs, chunksize=8))
    else:
        res = []
        for k, j in enumerate(jobs, 1):
            res.append(_audit_text(j))
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    return [r for r in res if r is not None]


def print_audit_report(results: List[MapAuditResult]) -> None:
    if not results:
        print("\n[!] No osu!mania 4K difficulties found to check.")
        return
    n = len(results)
    notes = sum(r.total_notes for r in results)
    lns = sum(r.total_notes * r.ln_ratio_pct / 100.0 for r in results)
    print("\n" + "=" * 78)
    print(" OSU!MANIA 4K DATASET CHECK (hybrid mapper)")
    print("=" * 78)
    print(f"Difficulties: {n:,} | notes: {notes:,} ({lns / max(1, notes):.1%} long notes)")
    print("-" * 78)
    print("1. EXACT REPRODUCTION (the only real data loss)")
    checked = [r for r in results if r.roundtrip_exact is not None]
    if checked:
        bad = [r for r in checked if not r.roundtrip_exact or r.overflow_notes]
        print(f"  exact: {len(checked) - len(bad)} of {len(checked)}" + ("  - all reproduce note for note" if not bad else ""))
        for r in bad[:10]:
            print(f"    ! {r.osz_name} :: {r.map_name}: {'; '.join(r.problems)}")
    else:
        print("  not checked (--fast)")
    print("-" * 78)
    print("2. RHYTHM GRID (per red line in effect, 3 ms; the network keeps every note to the ms anyway)")
    tot = Counter()
    for r in results:
        for k, v in r.grid_pct.items():
            tot[k] += v * r.total_notes
    tw = sum(r.total_notes for r in results)
    print("  " + " | ".join(f"{k} {tot[k] / tw:.1f}%" for k in GRID_NAMES))
    off = [r for r in results if r.grid_pct["off-grid"] > 25.0]
    print(f"  maps with > 25% off-grid notes: {len(off)} (unsnapped, 1/5, 1/7 - or a wrong red line; worth a look)")
    print("-" * 78)
    print("3. TEMPO AND METER")
    tc = sum(1 for r in results if r.tempo_changes)
    print(f"  real tempo changes: {tc} maps ({100 * tc / n:.1f}%) | maps with SV (green lines): "
          f"{sum(1 for r in results if r.sv_lines)} ({100 * sum(1 for r in results if r.sv_lines) / n:.1f}%)")
    only44 = sum(1 for r in results if r.meters == [4])
    has = Counter(x for r in results for x in r.meters if x != 4)
    changes = sum(1 for r in results if len(r.meters) > 1)
    print(f"  meter: only 4/4 in {only44} maps ({100 * only44 / n:.1f}%), meter changes inside the map: {changes}")
    if has:
        print("  maps with another meter: " + ", ".join(f"{x}/4: {c}" for x, c in sorted(has.items())))
    print("-" * 78)
    print("4. STAR RATINGS (rosu-pp) - where the dataset is thin")
    bins = Counter(int(min(11, max(0, r.star_rating))) for r in results)
    peak = max(bins.values())
    for b in range(0, 12):
        if bins.get(b):
            bar = "#" * max(1, int(40 * bins[b] / peak))
            print(f"  {b:>2}-{b + 1:<2}*  {bins[b]:>5}  {bar}")
    hard = sum(v for k, v in bins.items() if k >= 6)
    print(f"  6*+: {hard} ({100 * hard / n:.1f}%)")
    print("-" * 78)
    print("5. LONG NOTES")
    zg = sum(r.zero_gap_ln_count for r in results)
    ov = sum(r.overlap_note_count for r in results)
    print(f"  zero-gap chains (release -> press <= 5 ms): {zg} | notes inside a held LN: {ov}")
    print("-" * 78)
    print("6. DUPLICATES")
    dup = sum(c - 1 for c in Counter(r.notes_key for r in results).values() if c > 1)
    print(f"  difficulties with the same notes as another one: {dup}" + (" (remove: --clean --apply)" if dup else ""))
    print("=" * 78)
    print("WHAT IT MEANS FOR TRAINING")
    if checked and any(not r.roundtrip_exact or r.overflow_notes for r in checked):
        print("  * Maps that do not reproduce exactly are skipped by prepare_hybrid_dataset.py.")
    if hard / n < 0.2:
        print(f"  * Hard maps are rare ({100 * hard / n:.0f}% at 6*+): keep --rate_aug / --sr_rare on, run make_self_maps.py "
              "after the first training (HYBRID_GUIDE.md 4.9).")
    if off:
        print("  * Off-grid maps are fine for the network (ms targets); check their red lines if the share is unexpected.")
    if dup:
        print("  * Duplicates weigh one chart twice: python audit_dataset_compatibility.py --clean --apply")
    if ov:
        print("  * Notes inside held LNs cannot be played as written; those maps teach overlaps.")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", "--sample-dir", default="sample beatmaps", help="Directory containing .osz beatmaps or song folders.")
    parser.add_argument("--map", default=None, help="Path to a single .osz or .osu file to audit.")
    parser.add_argument("--out", default="dataset_audit_report.json", help="Path to save JSON audit report.")
    parser.add_argument("--clean", action="store_true",
                        help="cleanup report: what training does not need (videos, backgrounds, hitsounds, storyboards, "
                             "other modes / key counts, unusable audio, clone difficulties) and how much space it frees")
    parser.add_argument("--apply", action="store_true",
                        help="with --clean: really remove it (archives are rewritten in place; an archive without a usable "
                             "4K difficulty is deleted)")
    parser.add_argument("--out_dir", default=None,
                        help="with --clean --apply: write cleaned copies here instead of changing the originals")
    parser.add_argument("--keep_clones", action="store_true", help="with --clean: keep difficulties with identical notes")
    parser.add_argument("--strict_timing", action="store_true",
                        help="with --clean: identical notes with different timing points are not clones")
    parser.add_argument("--min_notes", type=int, default=20, help="with --clean: difficulties with fewer notes are removed")
    parser.add_argument("--verbose", action="store_true", help="with --clean: list every removed difficulty and why")
    parser.add_argument("--skip_audit", action="store_true", help="only the cleanup, no rhythm/grid audit afterwards")
    parser.add_argument("--fast", action="store_true", help="skip the exact-roundtrip check (the slow part)")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (__import__("os").cpu_count() or 2) - 1)),
                        help="parallel processes")
    args = parser.parse_args()

    if args.clean:
        from osu_mapper.dataset.cleaner import apply_plan, format_report, plan_dataset

        root = Path(args.dir)
        print(f"[Dataset Cleanup] Checking '{root.resolve()}' ...")
        rep = plan_dataset(root, min_notes=args.min_notes, keep_clones=args.keep_clones, strict_timing=args.strict_timing)
        print(format_report(rep, verbose=args.verbose))
        if args.apply:
            print("[Dataset Cleanup] Applying" + (f" -> {Path(args.out_dir).resolve()}" if args.out_dir else " in place") + " ...")
            apply_plan(rep, args.out_dir)
            print("[Dataset Cleanup] Done.")
            if args.out_dir:
                args.dir = args.out_dir
        else:
            print("[Dataset Cleanup] Dry run: nothing was changed. Add --apply to remove it "
                  "(or --apply --out_dir <folder> to write cleaned copies).")
        if args.skip_audit:
            return

    source = Path(args.map) if args.map else Path(args.dir)
    if not source.exists():
        print(f"[Error] not found: {source}")
        sys.exit(1)
    print(f"[Dataset Audit] Checking '{source.resolve()}' ...")
    results = audit_all_dataset(source, roundtrip=not args.fast, workers=args.workers)

    print_audit_report(results)

    # Save to JSON
    if results:
        out_path = Path(args.out)
        data = [asdict(r) for r in results]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Report] Full audit details written to '{out_path.resolve()}'")


if __name__ == "__main__":
    main()
