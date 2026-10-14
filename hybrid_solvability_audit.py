#!/usr/bin/env python3
"""
Pre-launch check #1: "does a parameter set exist that reproduces EVERY map of the dataset 100 %?"

For every 4K difficulty in the dataset sources this script runs  map -> solver -> parameters ->
universal algorithm -> map  and compares the result with the original bit by bit (lane, start ms,
end ms, tap/LN of every object). It needs no audio and no GPU and takes seconds per thousand maps.

It also measures how *well* the parameter language fits real maps (what the network will have to
learn): how many events are explained by relational lane operators vs explicit lane masks, how many
chords are staggered, how many notes are off the musical grid, how many objects needed the sparse
overflow list (degenerate objects: negative times, zero-length LNs, two notes in one lane < 10 ms).

    python hybrid_solvability_audit.py                          # sources from 'sample beatmaps'
    python hybrid_solvability_audit.py --input_dir "F:\\mania4k dataset" --out audit_hybrid.json

Exit code 1 if any map fails the exact round trip.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from osu_mapper.hybrid.params import LANE_CODE_NAMES, N_LANE_CODES, N_OPS
from osu_mapper.hybrid.renderer import GridSnapper, render_track
from osu_mapper.hybrid.solver import SolveStats, solve_map
from osu_mapper.hybrid.sources import iter_song_groups


def entropy_bits(counts: np.ndarray) -> float:
    p = counts[counts > 0] / max(1, counts.sum())
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact solvability audit of the hybrid parameter space")
    ap.add_argument("--input_dir", "-i", default="sample beatmaps")
    ap.add_argument("--out", default="hybrid_audit_report.json")
    ap.add_argument("--max_maps", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    per_map: List[Dict[str, Any]] = []
    code_hist = np.zeros(N_LANE_CODES, dtype=np.int64)
    offset_hist = np.zeros(10, dtype=np.int64)
    overflow_reasons: Counter = Counter()
    failures: List[str] = []
    n_maps = 0
    tot = Counter()

    for group in iter_song_groups(args.input_dir):
        for osu_name, _text, m in group.maps:
            if args.max_maps and n_maps >= args.max_maps:
                break
            n_maps += 1
            st = SolveStats()
            track = solve_map(m, stats=st)
            rendered = render_track(track)
            exact = rendered == sorted(m.notes)
            if not exact:
                failures.append(f"{group.source} :: {osu_name}")
            code_hist += np.array(st.code_hist, dtype=np.int64)
            ev = track.event_frames()
            offset_hist += np.bincount(track.time_class[ev].astype(np.int64) - 1, minlength=10)[:10]
            overflow_reasons.update(st.overflow_reasons)
            snapper = GridSnapper(m.timing_points)
            offgrid = 0
            if snapper.available and m.notes:
                _, dist = snapper.nearest_tick(np.array([n.start for n in m.notes], dtype=np.float64))
                offgrid = int((dist > 2.0).sum())
            relational = int(sum(st.code_hist[:N_OPS]))
            rec = {
                "source": group.source,
                "osu_file": osu_name,
                "version": m.metadata.get("Version", ""),
                "exact": exact,
                "notes": st.notes,
                "events": st.events,
                "ln_notes": st.ln_notes,
                "staggered_events": st.staggered_events,
                "overflow": st.overflow,
                "overflow_reasons": dict(st.overflow_reasons),
                "notes_inside_held_ln": st.overlaps_in_ln,
                "offgrid_notes_2ms": offgrid,
                "relational_events": relational,
                "explicit_events": int(sum(st.code_hist[N_OPS:])),
                "red_lines": len(m.red_lines()),
                "timing_points": len(m.timing_points),
                "duration_s": round(m.last_time_ms / 1000.0, 1),
            }
            per_map.append(rec)
            for k in ("notes", "events", "ln_notes", "staggered_events", "overflow", "notes_inside_held_ln", "offgrid_notes_2ms", "relational_events", "explicit_events"):
                tot[k] += rec[k]
            tot["exact"] += int(exact)

    if n_maps == 0:
        sys.exit(f"No 4K mania maps found in '{args.input_dir}' (or its sources.txt pointers).")

    rel_share = tot["relational_events"] / max(1, tot["events"])
    summary = {
        "maps": n_maps,
        "exact_roundtrip": tot["exact"],
        "exact_roundtrip_pct": round(100.0 * tot["exact"] / n_maps, 3),
        "notes": tot["notes"],
        "events": tot["events"],
        "overflow_notes": tot["overflow"],
        "overflow_pct_of_notes": round(100.0 * tot["overflow"] / max(1, tot["notes"]), 4),
        "overflow_reasons": dict(overflow_reasons),
        "maps_with_overflow": sum(1 for r in per_map if r["overflow"]),
        "staggered_events_pct": round(100.0 * tot["staggered_events"] / max(1, tot["events"]), 3),
        "ln_notes_pct": round(100.0 * tot["ln_notes"] / max(1, tot["notes"]), 2),
        "offgrid_notes_pct": round(100.0 * tot["offgrid_notes_2ms"] / max(1, tot["notes"]), 2),
        "notes_inside_held_ln": tot["notes_inside_held_ln"],
        "relational_lane_codes_pct": round(100.0 * rel_share, 2),
        "lane_code_histogram": {LANE_CODE_NAMES[i]: int(c) for i, c in enumerate(code_hist)},
        "lane_code_entropy_bits": round(entropy_bits(code_hist), 3),
        "offset_in_frame_histogram_ms": [int(x) for x in offset_hist],
        "seconds": round(time.time() - t0, 1),
    }
    report = {"summary": summary, "failures": failures, "maps": per_map}
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print(" HYBRID PARAMETER SPACE - EXACT SOLVABILITY AUDIT")
    print("=" * 72)
    print(f" maps checked                 : {n_maps}")
    print(f" bit-exact round trips        : {tot['exact']} / {n_maps} ({summary['exact_roundtrip_pct']}%)")
    print(f" notes / events               : {tot['notes']:,} / {tot['events']:,}")
    print(f" overflow (degenerate) notes  : {tot['overflow']} ({summary['overflow_pct_of_notes']}% of notes, {summary['maps_with_overflow']} maps) {dict(overflow_reasons)}")
    print(f" staggered chords             : {summary['staggered_events_pct']}% of events")
    print(f" off-grid notes (>2 ms)       : {summary['offgrid_notes_pct']}% (kept exactly - the language is not grid-bound)")
    print(f" notes inside a held LN       : {tot['notes_inside_held_ln']} (unplayable overlaps present in the source maps)")
    print(f" relational lane codes        : {summary['relational_lane_codes_pct']}% of events (rest = explicit masks)")
    print(f" lane code entropy            : {summary['lane_code_entropy_bits']} bits/event (max {math.log2(N_LANE_CODES):.2f})")
    top = sorted(summary["lane_code_histogram"].items(), key=lambda kv: -kv[1])[:8]
    print(" most used lane codes         : " + ", ".join(f"{k}={v}" for k, v in top))
    print(f" report                       : {args.out} ({summary['seconds']}s)")
    if failures:
        print("\n EXACT ROUND TRIP FAILED FOR:")
        for f in failures[:20]:
            print("  - " + f)
        sys.exit(1)
    print("\n RESULT: every map of the dataset is reproduced 100% by its parameters.")


if __name__ == "__main__":
    main()
