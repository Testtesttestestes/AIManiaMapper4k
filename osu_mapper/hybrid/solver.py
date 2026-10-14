"""
Exact inverse of the universal algorithm: any 4K map -> canonical ParamTrack.

Guarantee (checked by :func:`verify_roundtrip`, the fuzz tests and hybrid_solvability_audit.py):

    render_track(solve_map(m)) == sorted(m.notes)      for every map m

Objects that the dense language cannot express are kept verbatim in ``track.overflow`` so the
guarantee holds even for broken maps; the audit reports how many such objects exist.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .beatmap import ManiaMap, Note, RawTimingPoint
from .params import (
    FRAME_MS,
    LANE_OPS,
    N_LANE_CODES,
    N_OPS,
    NUM_LANES,
    ParamTrack,
    canonical_lane_code,
    lanes_to_mask,
)
from .renderer import GridSnapper, render_track

STYLE_KEYS: Tuple[str, ...] = (
    "sr",              # star rating / 10
    "nps",             # log1p(notes per second over the mapped span) / log1p(30)
    "ln_ratio",        # long notes / all notes
    "chord_mean",      # (mean notes per event - 1) / 3
    "jack_ratio",      # SAME codes / events
    "alt_ratio",       # ALT codes / events
    "roll_ratio",      # SHIFT_R/L, ROT_R/L, SHIFT_R2/L2 codes / events
    "symmetric_ratio", # MIRROR, COMPLEMENT codes / events
    "explicit_ratio",  # explicit masks / events
    "offgrid_ratio",   # notes further than 2 ms from any 1/1..1/16 tick of the red lines
    "od",              # overall difficulty / 10
)
STYLE_DIM = len(STYLE_KEYS)


@dataclass
class SolveStats:
    notes: int = 0
    events: int = 0
    overflow: int = 0
    overflow_reasons: Dict[str, int] = field(default_factory=dict)
    staggered_events: int = 0
    ln_notes: int = 0
    overlaps_in_ln: int = 0
    code_hist: List[int] = field(default_factory=lambda: [0] * N_LANE_CODES)

    def as_dict(self) -> Dict[str, object]:
        return {
            "notes": self.notes,
            "events": self.events,
            "overflow": self.overflow,
            "overflow_reasons": dict(self.overflow_reasons),
            "staggered_events": self.staggered_events,
            "ln_notes": self.ln_notes,
            "overlaps_in_ln": self.overlaps_in_ln,
            "code_hist": list(self.code_hist),
        }


def solve_map(
    m: ManiaMap | Sequence[Note],
    num_frames: Optional[int] = None,
    frame_ms: int = FRAME_MS,
    timing_points: Optional[Sequence[RawTimingPoint]] = None,
    stats: Optional[SolveStats] = None,
) -> ParamTrack:
    """Canonical parameter track of a map (or of a plain note list)."""
    if isinstance(m, ManiaMap):
        notes = sorted(m.notes)
        tps = list(m.timing_points) if timing_points is None else list(timing_points)
    else:
        notes = sorted(m)
        tps = list(timing_points or [])
    H = int(frame_ms)
    st = stats if stats is not None else SolveStats()

    regular: List[Note] = []
    overflow: List[Note] = []

    def _overflow(n: Note, reason: str) -> None:
        overflow.append(n)
        st.overflow_reasons[reason] = st.overflow_reasons.get(reason, 0) + 1

    for n in notes:
        if not (0 <= n.col < NUM_LANES):
            _overflow(n, "lane_out_of_range")
        elif n.start < 0:
            _overflow(n, "negative_time")
        elif n.ln and n.end <= n.start:
            _overflow(n, "ln_non_positive_length")
        elif (not n.ln) and n.end != n.start:
            _overflow(n, "tap_with_end_time")
        else:
            regular.append(n)

    groups: Dict[int, Dict[int, Note]] = defaultdict(dict)
    for n in regular:  # sorted by start, so the earliest note of a lane inside a frame is kept
        f = n.start // H
        if n.col in groups[f]:
            _overflow(n, "same_lane_twice_in_frame")
            continue
        groups[f][n.col] = n

    last_frame = max(groups) if groups else -1
    F = max(int(num_frames or 0), last_frame + 1)
    track = ParamTrack.empty(F, frame_ms=H)
    track.timing_points = tps

    prev = prev2 = 0
    for f in sorted(groups):
        members = [groups[f][c] for c in sorted(groups[f])]
        mask = lanes_to_mask([n.col for n in members])
        base = min(n.start for n in members)
        track.time_class[f] = base - f * H + 1
        track.event_mask[f] = mask
        code = canonical_lane_code(mask, prev, prev2)
        track.lane_code[f] = code
        st.code_hist[code] += 1
        staggered = False
        for mi, n in enumerate(members):
            track.stagger[f, mi] = n.start - base
            staggered = staggered or (n.start != base)
            if n.ln:
                track.ln_flag[f, mi] = True
                track.ln_len[f, mi] = n.end - n.start
                st.ln_notes += 1
        st.staggered_events += int(staggered)
        st.events += 1
        prev2, prev = prev, mask

    track.overflow = sorted(overflow)
    st.notes += len(notes)
    st.overflow += len(overflow)
    st.overlaps_in_ln += count_notes_inside_lns(notes)
    return track


def count_notes_inside_lns(notes: Sequence[Note]) -> int:
    """Number of notes that start while an LN is held on the same lane (unplayable overlaps)."""
    count = 0
    by_lane: Dict[int, List[Note]] = defaultdict(list)
    for n in notes:
        by_lane[n.col].append(n)
    for lane_notes in by_lane.values():
        lane_notes.sort()
        held_until = -10 ** 9
        for n in lane_notes:
            if n.start <= held_until:
                count += 1
            if n.ln:
                held_until = max(held_until, n.end)
    return count


def verify_roundtrip(m: ManiaMap | Sequence[Note], frame_ms: int = FRAME_MS) -> Tuple[bool, ParamTrack, List[Note]]:
    """Solve -> render and compare bit-exactly. Returns (ok, track, rendered_notes)."""
    track = solve_map(m, frame_ms=frame_ms)
    rendered = render_track(track)
    original = sorted(m.notes) if isinstance(m, ManiaMap) else sorted(m)
    return rendered == original, track, rendered


def compute_style(
    track: ParamTrack,
    notes: Sequence[Note],
    star_rating: Optional[float] = None,
    overall_difficulty: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Global style vector (normalized, ~[0, 1]) and its known-mask, derived from the exact track."""
    style = np.zeros(STYLE_DIM, dtype=np.float32)
    known = np.zeros(STYLE_DIM, dtype=np.float32)

    def _set(key: str, value: Optional[float]) -> None:
        if value is None or not np.isfinite(value):
            return
        i = STYLE_KEYS.index(key)
        style[i] = float(value)
        known[i] = 1.0

    n_notes = len(notes)
    ev = track.event_frames()
    n_ev = int(ev.size)
    _set("sr", None if star_rating is None else star_rating / 10.0)
    _set("od", None if overall_difficulty is None else overall_difficulty / 10.0)
    if n_notes >= 2:
        span_s = max(1.0, (max(n.start for n in notes) - min(n.start for n in notes)) / 1000.0)
        _set("nps", float(np.log1p(n_notes / span_s) / np.log1p(30.0)))
    if n_notes:
        _set("ln_ratio", sum(1 for n in notes if n.ln) / n_notes)
    if n_ev:
        _set("chord_mean", (track.chord_sizes()[ev].mean() - 1.0) / 3.0)
        codes = track.lane_code[ev].astype(np.int64)[1:]  # first event has no history
        if codes.size:
            names = np.array(LANE_OPS + ("EXPLICIT",))
            op_idx = np.minimum(codes, N_OPS)
            groups = {
                "jack_ratio": ("SAME",),
                "alt_ratio": ("ALT",),
                "roll_ratio": ("SHIFT_R", "SHIFT_L", "ROT_R", "ROT_L", "SHIFT_R2", "SHIFT_L2"),
                "symmetric_ratio": ("MIRROR", "COMPLEMENT"),
                "explicit_ratio": ("EXPLICIT",),
            }
            for key, ops in groups.items():
                _set(key, float(np.isin(names[op_idx], ops).mean()))
    snapper = GridSnapper(track.timing_points, divisors=(1, 2, 3, 4, 6, 8, 12, 16))
    if snapper.available and n_notes:
        _, dist = snapper.nearest_tick(np.array([n.start for n in notes], dtype=np.float64))
        _set("offgrid_ratio", float((dist > 2.0).mean()))
    return style, known
