"""
Musical thinning: easier versions of a real map on the same audio.

The network learns what changes between difficulties only if it sees the same music at several star
ratings, but most songs have one or two mapped difficulties and the hardest ranks have a handful of maps.
Thinning turns every hard map into a ladder of easier maps: events are removed from the weakest upwards
(off-beat 1/8-1/16 notes on quiet audio first, downbeats on loud attacks last) and chords are capped, the
way mappers simplify a difficulty. The star rating of every derived map is measured (rosu-pp), so the
network gets honest (audio, SR) -> map examples along the whole range.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .beatmap import Note, RawTimingPoint

# metric weight of a note on a tick of the given beat divisor (1 = on the beat)
DIVISOR_STRENGTH: Dict[int, float] = {1: 3.0, 2: 2.0, 4: 1.2, 3: 1.0, 6: 0.7, 8: 0.5, 12: 0.4, 16: 0.25}
OFF_GRID_STRENGTH = 0.2
DOWNBEAT_BONUS = 1.0


def metric_strength(t: float, reds: Sequence[RawTimingPoint], tol_ms: float = 3.0) -> float:
    if not reds:
        return 1.0
    times = [r.time for r in reds]
    i = max(0, int(np.searchsorted(times, t, side="right")) - 1)
    red = reds[i]
    bl = red.beat_len
    beats = (t - red.time) / bl
    for d, s in DIVISOR_STRENGTH.items():  # ordered from strongest to weakest
        tick = round(beats * d) / d
        if abs(tick - beats) * bl <= tol_ms:
            if d == 1 and int(round(beats)) % max(1, int(red.meter or 4)) == 0:
                return s + DOWNBEAT_BONUS
            return s
    return OFF_GRID_STRENGTH


def default_max_chord(keep: float) -> int:
    return 4 if keep > 0.75 else 3 if keep > 0.55 else 2 if keep > 0.35 else 1


def thin_map(
    notes: Sequence[Note],
    timing_points: Sequence[RawTimingPoint],
    keep: float,
    onset: Optional[np.ndarray] = None,
    frame_ms: int = 10,
    max_chord: Optional[int] = None,
    onset_weight: float = 0.6,
) -> List[Note]:
    """Keeps the strongest ``keep`` share of events (metric position + audio onset strength) and caps chords.

    ``onset``: per-frame onset strength of the audio (any scale, z-normalized internally), optional.
    """
    groups: Dict[int, List[Note]] = {}
    for n in notes:
        groups.setdefault(n.start, []).append(n)
    times = sorted(groups)
    if not times:
        return []
    reds = sorted((tp for tp in timing_points if tp.uninherited and tp.beat_len > 0), key=lambda r: r.time)
    if onset is not None and onset.size:
        z = (onset - float(onset.mean())) / (float(onset.std()) + 1e-6)
    else:
        z = None
    scores = []
    for t in times:
        s = metric_strength(t, reds)
        if z is not None:
            f = int(t // frame_ms)
            s += onset_weight * float(z[max(0, f - 1): f + 2].max()) if 0 <= f < z.size else 0.0
        s += 0.15 * (len(groups[t]) - 1)  # chords mark accents
        scores.append(s)
    n_keep = max(1, int(math.ceil(keep * len(times))))
    order = sorted(range(len(times)), key=lambda i: (-scores[i], times[i]))
    kept = sorted(times[i] for i in order[:n_keep])

    cap = max_chord if max_chord is not None else default_max_chord(keep)
    out: List[Note] = []
    prev_lanes: set = set()
    for t in kept:
        members = sorted(groups[t], key=lambda n: n.col)
        if len(members) > cap:
            members = _reduce_chord(members, cap, prev_lanes)
        out.extend(members)
        prev_lanes = {n.col for n in members}
    return _drop_overlaps(sorted(out))


def _reduce_chord(members: List[Note], cap: int, prev_lanes: set) -> List[Note]:
    # prefer long notes, then lanes not just hit (no accidental jacks), then the outer lanes (hand balance)
    def rank(n: Note) -> tuple:
        return (0 if n.ln else 1, 1 if n.col in prev_lanes else 0, min(n.col, 3 - n.col))
    if cap == 1:
        return [min(members, key=rank)]
    chosen = sorted(members, key=rank)[:cap]
    return sorted(chosen, key=lambda n: n.col)


def _drop_overlaps(notes: List[Note]) -> List[Note]:
    """Removing notes never creates overlaps, but keep the invariant explicit (one note per lane at a time)."""
    busy = [-1] * 8
    out = []
    for n in notes:
        if n.start <= busy[n.col]:
            continue
        out.append(n)
        busy[n.col] = n.end if n.ln else n.start
    return out


def onset_curve(feats: np.ndarray, onset_slice: Optional[tuple]) -> Optional[np.ndarray]:
    """Per-frame onset strength from the stacked feature matrix (total flux + superflux columns)."""
    if onset_slice is None:
        return None
    a, b = onset_slice
    block = np.asarray(feats[:, a:b], dtype=np.float32)
    return block[:, -2:].mean(axis=1) if block.shape[1] >= 2 else block.mean(axis=1)
