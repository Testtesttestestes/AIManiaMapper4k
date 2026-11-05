"""
Network-first maps at a requested star rating.

The network's own map (its event probabilities decoded by the universal algorithm) plays best: it only puts
notes where it hears something. Its weaknesses are (1) the star rating: one threshold gives whatever density the
network finds typical, often 0.5-3 stars away from the request, and (2) on some audio its milliseconds are shifted
against the template's grid (a constant per song, up to ~10 ms).

``fit_network_map`` keeps everything the network decides (which moments, chords, lanes, long notes) and only
(1) moves the event threshold until the rendered map has the requested star rating, never below ``min_threshold``
(below it the network does not hear a note: such notes would be on silence), and (2) subtracts the per-song lag.
If the requested star rating needs more notes than the network hears, the map stays at the closest reachable
rating and the result says so, instead of filling silence.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .beatmap import ManiaMap, Note, RawTimingPoint, map_to_osu_text
from .renderer import DecodeKnobs, HeadOutputs, decode_outputs


@dataclass
class NetworkFit:
    notes: List[Note]
    star_rating: float
    threshold: float
    lag_ms: int
    reached: bool          # False: the requested rating needs notes the network does not hear (or fewer than it is sure of)


def star_rating(notes: Sequence[Note], timing_points: Sequence[RawTimingPoint], od: float = 8.0) -> float:
    from osu_mapper.dataset.difficulty_calculator import calculate_osu_mania_star_rating

    if not notes:
        return 0.0
    tps = list(timing_points) or [RawTimingPoint(0.0, 500.0)]
    text = map_to_osu_text(ManiaMap(notes=list(notes), timing_points=tps, overall_difficulty=od))
    return float(calculate_osu_mania_star_rating(osu_text=text, hit_objects=[(n.col, n.start, n.end, n.ln) for n in notes],
                                                 overall_difficulty=od, keys=4))


def network_lag(out: HeadOutputs, timing_points: Sequence[RawTimingPoint], min_events: int = 30) -> int:
    """Median offset (ms) of the network's confident events against the beat / half-beat ticks of the red lines."""
    reds = sorted((tp for tp in timing_points if tp.uninherited and tp.beat_len > 0), key=lambda r: r.time)
    if not reds:
        return 0
    pe = 1.0 - out.time_probs[:, 0]
    fr = np.flatnonzero((pe >= 0.5) & (pe >= np.roll(pe, 1)) & (pe >= np.roll(pe, -1)))
    if fr.size < min_events:
        return 0
    ms = fr * out.frame_ms + np.argmax(out.time_probs[fr, 1:], axis=1)
    ticks = []
    end = float(ms.max()) + 1000.0
    for i, r in enumerate(reds):
        stop = reds[i + 1].time if i + 1 < len(reds) else end
        start = r.time - np.floor(r.time / (r.beat_len / 2)) * (r.beat_len / 2) if i == 0 else r.time
        ticks.append(np.arange(start, stop, r.beat_len / 2))
    t = np.sort(np.round(np.concatenate(ticks)).astype(np.int64))
    j = np.clip(np.searchsorted(t, ms), 1, t.size - 1)
    a, b = ms - t[j], ms - t[j - 1]
    d = np.where(np.abs(a) < np.abs(b), a, b)
    d = d[np.abs(d) <= 20]
    return int(round(float(np.median(d)))) if d.size >= min_events else 0


def fit_network_map(
    out: HeadOutputs,
    knobs: DecodeKnobs,
    timing_points: Sequence[RawTimingPoint],
    target_sr: float,
    od: float = 8.0,
    min_threshold: float = 0.15,
    max_threshold: float = 0.95,
    tolerance: float = 0.05,
    iterations: int = 14,
    fix_lag: bool = True,
) -> NetworkFit:
    """Decodes the network's map at the event threshold whose rendered star rating matches ``target_sr``."""
    lag = network_lag(out, timing_points) if fix_lag else 0
    base = dataclasses.replace(knobs, onset_shift_ms=int(knobs.onset_shift_ms) - lag)
    cache = {}

    def run(th: float):
        th = round(float(th), 4)
        if th not in cache:
            notes = decode_outputs(out, dataclasses.replace(base, onset_threshold=th), timing_points)
            cache[th] = (star_rating(notes, timing_points, od), notes)
        return cache[th]

    lo, hi = float(min_threshold), float(max_threshold)
    sr_lo, _ = run(lo)          # the most notes the network hears
    sr_hi, _ = run(hi)          # only what it is sure about
    if target_sr >= sr_lo:
        sr, notes = run(lo)
        return NetworkFit(notes, sr, lo, lag, abs(sr - target_sr) <= tolerance)
    if target_sr <= sr_hi:
        sr, notes = run(hi)
        return NetworkFit(notes, sr, hi, lag, abs(sr - target_sr) <= tolerance)
    for _ in range(iterations):  # star rating falls as the threshold rises
        mid = 0.5 * (lo + hi)
        sr, _ = run(mid)
        if abs(sr - target_sr) <= tolerance:
            break
        if sr > target_sr:
            lo = mid
        else:
            hi = mid
    th, (sr, notes) = min(cache.items(), key=lambda kv: abs(kv[1][0] - target_sr))
    return NetworkFit(notes, sr, th, lag, abs(sr - target_sr) <= tolerance)
