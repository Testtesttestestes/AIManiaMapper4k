"""
The universal deterministic map algorithm.

* :func:`render_track` - exact rendering of a :class:`ParamTrack` into notes. Pure function, no
  heuristics: whatever the track says is what the map gets. ``render_track(solve_map(m)) == m``.
* :func:`decode_outputs` - the same algorithm driven by network probabilities instead of exact
  parameters. It runs in one O(N) pass over the detected events and is the only place where
  inference-time knobs (density, chord limits, jack limits, LN behaviour, snapping) act.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .beatmap import Note, RawTimingPoint
from .params import (
    CODE_POPCOUNT,
    FRAME_MS,
    LANE_TABLE,
    N_LANE_CODES,
    NUM_LANES,
    ParamTrack,
    ln_bucket_to_length,
    mask_lanes,
)


class InvalidTrackError(ValueError):
    pass


def render_track(track: ParamTrack, check_masks: bool = True) -> List[Note]:
    """Exact rendering: params -> notes (sorted canonically)."""
    H = int(track.frame_ms)
    notes: List[Note] = []
    prev = prev2 = 0
    for f in track.event_frames():
        code = int(track.lane_code[f])
        if not (0 <= code < N_LANE_CODES):
            raise InvalidTrackError(f"frame {f}: lane code {code} out of range")
        mask = int(LANE_TABLE[prev, prev2, code])
        if mask == 0:
            raise InvalidTrackError(f"frame {f}: lane code {code} is invalid for prev={prev:04b} prev2={prev2:04b}")
        if check_masks and int(track.event_mask[f]) not in (0, mask):
            raise InvalidTrackError(f"frame {f}: lane code renders {mask:04b} but event_mask says {int(track.event_mask[f]):04b}")
        base = f * H + int(track.time_class[f]) - 1
        for m, c in enumerate(mask_lanes(mask)):
            s = base + int(track.stagger[f, m])
            if track.ln_flag[f, m]:
                notes.append(Note(s, c, s + int(track.ln_len[f, m]), True))
            else:
                notes.append(Note(s, c, s, False))
        prev2, prev = prev, mask
    notes.extend(track.overflow)
    notes.sort()
    return notes


# ----------------------------------------------------------------------------------------------
# Inference-time decoding
# ----------------------------------------------------------------------------------------------

@dataclass
class HeadOutputs:
    """Network outputs for a whole song (numpy, float32). Produced by inference.py."""

    time_probs: np.ndarray        # [F, 1 + H] softmax over {no event, event at ms 0..H-1}
    lane_logits: np.ndarray       # [F, N_LANE_CODES]
    stagger: np.ndarray           # [F, 4] int, most likely extra ms offset of each chord member
    ln_logits: np.ndarray         # [F, 4]
    ln_bucket: np.ndarray         # [F, 4] int, most likely log2 length bucket of each member
    ln_resid: np.ndarray          # [F, 4] in [0, 1], position inside the bucket
    frame_ms: int = FRAME_MS
    chord_logprobs: Optional[np.ndarray] = None  # [F, 4] log P(chord size = 1..4)

    @property
    def num_frames(self) -> int:
        return int(self.time_probs.shape[0])


@dataclass
class DecodeKnobs:
    """Inference controls of the algorithm. Defaults reproduce the network's own decisions."""

    onset_threshold: float = 0.5        # P(event in frame) needed to place an event
    merge_ms: int = 3                   # events closer than this are merged (frame-boundary duplicates)
    min_event_gap_ms: int = 0           # optional hard minimum distance between events
    lane_temperature: float = 0.0       # 0 = argmax over valid lane codes, >0 = sampling
    lane_bias: Optional[np.ndarray] = None     # [N_LANE_CODES] additive logit bias (style control)
    chord_bias: Optional[np.ndarray] = None    # [5] additive logit bias by chord size 0..4
    max_chord: int = 4
    chord_weight: float = 1.0           # weight of the chord-size head in the lane decision (0 = ignore it)
    jack_min_ms: int = 0                # forbid a lane repeat faster than this (0 = allowed)
    ln_threshold: float = 0.5
    ln_min_ms: int = 30                 # shorter predicted LNs become taps
    ln_conflict: str = "avoid"          # "avoid": choose free lanes while an LN is held; "truncate": cut the LN
    ln_release_gap_ms: int = 1          # an LN must end at least this long before the next note on its lane
    use_stagger: bool = True
    onset_shift_ms: int = 0             # constant correction of note times (calibrated on validation, e.g. MP3 decoder delay)
    ln_length_scale: float = 1.0        # multiplies predicted LN lengths (calibrated on validation)
    snap_ms: float = 0.0                # snap note times to the nearest musical tick within this tolerance
    ln_snap_ms: Optional[float] = None  # snap LN releases to the PREVIOUS tick within this tolerance (None = same as snap_ms)
    snap_divisors: Tuple[int, ...] = (1, 2, 4, 3, 6, 8, 12, 16)
    seed: int = 0


def extract_events(
    time_probs: np.ndarray,
    threshold: float,
    frame_ms: int = FRAME_MS,
    merge_ms: int = 3,
    min_event_gap_ms: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Event detection from per-frame {none, ms_0..ms_H-1} probabilities.

    Returns (times_ms, frames, probs) sorted by time.
    """
    p_event = 1.0 - time_probs[:, 0]
    frames = np.flatnonzero(p_event >= threshold)
    if frames.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0, dtype=np.float32)
    offsets = np.argmax(time_probs[frames, 1:], axis=1)
    times = frames.astype(np.int64) * int(frame_ms) + offsets
    probs = p_event[frames].astype(np.float32)
    gap = max(int(merge_ms), int(min_event_gap_ms))
    if gap > 0 and times.size > 1:
        keep_t, keep_f, keep_p = [int(times[0])], [int(frames[0])], [float(probs[0])]
        for t, f, p in zip(times[1:], frames[1:], probs[1:]):
            if int(t) - keep_t[-1] < gap:
                if p > keep_p[-1]:
                    keep_t[-1], keep_f[-1], keep_p[-1] = int(t), int(f), float(p)
            else:
                keep_t.append(int(t))
                keep_f.append(int(f))
                keep_p.append(float(p))
        times = np.array(keep_t, dtype=np.int64)
        frames = np.array(keep_f, dtype=np.int64)
        probs = np.array(keep_p, dtype=np.float32)
    return times, frames, probs


class GridSnapper:
    """Snaps millisecond times to the nearest musical tick of the given red lines."""

    def __init__(self, timing_points: Sequence[RawTimingPoint], divisors: Sequence[int] = (1, 2, 4, 3, 6, 8, 12, 16)):
        reds = sorted((tp for tp in timing_points if tp.uninherited and tp.beat_len > 0), key=lambda t: t.time)
        self.times = np.array([tp.time for tp in reds], dtype=np.float64)
        self.beat_lens = np.array([tp.beat_len for tp in reds], dtype=np.float64)
        self.divisors = tuple(int(d) for d in divisors)

    @property
    def available(self) -> bool:
        return self.times.size > 0

    def nearest_tick(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = np.asarray(t, dtype=np.float64)
        idx = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, self.times.size - 1)
        off = self.times[idx]
        bl = self.beat_lens[idx]
        beats = (t - off) / bl
        best = np.full(t.shape, np.inf)
        best_tick = t.copy()
        for d in self.divisors:  # earlier divisors win ties (simpler rhythm preferred)
            tick = off + np.round(beats * d) * bl / d
            dist = np.abs(tick - t)
            better = dist < best - 1e-9
            best = np.where(better, dist, best)
            best_tick = np.where(better, tick, best_tick)
        return best_tick, best

    def previous_tick(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Latest tick at or before t (used for long-note releases, which must never move later)."""
        t = np.asarray(t, dtype=np.float64)
        idx = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, self.times.size - 1)
        off = self.times[idx]
        bl = self.beat_lens[idx]
        beats = (t - off) / bl
        best = np.full(t.shape, np.inf)
        best_tick = t.copy()
        for d in self.divisors:
            tick = off + np.floor(beats * d + 1e-6) * bl / d
            dist = t - tick
            better = dist < best - 1e-9
            best = np.where(better, dist, best)
            best_tick = np.where(better, tick, best_tick)
        return best_tick, best

    def snap(self, t: np.ndarray, tol_ms: float, direction: str = "nearest") -> np.ndarray:
        t = np.asarray(t)
        if not self.available or tol_ms <= 0 or t.size == 0:
            return t.astype(np.int64)
        tick, dist = self.previous_tick(t) if direction == "down" else self.nearest_tick(t)
        # osu! stores integer ms; editor-placed notes sit on floor(tick) in most clients, round is within 1 ms
        snapped = np.where(dist <= tol_ms, np.round(tick), t)
        return snapped.astype(np.int64)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def resolve_lane_conflicts(notes: List[Note], release_gap_ms: int = 1, ln_min_ms: int = 30) -> List[Note]:
    """Playability safety net: one note per lane per ms, no note inside a held LN on the same lane."""
    by_lane: List[List[Note]] = [[] for _ in range(NUM_LANES)]
    for n in notes:
        if 0 <= n.col < NUM_LANES:
            by_lane[n.col].append(n)
    out: List[Note] = []
    for lane_notes in by_lane:
        lane_notes.sort(key=lambda n: (n.start, 0 if n.ln else 1))
        dedup: List[Note] = []
        for n in lane_notes:
            if dedup and dedup[-1].start == n.start:
                continue  # LN sorted first wins over a duplicate tap
            dedup.append(n)
        for i, n in enumerate(dedup):
            if n.ln and i + 1 < len(dedup):
                limit = dedup[i + 1].start - release_gap_ms
                if n.end > limit:
                    n = Note(n.start, n.col, limit, True)
            if n.ln and n.end - n.start < ln_min_ms:
                n = Note(n.start, n.col, n.start, False)
            out.append(n)
    out.sort()
    return out


def decode_outputs(
    out: HeadOutputs,
    knobs: Optional[DecodeKnobs] = None,
    timing_points: Optional[Sequence[RawTimingPoint]] = None,
    events: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> List[Note]:
    """Network probabilities -> notes, through the same lane algebra as :func:`render_track`."""
    k = knobs or DecodeKnobs()
    H = int(out.frame_ms)
    if events is None:
        events = extract_events(out.time_probs, k.onset_threshold, H, k.merge_ms, k.min_event_gap_ms)
    ev_t, ev_f, ev_p = events
    if k.onset_shift_ms:
        ev_t = ev_t + int(k.onset_shift_ms)
    ln_snap = k.snap_ms if k.ln_snap_ms is None else k.ln_snap_ms

    snapper = GridSnapper(timing_points or [], k.snap_divisors) if ((k.snap_ms > 0 or ln_snap > 0) and timing_points) else None
    if snapper is not None and snapper.available and ev_t.size and k.snap_ms > 0:
        ev_t = snapper.snap(ev_t, k.snap_ms)
        order = np.argsort(ev_t, kind="stable")
        ev_t, ev_f, ev_p = ev_t[order], ev_f[order], ev_p[order]
        keep = np.ones(ev_t.size, dtype=bool)
        keep[1:] = np.diff(ev_t) > 0
        ev_t, ev_f, ev_p = ev_t[keep], ev_f[keep], ev_p[keep]

    rng = np.random.default_rng(k.seed)
    lane_bias = np.zeros(N_LANE_CODES) if k.lane_bias is None else np.asarray(k.lane_bias, dtype=np.float64)
    chord_bias = np.zeros(5) if k.chord_bias is None else np.asarray(k.chord_bias, dtype=np.float64)
    lane_bits = (1 << np.arange(NUM_LANES))

    prev = prev2 = 0
    last_start = np.full(NUM_LANES, -10 ** 9, dtype=np.int64)
    busy_until = np.full(NUM_LANES, -10 ** 9, dtype=np.int64)
    notes: List[Note] = []

    for t, f in zip(ev_t.tolist(), ev_f.tolist()):
        masks = LANE_TABLE[prev, prev2].astype(np.int64)
        pops = CODE_POPCOUNT[prev, prev2].astype(np.int64)
        logits = out.lane_logits[f].astype(np.float64) + lane_bias + chord_bias[pops]
        if out.chord_logprobs is not None and k.chord_weight > 0:
            logits = logits + k.chord_weight * out.chord_logprobs[f].astype(np.float64)[np.clip(pops - 1, 0, 3)]
        valid = (masks > 0) & (pops <= k.max_chord)

        busy_mask = int(lane_bits[busy_until + k.ln_release_gap_ms > t].sum()) if k.ln_conflict == "avoid" else 0
        jack_mask = int(lane_bits[(t - last_start) < k.jack_min_ms].sum()) if k.jack_min_ms > 0 else 0
        candidates = None
        for forbidden in (busy_mask | jack_mask, busy_mask, 0):
            cand = valid & ((masks & forbidden) == 0)
            if cand.any():
                candidates = cand
                break
        if candidates is None:
            continue
        cand_idx = np.flatnonzero(candidates)
        if k.lane_temperature > 0:
            probs = _softmax_np(logits[cand_idx] / k.lane_temperature)
            code = int(cand_idx[rng.choice(cand_idx.size, p=probs)])
        else:
            code = int(cand_idx[np.argmax(logits[cand_idx])])
        mask = int(masks[code])

        for m, c in enumerate(mask_lanes(mask)):
            st = int(out.stagger[f, m]) if k.use_stagger else 0
            s = int(t) + st
            p_ln = 1.0 / (1.0 + math.exp(-float(out.ln_logits[f, m])))
            if p_ln >= k.ln_threshold:
                b = int(out.ln_bucket[f, m])
                L = int(round(float(ln_bucket_to_length(b, float(out.ln_resid[f, m]))) * k.ln_length_scale))
                e = s + L
                if snapper is not None and snapper.available and ln_snap > 0:
                    e = int(snapper.snap(np.array([e]), ln_snap, direction="down")[0])
                if e - s >= k.ln_min_ms:
                    notes.append(Note(s, c, e, True))
                    busy_until[c] = e
                    last_start[c] = s
                    continue
            notes.append(Note(s, c, s, False))
            last_start[c] = s
            busy_until[c] = s
        prev2, prev = prev, mask

    return resolve_lane_conflicts(notes, k.ln_release_gap_ms, k.ln_min_ms)


def oracle_outputs(track: ParamTrack, sharpness: float = 30.0) -> HeadOutputs:
    """One-hot 'perfect network' outputs for a track (used to test decoding against exact rendering)."""
    from .params import ln_length_to_bucket

    F, H = track.num_frames, int(track.frame_ms)
    time_probs = np.zeros((F, 1 + H), dtype=np.float32)
    time_probs[np.arange(F), track.time_class.astype(np.int64)] = 1.0
    lane_logits = np.full((F, N_LANE_CODES), -sharpness, dtype=np.float32)
    ev = track.event_frames()
    lane_logits[ev, track.lane_code[ev].astype(np.int64)] = sharpness
    ln_logits = np.where(track.ln_flag, sharpness, -sharpness).astype(np.float32)
    bucket, resid = ln_length_to_bucket(np.maximum(track.ln_len, 1))
    chord = np.full((F, 4), -sharpness, dtype=np.float32)
    sizes = track.chord_sizes()
    chord[ev, np.clip(sizes[ev] - 1, 0, 3)] = 0.0
    return HeadOutputs(
        time_probs=time_probs,
        lane_logits=lane_logits,
        stagger=np.clip(track.stagger, 0, H - 1).astype(np.int64),
        ln_logits=ln_logits,
        ln_bucket=bucket.astype(np.int64),
        ln_resid=resid.astype(np.float32),
        frame_ms=H,
        chord_logprobs=chord,
    )
