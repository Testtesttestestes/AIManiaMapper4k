"""
Algorithmic lane polishing ("make it playable and nice").

The network decides WHEN notes happen, how many notes each moment has and which of them are long notes.
Its lane choice is the weakest part: it can produce runs of fast jacks on one lane or long passages for one
hand. This module keeps every time, chord size and long-note length exactly as decoded and re-assigns only
the lanes with a beam search over an ergonomic cost:

* fast repeats of the same lane (minijacks / jackhammers)          -> ``jack``
* one lane hammered in a short window (anchors)                     -> ``anchor``
* one hand carrying the load for too long (decaying hand balance)  -> ``balance``
* consecutive fast notes on the same hand (no hand alternation)    -> ``same_hand``
* two-note chords on one hand instead of split between hands       -> ``chord_shape``
* moving away from the network's own lane choice                   -> ``deviation`` (weighted by 1 - strength)

``strength`` 0 keeps the network's lanes, 1 only optimizes ergonomics; values in between keep the network's
patterns where they are already fine and repair the bad places.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .beatmap import Note, RawTimingPoint

NUM_LANES = 4
LEFT = (0, 1)
RIGHT = (2, 3)
MASKS_BY_SIZE: Dict[int, List[int]] = {k: [m for m in range(1, 16) if bin(m).count("1") == k] for k in range(1, 5)}


@dataclass
class ErgoWeights:
    jack: float = 4.0          # same lane in two consecutive events faster than jack_ms
    jack_ms: float = 180.0
    anchor: float = 1.0        # a lane used 3+ times within the last 4 events (fast)
    balance: float = 2.0       # squared hand imbalance of the decaying load
    balance_tau_ms: float = 700.0
    same_hand: float = 0.8     # two fast consecutive events entirely on the same hand
    same_hand_ms: float = 160.0
    chord_shape: float = 0.3   # two-note chord on one hand
    repeat: float = 0.35       # a back-and-forth pattern that keeps going (variety)
    deviation: float = 1.5     # per lane moved away from the network's choice (x (1 - strength))


def _hand_load(mask: int) -> Tuple[int, int]:
    return (mask & 1) + ((mask >> 1) & 1), ((mask >> 2) & 1) + ((mask >> 3) & 1)


def _group_events(notes: Sequence[Note], window_ms: int) -> List[List[Note]]:
    groups: List[List[Note]] = []
    for n in sorted(notes):
        if groups and n.start - groups[-1][0].start <= window_ms and all(m.col != n.col for m in groups[-1]):
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def improve_ergonomics(
    notes: Sequence[Note],
    timing_points: Optional[Sequence[RawTimingPoint]] = None,
    strength: float = 0.7,
    beam: int = 32,
    weights: Optional[ErgoWeights] = None,
    release_gap_ms: int = 40,
    release_gap_beats: float = 0.25,
    group_ms: int = 8,
) -> List[Note]:
    """Re-assigns lanes; times, chord sizes and LN lengths stay exactly as given."""
    if strength <= 0 or not notes:
        return sorted(notes)
    from .renderer import release_gap_fn, resolve_lane_conflicts

    w = weights or ErgoWeights()
    gap_fn = release_gap_fn(timing_points, release_gap_ms, release_gap_beats)
    dev_w = w.deviation * (1.0 - min(1.0, strength)) / max(0.05, min(1.0, strength))
    events = _group_events(notes, group_ms)

    # beam state: (cost, last_hit[4], busy_until[4], hand_load(L, R), last_masks(3), t_prev, backpointer)
    BIG = -10 ** 9
    states = [(0.0, (BIG,) * 4, (BIG,) * 4, (0.0, 0.0), (0, 0, 0), BIG, None)]
    history: List[List[Tuple[int, int]]] = []  # per event: list of (mask, parent index) of the kept states
    for ev in events:
        t = ev[0].start
        net_mask = sum(1 << n.col for n in ev)
        k = len(ev)
        gap_t = gap_fn(t)
        cand: List[Tuple[float, int, int, tuple]] = []
        for si, (cost, last_hit, busy, load, masks, t_prev, _) in enumerate(states):
            free = [c for c in range(NUM_LANES) if busy[c] + gap_t <= t]
            kk = min(k, len(free))
            if kk == 0:
                continue
            free_mask = sum(1 << c for c in free)
            dt = t - t_prev
            decay = math.exp(-max(0.0, dt) / w.balance_tau_ms)
            for mask in MASKS_BY_SIZE[kk]:
                if mask & ~free_mask:
                    continue
                c = 0.0
                lanes = [b for b in range(NUM_LANES) if mask >> b & 1]
                for b in lanes:
                    if masks[-1] >> b & 1 and dt < w.jack_ms:  # a jack: the previous event used this lane too
                        c += w.jack * (1.0 - dt / w.jack_ms) ** 0.5
                    if dt < 250 and sum(1 for m in masks if m >> b & 1) >= 2:
                        c += w.anchor
                l_, r_ = _hand_load(mask)
                L, R = load[0] * decay + l_, load[1] * decay + r_
                c += w.balance * ((L - R) / (L + R + 1.0)) ** 2
                prev = masks[-1]
                if dt < w.same_hand_ms and prev:
                    pl, pr = _hand_load(prev)
                    if (r_ == 0 and pr == 0) or (l_ == 0 and pl == 0):  # both events on one and the same hand
                        c += w.same_hand
                if mask == masks[-2] and masks[-1] == masks[-3] and masks[-1]:
                    c += w.repeat
                if kk == 2 and mask in (0b0011, 0b1100):
                    c += w.chord_shape
                if dev_w > 0:
                    c += dev_w * bin(mask ^ net_mask).count("1") / 2.0
                cand.append((cost + c, si, mask, (L, R)))
        if not cand:
            history.append([])
            continue
        cand.sort(key=lambda x: x[0])
        new_states, kept, seen = [], [], set()
        for total, si, mask, load in cand:
            cost, last_hit, busy, _load, masks, _t_prev, _bp = states[si]
            key = (mask, masks[-1], masks[-2], round(load[0] - load[1], 1))
            if key in seen:
                continue
            seen.add(key)
            lh, bu = list(last_hit), list(busy)
            lanes = [b for b in range(NUM_LANES) if mask >> b & 1]
            members = sorted(ev, key=lambda n: n.col)[: len(lanes)]
            for b, n in zip(lanes, members):
                lh[b] = n.start
                bu[b] = n.end if n.ln else BIG  # only held long notes block a lane (taps: the jack cost decides)
            new_states.append((total, tuple(lh), tuple(bu), load, masks[1:] + (mask,), t, len(kept)))
            kept.append((mask, si))
            if len(new_states) >= beam:
                break
        states = new_states
        history.append(kept)

    # backtrack the best path
    best = 0
    masks_path: List[Optional[int]] = [None] * len(events)
    idx = best
    for e in range(len(events) - 1, -1, -1):
        if not history[e]:
            continue
        mask, parent = history[e][idx]
        masks_path[e] = mask
        idx = parent
    out: List[Note] = []
    for ev, mask in zip(events, masks_path):
        members = sorted(ev, key=lambda n: n.col)
        if mask is None:
            out.extend(members)
            continue
        lanes = [b for b in range(NUM_LANES) if mask >> b & 1]
        for b, n in zip(lanes, members):
            out.append(Note(n.start, b, n.end, n.ln))
    return resolve_lane_conflicts(out, release_gap_ms, 30, gap_fn)


def ergonomics_report(notes: Sequence[Note], jack_ms: float = 150.0, run_ms: float = 200.0) -> Dict[str, float]:
    """Simple playability numbers: fast jacks, longest one-hand run, hand balance in 2 s windows."""
    events = _group_events(notes, 8)
    if len(events) < 2:
        return {"events": float(len(events))}
    jacks = 0
    prev_cols: set = set()
    run = best_run = 0
    prev_hand = None
    prev_t = -10 ** 9
    for ev in events:
        t = ev[0].start
        cols = {n.col for n in ev}
        if cols & prev_cols and t - prev_t < jack_ms:  # same lane in two consecutive events
            jacks += 1
        prev_cols = cols
        hand = "L" if cols <= set(LEFT) else "R" if cols <= set(RIGHT) else "B"
        if hand != "B" and hand == prev_hand and t - prev_t < run_ms:
            run += 1
        else:
            run = 1 if hand != "B" else 0
        best_run = max(best_run, run)
        prev_hand, prev_t = hand, t
    times = np.array([n.start for n in notes])
    left = np.array([n.col in LEFT for n in notes])
    imb = []
    for t0 in range(int(times.min()), int(times.max()) + 1, 2000):
        sel = (times >= t0) & (times < t0 + 2000)
        if sel.sum() >= 4:
            imb.append(abs(left[sel].mean() - 0.5) * 2.0)
    return {
        "events": float(len(events)),
        "fast_jack_pct": 100.0 * jacks / len(events),
        "max_one_hand_run": float(best_run),
        "hand_imbalance": float(np.mean(imb)) if imb else 0.0,   # 0 = both hands equal, 1 = one hand only
    }


def format_report(r: Dict[str, float]) -> str:
    return (f"fast jacks {r.get('fast_jack_pct', 0):.1f}% | longest one-hand run {int(r.get('max_one_hand_run', 0))} "
            f"| hand imbalance {r.get('hand_imbalance', 0):.2f}")
