"""
Two-sided lane choice: the lanes of every chord are re-chosen knowing the chords before AND after it.

The network decides where notes go well (its moments match human maps at F1 0.70-0.77 on maps it never saw) but not
which lanes: with the chord size known, its lane choice from audio hits the human chord for 0.31 of chords (random:
0.23). The lanes of human maps follow the pattern around them - and the chord AFTER matters more than older ones:
2 chords before 0.51, 3-8 before 0.52, 1 before + 1 after 0.61, 2 before + 1 after 0.64 (statistics from 48 maps,
measured on the 15 unseen maps). A decoder that goes left to right never sees the chord after; this pass does.

``two_sided_lanes`` keeps every note's time, the chord sizes and long notes, and in a few sweeps re-chooses each
chord's lane mask by

    log P_patterns(mask | 2 chords before, 1 after, gaps)  +  net_weight * log P_network(mask)

(the pattern model's two-sided counts of real maps of this star rating, ``PatternModel.lane_logprob_bi``; the network's
lane probabilities of that frame, summed over the lane codes that give the mask). Masks that would put a note into a
held long note, or on a lane hit less than ``min_same_lane_ms`` before or after, are not allowed. Every sweep sees the
choices of the previous one, so a pattern settles from both sides (iterated conditional modes).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .beatmap import Note
from .params import LANE_TABLE, NUM_LANES, mask_lanes

POP = np.array([bin(m).count("1") for m in range(16)])


def _events(notes: Sequence[Note]) -> Tuple[List[int], List[List[Note]]]:
    groups: Dict[int, List[Note]] = {}
    for n in notes:
        groups.setdefault(int(n.start), []).append(n)
    times = sorted(groups)
    return times, [sorted(groups[t], key=lambda n: (n.col, n.end)) for t in times]


def _net_mask_logp(out, frame: int, prev: int, prev2: int) -> Optional[np.ndarray]:
    """log P(mask) of the network at ``frame``: softmax over lane codes, summed per resulting mask."""
    if out is None or not (0 <= frame < out.num_frames):
        return None
    lg = out.lane_logits[frame].astype(np.float64)
    res = LANE_TABLE[prev & 15, prev2 & 15].astype(np.int64)
    ok = res > 0
    if not ok.any():
        return None
    e = np.exp(lg - lg[ok].max()) * ok
    p = np.bincount(res[ok], weights=e[ok], minlength=16)
    with np.errstate(divide="ignore"):
        return np.log(np.maximum(p / max(1e-12, p.sum()), 1e-9))


def two_sided_lanes(
    notes: Sequence[Note],
    patterns,
    star_rating: float,
    out=None,
    shift_ms: int = 0,
    net_weight: float = 0.5,
    sweeps: int = 3,
    min_same_lane_ms: int = 50,
    gap_fn: Optional[Callable[[int], int]] = None,
    temperature: float = 0.0,
    seed: int = 0,
) -> List[Note]:
    """See module docstring. ``out``: the network outputs (HeadOutputs) the notes came from, ``shift_ms`` the ms the
    decoder added to frame times (onset shift - lag); without ``out`` only the pattern statistics decide.
    ``temperature`` > 0 samples among the masks instead of taking the best (more variety, less typical)."""
    from .renderer import resolve_lane_conflicts

    if patterns is None or not getattr(patterns, "has_two_sided", False):
        return list(notes)
    times, evs = _events(notes)
    n = len(times)
    if n < 3:
        return list(notes)
    masks = np.array([sum(1 << nt.col for nt in e) & 15 for e in evs], dtype=np.int64)
    sizes = np.array([len({nt.col for nt in e}) for e in evs])
    if (sizes != np.array([len(e) for e in evs])).any():
        return list(notes)  # two notes on one lane at one time: leave such a map alone
    t = np.array(times, dtype=np.int64)
    H = int(out.frame_ms) if out is not None else 10
    pe = (1.0 - out.time_probs[:, 0]) if out is not None else None
    frames = np.zeros(n, dtype=np.int64)
    if out is not None:
        for i, x in enumerate(times):
            f = (x - int(shift_ms)) // H
            lo, hi = max(0, f - 2), min(out.num_frames, f + 3)
            frames[i] = lo + int(np.argmax(pe[lo:hi])) if hi > lo else -1
    gap = (lambda x: 30) if gap_fn is None else gap_fn
    rng = np.random.default_rng(seed)
    max_len = max([nt.end - nt.start for e in evs for nt in e if nt.ln] + [0])

    def lane_free(i: int, lane: int, member: Note) -> bool:
        s = int(t[i])
        e_ = member.end if member.ln else s
        # neighbours: everything that can reach this time on this lane (long notes up to max_len before)
        j = i - 1
        while j >= 0 and t[j] >= s - max_len - 500:
            if (masks[j] >> lane) & 1:
                q = evs[j][mask_lanes(int(masks[j])).index(lane)]
                q_end = q.end if q.ln else q.start
                if s - q.start < min_same_lane_ms or q_end + (gap(s) if q.ln else 0) > s:
                    return False
            j -= 1
        j = i + 1
        while j < n and t[j] <= e_ + 500:
            if (masks[j] >> lane) & 1:
                if t[j] - s < min_same_lane_ms or (member.ln and e_ + gap(int(t[j])) > t[j]):
                    return False
            j += 1
        return True

    lp_cache: Dict[tuple, np.ndarray] = {}
    for sweep in range(max(1, int(sweeps))):
        changed = 0
        for i in range(n):
            k = int(sizes[i])
            if k >= 4:
                continue
            p1 = int(masks[i - 1]) if i >= 1 else 0
            p2 = int(masks[i - 2]) if i >= 2 else 0
            nx = int(masks[i + 1]) if i + 1 < n else 0
            g_in = float(t[i] - t[i - 1]) if i >= 1 else 1e6
            g_out = float(t[i + 1] - t[i]) if i + 1 < n else 1e6
            key = (p2, p1, nx, int(np.searchsorted([90, 150, 250], g_in, side="right")),
                   int(np.searchsorted([90, 150, 250], g_out, side="right")), k, round(g_in / 25))
            if key not in lp_cache:
                lp_cache[key] = patterns.lane_logprob_bi(star_rating, p2, p1, nx, g_in, g_out, k)
            score = lp_cache[key].copy()
            if out is not None and net_weight > 0 and frames[i] >= 0:
                nl = _net_mask_logp(out, int(frames[i]), p1, p2)
                if nl is not None:
                    score = score + net_weight * nl
            cand = [m for m in np.flatnonzero((POP == k) & np.isfinite(score))]
            members = evs[i]
            allowed = []
            for m in cand:
                lanes = mask_lanes(int(m))
                if m == masks[i] or all(lane_free(i, c, members[q]) for q, c in enumerate(lanes)):
                    allowed.append(int(m))
            if not allowed:
                continue
            sc = np.array([score[m] for m in allowed])
            if temperature > 0:
                w = np.exp((sc - sc.max()) / temperature)
                pick = allowed[int(rng.choice(len(allowed), p=w / w.sum()))]
            else:
                pick = allowed[int(np.argmax(sc))]
            if pick != masks[i]:
                masks[i] = pick
                changed += 1
        if changed == 0:
            break
    out_notes: List[Note] = []
    for i, e in enumerate(evs):
        for q, c in enumerate(mask_lanes(int(masks[i]))):
            nt = e[q]
            out_notes.append(Note(nt.start, c, nt.end, nt.ln))
    return resolve_lane_conflicts(sorted(out_notes), 40, 30, gap_fn)
