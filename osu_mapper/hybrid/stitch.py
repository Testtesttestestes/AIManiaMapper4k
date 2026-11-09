"""
Stitching the best bars of several candidate maps of the same song and request (make_self_maps.py).

The reward of a whole map hides where it is bad: one candidate has a broken stream in the chorus, another a long
one-hand run in the bridge. With the per-event reward (reward.note_rewards) every bar (4 beats of the red lines) is
taken from the candidate that is best THERE, so the result can be better than every candidate as a whole.

* quality of a bar = the mean penalty of its events (not the sum: a denser bar is not worse for having more notes);
* only candidates with at least the median amount of notes of the candidates in that bar compete (``min_density``),
  so a sparse bar does not win for having less to penalize, and the star rating stays near the request (0.8 x the
  median lost 0.28* on average);
* switching costs ``switch_margin`` per event: the map stays with one candidate unless another one is clearly better
  (fewer seams, phrases stay whole).

Measured on 6 songs x 6.5/7.5* with the test_NN network (6 candidates): better than the best single candidate in
10 of 12 requests (reward +0.23 on average, ~24 seams per map, stars -0.07 from the request). The reward is what is
optimized here, so this shows the stitching works, not that the maps are better to play: seams are not judged
beyond jacks, runs and patterns;
* at a seam a note that would start on a lane still held by a long note of the previous bar is dropped.

The caller checks the result as a whole (real star rating, map reward) and keeps it only when it beats the best
single candidate.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .beatmap import Note, RawTimingPoint


def bar_index(times: np.ndarray, timing_points: Sequence[RawTimingPoint], beats_per_bar: int = 4) -> np.ndarray:
    from .motifs import beat_positions

    beats, _, _ = beat_positions(np.asarray(times, dtype=np.float64), timing_points)
    return np.floor((beats + 1e-6) / beats_per_bar).astype(np.int64)


def stitch_candidates(candidates: Sequence[Sequence[Note]], timing_points: Sequence[RawTimingPoint], target_sr: float,
                      lane_predictor=None, power: Optional[np.ndarray] = None, switch_margin: float = 1.0,
                      min_density: float = 1.0, release_gap_ms: int = 30) -> Tuple[List[Note], Dict[str, object]]:
    """(stitched notes, info: ``choice`` per bar, ``switches``, ``from`` = share of bars per candidate)."""
    from .reward import note_rewards

    cands = [sorted(c) for c in candidates if len(c) >= 8]
    if len(cands) < 2:
        return (list(cands[0]) if cands else []), {"choice": {}, "switches": 0, "from": [1.0]}
    per: List[Dict[int, Tuple[float, int]]] = []          # candidate -> bar -> (sum of penalties, events)
    bars_all = set()
    for c in cands:
        nr = note_rewards(c, target_sr, lane_predictor, power)
        bars = bar_index(nr["times"], timing_points)
        d: Dict[int, Tuple[float, int]] = {}
        for b, p in zip(bars.tolist(), nr["penalty"].tolist()):
            s, k = d.get(b, (0.0, 0))
            d[b] = (s + p, k + 1)
        per.append(d)
        bars_all.update(d)
    order = sorted(bars_all)
    # dynamic programming over bars: maximize sum over bars of (events x mean penalty of the chosen candidate) minus
    # switch_margin x events at every switch; candidates far below the usual density of a bar do not compete
    K = len(cands)
    NEG = -1e18
    score = np.zeros(K)
    back: List[np.ndarray] = []
    for b in order:
        counts = np.array([per[k].get(b, (0.0, 0))[1] for k in range(K)], dtype=np.float64)
        med = float(np.median(counts))
        local = np.full(K, NEG)
        for k in range(K):
            s, n = per[k].get(b, (0.0, 0))
            if n >= min_density * med:
                local[k] = s if n else 0.0
        n_bar = max(1.0, med)
        stay = score
        switch = score.max() - switch_margin * n_bar
        best_prev = np.where(stay >= switch, np.arange(K), int(score.argmax()))
        score = np.maximum(stay, switch) + local
        back.append(best_prev)
    choice_seq = [int(score.argmax())]
    for bp in reversed(back[1:]):
        choice_seq.append(int(bp[choice_seq[-1]]))
    choice_seq.reverse()
    choice = dict(zip(order, choice_seq))
    out: List[Note] = []
    held_until = {c: -10 ** 9 for c in range(4)}
    by_bar: Dict[int, List[Note]] = {}
    for k, c in enumerate(cands):
        bars = bar_index(np.array([n.start for n in c], dtype=np.float64), timing_points)
        for n, b in zip(c, bars.tolist()):
            if choice.get(b) == k:
                by_bar.setdefault(b, []).append(n)
    for b in sorted(by_bar):
        for n in sorted(by_bar[b]):
            if n.start < held_until.get(n.col, -10 ** 9) + release_gap_ms:
                continue                                    # lane still held by a long note of the previous bar
            out.append(n)
            held_until[n.col] = n.end if n.ln else n.start
    switches = sum(1 for a, b in zip(choice_seq, choice_seq[1:]) if a != b)
    share = [choice_seq.count(k) / max(1, len(choice_seq)) for k in range(K)]
    return out, {"choice": choice, "switches": switches, "from": share}


def preference_segments(chosen: Sequence[Note], rejected: Sequence[Note], timing_points: Sequence[RawTimingPoint],
                        target_sr: float, lane_predictor=None, power: Optional[np.ndarray] = None, margin: float = 0.5,
                        frame_ms: int = 10) -> List[Tuple[int, int]]:
    """Bars (as [start, end) frames) where the chosen map is clearly better than the rejected one by the per-event
    reward: the mean penalty per event differs by at least ``margin``. DPO trains on windows around them instead of
    random windows of the whole map, where the two maps are often equally good (the preference there is noise)."""
    from .reward import note_rewards

    def per_bar(notes):
        nr = note_rewards(notes, target_sr, lane_predictor, power)
        bars = bar_index(nr["times"], timing_points)
        d: Dict[int, List[float]] = {}
        for b, p in zip(bars.tolist(), nr["penalty"].tolist()):
            d.setdefault(b, []).append(p)
        return {b: float(np.mean(v)) for b, v in d.items()}

    if len(chosen) < 8 or len(rejected) < 8:
        return []
    pc, pr = per_bar(chosen), per_bar(rejected)
    good = sorted(b for b in set(pc) & set(pr) if pc[b] - pr[b] >= margin)
    if not good:
        return []
    # bar b -> its time span: invert beat_positions on a fine grid of the map's time range
    t_all = np.array([n.start for n in list(chosen) + list(rejected)], dtype=np.float64)
    grid = np.arange(max(0.0, t_all.min() - 4000.0), t_all.max() + 4000.0, float(frame_ms))
    gb = bar_index(grid, timing_points)
    out: List[Tuple[int, int]] = []
    for b in good:
        sel = np.flatnonzero(gb == b)
        if sel.size:
            out.append((int(grid[sel[0]] // frame_ms), int(grid[sel[-1]] // frame_ms) + 1))
    return out
