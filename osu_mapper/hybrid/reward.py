"""
Objective reward of a generated map (no human rating needed): the "RL from feedback" part of the training plan.

    r = w_sr * stars + w_play * playability + w_pattern * patterns + w_silence * silence

* stars      -|SR - target| (rosu-pp): the network must give what was asked;
* playability  fast jacks, the share of the map in one-hand runs longer than 8 notes (10 % costs 1), hand balance
               (ergonomics.ergonomics_report);
* patterns   how real maps of that star rating continue the chords (motif library, LanePredictor): mean
             log-probability of every chord after the six before it, against human maps of that rating (less typical
             than them costs, more typical - the same few figures everywhere - earns nothing);
* silence    no notes where the audio is silent (6 standard deviations of the song's log power below its loud parts,
             the composer's rule; features are standardized per song).

Used by make_self_maps.py to choose the best of several generated maps (ReST: the chosen ones are added to the
dataset with their REAL star rating, so the network gets hard examples where human ones are missing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .beatmap import Note, RawTimingPoint

def human_pattern_logp(sr: float) -> float:
    """Mean log P(chord | 6 chords before) of human maps under the library at a star rating: harder maps are more
    varied. Measured on 48 human maps of 9 songs: 1-3* -1.32, 3-5* -1.69, 5-7* -1.88, 7*+ -2.14 (linear fit)."""
    return -1.08 - 0.13 * float(sr)


@dataclass
class RewardWeights:
    stars: float = 3.0
    play: float = 1.0
    pattern: float = 1.0
    silence: float = 2.0


def pattern_logp(notes: Sequence[Note], predictor) -> float:
    """Mean log-probability of every chord after the six before it under the motif library."""
    groups: Dict[int, int] = {}
    for n in notes:
        groups[int(n.start)] = groups.get(int(n.start), 0) | (1 << int(n.col))
    masks = [groups[t] for t in sorted(groups)]
    if len(masks) < 8:
        return float("nan")
    lp = [float(predictor.logprob(masks[max(0, i - predictor.max_order):i])[masks[i]]) for i in range(1, len(masks))]
    return float(np.mean(lp))


def silent_share(notes: Sequence[Note], power: Optional[np.ndarray], frame_ms: int = 10, silence_std: float = 6.0) -> float:
    if power is None or not len(notes):
        return 0.0
    top = float(np.percentile(power, 95))
    silent = power < top - float(silence_std)
    f = np.clip(np.array([n.start for n in notes]) // frame_ms, 0, power.size - 1)
    return float(silent[f].mean())


def map_reward(notes: Sequence[Note], timing_points: Sequence[RawTimingPoint], target_sr: float, od: float = 8.0,
               lane_predictor=None, power: Optional[np.ndarray] = None, jack_target: Optional[float] = None,
               weights: Optional[RewardWeights] = None, sr: Optional[float] = None) -> Dict[str, float]:
    """Reward and its parts (higher is better)."""
    from .ergonomics import ergonomics_report
    from .inference import star_rating_of

    w = weights or RewardWeights()
    if len(notes) < 8:
        return {"reward": -1e9, "sr": 0.0}
    sr = float(star_rating_of(notes, timing_points, od)) if sr is None else float(sr)
    er = ergonomics_report(notes)
    jack = float(er.get("fast_jack_pct", 0.0)) / 100.0
    # long one-hand runs as the SHARE of the map inside them (human maps: ~0 %), not the single longest run: one bad
    # spot in a 4000-note map used to cost up to -25 and decided the choice between candidates on its own
    long_runs = float(er.get("long_run_pct", 0.0)) / 100.0
    r_play = -(abs(jack - jack_target) * 4.0 if jack_target is not None else jack) \
             - 10.0 * long_runs - float(er.get("hand_imbalance", 0.0))
    r_pat = 0.0
    if lane_predictor is not None:
        lp = pattern_logp(notes, lane_predictor)
        # as typical as a human map of this rating = 0; much less typical (random lanes) is bad, more typical than humans
        # (the same few figures everywhere) is not rewarded
        r_pat = min(0.0, lp - human_pattern_logp(target_sr)) if np.isfinite(lp) else 0.0
    sil = silent_share(notes, power)
    parts = {"sr": sr, "stars": -abs(sr - float(target_sr)), "play": r_play, "pattern": r_pat, "silence": -10.0 * sil}
    parts["reward"] = (w.stars * parts["stars"] + w.play * parts["play"] + w.pattern * parts["pattern"]
                       + w.silence * parts["silence"])
    return parts


def note_rewards(notes: Sequence[Note], target_sr: float, lane_predictor=None, power: Optional[np.ndarray] = None,
                 weights: Optional[RewardWeights] = None, frame_ms: int = 10, silence_std: float = 6.0,
                 pattern_window: int = 8) -> Dict[str, np.ndarray]:
    """The reward of ``map_reward`` spread over the map: a penalty (<= 0) for every event (chord), so that a bad spot
    can be found and replaced instead of costing the whole map.

    Per event: a fast jack (a lane of the previous event again within 150 ms), being inside a one-hand run longer than
    8 events (10 each, so their mean is the map's share term), the hand imbalance of its 2 s window, a note in silence
    (10 per note, the map's term is 10 x the share) and how much less typical its chords are than human maps of the
    rating (the motif library's log-probability, averaged over ``pattern_window`` events - a single unusual chord is
    variety, a run of them is a broken pattern). The stars are a property of the whole map and are not spread.

    Returns ``times`` (event start, ms), ``penalty`` (weighted sum per event), the parts (``jack``, ``long_run``,
    ``imbalance``, ``pattern``, ``silence``, unweighted) and ``event_of_note`` (index into the events for every note
    of ``sorted(notes)``)."""
    from .ergonomics import LEFT, RIGHT, _group_events

    w = weights or RewardWeights()
    srt = sorted(notes)
    events = _group_events(srt, 8)
    n = len(events)
    out: Dict[str, np.ndarray] = {k: np.zeros(n) for k in ("jack", "long_run", "imbalance", "pattern", "silence")}
    out["times"] = np.array([ev[0].start for ev in events], dtype=np.float64)
    if n == 0:
        out["penalty"] = np.zeros(0)
        out["event_of_note"] = np.zeros(0, dtype=np.int64)
        return out
    cols = [{m.col for m in ev} for ev in events]
    hands = ["L" if c <= set(LEFT) else "R" if c <= set(RIGHT) else "B" for c in cols]
    t = out["times"]
    run_start = 0
    run = 0
    for i in range(n):
        if i and cols[i] & cols[i - 1] and t[i] - t[i - 1] < 150.0:
            out["jack"][i] = 1.0
        if i and hands[i] != "B" and hands[i] == hands[i - 1] and t[i] - t[i - 1] < 200.0:
            run += 1
        else:
            run, run_start = (1 if hands[i] != "B" else 0), i
        if run > 8:
            out["long_run"][run_start:i + 1] = 10.0
    # hand balance of the 2 s window of each event
    left = np.array([sum(c in LEFT for c in cs) for cs in cols], dtype=np.float64)
    size = np.array([len(cs) for cs in cols], dtype=np.float64)
    win = ((t - t[0]) // 2000).astype(np.int64)
    for b in np.unique(win):
        sel = win == b
        if size[sel].sum() >= 4:
            out["imbalance"][sel] = abs(left[sel].sum() / size[sel].sum() - 0.5) * 2.0
    if lane_predictor is not None and n >= 8:
        masks = [sum(1 << c for c in cs) for cs in cols]
        lp = np.zeros(n)
        for i in range(1, n):
            lp[i] = float(lane_predictor.logprob(masks[max(0, i - lane_predictor.max_order):i])[masks[i]])
        lp[0] = lp[1]
        k = max(1, int(pattern_window))
        smooth = np.convolve(np.pad(lp, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
        out["pattern"] = np.minimum(0.0, smooth - human_pattern_logp(target_sr))
    if power is not None and power.size:
        top = float(np.percentile(power, 95))
        silent = power < top - float(silence_std)
        for i, ev in enumerate(events):
            f = np.clip(np.array([m.start for m in ev]) // frame_ms, 0, power.size - 1)
            out["silence"][i] = 10.0 * float(silent[f].sum())
    out["penalty"] = -(w.play * (out["jack"] + out["long_run"] + out["imbalance"]) + w.pattern * (-out["pattern"])
                       + w.silence * out["silence"])
    note_ev = {}
    for i, ev in enumerate(events):
        for m in ev:
            note_ev[id(m)] = i
    out["event_of_note"] = np.array([note_ev[id(m)] for m in srt], dtype=np.int64)
    return out
