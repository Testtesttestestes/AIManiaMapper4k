"""
Objective reward of a generated map (no human rating needed): the "RL from feedback" part of the training plan.

    r = w_sr * stars + w_play * playability + w_pattern * patterns + w_silence * silence

* stars      -|SR - target| (rosu-pp): the network must give what was asked;
* playability  fast jacks against what real maps of that star rating have, the longest one-hand run, hand balance
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
    r_play = -(abs(jack - jack_target) * 4.0 if jack_target is not None else jack) \
             - 0.25 * max(0.0, float(er.get("max_one_hand_run", 0.0)) - 6.0) - float(er.get("hand_imbalance", 0.0))
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
