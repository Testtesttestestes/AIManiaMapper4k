"""
How much of a map would mappers make long notes, for this music and this star rating? (as the climax model of
sections.py: learned from the dataset's maps, kept only if it beats the plain average on the validation maps)

Whether a song gets an LN map or a rice map is partly the mapper's choice, partly the music's: sustained sounds
(singing, pads, held melody notes) invite long notes, sharp percussive ones invite taps; and long notes give stars with
fewer notes. The network itself is told the LN share as part of the style; when the user does not give one
(generate_v5.py without --style ln_ratio=...), this model's estimate is given.

The features are standardized per song, so absolute levels say nothing; the descriptor uses what survives that:
how much each representation and each demucs stem changes from frame to frame (sustained music changes slowly),
how often each stem attacks, the tempo and the star rating. Ridge regression, a dozen numbers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

GROUPS = ("mel_mid", "pitch", "chroma", "hpss")


def song_descriptor(feats: np.ndarray, slices: Dict[str, Tuple[int, int]], timing_points, sr: float) -> np.ndarray:
    from ..hybrid.causes import stem_attacks

    x = []
    F = feats.shape[0]
    step = max(1, F // 6000)                                       # ~6000 frames are enough for the statistics
    for g in GROUPS:
        if g in slices:
            a, b = slices[g]
            v = np.asarray(feats[::step, a:b], dtype=np.float32)
            x.append(float(np.abs(np.diff(v, axis=0)).mean()) if v.shape[0] > 1 else 0.0)
        else:
            x.append(0.0)
    if "demucs_stems" in slices and slices["demucs_stems"][1] - slices["demucs_stems"][0] == 128:
        a, _ = slices["demucs_stems"]
        st = np.asarray(feats[::step, a:a + 128], dtype=np.float32).reshape(-1, 4, 32)
        x += [float(np.abs(np.diff(st[:, k], axis=0)).mean()) if st.shape[0] > 1 else 0.0 for k in range(4)]
        att = stem_attacks(feats, slices)
        x += [float((att[:, k] > 3.0).mean() * 100.0) for k in range(4)]       # attacks per second
    else:
        x += [0.0] * 8
    bpm = 120.0
    reds = [tp for tp in (timing_points or []) if getattr(tp, "uninherited", False) and tp.beat_len > 1]
    if reds:
        bpm = 60000.0 / float(reds[0].beat_len)
    x += [bpm / 200.0, float(sr) / 10.0, 1.0]
    return np.array(x, dtype=np.float64)


def fit(samples: Sequence[Tuple[np.ndarray, float]], l2: float = 1.0) -> List[float]:
    """Ridge regression descriptor -> LN share (the bias, last column, is not penalized)."""
    X = np.stack([d for d, _ in samples])
    y = np.array([t for _, t in samples], dtype=np.float64)
    mu = X[:, :-1].mean(0)
    sd = X[:, :-1].std(0) + 1e-6
    Z = np.concatenate([(X[:, :-1] - mu) / sd, X[:, -1:]], axis=1)
    R = l2 * np.eye(Z.shape[1])
    R[-1, -1] = 0.0
    w = np.linalg.solve(Z.T @ Z + R, Z.T @ y)
    return [float(v) for v in np.concatenate([w, mu, sd])]


def predict(model: Sequence[float], desc: np.ndarray) -> float:
    m = np.asarray(model, dtype=np.float64)
    n = desc.size
    w, mu, sd = m[:n], m[n:2 * n - 1], m[2 * n - 1:]
    z = np.concatenate([(desc[:-1] - mu) / sd, desc[-1:]])
    return float(np.clip(z @ w, 0.0, 1.0))
