"""
Why is a note there? The sound source it follows (the "why" of the v5 network).

Mappers put notes on something they hear: a syllable of the singer (the burst of notes on "open your eyes"), a kick or
a snare, a bass line, a synth melody - or simply on the beat, with nothing in particular sounding. With the
``demucs_stems`` features (htdemucs: drums, bass, other, vocals; 32-band log-mel each, prepare_hybrid_dataset.py
--external demucs_stems) every note gets the source with the strongest attack around it:

* attack strength of a stem = positive spectral flux of its 32 bands (mean over the bands) as a robust z-score within
  the song ((flux - median) / (1.4826 x median absolute deviation)), so a quiet singer counts as much as a loud drum
  kit and a stem that is silent most of the time is not judged by its noise;
* a note's cause = the stem with the highest attack within ``window_ms`` of it, if that attack is at least
  ``min_strength`` (z >= 3 by default); otherwise "grid" (a note on the rhythm, not on a sound).

The labels train the cause head of the v5 network; the attack curves are also an input to it, and the share of
causes per phrase is part of its plan (which source this phrase follows).

    from osu_mapper.hybrid.causes import stem_attacks, note_causes, CAUSES
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

CAUSES: Tuple[str, ...] = ("drums", "bass", "other", "vocals", "grid")
STEMS: Tuple[str, ...] = CAUSES[:4]          # htdemucs order, as in external_reps._demucs
GRID = len(STEMS)


def stem_attacks(feats: np.ndarray, slices: Dict[str, Tuple[int, int]]) -> Optional[np.ndarray]:
    """[frames, 4] attack strength of drums / bass / other / vocals (robust z-score of the stem's spectral flux in this
    song), or None when the features have no demucs stems."""
    if "demucs_stems" not in slices:
        return None
    a, b = slices["demucs_stems"]
    x = np.asarray(feats[:, a:b], dtype=np.float32)
    if x.shape[1] != 128:
        return None
    x = x.reshape(x.shape[0], 4, 32)
    flux = np.maximum(0.0, np.diff(x, axis=0, prepend=x[:1])).mean(axis=2)          # [F, 4]
    med = np.median(flux, axis=0)
    mad = np.median(np.abs(flux - med), axis=0) * 1.4826
    return ((flux - med) / np.maximum(mad, 1e-6)).astype(np.float32)


def note_causes(times_ms: Sequence[float], attacks: Optional[np.ndarray], frame_ms: int = 10, window_ms: int = 20,
                min_strength: float = 3.0) -> np.ndarray:
    """Cause index (into CAUSES) of every time; all "grid" without attack curves."""
    t = np.asarray(times_ms, dtype=np.float64)
    out = np.full(t.size, GRID, dtype=np.int64)
    if attacks is None or not t.size:
        return out
    F = attacks.shape[0]
    w = max(0, int(round(window_ms / frame_ms)))
    f = np.clip((t // frame_ms).astype(np.int64), 0, F - 1)
    best = np.full((t.size, attacks.shape[1]), -np.inf, dtype=np.float32)
    for d in range(-w, w + 1):
        best = np.maximum(best, attacks[np.clip(f + d, 0, F - 1)])
    k = best.argmax(axis=1)
    strong = best[np.arange(t.size), k] >= min_strength
    out[strong] = k[strong]
    return out


def cause_shares(causes: np.ndarray) -> np.ndarray:
    """Share of every cause among the given notes ([5], sums to 1; uniform for no notes)."""
    if not len(causes):
        return np.full(len(CAUSES), 1.0 / len(CAUSES))
    c = np.bincount(np.asarray(causes, dtype=np.int64), minlength=len(CAUSES)).astype(np.float64)
    return c / c.sum()
