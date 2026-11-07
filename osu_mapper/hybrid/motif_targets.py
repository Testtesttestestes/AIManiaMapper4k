"""
Pattern knowledge for training the hybrid network: per-event targets from the motif analysis and the motif library.

The plain network learns lanes from one map at a time and never sees the structure of a map: which figure a chord
continues, whether a passage copies an earlier one, how real maps of the same star rating continue the last six
chords. This module turns all of that into per-event training targets (``train_hybrid.py --pattern_weight /
--prior_weight``):

auxiliary heads (``pattern_heads``), one class per event, computed over the WHOLE map (full context, not the crop):
    rhythm     gap to the previous chord in beats (1/16 .. 4, rest, off-grid)             18 classes  (motifs.py)
    shape      movement from the previous chord: size, shared lanes, direction, near/far  96 classes
    hand       left / right / both hands                                                    3 classes
    flow       gap in ms speeding up / steady / slowing x hold getting shorter / longer    80 classes
    copy       the chord lies in an earlier passage copied exactly / mirrored / with a      5 classes
               few lanes changed / with the same rhythm only / no copy (LZ77 parse)
    copy_dist  how far back that passage is (beats: <4 .. >256, 0 = no copy)                8 classes

library prior (distillation): for every event the motif library's distribution over the next chord mask given the
six chords before it, for maps of this star rating (``LanePredictor``). The lane loss then also pulls the network's
lane distribution (codes mapped to masks with the true previous chords) towards what 1606 real maps do there.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .beatmap import Note
from .motifs import (COPY_DIST_BEATS, COPY_KINDS, HAND, MotifLibrary, LanePredictor, chord_seq, self_repeats,
                     tokenize, VOCAB)

PATTERN_HEADS: Dict[str, int] = {
    "rhythm": VOCAB["rhythm"] // 4,       # 18
    "shape": VOCAB["shape"],              # 96
    "hand": 3,
    "flow": VOCAB["flow"],                # 80
    "copy": len(COPY_KINDS) + 1,          # none + exact / mirror / variation / rhythm
    "copy_dist": COPY_DIST_BEATS.size + 2,  # 0 = no copy, 1.. = distance bins
}


def event_notes(frames: np.ndarray, time_class: np.ndarray, event_mask: np.ndarray, ln_flag: np.ndarray,
                ln_len: np.ndarray, frame_ms: int) -> Sequence[Note]:
    """Notes of the events (chord members on their event's millisecond: staggers are ignored, one chord = one event)."""
    out = []
    for f, tc, m, lf, ll in zip(frames, time_class, event_mask, ln_flag, ln_len):
        t = int(f) * frame_ms + int(tc) - 1
        lanes = [c for c in range(4) if (int(m) >> c) & 1]
        for j, c in enumerate(lanes):
            if j < len(lf) and lf[j] and ll[j] > 0:
                out.append(Note(t, c, t + int(ll[j]), True))
            else:
                out.append(Note(t, c, t, False))
    return out


def compute_targets(frames: np.ndarray, time_class: np.ndarray, event_mask: np.ndarray, ln_flag: np.ndarray,
                    ln_len: np.ndarray, timing_points, frame_ms: int, sr: float,
                    lib: Optional[MotifLibrary] = None) -> Dict[str, np.ndarray]:
    """Per-event targets (arrays aligned with ``frames``); -1 where an event has no target."""
    n = int(frames.size)
    out = {k: np.full(n, -1, dtype=np.int16) for k in PATTERN_HEADS}
    out["prev"] = np.zeros((n, 2), dtype=np.uint8)
    if n == 0:
        out["prior"] = np.zeros((0, 16), dtype=np.float16)
        return out
    masks = event_mask.astype(np.int64) & 15
    out["prev"][1:, 0] = masks[:-1]
    out["prev"][2:, 1] = masks[:-2]
    notes = event_notes(frames, time_class, event_mask, ln_flag, ln_len, frame_ms)
    cs = chord_seq(notes, timing_points)
    if cs.size == n:  # one chord per event (always, unless two events share a millisecond)
        tok = tokenize(cs)
        out["rhythm"][:] = tok["rhythm"][0] // 4
        out["shape"][:] = tok["shape"][0]
        out["flow"][:] = tok["flow"][0]
        out["hand"][:] = HAND[cs.masks]
        copy = np.zeros(n, dtype=np.int16)
        dist = np.zeros(n, dtype=np.int16)
        if n >= 16:
            for c in self_repeats(cs, tok):
                k = COPY_KINDS.index(c.kind) + 1
                d = float(cs.beats[c.start] - cs.beats[c.src])
                copy[c.start:c.start + c.length] = k
                dist[c.start:c.start + c.length] = int(np.searchsorted(COPY_DIST_BEATS, d)) + 1
        out["copy"][:] = copy
        out["copy_dist"][:] = dist
    prior = np.zeros((n, 16), dtype=np.float16)
    if lib is not None and "lanes" in lib.tables:
        pred = LanePredictor.cached(lib, sr)
        for i in range(1, n):
            prior[i] = np.exp(pred.logprob(masks[max(0, i - pred.max_order):i]))
        prior[0] = 0.0          # no context: no prior
    out["prior"] = prior
    return out


TARGETS_VERSION = "motif-targets-v1"


def targets_for_track(track, sr: float, lib: Optional[MotifLibrary], cache_path=None) -> Dict[str, np.ndarray]:
    """``compute_targets`` for the events of a ParamTrack / SparseTrack, cached as .npz next to the dataset."""
    from pathlib import Path

    tag = f"{TARGETS_VERSION}:{getattr(lib, 'n_maps', 0)}:{getattr(lib, 'n_sets', 0)}:{round(float(sr), 3)}"
    if cache_path is not None and Path(cache_path).exists():
        try:
            z = np.load(cache_path, allow_pickle=False)
            if str(z["tag"]) == tag:
                return {k: z[k] for k in z.files if k != "tag"}
        except (OSError, KeyError, ValueError):
            pass
    ev = track.event_frames()
    if hasattr(track, "time_class") and track.time_class.shape[0] == ev.shape[0]:  # SparseTrack: arrays are per event
        tc, em, lf, ll = track.time_class, track.event_mask, track.ln_flag, track.ln_len
    else:
        tc, em, lf, ll = track.time_class[ev], track.event_mask[ev], track.ln_flag[ev], track.ln_len[ev]
    out = compute_targets(ev, tc, em, lf, ll, track.timing_points, track.frame_ms, sr, lib)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, tag=np.array(tag), **out)
    return out
