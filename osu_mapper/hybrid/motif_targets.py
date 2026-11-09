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

import os
import zipfile

from typing import Dict, Optional, Sequence, Tuple

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


# plan of a map ("think first, then map"): the technique family of every bar (4 beats) - what a mapper decides before
# placing notes. Learnt as a target (plan head) and given back as an input (plan embedding), so a technique learnt at
# any difficulty can be asked for at any other.
PLAN_FAMILIES: Tuple[str, ...] = ("sparse", "stream", "roll/stairs", "trill", "jacks", "jumpstream", "handstream",
                                  "chordjack", "ln")
N_PLAN = len(PLAN_FAMILIES)
N_PLAN_COPY = len(COPY_KINDS) + 1       # no copy / exact / mirror / variation / rhythm


def bar_family(masks: Sequence[int], lnmasks: Sequence[int]) -> int:
    """Technique family of the chords of one bar (rules a mapper would use to name it)."""
    m = [int(x) & 15 for x in masks]
    n = len(m)
    if n < 3:
        return PLAN_FAMILIES.index("sparse")
    pop = [bin(x).count("1") for x in m]
    ln_share = float(np.mean([bool(x) for x in lnmasks])) if len(lnmasks) else 0.0
    if ln_share >= 0.4:
        return PLAN_FAMILIES.index("ln")
    big = float(np.mean([p >= 3 for p in pop]))
    two = float(np.mean([p >= 2 for p in pop]))
    overlap = float(np.mean([bool(m[i] & m[i - 1]) for i in range(1, n)]))
    if big >= 0.3:
        return PLAN_FAMILIES.index("chordjack" if overlap >= 0.5 else "handstream")
    if two >= 0.35:
        return PLAN_FAMILIES.index("chordjack" if overlap >= 0.5 else "jumpstream")
    same = float(np.mean([m[i] == m[i - 1] for i in range(1, n)]))
    if same >= 0.3:
        return PLAN_FAMILIES.index("jacks")
    if n >= 4 and float(np.mean([m[i] == m[i - 2] and m[i] != m[i - 1] for i in range(2, n)])) >= 0.5:
        return PLAN_FAMILIES.index("trill")
    cols = [int(np.mean([c for c in range(4) if (x >> c) & 1])) for x in m]
    steps = [cols[i] - cols[i - 1] for i in range(1, n)]
    mono = float(np.mean([abs(d) == 1 and (i == 0 or d == steps[i - 1]) for i, d in enumerate(steps)]))
    if mono >= 0.5:
        return PLAN_FAMILIES.index("roll/stairs")
    return PLAN_FAMILIES.index("stream")


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
    out["family"] = np.full(n, -1, dtype=np.int16)
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
        bar = np.floor((cs.beats - cs.beats[0]) / 4.0).astype(np.int64)
        for b in np.unique(bar):
            sel = np.flatnonzero(bar == b)
            out["family"][sel] = bar_family(cs.masks[sel], cs.lnmasks[sel])
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


TARGETS_VERSION = "motif-targets-v2"


def targets_for_track(track, sr: float, lib: Optional[MotifLibrary], cache_path=None) -> Dict[str, np.ndarray]:
    """``compute_targets`` for the events of a ParamTrack / SparseTrack, cached as .npz next to the dataset."""
    from pathlib import Path

    tag = f"{TARGETS_VERSION}:{getattr(lib, 'n_maps', 0)}:{getattr(lib, 'n_sets', 0)}:{round(float(sr), 3)}"
    if cache_path is not None and Path(cache_path).exists():
        try:
            z = np.load(cache_path, allow_pickle=False)
            if str(z["tag"]) == tag:
                return {k: z[k] for k in z.files if k != "tag"}
        except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile):
            pass                        # unreadable (an interrupted write): computed again below
    ev = track.event_frames()
    if hasattr(track, "time_class") and track.time_class.shape[0] == ev.shape[0]:  # SparseTrack: arrays are per event
        tc, em, lf, ll = track.time_class, track.event_mask, track.ln_flag, track.ln_len
    else:
        tc, em, lf, ll = track.time_class[ev], track.event_mask[ev], track.ln_flag[ev], track.ln_len[ev]
    out = compute_targets(ev, tc, em, lf, ll, track.timing_points, track.frame_ms, sr, lib)
    if cache_path is not None:
        # atomic: several training processes (DDP ranks, loader workers) build the same caches at the same time, and a
        # reader must never see a half-written file - each writes its own temporary file and renames it into place
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npz")
        try:
            np.savez_compressed(tmp, tag=np.array(tag), **out)
            os.replace(tmp, cache_path)
        except OSError:
            tmp.unlink(missing_ok=True)
    return out


def plan_from_logits(plan_logits: np.ndarray, timing_points, frame_ms: int = 10,
                     force_family: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """The network's plan as per-frame inputs for the second pass: per bar (4 beats of the red lines) the family and the
    copy kind with the highest summed log-probability; ``force_family`` puts one technique on every bar that is not
    'sparse' (any technique at any difficulty)."""
    from .motifs import beat_positions

    F = plan_logits.shape[0]
    lp = plan_logits.astype(np.float64)
    fam_lp = lp[:, :N_PLAN] - np.logaddexp.reduce(lp[:, :N_PLAN], axis=1, keepdims=True)
    cpy_lp = lp[:, N_PLAN:] - np.logaddexp.reduce(lp[:, N_PLAN:], axis=1, keepdims=True)
    beats, _, _ = beat_positions((np.arange(F) + 0.5) * frame_ms, timing_points)
    bar = np.floor(beats / 4.0).astype(np.int64)
    fam = np.zeros(F, np.int64)
    cpy = np.zeros(F, np.int64)
    for b in np.unique(bar):
        sel = bar == b
        fam[sel] = int(np.argmax(fam_lp[sel].sum(axis=0)))
        cpy[sel] = int(np.argmax(cpy_lp[sel].sum(axis=0)))
    if force_family:
        k = PLAN_FAMILIES.index(force_family)
        fam = np.where(fam == PLAN_FAMILIES.index("sparse"), fam, k)
    return fam, cpy
