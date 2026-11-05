"""
Masked discrete diffusion over the map ("canvas"), MaskGIT-style.

The network normally decides every frame on its own: notes do not see each other. With the canvas it also gets a
partially known map of the same window - for every frame: unknown (masked), known empty, or a known event with its
absolute chord mask, millisecond inside the frame and which of its lanes start a long note. It sees known notes on
both sides of a frame, not only earlier ones, and nothing is generated step by step.

Training: a random share of frames is masked (cosine schedule, as MaskGIT: from almost nothing to everything; half the
crops mask random frames, half mask contiguous spans of 0.5-4 s), the time / lane / member losses count only the masked
frames. Some crops are fully masked: the plain one-pass network is the first step of the generation.

Generation (``refine``): K passes over the song. After every pass the map is fitted to the requested star rating as
usual; then the most confident share of its events and of its empty frames (cosine schedule) is fixed on the canvas,
and the next pass predicts the rest seeing them. The last pass fixes nothing: fixed frames keep their notes (their
event probability becomes 1 or 0), and the threshold fit still moves the remaining ones.

Research: MaskGIT (Chang et al., CVPR 2022), SoundStorm (2023), MaskGCT (ICLR 2025); for 4K charts the same masking
with an audio-attending denoiser in github.com/qxxiit/rhythm-chart-diffusion (2026, no comparison to other methods).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .beatmap import Note
from .params import NUM_LANES, mask_lanes

MASK, EMPTY = 0, 1
N_CANVAS_STATES = 17          # mask, empty, event with chord mask 1..15 (state = 1 + mask)


def event_state(mask: int) -> int:
    return 1 + int(mask)


def canvas_from_targets(tgt: Dict[str, np.ndarray], known: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canvas of a training crop: (state [T] int64, ms [T] int64, ln start bits per lane [T, 4] float32)."""
    tc = tgt["time_class"]
    em = tgt["event_mask"].astype(np.int64)
    T = tc.shape[0]
    state = np.where(tc > 0, 1 + em, EMPTY).astype(np.int64)
    ms = np.where(tc > 0, tc - 1, 0).astype(np.int64)
    ln = np.zeros((T, NUM_LANES), np.float32)
    ev = np.flatnonzero(tc > 0)
    flag = tgt["ln_flag"]
    for f in ev:
        for m, c in enumerate(mask_lanes(int(em[f]))):
            if m < flag.shape[1] and flag[f, m] > 0.5:
                ln[f, c] = 1.0
    state[~known] = MASK
    ms[~known] = 0
    ln[~known] = 0.0
    return state, ms, ln


def sample_known(T: int, rng: np.random.Generator, frame_ms: int = 10) -> np.ndarray:
    """Which frames of a training crop are given (True) - MaskGIT cosine schedule of the masked share."""
    masked_share = math.cos(0.5 * math.pi * float(rng.random()))
    known = np.ones(T, bool)
    if rng.random() < 0.5:
        known[rng.random(T) < masked_share] = False
    else:
        need = int(round(masked_share * T))
        tries = 0
        while (~known).sum() < need and tries < 64:
            L = int(rng.integers(int(500 / frame_ms), int(4000 / frame_ms) + 1))
            a = int(rng.integers(0, max(1, T - L + 1)))
            known[a:a + L] = False
            tries += 1
    return known


def empty_canvas(n_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.zeros(n_frames, np.int64), np.zeros(n_frames, np.int64), np.zeros((n_frames, NUM_LANES), np.float32)


def _note_frames(notes: Sequence[Note], shift_ms: int, frame_ms: int, n_frames: int, p_event: Optional[np.ndarray] = None):
    """Group notes by start time -> (frame, ms inside the frame, chord mask, LN bits). Snapping to the grid can move a
    note into a neighbouring frame: with ``p_event`` the frame is the most probable one within +-2 frames."""
    groups: Dict[int, List[Note]] = {}
    for n in notes:
        groups.setdefault(int(n.start), []).append(n)
    out = []
    for t, g in sorted(groups.items()):
        raw = t - int(shift_ms)
        f = raw // frame_ms
        if p_event is not None:
            lo, hi = max(0, f - 2), min(n_frames, f + 3)
            if hi > lo:
                f = lo + int(np.argmax(p_event[lo:hi]))
        if not 0 <= f < n_frames:
            continue
        mask = 0
        ln = np.zeros(NUM_LANES, np.float32)
        for n in g:
            mask |= 1 << int(n.col)
            if n.ln:
                ln[int(n.col)] = 1.0
        out.append((int(f), int(min(frame_ms - 1, max(0, raw - f * frame_ms))), mask, ln))
    return out


def refine(
    predict: Callable[[Tuple[np.ndarray, np.ndarray, np.ndarray]], object],
    fit: Callable[[object], object],
    n_frames: int,
    frame_ms: int,
    shift_of: Callable[[object], int],
    steps: int = 4,
    log=None,
    first=None,
):
    """K passes (see module docstring). ``predict(canvas) -> HeadOutputs``, ``fit(HeadOutputs) -> NetworkFit``,
    ``shift_of(fit)``: the ms the decoder added to frame times (onset shift minus lag). Returns (HeadOutputs, fit)."""
    from .params import code_for_mask

    state, ms, ln = empty_canvas(n_frames)
    frozen = None
    out = res = None
    for s in range(1, max(1, int(steps)) + 1):
        out = first if (s == 1 and first is not None) else predict((state, ms, ln))
        out = _force(out, state, ms, ln, frozen, code_for_mask)
        res = fit(out)
        if s >= steps:
            break
        share = 1.0 - math.cos(0.5 * math.pi * s / steps)
        p = 1.0 - out.time_probs[:, 0].astype(np.float64)
        evs = _note_frames(res.notes, shift_of(res), frame_ms, n_frames, p)
        is_ev = np.zeros(n_frames, bool)
        for f, _, _, _ in evs:
            is_ev[f] = True
        free = state == MASK
        new_state = state.copy()
        # events and empty frames are fixed by the same share, each by its own confidence
        cand_e = [e for e in evs if free[e[0]]]
        n_e = int(round(share * len(evs))) - int(((state >= 2)).sum())
        if n_e > 0 and cand_e:
            order = sorted(cand_e, key=lambda e: -p[e[0]])[:n_e]
            for f, m_, mask, lnb in order:
                new_state[f], ms[f], ln[f] = event_state(mask), m_, lnb
        # a frame the fitted map could still use (at or above its threshold, +-1 frame) is never fixed as empty
        live = p >= float(getattr(res, "threshold", 1.0))
        live[1:] |= live[:-1].copy()
        live[:-1] |= live[1:].copy()
        empt = np.flatnonzero(free & ~is_ev & ~live)
        n_x = int(round(share * (n_frames - len(evs)))) - int((state == EMPTY).sum())
        if n_x > 0 and empt.size:
            pick = empt[np.argsort(p[empt], kind="stable")[:n_x]]
            new_state[pick] = EMPTY
        fixed_now = (new_state != state)
        frozen = _freeze(frozen, out, fixed_now)
        state = new_state
        if log:
            log(f"    refine {s}/{steps}: fixed {int((state >= 2).sum())} events, {int((state == EMPTY).sum())} empty frames")
    return out, res


def _freeze(frozen, out, rows: np.ndarray):
    """Copy the head outputs of newly fixed frames (later passes are not trained there: only masked frames are)."""
    if frozen is None:
        frozen = {k: v.copy() for k, v in _rows(out).items()}
    else:
        for k, v in _rows(out).items():
            frozen[k][rows] = v[rows]
    return frozen


def _rows(out) -> Dict[str, np.ndarray]:
    d = {"lane_logits": out.lane_logits, "stagger": out.stagger, "ln_logits": out.ln_logits,
         "ln_bucket": out.ln_bucket, "ln_resid": out.ln_resid}
    if out.chord_logprobs is not None:
        d["chord_logprobs"] = out.chord_logprobs
    if out.importance is not None:
        d["importance"] = out.importance
    return d


def _force(out, state, ms, ln, frozen, code_for_mask):
    """Fixed frames keep their decision: event -> probability 1 at its ms with its chord, empty -> 0."""
    if frozen is None or not (state != MASK).any():
        return out
    fixed = state != MASK
    rows = {k: v.copy() for k, v in _rows(out).items()}
    for k in rows:
        rows[k][fixed] = frozen[k][fixed]
    tp = out.time_probs.copy()
    ev = np.flatnonzero(state >= 2)
    em = np.flatnonzero(state == EMPTY)
    tp[em] = 0.0
    tp[em, 0] = 1.0
    tp[ev] = 0.0
    tp[ev, 1 + ms[ev]] = 1.0
    lane = rows["lane_logits"]
    for f in ev:
        mask = int(state[f]) - 1
        lane[f, code_for_mask(mask)] += 12.0
        n = len(mask_lanes(mask))
        if "chord_logprobs" in rows:
            cl = np.full(rows["chord_logprobs"].shape[1], -12.0, np.float32)
            cl[min(n, cl.size) - 1] = 0.0
            rows["chord_logprobs"][f] = cl
        for m, c in enumerate(mask_lanes(mask)):
            if m < rows["ln_logits"].shape[1]:
                rows["ln_logits"][f, m] = 12.0 if ln[f, c] > 0.5 else -12.0
    return dataclasses.replace(out, time_probs=tp, **rows)
