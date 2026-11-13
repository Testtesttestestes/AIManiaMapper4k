"""
Generating a map with the v5 canvas (as the v4 canvas: everything that is known stays visible, what is missing is
restored).

The whole song is one canvas (frames x lanes, osu_mapper/v5/frames.py); every frame is either known or hidden.

1. Windows of ~10 s (half a window apart) go over the song. In each, the hidden frames are revealed in ``steps``
   rounds (MaskGIT, Chang et al. 2022): the network predicts every hidden frame from everything visible on BOTH sides,
   a state is sampled for every lane, and the frames it is surest about are kept, the others hidden again for the
   next round - so a burst is decided as a whole, with what is around it in view.
2. Self-correction (the v4 ``correct_steps``): the error head marks the visible frames it thinks are wrong; at most
   ``refine_frac`` of the note frames of a window (and their neighbours) are hidden and restored again; the change
   is kept only if the error head then finds fewer errors there (and the score, when a critic / reward is given, is
   not worse) - no endless polishing.
3. Search: ``candidates`` K restorations of every window are sampled at once, ``score_fn`` (critic + per-note
   reward) keeps the best.
4. Control: ``fixed`` notes (e.g. the user's own map) stay exactly as they are; ``regen`` time ranges are hidden and
   restored around them - edit any part of a map with the rest in view.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..hybrid.beatmap import Note
from . import frames as fr
from .model import V5Net


def _rows(a: np.ndarray, start: int, length: int) -> np.ndarray:
    from ..hybrid.dataset import _crop_rows
    return _crop_rows(a, start, length)


class SongCanvas:
    """The whole song's canvas: state / start offset / end offset per frame and lane, which frames are known, which
    are locked (never hidden again)."""

    def __init__(self, n_frames: int, frame_ms: int = 10):
        self.F = int(n_frames)
        self.frame_ms = frame_ms
        self.state = np.zeros((self.F, fr.NUM_LANES), dtype=np.int8)
        self.start = np.full((self.F, fr.NUM_LANES), -1, dtype=np.int8)
        self.end = np.full((self.F, fr.NUM_LANES), -1, dtype=np.int8)
        self.known = np.zeros(self.F, dtype=bool)
        self.locked = np.zeros(self.F, dtype=bool)

    def put_notes(self, notes: Sequence[Note], lock: bool = True, frames: Optional[np.ndarray] = None) -> None:
        st, so, eo, _ = fr.encode(notes, 0, self.F, self.frame_ms)
        sel = np.ones(self.F, dtype=bool) if frames is None else frames
        self.state[sel], self.start[sel], self.end[sel] = st[sel], so[sel], eo[sel]
        self.known |= sel
        if lock:
            self.locked |= sel

    def window(self, s: int, W: int):
        e = min(self.F, s + W)
        pad = W - (e - s)
        st, so, eo = self.state[s:e], self.start[s:e], self.end[s:e]
        known = self.known[s:e]
        if pad:
            st = np.concatenate([st, np.zeros((pad, fr.NUM_LANES), np.int8)])
            so = np.concatenate([so, np.full((pad, fr.NUM_LANES), -1, np.int8)])
            eo = np.concatenate([eo, np.full((pad, fr.NUM_LANES), -1, np.int8)])
            known = np.concatenate([known, np.ones(pad, bool)])          # past the end: empty and known
        return st.copy(), so.copy(), eo.copy(), known.copy()

    def notes(self) -> List[Note]:
        return fr.decode(self.state, self.start, self.end, 0, self.frame_ms)


@torch.no_grad()
def _decode_logits(model: V5Net, memory, cond, cond_null, st, so, eo, hidden, cfg_scale: float):
    """Canvas logits for a batch of rows (classifier-free guidance when cfg_scale != 1)."""
    dev = memory.device
    T = lambda a: torch.from_numpy(np.asarray(a)).to(dev)
    s_in, a_in, e_in = zip(*[fr.inputs(st[r], so[r], eo[r], hidden[r]) for r in range(len(st))])
    s_in, a_in, e_in = T(np.stack(s_in)), T(np.stack(a_in)), T(np.stack(e_in))
    h = T(np.stack(hidden).astype(np.int64))
    out = model.canvas_decode(memory, cond, s_in, a_in, e_in, h)
    if cfg_scale != 1.0:
        un = model.canvas_decode(memory, cond_null, s_in, a_in, e_in, h)
        out = {k: (un[k] + cfg_scale * (out[k] - un[k])) if k != "error" else out[k] for k in out}
    return out


def allowed_states(st: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    """[K, W, 4, 5] bool: the states a lane may take in a frame given its KNOWN neighbours (a long note is HEAD,
    HOLD..., TAIL; nothing starts within 30 ms after a note on the same lane). Unknown neighbours allow everything; a
    contradiction allows everything (the decoder repairs it)."""
    K, W, L = st.shape
    ok = np.ones((K, W, L, fr.N_STATES), dtype=bool)
    known = ~hidden[..., None]                                                    # [K, W, 1]
    prev = np.concatenate([np.full((K, 1, L), -1, st.dtype), st[:, :-1]], axis=1)
    prev_known = np.concatenate([np.zeros((K, 1, 1), bool), known[:, :-1]], axis=1)
    nxt = np.concatenate([st[:, 1:], np.full((K, 1, L), -1, st.dtype)], axis=1)
    nxt_known = np.concatenate([known[:, 1:], np.zeros((K, 1, 1), bool)], axis=1)
    open_before = prev_known & ((prev == fr.HEAD) | (prev == fr.HOLD))
    closed_before = prev_known & ((prev == fr.EMPTY) | (prev == fr.TAP) | (prev == fr.TAIL))
    for s_ in (fr.EMPTY, fr.TAP, fr.HEAD):
        ok[..., s_] &= ~open_before                                               # a long note is still held
    for s_ in (fr.HOLD, fr.TAIL):
        ok[..., s_] &= ~closed_before                                             # no hold without a head
    holds_after = nxt_known & ((nxt == fr.HOLD) | (nxt == fr.TAIL))
    free_after = nxt_known & ((nxt == fr.EMPTY) | (nxt == fr.TAP) | (nxt == fr.HEAD))
    for s_ in (fr.EMPTY, fr.TAP, fr.TAIL):
        ok[..., s_] &= ~holds_after
    for s_ in (fr.HEAD, fr.HOLD):
        ok[..., s_] &= ~free_after
    # 30 ms: no TAP / HEAD within two frames after a known TAP / HEAD / TAIL of the same lane
    near = np.zeros((K, W, L), dtype=bool)
    for d in (1, 2):
        p_ = np.concatenate([np.full((K, d, L), -1, st.dtype), st[:, :-d]], axis=1)
        pk = np.concatenate([np.zeros((K, d, 1), bool), known[:, :-d]], axis=1)
        near |= pk & ((p_ == fr.TAP) | (p_ == fr.HEAD) | (p_ == fr.TAIL))
    ok[..., fr.TAP] &= ~near
    ok[..., fr.HEAD] &= ~near
    none = ~ok.any(-1)
    ok[none] = True
    return ok


@torch.no_grad()
def restore_window(model: V5Net, memory, cond, cond_null, st, so, eo, hidden, steps: int = 8, temperature: float = 1.0,
                   cfg_scale: float = 1.0, rng: Optional[np.random.Generator] = None):
    """Reveals the ``hidden`` [K, W] frames of K rows of one window in ``steps`` rounds; returns (st, so, eo) [K, W, 4]."""
    rng = rng or np.random.default_rng(0)
    st, so, eo = st.copy(), so.copy(), eo.copy()
    hidden = hidden.copy()
    K = st.shape[0]
    todo = hidden.sum(axis=1)
    for t in range(max(1, steps)):
        if not hidden.any():
            break
        out = _decode_logits(model, memory, cond, cond_null, st, so, eo, hidden, cfg_scale)
        lp = torch.log_softmax(out["state"].float() / max(1e-4, temperature), dim=-1).cpu().numpy()   # [K, W, 4, 5]
        ok = allowed_states(st, hidden)
        lp = np.where(ok, lp, -1e9)
        lp = lp - np.logaddexp.reduce(lp, axis=-1, keepdims=True)
        p = np.exp(lp)
        # sample a state for every lane of every frame
        u = rng.random(p.shape[:-1])[..., None]
        samp = (p.cumsum(-1) < u).sum(-1).clip(0, fr.N_STATES - 1)                                    # [K, W, 4]
        conf = np.take_along_axis(lp, samp[..., None], -1)[..., 0].sum(-1)                            # [K, W]
        conf = conf + rng.gumbel(size=conf.shape) * temperature * (1.0 - (t + 1) / steps)             # MaskGIT noise
        a_off = out["start"].float().argmax(-1).cpu().numpy()
        e_off = out["end"].float().argmax(-1).cpu().numpy()
        frac = 1.0 - math.cos(math.pi / 2 * (t + 1) / steps)                                        # cumulative share
        for r in range(K):
            h = np.flatnonzero(hidden[r])
            if not h.size:
                continue
            want = int(math.ceil(todo[r] * frac)) - int(todo[r] - h.size)
            n = h.size if t == steps - 1 else max(1, min(h.size, want))
            pick = h[np.argsort(-conf[r, h])[:n]]
            s_ = samp[r, pick]
            st[r, pick] = s_
            so[r, pick] = np.where((s_ == fr.TAP) | (s_ == fr.HEAD), a_off[r, pick], -1)
            eo[r, pick] = np.where(s_ == fr.TAIL, e_off[r, pick], -1)
            hidden[r, pick] = False
    return st, so, eo


@torch.no_grad()
def generate_canvas(model: V5Net, feats_all: np.ndarray, beat_all: np.ndarray, timing_points, sr: float,
                    style: Optional[np.ndarray] = None, style_mask: Optional[np.ndarray] = None, steps: int = 8,
                    temperature: float = 1.0, cfg_scale: float = 1.0, candidates: int = 1,
                    score_fn: Optional[Callable[[List[Note], List[Note], int, int], float]] = None,
                    refine_passes: int = 1, refine_frac: float = 0.1, fixed: Optional[Sequence[Note]] = None,
                    regen: Optional[Sequence[Tuple[float, float]]] = None, seed: int = 0,
                    device: Optional[torch.device] = None, log: Optional[Callable[[str], None]] = None) -> List[Note]:
    """Notes of the whole song (see the module docstring). ``fixed``: notes kept as they are (with ``regen``: the notes
    outside the regen ranges are kept, the ranges (seconds) are restored)."""
    model.eval()
    dev = device or next(model.parameters()).device
    cfg = model.cfg
    feats_all = feats_all[:, :sum(cfg.groups.values())]
    W, H = cfg.window_frames, cfg.frame_ms
    hop = W // 2
    F_ = feats_all.shape[0]
    sd = cfg.style_dim
    style = np.zeros(sd, np.float32) if style is None else np.asarray(style, np.float32)
    style_mask = np.zeros(sd, np.float32) if style_mask is None else np.asarray(style_mask, np.float32)
    rng = np.random.default_rng(seed)
    canvas = SongCanvas(F_, H)
    if fixed is not None:
        keep = np.ones(F_, dtype=bool)
        for a, b in (regen or []):
            keep[max(0, int(a * 1000 // H)):min(F_, int(b * 1000 // H) + 1)] = False
        # with ranges to map again the rest is locked; without them the whole map is only a start: self-correction
        # may still change what its error head finds wrong
        canvas.put_notes(fixed, lock=bool(regen), frames=keep)
    K = max(1, int(candidates))

    def encode_window(s: int, rows: int):
        f = torch.from_numpy(_rows(feats_all, s, W).astype(np.float32)).to(dev)[None]
        b = torch.from_numpy(_rows(beat_all, s, W).astype(np.float32)).to(dev)[None]
        mem = model.encode(f, b)["memory"].expand(rows, -1, -1)
        st_t = torch.from_numpy(style).to(dev)[None].expand(rows, -1)
        sm_t = torch.from_numpy(style_mask).to(dev)[None].expand(rows, -1)
        srt = torch.full((rows, 1), float(sr), device=dev)
        cond = model.condition(srt, st_t, sm_t)
        cond_null = model.condition(srt, st_t, sm_t, drop=torch.ones(rows, dtype=torch.bool, device=dev))
        return mem, cond, cond_null

    starts = list(range(0, max(1, F_), hop))
    # 1. reveal the song window by window
    for s in starts:
        st, so, eo, known = canvas.window(s, W)
        hidden = ~known
        if not hidden.any():
            continue
        mem, cond, cnull = encode_window(s, K)
        rs, ro, re_ = restore_window(model, mem, cond, cnull, np.repeat(st[None], K, 0), np.repeat(so[None], K, 0),
                                     np.repeat(eo[None], K, 0), np.repeat(hidden[None], K, 0), steps, temperature,
                                     cfg_scale, rng)
        best = 0
        if K > 1 and score_fn is not None:
            ctx = max(0, s - hop)
            so_far = canvas.notes()
            scores = []
            for r in range(K):
                cand = fr.decode(rs[r], ro[r], re_[r], s * H, H)
                new = [n for n in cand if hidden[(n.start - s * H) // H]] if cand else []
                scores.append(score_fn(new, [n for n in so_far if n.start >= ctx * H], s * H, (s + W) * H))
            best = int(np.argmax(scores))
        e = min(F_, s + W)
        sel = hidden[: e - s]
        canvas.state[s:e][sel], canvas.start[s:e][sel], canvas.end[s:e][sel] = rs[best, :e - s][sel], ro[best, :e - s][sel], re_[best, :e - s][sel]
        canvas.known[s:e] = True
    # 2. self-correction
    for p in range(max(0, int(refine_passes))):
        changed = 0
        for s in starts:
            st, so, eo, known = canvas.window(s, W)
            e = min(F_, s + W)
            lock = np.zeros(W, dtype=bool)
            lock[: e - s] = canvas.locked[s:e]
            lock[e - s:] = True
            mem, cond, cnull = encode_window(s, 1)
            none = np.zeros((1, W), dtype=bool)
            out = _decode_logits(model, mem, cond, cnull, st[None], so[None], eo[None], none, cfg_scale)
            err = torch.sigmoid(out["error"].float())[0].cpu().numpy()
            err[lock] = 0.0
            notes_f = int((st != fr.EMPTY).any(axis=1).sum())
            cand = np.flatnonzero(err > 0.5)
            if not cand.size or notes_f == 0:
                continue
            cand = cand[np.argsort(-err[cand])][: max(1, int(refine_frac * notes_f))]
            hide = np.zeros(W, dtype=bool)
            for f in cand:
                hide[max(0, f - 2):f + 3] = True
            hide &= ~lock
            before = float(err[hide].sum())
            rs, ro, re_ = restore_window(model, mem, cond, cnull, st[None], so[None], eo[None], hide[None],
                                         max(2, steps // 2), temperature, cfg_scale, rng)
            out2 = _decode_logits(model, mem, cond, cnull, rs, ro, re_, none, cfg_scale)
            err2 = torch.sigmoid(out2["error"].float())[0].cpu().numpy()
            after = float(err2[hide].sum())
            ok = after < before
            if ok and score_fn is not None:
                old_n = fr.decode(st, so, eo, s * H, H)
                new_n = fr.decode(rs[0], ro[0], re_[0], s * H, H)
                ok = score_fn(new_n, [], s * H, (s + W) * H) >= score_fn(old_n, [], s * H, (s + W) * H)
            if ok:
                sel = hide[: e - s]
                canvas.state[s:e][sel], canvas.start[s:e][sel], canvas.end[s:e][sel] = rs[0, :e - s][sel], ro[0, :e - s][sel], re_[0, :e - s][sel]
                changed += int(sel.sum())
        if log is not None:
            log(f"  self-correction pass {p + 1}: {changed} frames changed")
        if not changed:
            break
    return canvas.notes()
