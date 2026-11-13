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
5. Overlaps (``overlap``): windows overlap by half (as in Mapperatorinator); the later half of the overlap was placed
   by the previous window without seeing what follows, so the next window proposes its own version there. Where the
   two versions are the same nothing happens; where they differ, every differing stretch is decided with both sides
   in view by the seam head (learned on exactly this: the network's own versions of overlap zones made by the earlier
   and the later window, against the human map; older checkpoints: the error head): the version it finds fewer
   errors in stays.
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

    def context(self, s: int, W: int, n: int = 3):
        """States of the ``n`` frames before and after the window [s, s + W) ([n, 4] each; frames before the song
        start are empty, unknown frames after it -1): restore_window glues long notes and gaps across its edges."""
        bef = np.zeros((n, fr.NUM_LANES), dtype=np.int8)
        for k in range(n):
            f = s - n + k
            if 0 <= f < self.F and self.known[f]:
                bef[k] = self.state[f]
        aft = np.full((n, fr.NUM_LANES), -1, dtype=np.int8)
        for k in range(n):
            f = s + W + k
            if f >= self.F:
                aft[k] = fr.EMPTY                         # past the end of the song: empty
            elif self.known[f]:
                aft[k] = self.state[f]
        return bef, aft

    def open_before(self, s: int) -> np.ndarray:
        """[4] bool: lanes whose long note is held in the frame before ``s`` (known HEAD / HOLD)."""
        if s <= 0 or not self.known[s - 1]:
            return np.zeros(fr.NUM_LANES, dtype=bool)
        return np.isin(self.state[s - 1], (fr.HEAD, fr.HOLD))

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
        out = {k: (un[k] + cfg_scale * (out[k] - un[k])) if k not in ("error", "seam") else out[k] for k in out}
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
    # a long note held up to three frames before must still end and leave its 30 ms
    held = np.zeros((K, W, L), dtype=bool)
    for d in (1, 2, 3):
        p_ = np.concatenate([np.full((K, d, L), -1, st.dtype), st[:, :-d]], axis=1)
        pk = np.concatenate([np.zeros((K, d, 1), bool), known[:, :-d]], axis=1)
        held |= pk & ((p_ == fr.HEAD) | (p_ == fr.HOLD))
    ok[..., fr.TAP] &= ~(near | held)
    ok[..., fr.HEAD] &= ~(near | held)
    # ... and the same looking ahead: a known TAP / HEAD two frames ahead leaves no room to start or end a note here,
    # three frames ahead no room to open (or keep) a long note
    for d, states in ((1, (fr.TAP, fr.TAIL, fr.HEAD, fr.HOLD)), (2, (fr.TAP, fr.TAIL, fr.HEAD, fr.HOLD)), (3, (fr.HEAD, fr.HOLD))):
        n_ = np.concatenate([st[:, d:], np.full((K, d, L), -1, st.dtype)], axis=1)
        nk = np.concatenate([known[:, d:], np.zeros((K, d, 1), bool)], axis=1)
        nxt_note = nk & ((n_ == fr.TAP) | (n_ == fr.HEAD))
        for s_ in states:
            ok[..., s_] &= ~nxt_note
    none = ~ok.any(-1)
    ok[none] = True
    return ok


_OPEN = (fr.HEAD, fr.HOLD)
_CONT = (fr.HOLD, fr.TAIL)


def _commit_in_order(st: np.ndarray, hidden: np.ndarray, lp: np.ndarray, samp: np.ndarray, pick: np.ndarray,
                     prev_open: np.ndarray, rng: np.random.Generator, final: bool) -> np.ndarray:
    """Commits the sampled states of the ``pick`` [K, W] frames in time order, so a long note is always HEAD,
    HOLD..., TAIL: a lane holds (HOLD / TAIL) only right after an open long note - known, or committed just before in
    this pass - and must hold then; a long note known to be held further on (the next known frame holds the lane) is
    carried on to it; nothing starts within 30 ms (2 frames) after a note of its lane or before a known one, and a long
    note is not opened where it could not end in time. A frame that samples a hold, or lies before a known hold of a
    lane, while the frame before it is still hidden waits for a later round (in the last round every frame has its
    neighbour by then). Where a sample breaks
    these rules the state is drawn again from the allowed ones (``lp`` [K, W, 4, 5] log-probabilities, Gumbel-max).
    Independent per-frame sampling otherwise scatters single HOLDs - the most common non-empty state of long-note maps
    - that turn into long notes no one asked for, and cuts long notes at the edges of hidden stretches.
    ``st`` and ``hidden`` are updated in place; returns the committed frames [K, W] bool."""
    K, W, L = st.shape
    E, TP, HD, HL, TL = fr.EMPTY, fr.TAP, fr.HEAD, fr.HOLD, fr.TAIL
    known = ~hidden
    committed = np.zeros((K, W), dtype=bool)
    # the next frame known before this pass, after every frame (frames after f committed in this pass come later)
    idx = np.where(known, np.arange(W)[None], W)
    nxt = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
    nxt_after = np.concatenate([nxt[:, 1:], np.full((K, 1), W)], axis=1)
    st_n = st[np.arange(K)[:, None], np.clip(nxt_after, 0, W - 1)]
    held_later = (nxt_after < W)[..., None] & ((st_n == HL) | (st_n == TL))                      # [K, W, 4]
    next_is_next = nxt_after == np.arange(W)[None] + 1
    for r in range(K):
        frames = np.flatnonzero(pick[r])
        if not frames.size:
            continue
        st_l, kn, sp = st[r].tolist(), known[r].tolist(), samp[r].tolist()
        held, nn, po = held_later[r].tolist(), next_is_next[r].tolist(), prev_open[r].tolist()
        gum = None

        def state(x, c):
            if x == -1:
                return HL if po[c] else -1
            if 0 <= x < W and kn[x]:
                return st_l[x][c]
            return -1

        for f in frames.tolist():
            row = sp[f]
            if not final and f > 0 and not kn[f - 1] and (any(v in (HL, TL) for v in row) or any(held[f])):
                # a hold - or a frame before a known hold (carry on the long note from before, or start one?) -
                # waits until the frame before it is known
                continue
            open_before = [state(f - 1, c) in (HD, HL) for c in range(L)]
            if not any(row) and not any(open_before) and not any(held[f]):
                st_l[f] = [E] * L                       # an empty frame with nothing held into it: nothing to check
                kn[f] = True
                committed[r, f] = True
                continue
            out = []
            for c in range(L):
                s_ = row[c]
                a1, a2, a3 = state(f + 1, c), state(f + 2, c), state(f + 3, c)
                na1, na2, na3 = a1 in (TP, HD), a2 in (TP, HD), a3 in (TP, HD)
                if open_before[c]:                      # the long note goes on: hold or end
                    if held[f][c]:
                        allowed = (HL,)
                    else:
                        allowed = ((HL,) if not (na1 or na2 or na3) else ()) + ((TL,) if not (na1 or na2) else ())
                        allowed = allowed or (TL,)
                elif held[f][c] and nn[f]:
                    allowed = (HD,)                     # the held note must start right here
                else:
                    b1, b2, b3 = state(f - 1, c), state(f - 2, c), state(f - 3, c)
                    start_ok = (b1 not in (TP, TL) and b2 not in (TP, HD, HL, TL) and b3 not in (HD, HL)
                                and not (na1 or na2))
                    allowed = (E,) + ((TP,) if start_ok and not held[f][c] else ()) + ((HD,) if start_ok and not na3 else ())
                if s_ not in allowed:
                    if gum is None:
                        gum = rng.gumbel(size=(W, L, fr.N_STATES))
                    sc = [lp[r, f, c, a] + gum[f, c, a] for a in allowed]
                    s_ = allowed[int(np.argmax(sc))]
                out.append(s_)
            st_l[f] = out
            kn[f] = True
            committed[r, f] = True
        st[r] = np.asarray(st_l, dtype=st.dtype)
    hidden &= ~committed
    return committed


@torch.no_grad()
def restore_window(model: V5Net, memory, cond, cond_null, st, so, eo, hidden, steps: int = 8, temperature: float = 1.0,
                   cfg_scale: float = 1.0, rng: Optional[np.random.Generator] = None, prev_open=None,
                   before=None, after=None):
    """Reveals the ``hidden`` [K, W] frames of K rows of one window in ``steps`` rounds; returns (st, so, eo) [K, W, 4].
    The song around the window, so long notes and the 30 ms gaps glue across its edges: ``before`` / ``after``
    [K, n, 4] states of the known frames right before / after it (``after`` may hold -1 for unknown frames);
    ``prev_open`` [K, 4] bool (without ``before``): lanes whose long note is held in the frame before the window."""
    rng = rng or np.random.default_rng(0)
    st, so, eo = st.copy(), so.copy(), eo.copy()
    hidden = hidden.copy()
    K, W = hidden.shape
    if before is None and prev_open is not None:
        before = np.where(np.asarray(prev_open, dtype=bool), fr.HOLD, fr.EMPTY)[:, None, :]
    B_ = 0 if before is None else int(before.shape[1])
    A_ = 0 if after is None else int(after.shape[1])
    # the window with its context frames on both sides (known; unknown context frames stay hidden and are never picked)
    ctx_b = np.zeros((K, B_, fr.NUM_LANES), st.dtype) if B_ == 0 else np.asarray(before).astype(st.dtype)
    ctx_a = np.zeros((K, A_, fr.NUM_LANES), st.dtype) if A_ == 0 else np.clip(np.asarray(after), 0, None).astype(st.dtype)
    hid_a = np.zeros((K, A_), bool) if A_ == 0 else (np.asarray(after) < 0).any(-1)
    win = slice(B_, B_ + W)
    no_open = np.zeros((K, fr.NUM_LANES), dtype=bool)
    todo = hidden.sum(axis=1)
    for t in range(max(1, steps)):
        if not hidden.any():
            break
        out = _decode_logits(model, memory, cond, cond_null, st, so, eo, hidden, cfg_scale)
        lp = torch.log_softmax(out["state"].float() / max(1e-4, temperature), dim=-1).cpu().numpy()   # [K, W, 4, 5]
        st_x = np.concatenate([ctx_b, st, ctx_a], axis=1)
        hid_x = np.concatenate([np.zeros((K, B_), bool), hidden, hid_a], axis=1)
        ok = allowed_states(st_x, hid_x)[:, win]
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
        last = t == max(1, steps) - 1
        pick = np.zeros((K, W), dtype=bool)
        for r in range(K):
            h = np.flatnonzero(hidden[r])
            if not h.size:
                continue
            want = int(math.ceil(todo[r] * frac)) - int(todo[r] - h.size)
            n = h.size if last else max(1, min(h.size, want))
            pick[r, h[np.argsort(-conf[r, h])[:n]]] = True
        pad = lambda a, v: np.concatenate([np.full((K, B_) + a.shape[2:], v, a.dtype), a,
                                           np.full((K, A_) + a.shape[2:], v, a.dtype)], axis=1)
        done = _commit_in_order(st_x, hid_x, pad(lp, -1e9), pad(samp, fr.EMPTY), pad(pick, False), no_open, rng,
                                last)[:, win]
        st[:] = st_x[:, win]
        hidden &= ~done
        so = np.where(done[..., None], np.where((st == fr.TAP) | (st == fr.HEAD), a_off, -1), so).astype(so.dtype)
        eo = np.where(done[..., None], np.where(st == fr.TAIL, e_off, -1), eo).astype(eo.dtype)
    return st, so, eo


def glues(state: np.ndarray, R: int) -> bool:
    """True when frame ``R`` of the canvas ``state`` [W, 4] follows frame R-1 on every lane: a long note open at R-1 goes
    on at R (HOLD / TAIL) and nothing else does, and no note starts within 30 ms (2 frames) after a note of its lane."""
    W = state.shape[0]
    if R <= 0 or R >= W:
        return True
    for c in range(state.shape[1]):
        if (int(state[R - 1, c]) in _OPEN) != (int(state[R, c]) in _CONT):
            return False
        last = [f for f in (R - 2, R - 1) if f >= 0 and int(state[f, c]) in (fr.TAP, fr.HEAD, fr.TAIL)]
        first = [f for f in (R, R + 1) if f < W and int(state[f, c]) in (fr.TAP, fr.HEAD)]
        if last and first and first[0] - last[-1] < 3:
            return False
    return True


def arbitrate_overlap(prev, new, zone: np.ndarray, judge: Callable) -> Tuple[tuple, Dict[str, int]]:
    """The overlap zone of two windows: ``prev`` = the earlier window's version (placed without seeing what follows),
    ``new`` = the later window's (with the continuation in view); (state, start, end) [W, 4] each. Nothing happens where
    they are the same; every differing stretch (+-2 frames, inside the zone) is decided by ``judge`` (version ->
    per-frame error probability [W], the seam head) with the later version around it: the one with fewer errors
    stays. The earlier version of a stretch is taken only if it glues with what is around it on every lane - a long
    note held across the edge of the stretch goes on in the other version (and the 30 ms gap holds) - so a long note
    is never cut at the seam. Returns (the merged version, counts)."""
    cnt = {"differed": 0, "kept_earlier": 0, "took_later": 0}
    new = (new[0].copy(), new[1].copy(), new[2].copy())
    diff = fr.frame_differs(prev, new) & zone
    if not diff.any():
        return new, cnt
    cnt["differed"] = 1
    W = zone.shape[0]
    mark = np.zeros(W, dtype=bool)
    for f in np.flatnonzero(diff):
        mark[max(0, f - 2):f + 3] = True
    mark &= zone
    labels = np.cumsum(np.r_[mark[0], mark[1:] & ~mark[:-1]]) * mark
    for seg in range(1, int(labels.max()) + 1):
        sl = labels == seg
        idx = np.flatnonzero(sl)
        L, R = int(idx[0]), int(idx[-1]) + 1
        cand = (new[0].copy(), new[1].copy(), new[2].copy())
        cand[0][sl], cand[1][sl], cand[2][sl] = prev[0][sl], prev[1][sl], prev[2][sl]
        if not (glues(cand[0], L) and glues(cand[0], R)):
            cnt["took_later"] += 1                     # the earlier version would cut a long note at the seam
            continue
        if float(judge(cand)[sl].sum()) <= float(judge(new)[sl].sum()):
            new = cand
            cnt["kept_earlier"] += 1
        else:
            cnt["took_later"] += 1
    return new, cnt


@torch.no_grad()
def generate_canvas(model: V5Net, feats_all: np.ndarray, beat_all: np.ndarray, timing_points, sr: float,
                    style: Optional[np.ndarray] = None, style_mask: Optional[np.ndarray] = None, steps: int = 8,
                    temperature: float = 1.0, cfg_scale: float = 1.0, candidates: int = 1,
                    score_fn: Optional[Callable[[List[Note], List[Note], int, int], float]] = None,
                    refine_passes: int = 1, refine_frac: float = 0.1, fixed: Optional[Sequence[Note]] = None,
                    regen: Optional[Sequence[Tuple[float, float]]] = None, seed: int = 0, overlap: bool = True,
                    device: Optional[torch.device] = None, log: Optional[Callable[[str], None]] = None,
                    stats: Optional[dict] = None) -> List[Note]:
    """Notes of the whole song (see the module docstring). ``fixed``: notes kept as they are (with ``regen``: the notes
    outside the regen ranges are kept, the ranges (seconds) are restored). ``stats``: a dict the overlap counts are
    added to (windows, differed, kept_earlier, took_later)."""
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
    generated = np.zeros(F_, dtype=bool)          # frames this run produced (not given by the user)
    seams = {"windows": 0, "differed": 0, "kept_earlier": 0, "took_later": 0}
    # 1. reveal the song window by window
    for s in starts:
        st, so, eo, known = canvas.window(s, W)
        e = min(F_, s + W)
        # overlap: the later half of what the previous window placed was decided without seeing what comes after it;
        # this window proposes its own version there (with the continuation in view)
        zone = np.zeros(W, dtype=bool)
        if overlap and s > 0:
            a_, b_ = hop // 2, min(hop, e - s)
            zone[a_:b_] = generated[s + a_:s + b_] & ~canvas.locked[s + a_:s + b_]
        hidden = ~known | zone
        if not hidden.any():
            continue
        prev = (st.copy(), so.copy(), eo.copy())
        bef, aft = canvas.context(s, W)
        mem, cond, cnull = encode_window(s, K)
        rs, ro, re_ = restore_window(model, mem, cond, cnull, np.repeat(st[None], K, 0), np.repeat(so[None], K, 0),
                                     np.repeat(eo[None], K, 0), np.repeat(hidden[None], K, 0), steps, temperature,
                                     cfg_scale, rng, before=np.repeat(bef[None], K, 0), after=np.repeat(aft[None], K, 0))
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
        new = (rs[best].copy(), ro[best].copy(), re_[best].copy())
        if zone.any():
            seams["windows"] += 1
            mem1, cond1, cnull1 = (mem[:1], cond[:1], cnull[:1])
            none = np.zeros((1, W), dtype=bool)

            def judge(ver):
                out = _decode_logits(model, mem1, cond1, cnull1, ver[0][None], ver[1][None], ver[2][None], none, cfg_scale)
                j = out["seam"] if out.get("seam") is not None else out["error"]
                return torch.sigmoid(j.float())[0].cpu().numpy()

            new, cnt = arbitrate_overlap(prev, new, zone, judge)
            for k_, v_ in cnt.items():
                seams[k_] += v_
        sel = hidden[: e - s]
        canvas.state[s:e][sel], canvas.start[s:e][sel], canvas.end[s:e][sel] = new[0][: e - s][sel], new[1][: e - s][sel], new[2][: e - s][sel]
        canvas.known[s:e] = True
        generated[s:e] |= sel
    if stats is not None:
        for k_, v_ in seams.items():
            stats[k_] = stats.get(k_, 0) + v_
    if log is not None and seams["windows"]:
        judge = "seam head" if getattr(model, "seam_head", None) is not None else "error head"
        log(f"  overlaps: {seams['differed']} of {seams['windows']} differed between the two windows; differing stretches "
            f"decided by the {judge}: kept the earlier {seams['kept_earlier']}, took the later {seams['took_later']}")
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
            bef, aft = canvas.context(s, W)
            rs, ro, re_ = restore_window(model, mem, cond, cnull, st[None], so[None], eo[None], hide[None],
                                         max(2, steps // 2), temperature, cfg_scale, rng, before=bef[None], after=aft[None])
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
