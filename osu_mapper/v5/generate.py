"""
Generating a whole song with the v5 network: windows of 10 s, each one continuing the second half of the previous
one (as in Mapperatorinator), token by token with a key/value cache, sampling (temperature, top-p) inside the event
grammar, classifier-free guidance on the condition, and search: ``candidates`` K continuations of every half-window
("phrase") are sampled at once and the best one by ``score_fn`` (critic + per-note reward) is kept - policy + value +
search, the AlphaZero recipe at the scale of one decision per phrase.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..hybrid.beatmap import Note
from ..hybrid.metrics import subdivision_classes
from .data import PHASE_NONE
from .model import V5Net
from .tokens import EOS, SOS, GrammarState, Vocab, decode, encode


def _top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    s, idx = torch.sort(logits, descending=True, dim=-1)
    cum = torch.softmax(s, dim=-1).cumsum(-1)
    cut = cum - torch.softmax(s, dim=-1) > p
    s = s.masked_fill(cut, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, idx, s)


@torch.no_grad()
def sample_window(model: V5Net, feats: torch.Tensor, beat: torch.Tensor, sr: float, style: np.ndarray,
                  style_mask: np.ndarray, prefix: Sequence[int], min_time: int, start_ms: int, timing_points,
                  n: int = 1, temperature: float = 0.9, top_p: float = 0.95, cfg_scale: float = 1.0,
                  max_new: int = 1400, generator: Optional[torch.Generator] = None) -> List[List[int]]:
    """``n`` token sequences of one window (each starting with ``prefix``); feats [T, D], beat [T, 13] of the window."""
    v: Vocab = model.cfg.vocab()
    dev = feats.device
    enc = model.encode(feats[None], beat[None])
    use_cfg = cfg_scale != 1.0
    rows = 2 * n if use_cfg else n
    memory = enc["memory"].expand(rows, -1, -1)
    mem_kv = model.memory_kv(memory)
    st = torch.from_numpy(np.asarray(style, np.float32)).to(dev)[None].expand(rows, -1)
    sm = torch.from_numpy(np.asarray(style_mask, np.float32)).to(dev)[None].expand(rows, -1)
    drop = torch.zeros(rows, dtype=torch.bool, device=dev)
    if use_cfg:
        drop[n:] = True
    cond = model.condition(torch.full((rows, 1), float(sr), device=dev), st, sm, drop)
    caches = [dict() for _ in model.dec]

    def phase_frame(tokens: List[int]):
        ph, fr = [], []
        cur = -1
        for t in tokens:
            if v.time0 <= t < v.mask0:
                cur = t - v.time0
                ph.append(int(subdivision_classes(np.array([start_ms + cur]), timing_points)[0]) if timing_points else PHASE_NONE)
            else:
                ph.append(PHASE_NONE)
            fr.append(cur // model.cfg.frame_ms if cur >= 0 else -1)
        return ph, fr, cur

    pre = list(prefix) if prefix else [SOS]
    ph, fr, cur_ms = phase_frame(pre)
    T = lambda a: torch.tensor(a, dtype=torch.long, device=dev)[None].expand(rows, -1)
    x = model.embed_tokens(T(pre), T(ph), T(fr), memory, cond, pos0=0)
    logits = model.decode(x, mem_kv, caches, tok_frame=T(fr))[:, -1].float()
    states = []
    for _ in range(n):
        g = GrammarState(v, min_time=min_time)
        for t in pre[1:]:
            g.push(t)
        states.append(g)
    seqs = [list(pre) for _ in range(n)]
    cur = [cur_ms] * n
    pos = len(pre)
    for _step in range(max_new):
        if all(g.done for g in states) or pos >= model.cfg.max_tokens - 1:
            break
        lg = logits[:n] if not use_cfg else logits[n:] + cfg_scale * (logits[:n] - logits[n:])
        mask = torch.full_like(lg, float("-inf"))
        for r, g in enumerate(states):
            if g.done:
                mask[r, EOS] = 0.0
                continue
            for lo, hi in g.ranges(seqs[r]):
                mask[r, lo:hi] = 0.0
        lg = _top_p((lg + mask) / max(1e-4, temperature), top_p)
        nxt = torch.multinomial(torch.softmax(lg, dim=-1), 1, generator=generator).view(-1).tolist()
        new_ph, new_fr = [], []
        for r, t in enumerate(nxt):
            if states[r].done:
                t = EOS
            else:
                states[r].push(t)
                seqs[r].append(t)
            if v.time0 <= t < v.mask0:
                cur[r] = t - v.time0
                new_ph.append(int(subdivision_classes(np.array([start_ms + cur[r]]), timing_points)[0]) if timing_points else PHASE_NONE)
            else:
                new_ph.append(PHASE_NONE)
            new_fr.append(cur[r] // model.cfg.frame_ms if cur[r] >= 0 else -1)
        tok = torch.tensor(nxt * (2 if use_cfg else 1), device=dev).view(rows, 1)
        phs = torch.tensor(new_ph * (2 if use_cfg else 1), device=dev).view(rows, 1)
        frs = torch.tensor(new_fr * (2 if use_cfg else 1), device=dev).view(rows, 1)
        x = model.embed_tokens(tok, phs, frs, memory, cond, pos0=pos)
        logits = model.decode(x, mem_kv, caches, tok_frame=frs)[:, -1].float()
        pos += 1
    return seqs


def _rows(a: np.ndarray, start: int, length: int) -> np.ndarray:
    from ..hybrid.dataset import _crop_rows
    return _crop_rows(a, start, length)


@torch.no_grad()
def generate_song(model: V5Net, feats_all: np.ndarray, beat_all: np.ndarray, timing_points, sr: float,
                  style: Optional[np.ndarray] = None, style_mask: Optional[np.ndarray] = None, candidates: int = 1,
                  score_fn: Optional[Callable[[List[Note], List[Note], int, int], float]] = None,
                  temperature: float = 0.9, top_p: float = 0.95, cfg_scale: float = 1.0, seed: int = 0,
                  device: Optional[torch.device] = None, progress: Optional[Callable[[int, int], None]] = None,
                  log: Optional[Callable[[str], None]] = None) -> List[Note]:
    """Notes of the whole song. ``score_fn(new_notes, notes_so_far, start_ms, end_ms)`` ranks the ``candidates``
    continuations of every phrase (higher = better); without it the first sample is kept."""
    model.eval()
    dev = device or next(model.parameters()).device
    # inputs are built as features + attacks + sections (data.song_inputs); a model without some of them reads the
    # first columns only
    feats_all = feats_all[:, :sum(model.cfg.groups.values())]
    v = model.cfg.vocab()
    W = model.cfg.window_frames
    H = model.cfg.frame_ms
    hop = W // 2
    F_ = feats_all.shape[0]
    sd = model.cfg.style_dim
    style = np.zeros(sd, np.float32) if style is None else style
    style_mask = np.zeros(sd, np.float32) if style_mask is None else style_mask
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    song_ms = F_ * H
    notes: List[Note] = []
    # the cursor moves to the last note that was placed: when the network ends a window early (EOS before its end),
    # the next window starts there and decides again; at least ``min_step_ms`` of progress per window
    cursor = 0
    min_step_ms = 1000
    w = 0
    while cursor < song_ms:
        s = max(0, cursor // H - hop)
        s_ms = s * H
        min_time = cursor - s_ms
        f = torch.from_numpy(_rows(feats_all, s, W).astype(np.float32)).to(dev)
        b = torch.from_numpy(_rows(beat_all, s, W).astype(np.float32)).to(dev)
        prefix_notes = [n for n in notes if s_ms <= n.start < cursor]
        prefix, _, _ = encode(prefix_notes, s_ms, v, add_eos=False)
        k = max(1, int(candidates))
        seqs = _sample_cpu_gen(model, f, b, sr, style, style_mask, prefix, min_time, s_ms, timing_points, k,
                               temperature, top_p, cfg_scale, gen)
        end_ms = s_ms + W * H
        cands = [[n for n in decode(q, s_ms, v) if cursor <= n.start < end_ms] for q in seqs]
        if k > 1 and score_fn is not None:
            scores = [score_fn(c, notes, cursor, end_ms) for c in cands]
            best = int(np.argmax(scores))
            if log is not None:
                log(f"  phrase at {cursor / 1000:.1f}s: scores " + " ".join(f"{x:+.2f}" for x in scores) + f" -> {best + 1}")
        else:
            best = 0
        chosen = cands[best]
        notes.extend(chosen)
        w += 1
        if progress is not None:
            progress(min(cursor, song_ms), song_ms)
        if end_ms >= song_ms:
            break
        last = max((n.start for n in chosen), default=cursor)
        cursor = min(end_ms, max(last + 1, cursor + min_step_ms))
    return _resolve_overlaps(sorted(notes))


def _sample_cpu_gen(model, f, b, sr, style, style_mask, prefix, min_time, s_ms, tps, k, temperature, top_p, cfg, gen):
    # torch.multinomial with a CPU generator needs CPU probabilities: sample on CPU when the model is elsewhere
    if f.device.type != "cpu":
        return sample_window(model, f, b, sr, style, style_mask, prefix, min_time, s_ms, tps, k, temperature, top_p,
                             cfg, generator=None)
    return sample_window(model, f, b, sr, style, style_mask, prefix, min_time, s_ms, tps, k, temperature, top_p, cfg,
                         generator=gen)


def _resolve_overlaps(notes: List[Note], gap_ms: int = 30) -> List[Note]:
    """A note on a lane still held by a long note (or closer than ``gap_ms`` to its end) is dropped."""
    held = {c: -10 ** 9 for c in range(4)}
    out: List[Note] = []
    for n in notes:
        if n.start < held[n.col] + gap_ms:
            continue
        out.append(n)
        held[n.col] = n.end if n.ln else n.start
    return out
