"""
Why is this note here? Attribution of every decision of the v5 network to its inputs (the per-trade explanation of
trading models: which inputs moved this one decision - SHAP / integrated gradients, Sundararajan et al. 2017).

For an event (a chord at some millisecond) the decision is the log-probability the network gives its TIME token, given
everything before it. Integrated gradients from a zero input (the features are standardized per song, so zero = an
average moment of the song) to the real one: attribution = input x mean gradient along the way, summed over the
columns of each input group and over the frames within ``local_ms`` of the note. The groups are folded into sources:
vocals / drums / bass / other (each demucs stem + its attack curve), the mix (all other audio representations), the
part of the song (section) and the beat grid.

    from osu_mapper.v5.explain import explain_events
    rows = explain_events(model, feats_win, beat_win, tokens, sr, start_ms, events=[3, 4, 5])
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..hybrid.metrics import subdivision_classes
from .data import PHASE_NONE
from .model import V5Net

SOURCES = ("vocals", "drums", "bass", "other", "mix", "section", "beat")
_STEM_OF_ATTACK = ("drums", "bass", "other", "vocals")


def _source_of_group(name: str) -> str:
    if name.startswith("stem_"):
        return name[5:]
    if name == "section":
        return "section"
    if name == "beat":
        return "beat"
    return "mix"


def _token_tensors(tokens: Sequence[int], model: V5Net, start_ms: int, timing_points):
    v = model.cfg.vocab()
    ph, fr, cur = [], [], -1
    for t in tokens:
        if v.time0 <= t < v.mask0:
            cur = t - v.time0
            ph.append(int(subdivision_classes(np.array([start_ms + cur]), timing_points)[0]) if timing_points else PHASE_NONE)
        else:
            ph.append(PHASE_NONE)
        fr.append(cur // model.cfg.frame_ms if cur >= 0 else -1)
    return ph, fr


def explain_events(model: V5Net, feats: np.ndarray, beat: np.ndarray, tokens: Sequence[int], sr: float, start_ms: int,
                   timing_points=None, events: Optional[Sequence[int]] = None, steps: int = 8, local_ms: int = 200,
                   device: Optional[torch.device] = None) -> List[Dict[str, float]]:
    """One row per explained event: ``time_ms`` (absolute), ``logp`` of its TIME token, and the share of every source
    in SOURCES of the attribution near the note (sums to 1). ``events``: indices of the events in the window (default
    all); ``tokens``: the window's tokens starting with SOS (osu_mapper/v5/tokens.encode)."""
    dev = device or next(model.parameters()).device
    model.eval()
    v = model.cfg.vocab()
    width = sum(model.cfg.groups.values())
    f0 = torch.from_numpy(np.asarray(feats[:, :width], dtype=np.float32)).to(dev)
    b0 = torch.from_numpy(np.asarray(beat, dtype=np.float32)).to(dev)
    toks = list(tokens)
    time_pos = [i for i, t in enumerate(toks) if v.time0 <= t < v.mask0]
    pick = range(len(time_pos)) if events is None else [e for e in events if 0 <= e < len(time_pos)]
    ph, fr = _token_tensors(toks, model, start_ms, timing_points)
    T = lambda a: torch.tensor(a, dtype=torch.long, device=dev)[None].expand(steps, -1)
    tok_t, ph_t, fr_t = T(toks), T(ph), T(fr)
    alphas = (torch.arange(steps, device=dev, dtype=torch.float32) + 0.5) / steps
    names = list(model.cfg.groups)
    bounds, a = [], 0
    for n in names:
        bounds.append((a, a + model.cfg.groups[n]))
        a += model.cfg.groups[n]
    rows: List[Dict[str, float]] = []
    zero_style = torch.zeros(steps, model.cfg.style_dim, device=dev)
    for e in pick:
        p = time_pos[e]
        if p == 0:
            continue
        fx = (alphas[:, None, None] * f0[None]).requires_grad_(True)
        bx = (alphas[:, None, None] * b0[None]).requires_grad_(True)
        with torch.enable_grad():
            enc = model.encode(fx, bx)
            cond = model.condition(torch.full((steps, 1), float(sr), device=dev), zero_style, zero_style)
            x = model.embed_tokens(tok_t[:, :p], ph_t[:, :p], fr_t[:, :p], enc["memory"], cond)
            logits = model.decode(x, model.memory_kv(enc["memory"]), tok_frame=fr_t[:, :p])
            lp = torch.log_softmax(logits[:, -1].float(), dim=-1)[:, toks[p]]
            gf, gb = torch.autograd.grad(lp.sum(), [fx, bx])
        attr_f = (f0 * gf.mean(0)).abs()                                       # [T, width]
        attr_b = (b0 * gb.mean(0)).abs()
        t_ms = toks[p] - v.time0
        c = t_ms // model.cfg.frame_ms
        w = max(1, local_ms // model.cfg.frame_ms)
        lo, hi = max(0, c - w), min(f0.shape[0], c + w + 1)
        share = {s: 0.0 for s in SOURCES}
        for n, (ga, gb_) in zip(names, bounds):
            val = float(attr_f[lo:hi, ga:gb_].sum())
            if n == "attacks":
                per = attr_f[lo:hi, ga:gb_].sum(0).tolist()
                for k, s in enumerate(_STEM_OF_ATTACK):
                    share[s] += float(per[k]) if k < len(per) else 0.0
            else:
                share[_source_of_group(n)] += val
        share["beat"] += float(attr_b[lo:hi].sum())
        tot = sum(share.values()) or 1.0
        row = {"time_ms": float(start_ms + t_ms), "logp": float(lp.detach().mean())}
        row.update({s: share[s] / tot for s in SOURCES})
        rows.append(row)
    return rows


def summarize(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    return {s: float(np.mean([r[s] for r in rows])) for s in SOURCES}


def format_shares(sh: Dict[str, float], top: int = 4) -> str:
    return ", ".join(f"{s} {100 * sh[s]:.0f}%" for s in sorted(sh, key=lambda k: -sh[k])[:top])
