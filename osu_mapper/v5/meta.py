"""
Meta-labeling of bursts (López de Prado, "Advances in Financial Machine Learning", ch. 3): the primary model decides
the side, a secondary model decides whether to act on it.

Here the primary model proposes a burst - a run of 1/4 and faster ("rice") - and the meta model answers: would a
human mapper put a burst HERE, at THIS star rating? It sees the sound of the burst's span (and half a second around
it: every representation, the stems' attacks, the part of the song) and the burst's shape (length, notes, speed in
beats, chords, long notes, the requested stars).

Proposals to learn from (train_meta.py): bursts of the other difficulties of the same song (a mapper put a burst here
at another star rating - does this difficulty have it too?) and bursts of the v5 network's own maps; the label is
whether the human map of that difficulty has a burst over at least half of the span.

At generation (generate_v5.py --meta) every burst of the network's map gets its probability; a burst below the
threshold is not deleted but "sized down" (the position is reduced, not closed): its notes collapse to 1/2 beats.
Bursts that are mostly long notes (LN stairs, LN streams) are never touched: they are not rice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..hybrid.beatmap import Note

N_SHAPE = 9
FRAMES = 48          # the span (+ margins) resampled to this many frames
SKIP = ("mert95m", "encodec24")   # large neural representations: left out (memory; the DSP ones + stems suffice)
MARGIN = 50          # frames (0.5 s) of sound before and after the span


@dataclass
class MetaConfig:
    feat_dim: int
    beat_dim: int = 13
    d: int = 64
    cols: Optional[List[int]] = None        # columns of the network input the meta model reads (None = feat_dim first)

    def to_dict(self) -> Dict:
        return asdict(self)


class BurstMeta(nn.Module):
    def __init__(self, cfg: MetaConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.inp = nn.Linear(cfg.feat_dim + cfg.beat_dim, d)
        self.conv = nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, d, 5, padding=2, stride=2), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * d + N_SHAPE, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))

    def forward(self, frames: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        """frames [B, FRAMES, feat_dim + beat_dim], shape [B, N_SHAPE] -> logit "a human would burst here"."""
        x = self.conv(self.inp(frames).transpose(1, 2))
        return self.head(torch.cat([x.mean(-1), x.amax(-1), shape], dim=-1)).squeeze(-1)


def input_columns(slices: Dict[str, Tuple[int, int]], width: int) -> List[int]:
    """Columns of ``song_inputs`` (features + attacks + sections) without the large neural representations."""
    drop = set()
    for name in SKIP:
        if name in slices:
            drop.update(range(slices[name][0], slices[name][1]))
    return [c for c in range(width) if c not in drop]


def burst_inputs(feats_all: np.ndarray, beat_all: np.ndarray, span: Tuple[float, float], notes: Sequence[Note],
                 timing_points, sr: float, cols, frame_ms: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """(frames [FRAMES, len(cols) + 13] float16, shape [N_SHAPE]) of one burst."""
    from ..hybrid.dataset import _crop_rows
    from ..hybrid.motifs import beat_positions

    a = int(span[0] // frame_ms) - MARGIN
    b = int(span[1] // frame_ms) + MARGIN + 1
    L = max(2, b - a)
    x = np.concatenate([_crop_rows(feats_all, a, L)[:, cols], _crop_rows(beat_all, a, L)], axis=1).astype(np.float32)
    pos = np.linspace(0, L - 1, FRAMES)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, L - 1)
    w = (pos - lo)[:, None]
    frames = x[lo] * (1 - w) + x[hi] * w
    ts = np.array(sorted({n.start for n in notes}), dtype=np.float64)
    beats, beat_len, _ = beat_positions(ts, timing_points) if ts.size else (np.zeros(1), np.full(1, 500.0), None)
    steps = np.diff(beats) if ts.size > 1 else np.array([1.0])
    shape = np.array([
        (span[1] - span[0]) / 1000.0, len(ts) / 16.0, len(notes) / max(1, len(ts)) / 4.0,
        float(np.mean(steps)), float(np.min(steps)), float(np.mean([n.ln for n in notes])) if notes else 0.0,
        sr / 10.0, float(np.mean(beat_len)) / 1000.0, 1.0,
    ], dtype=np.float32)
    return frames.astype(np.float16), shape


def span_has_burst(span: Tuple[float, float], human: Sequence[Tuple[float, float]], min_cover: float = 0.5) -> bool:
    d = max(1.0, span[1] - span[0])
    cov = sum(max(0.0, min(span[1], e) - max(span[0], s)) for s, e in human)
    return cov / d >= min_cover


class MetaFilter:
    """A trained meta model applied to a generated map."""

    def __init__(self, net: BurstMeta, device: torch.device, threshold: float = 0.35):
        self.net = net.eval().to(device)
        self.dev = device
        self.threshold = float(threshold)

    @staticmethod
    def load(path, device: torch.device, threshold: float = 0.35) -> "MetaFilter":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = BurstMeta(MetaConfig(**ck["config"]))
        net.load_state_dict(ck["model"])
        return MetaFilter(net, device, threshold)

    @torch.no_grad()
    def probs(self, feats_all, beat_all, notes, timing_points, sr) -> List[Tuple[Tuple[float, float], float]]:
        from ..hybrid.metrics import bursts

        t = np.array(sorted({n.start for n in notes}), dtype=np.float64)
        out = []
        for span in bursts(t, timing_points):
            members = [n for n in notes if span[0] <= n.start <= span[1]]
            cols = self.net.cfg.cols if self.net.cfg.cols is not None else list(range(self.net.cfg.feat_dim))
            fr, sh = burst_inputs(feats_all, beat_all, span, members, timing_points, sr, cols)
            p = torch.sigmoid(self.net(torch.from_numpy(fr.astype(np.float32))[None].to(self.dev),
                                       torch.from_numpy(sh)[None].to(self.dev)))
            out.append((span, float(p[0])))
        return out

    def apply(self, feats_all, beat_all, notes, timing_points, sr) -> Tuple[List[Note], Dict[str, float]]:
        """Bursts below the threshold collapse to 1/2 beats; returns (notes, report)."""
        from .critic import _no_overlap, blandify

        ps = self.probs(feats_all, beat_all, notes, timing_points, sr)
        keep = list(notes)
        cut = 0
        ln_kept = 0
        for (a, b), p in ps:
            if p >= self.threshold:
                continue
            inside = [n for n in keep if a <= n.start <= b]
            if inside and sum(n.ln for n in inside) / len(inside) > 0.5:
                ln_kept += 1           # LN stairs / LN bursts are not rice: never collapsed
                continue
            outside = [n for n in keep if not (a <= n.start <= b)]
            keep = _no_overlap(sorted(outside + blandify(inside, timing_points)))
            cut += 1
        rep = {"bursts": float(len(ps)), "sized_down": float(cut), "ln_bursts_kept": float(ln_kept),
               "mean_p": float(np.mean([p for _, p in ps])) if ps else float("nan")}
        return keep, rep
