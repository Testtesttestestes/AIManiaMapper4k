"""
The critic ("value" of the AlphaZero recipe): given the audio of a window and the notes in it, how likely is it that a
human mapper made this map?

It is trained on human windows against windows that are not human-like (train_critic.py):

* "bland" versions of the same human window: every note moved to the nearest 1/2 beat, and only one note per 1/2 kept -
  exactly the default rhythm the critic must learn to dislike when the music asks for more (bursts, 1/4, 1/6 ...);
* broken versions: lanes shuffled, timing jittered by 10-30 ms, 30 % of the notes removed, random extra notes;
* maps of the network itself (v5 samples, or self-generated maps in the dataset), so the critic keeps up with what the
  network actually does (as the discriminator of GAIL).

The search in generation (osu_mapper/v5/generate.py) keeps the continuation of a phrase the critic likes most.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..hybrid.beatmap import Note

MAP_DIM = 9          # onset per lane (4), long note held per lane (4), ms offset of the onset in the frame


def map_frames(notes: Sequence[Note], start_ms: int, W: int, frame_ms: int = 10) -> np.ndarray:
    """[W, MAP_DIM] picture of the notes of a window, frame by frame."""
    x = np.zeros((W, MAP_DIM), dtype=np.float32)
    end_ms = start_ms + W * frame_ms
    for n in notes:
        if n.start >= end_ms or (n.end if n.ln else n.start) < start_ms:
            continue
        f = (n.start - start_ms) // frame_ms
        if 0 <= f < W:
            x[f, n.col] = 1.0
            x[f, 8] = ((n.start - start_ms) % frame_ms) / frame_ms
        if n.ln:
            a = max(0, (n.start - start_ms) // frame_ms + 1)
            b = min(W, (n.end - start_ms) // frame_ms + 1)
            if b > a:
                x[a:b, 4 + n.col] = 1.0
    return x


@dataclass
class CriticConfig:
    feat_dim: int
    beat_dim: int = 13
    d_model: int = 128
    layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    window_frames: int = 1024

    def to_dict(self) -> Dict:
        return asdict(self)


class CriticNet(nn.Module):
    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.audio = nn.Sequential(nn.Linear(cfg.feat_dim + cfg.beat_dim, d), nn.GELU(), nn.Linear(d, d))
        self.map = nn.Sequential(nn.Linear(MAP_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.conv = nn.Sequential(nn.Conv1d(2 * d, d, 5, padding=2, stride=2), nn.GELU(),
                                  nn.Conv1d(d, d, 5, padding=2, stride=2), nn.GELU())
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, feats: torch.Tensor, beat: torch.Tensor, mapx: torch.Tensor) -> torch.Tensor:
        """Logit of "human" per window: feats [B, T, F], beat [B, T, 13], mapx [B, T, MAP_DIM]."""
        a = self.audio(torch.cat([feats, beat], dim=-1))
        m = self.map(mapx)
        x = self.conv(torch.cat([a, m], dim=-1).transpose(1, 2)).transpose(1, 2)
        x = self.enc(x)
        return self.head(x.mean(dim=1)).squeeze(-1)


# --- negatives ------------------------------------------------------------------------------------------------------

def blandify(notes: Sequence[Note], timing_points) -> List[Note]:
    """The default-rhythm version of a map: every event moved to the nearest 1/2 beat, and of the events that land on
    the same 1/2 only the first one kept (bursts of 1/4, 1/6, 1/8 collapse to plain 1/2)."""
    from ..hybrid.motifs import beat_positions

    if not notes:
        return []
    starts = np.array(sorted({n.start for n in notes}), dtype=np.float64)
    beats, beat_len, _ = beat_positions(starts, timing_points)
    half = np.round(beats * 2) / 2
    snapped = np.round(starts + (half - beats) * beat_len).astype(np.int64)
    keep_start = {}
    for t, h, s in zip(starts.astype(np.int64).tolist(), half.tolist(), snapped.tolist()):
        keep_start.setdefault(round(h, 3), (t, s))
    move = {t: s for t, s in keep_start.values()}
    out = [Note(move[n.start], n.col, move[n.start] + (n.end - n.start), n.ln) for n in notes if n.start in move]
    return _no_overlap(out)


def corrupt(notes: Sequence[Note], rng: random.Random) -> List[Note]:
    """One random kind of damage: lanes shuffled / timing jittered / notes removed / random extra notes, and for
    long notes: every LN made a tap / taps made LNs of random length / LN lengths scrambled (x0.3..3) / releases
    moved off the beat by 20-60 ms."""
    ns = list(notes)
    has_ln = any(n.ln for n in ns)
    kind = rng.randrange(8 if has_ln else 5)
    if kind == 4:                      # taps -> random long notes
        out = [Note(n.start, n.col, n.start + rng.randint(80, 1500), True) if not n.ln and rng.random() < 0.5 else n for n in ns]
        return _no_overlap(sorted(out))
    if kind == 5:                      # LN -> tap
        return _no_overlap(sorted(Note(n.start, n.col, n.start, False) if n.ln else n for n in ns))
    if kind == 6:                      # LN lengths scrambled
        out = [Note(n.start, n.col, n.start + max(30, int((n.end - n.start) * rng.uniform(0.3, 3.0))), True) if n.ln else n
               for n in ns]
        return _no_overlap(sorted(out))
    if kind == 7:                      # releases off the beat
        out = [Note(n.start, n.col, max(n.start + 30, n.end + rng.choice([-1, 1]) * rng.randint(20, 60)), True) if n.ln else n
               for n in ns]
        return _no_overlap(sorted(out))
    if kind == 0:
        out = [Note(n.start, rng.randrange(4), n.end, n.ln) for n in ns]
    elif kind == 1:
        out = []
        for n in ns:
            d = rng.choice([-1, 1]) * rng.randint(10, 30)
            out.append(Note(n.start + d, n.col, n.end + d, n.ln))
    elif kind == 2:
        out = [n for n in ns if rng.random() > 0.3]
    else:
        out = list(ns)
        if ns:
            lo, hi = min(n.start for n in ns), max(n.start for n in ns)
            for _ in range(max(1, len(ns) // 4)):
                t = rng.randint(lo, max(lo, hi))
                out.append(Note(t, rng.randrange(4), t, False))
    return _no_overlap(sorted(out))


def _no_overlap(notes: List[Note], gap: int = 30) -> List[Note]:
    held = {c: -10 ** 9 for c in range(4)}
    out = []
    for n in sorted(notes):
        if n.start < held[n.col] + gap:
            continue
        out.append(n)
        held[n.col] = n.end if n.ln else n.start
    return out


class Critic:
    """A trained critic for scoring candidate phrases (log-odds of "human")."""

    def __init__(self, net: CriticNet, device: torch.device):
        self.net = net.eval().to(device)
        self.dev = device

    @staticmethod
    def load(path, device: torch.device) -> "Critic":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = CriticNet(CriticConfig(**ck["config"]))
        net.load_state_dict(ck["model"])
        return Critic(net, device)

    @torch.no_grad()
    def score(self, feats_all: np.ndarray, beat_all: np.ndarray, notes: Sequence[Note], end_ms: int,
              frame_ms: int = 10) -> float:
        """Log-odds of "human" of the window that ends at ``end_ms`` (the phrase and what came before it)."""
        from ..hybrid.dataset import _crop_rows

        W = self.net.cfg.window_frames
        feats_all = feats_all[:, :self.net.cfg.feat_dim]
        s = max(0, end_ms // frame_ms - W)
        f = torch.from_numpy(_crop_rows(feats_all, s, W).astype(np.float32))[None].to(self.dev)
        b = torch.from_numpy(_crop_rows(beat_all, s, W).astype(np.float32))[None].to(self.dev)
        m = torch.from_numpy(map_frames(notes, s * frame_ms, W, frame_ms))[None].to(self.dev)
        return float(self.net(f, b, m)[0])
