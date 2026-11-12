"""
The v5 network: why (sound sources) -> encoder -> autoregressive event decoder.

* SourceSelector (the variable selection network of the Temporal Fusion Transformer, Lim et al. 2019): every input
  group (each audio representation, each demucs stem, the stems' attack curves, the beat grid) is embedded on its own,
  and a softmax over the groups, computed per frame, mixes them. The weights are returned: on every frame they say
  which sources the network listens to - the "why" that can be read out.
* Encoder: convolutions + transformer over the 10 ms frames of a window; a cause head predicts the source of the note
  on every event frame (labels from osu_mapper/hybrid/causes.py).
* Decoder: transformer over event tokens (osu_mapper/v5/tokens.py) with self-attention, cross-attention to the encoder,
  the encoder state at the token's own event time added to its embedding, and a beat-phase embedding of TIME tokens
  (on which subdivision of the beat the event sits). Conditioned on the star rating and the style vector, which are
  randomly hidden in training (classifier-free guidance at generation).
  The decoder keeps a key/value cache, so sampling a window costs one step per token.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..hybrid.causes import CAUSES
from ..hybrid.metrics import SUBDIV_NAMES
from .tokens import Vocab

N_PHASE = len(SUBDIV_NAMES) + 1          # subdivision class of a TIME token, + "not a time token"
MAX_DIST = 400                           # frames (4 s) of the distance bias of the time pointer


@dataclass
class V5Config:
    groups: Dict[str, int] = field(default_factory=dict)     # input group -> width (ordered)
    beat_dim: int = 13
    style_dim: int = 11
    d_model: int = 256
    enc_layers: int = 4
    dec_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    cond_dropout: float = 0.1            # star rating + style hidden (CFG)
    window_frames: int = 1024
    frame_ms: int = 10
    max_tokens: int = 1536
    window_ms: int = 10240
    len_steps: int = 1000

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "V5Config":
        return V5Config(**d)

    def vocab(self) -> Vocab:
        return Vocab(self.window_ms, self.len_steps)


PRESETS = {
    "tiny": dict(d_model=96, enc_layers=2, dec_layers=2, n_heads=4),
    "base": dict(d_model=256, enc_layers=4, dec_layers=6, n_heads=8),
    "large": dict(d_model=384, enc_layers=6, dec_layers=8, n_heads=8),
}


class GRN(nn.Module):
    """Gated residual network (TFT)."""

    def __init__(self, d_in: int, d: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d)
        self.fc2 = nn.Linear(d, 2 * d)
        self.skip = nn.Linear(d_in, d) if d_in != d else nn.Identity()
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.elu(self.fc1(x)))
        a, b = self.fc2(h).chunk(2, dim=-1)
        return self.norm(self.skip(x) + a * torch.sigmoid(b))


class SourceSelector(nn.Module):
    def __init__(self, groups: Dict[str, int], d: int, dropout: float):
        super().__init__()
        self.names = list(groups)
        self.embed = nn.ModuleList([GRN(w, d, dropout) for w in groups.values()])
        self.score = GRN(d * len(self.names), len(self.names), dropout)

    def forward(self, parts: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        e = torch.stack([m(x) for m, x in zip(self.embed, parts)], dim=2)          # [B, T, G, d]
        w = torch.softmax(self.score(e.flatten(2)), dim=-1)                          # [B, T, G]
        return (e * w.unsqueeze(-1)).sum(dim=2), w


class MHA(nn.Module):
    def __init__(self, d: int, h: int, dropout: float):
        super().__init__()
        self.h, self.dh = h, d // h
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
        self.p = dropout

    def project_kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        k, v = self.kv(x).view(B, L, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        return k, v

    def forward(self, x: torch.Tensor, kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, causal: bool = False,
                key_pad: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q(x).view(B, L, self.h, self.dh).transpose(1, 2)
        k, v = kv if kv is not None else self.project_kv(x)
        mask = None
        if key_pad is not None:
            mask = ~key_pad[:, None, None, :]                                       # True = attend
        if causal and L > 1:
            # query i (at absolute position Lk - L + i) sees the keys up to itself (also with a key/value cache)
            Lk = k.shape[2]
            cm = torch.arange(Lk, device=x.device)[None, :] <= (torch.arange(L, device=x.device)[:, None] + Lk - L)
            mask = cm[None, None] if mask is None else (mask & cm[None, None])
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, L, -1))


class EncLayer(nn.Module):
    def __init__(self, d: int, h: int, ff: int, dropout: float):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = MHA(d, h, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.att(self.n1(x)))
        return x + self.drop(self.ff(self.n2(x)))


class DecLayer(nn.Module):
    def __init__(self, d: int, h: int, ff: int, dropout: float):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_att = MHA(d, h, dropout)
        self.cross = MHA(d, h, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem_kv, cache: Optional[Dict[str, torch.Tensor]] = None, key_pad=None):
        h = self.n1(x)
        if cache is None:
            x = x + self.drop(self.self_att(h, causal=True, key_pad=key_pad))
        else:
            k, v = self.self_att.project_kv(h)
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
            x = x + self.self_att(h, kv=(k, v), causal=True)
        x = x + self.drop(self.cross(self.n2(x), kv=mem_kv))
        return x + self.drop(self.ff(self.n3(x)))


def sinusoid(n: int, d: int, device=None) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    ang = pos / (10000.0 ** (i / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)[:, : d // 2]
    return pe


class V5Net(nn.Module):
    def __init__(self, cfg: V5Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        groups = dict(cfg.groups)
        groups["beat"] = cfg.beat_dim
        self.selector = SourceSelector(groups, d, cfg.dropout)
        self.conv = nn.Sequential(nn.Conv1d(d, d, 7, padding=3), nn.GELU(), nn.Conv1d(d, d, 7, padding=3), nn.GELU())
        self.enc = nn.ModuleList([EncLayer(d, cfg.n_heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.enc_layers)])
        self.enc_norm = nn.LayerNorm(d)
        self.cause_head = nn.Linear(d, len(CAUSES))
        V = cfg.vocab().size
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(cfg.max_tokens, d)
        self.phase = nn.Embedding(N_PHASE, d)
        self.sr_mlp = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.style_mlp = nn.Sequential(nn.Linear(2 * cfg.style_dim, d), nn.GELU(), nn.Linear(d, d))
        self.null_cond = nn.Parameter(torch.zeros(d))
        self.at_time = nn.Linear(d, d)
        self.dec = nn.ModuleList([DecLayer(d, cfg.n_heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.dec_layers)])
        self.dec_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, V)
        # TIME tokens are a pointer at the encoder frames (which 10 ms frame) + the millisecond inside the frame
        # (from the frame's own sound): the choice of "where" is tied to the audio directly
        self.ptr_q = nn.Linear(d, d)
        self.ptr_k = nn.Linear(d, d)
        self.ptr_bias = nn.Linear(d, 1)
        self.off_head = nn.Linear(d, cfg.frame_ms)
        # learned bias by the distance (frames) from the current event: the next note is usually near, and a pointer
        # that jumps far ahead skips notes it can never go back to
        self.dist_bias = nn.Embedding(MAX_DIST + 2, 1)
        nn.init.zeros_(self.dist_bias.weight)

    @torch.no_grad()
    def init_distance_prior(self, gaps_frames) -> None:
        """Starts the distance bias of the time pointer at the log-frequency of the gaps between events in the
        training maps (a learned bias from zero moves ~1e-3 per step and would take thousands of steps to say "the
        next note is near"); earlier frames start very unlikely."""
        import numpy as np

        g = np.clip(np.asarray(gaps_frames, dtype=np.int64), 0, MAX_DIST)
        h = np.bincount(g, minlength=MAX_DIST + 1).astype(np.float64) + 1.0
        logp = np.log(h / h.sum())
        w = self.dist_bias.weight
        w[0, 0] = float(logp.min()) - 5.0
        w[1:, 0] = torch.from_numpy(logp - logp.max()).to(w)

    # --- encoder ---------------------------------------------------------------------------------------------------
    def split_groups(self, feats: torch.Tensor, beat: torch.Tensor) -> List[torch.Tensor]:
        parts, a = [], 0
        for name, w in self.cfg.groups.items():
            parts.append(feats[..., a:a + w])
            a += w
        parts.append(beat)
        return parts

    def encode(self, feats: torch.Tensor, beat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """feats [B, T, sum(groups)] (the groups' columns in order, see data.group_columns), beat [B, T, beat_dim]."""
        x, w = self.selector(self.split_groups(feats, beat))
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + sinusoid(x.shape[1], x.shape[2], x.device)[None]
        for layer in self.enc:
            x = layer(x)
        x = self.enc_norm(x)
        return {"memory": x, "source_weights": w, "cause_logits": self.cause_head(x)}

    def condition(self, sr: torch.Tensor, style: torch.Tensor, style_mask: torch.Tensor,
                  drop: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[B, d]; ``drop`` [B] bool = unconditional (null) rows."""
        c = self.sr_mlp(sr.view(-1, 1) / 10.0) + self.style_mlp(torch.cat([style * style_mask, style_mask], dim=-1))
        if drop is not None:
            c = torch.where(drop[:, None], self.null_cond[None].expand_as(c), c)
        return c

    def embed_tokens(self, tokens, phase, tok_frame, memory, cond, pos0: int = 0):
        B, L = tokens.shape
        pos = torch.arange(pos0, pos0 + L, device=tokens.device).clamp(max=self.cfg.max_tokens - 1)
        x = self.tok(tokens) + self.pos(pos)[None] + self.phase(phase) + cond[:, None]
        fr = tok_frame.clamp(min=0, max=memory.shape[1] - 1)
        at = torch.gather(memory, 1, fr.unsqueeze(-1).expand(-1, -1, memory.shape[2]))
        return x + self.at_time(at) * (tok_frame >= 0).unsqueeze(-1).float()

    def memory_kv(self, memory: torch.Tensor):
        """Everything the decoder needs from the encoder, computed once per window."""
        kv = [layer.cross.project_kv(memory) for layer in self.dec]
        ptr_k = self.ptr_k(memory)                                                    # [B, T, d]
        ptr_b = self.ptr_bias(memory).squeeze(-1)                                     # [B, T]
        off = torch.log_softmax(self.off_head(memory).float(), dim=-1)                # [B, T, frame_ms]
        return {"kv": kv, "ptr_k": ptr_k, "ptr_b": ptr_b, "off": off}

    def decode(self, x: torch.Tensor, mem, caches: Optional[List[Dict]] = None, key_pad=None,
               tok_frame: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Logits of the next token for every position; ``tok_frame`` [B, L] = frame of the event each position is in
        (-1 before the first event), for the distance bias of the time pointer."""
        for i, layer in enumerate(self.dec):
            x = layer(x, mem["kv"][i], None if caches is None else caches[i], key_pad=key_pad)
        h = self.dec_norm(x)
        logits = self.out(h).float()
        v = self.cfg.vocab()
        T = mem["ptr_k"].shape[1]
        n_frames = min(T, v.window_ms // self.cfg.frame_ms)
        ptr = torch.einsum("bld,btd->blt", self.ptr_q(h), mem["ptr_k"][:, :n_frames]).float() / math.sqrt(h.shape[-1])
        ptr = ptr + mem["ptr_b"][:, None, :n_frames].float()                          # [B, L, frames]
        if tok_frame is not None:
            dist = torch.arange(n_frames, device=h.device)[None, None, :] - tok_frame[:, :, None]
            ptr = ptr + self.dist_bias(dist.clamp(-1, MAX_DIST) + 1).squeeze(-1).float()
        t_logits = (ptr.unsqueeze(-1) + mem["off"][:, None, :n_frames]).flatten(2)   # [B, L, frames x ms]
        full = torch.full((*logits.shape[:2], v.window_ms), float("-inf"), device=logits.device, dtype=logits.dtype)
        full[..., : t_logits.shape[-1]] = t_logits
        return torch.cat([logits[..., : v.time0], full, logits[..., v.mask0:]], dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enc = self.encode(batch["feats"], batch["beat"])
        B = batch["tokens"].shape[0]
        drop = None
        if self.training:      # always a mask in training, so the null condition is part of the graph (DDP)
            drop = torch.rand(B, device=batch["tokens"].device) < self.cfg.cond_dropout
        cond = self.condition(batch["sr"], batch["style"], batch["style_mask"], drop)
        x = self.embed_tokens(batch["tokens"], batch["phase"], batch["tok_frame"], enc["memory"], cond)
        logits = self.decode(x, self.memory_kv(enc["memory"]), key_pad=batch["tokens"] == 0, tok_frame=batch["tok_frame"])
        return {"logits": logits, **enc}


def v5_losses(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cause_weight: float = 0.2,
              label_smoothing: float = 0.02) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Next-token cross-entropy (+ the cause head on event frames). Per-item weights (hard mining) via
    batch['item_weight']; returns (loss, logs) with the per-item token loss in logs['_per_item'] for hard mining."""
    logits = out["logits"][:, :-1].float()
    tgt = batch["tokens"][:, 1:]
    valid = (tgt != 0).float()
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none",
                         label_smoothing=label_smoothing).view_as(tgt) * valid
    per_item = ce.sum(1) / valid.sum(1).clamp(min=1.0)
    w = batch.get("item_weight")
    w = torch.ones_like(per_item) if w is None else w.float().view(-1)
    w = w / w.mean().clamp(min=1e-8)
    loss_tok = (per_item * w).mean()
    total = loss_tok
    logs: Dict[str, float] = {"tokens": float(loss_tok.detach())}
    cl = batch["cause"]
    m = cl >= 0
    if m.any() and cause_weight > 0:
        lc = F.cross_entropy(out["cause_logits"][m].float(), cl[m])
        total = total + cause_weight * lc
        logs["cause"] = float(lc.detach())
        logs["cause_acc"] = float((out["cause_logits"][m].argmax(-1) == cl[m]).float().mean())
    else:
        total = total + out["cause_logits"].sum() * 0.0
    # time tokens alone (where the notes go) and the rest (lanes, long notes)
    with torch.no_grad():
        is_t = batch["is_time"][:, 1:] > 0
        if is_t.any():
            logs["time_tokens"] = float((ce * is_t.float()).sum() / is_t.float().sum().clamp(min=1.0))
    logs["total"] = float(total.detach())
    out["_per_item"] = per_item.detach()
    return total, logs


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
