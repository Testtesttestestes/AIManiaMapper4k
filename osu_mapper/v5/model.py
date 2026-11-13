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
    regimes: int = 16                    # slots of latent regimes of the song's parts (RegimeModule)
    regimes_active: int = 4              # slots in use; train_v5.py --grow_every splits the worst one (MarI/O-like)
    regime_rank: int = 32
    highlight: List[float] = field(default_factory=list)   # climax model of osu_mapper/v5/sections.py (from kiai)
    ln_model: List[float] = field(default_factory=list)    # LN share from the audio (osu_mapper/v5/lnstyle.py)
    draft: bool = False                  # re-mapping: the encoder also sees a draft of the map (critic.map_frames)

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "V5Config":
        d = dict(d)
        d.setdefault("regimes", 0)          # checkpoints made before the regimes existed
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


class RegimeModule(nn.Module):
    """Latent regimes of the song's parts, learned only from the maps (no names, no labels).

    The encoder states of every part of the song (boundaries from osu_mapper/v5/sections.py; 2 s blocks without them)
    are pooled; a part picks one of the active regimes (straight-through Gumbel-softmax in training, argmax at
    generation), and the regime adds its own embedding and its own low-rank adapter ("its own handwriting") to every
    frame of the part. Nothing tells the network what a solo or a drop is: if mappers treat two kinds of parts
    differently, separating them into two regimes lowers the loss, so it learns to.

    ``grow(src)`` copies regime ``src`` into the next free slot with a little noise and splits its probability in
    two (Net2Net-style growth, the "grow where it errs" of MarI/O / NEAT): the copies then specialise."""

    def __init__(self, d: int, slots: int, active: int, rank: int):
        super().__init__()
        self.slots = int(slots)
        self.register_buffer("active", torch.tensor(int(min(active, slots))))
        self.score = nn.Linear(d, slots)
        self.emb = nn.Parameter(torch.randn(slots, d) * 0.02)
        self.down = nn.Parameter(torch.randn(slots, d, rank) * (1.0 / math.sqrt(d)))
        self.up = nn.Parameter(torch.zeros(slots, rank, d))
        self.tau = 1.0

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x [B, T, d], seg [B, T] part index inside the window -> (x with the regimes added, regime of every frame)."""
        B, T, d = x.shape
        S = int(seg.max()) + 1
        pooled = torch.zeros(B, S, d, device=x.device, dtype=x.dtype).scatter_add_(1, seg[..., None].expand(-1, -1, d), x)
        cnt = torch.zeros(B, S, device=x.device, dtype=x.dtype).scatter_add_(1, seg, torch.ones_like(seg, dtype=x.dtype))
        pooled = pooled / cnt.clamp(min=1.0)[..., None]
        logits = self.score(pooled).float()
        mask = torch.arange(self.slots, device=x.device) >= self.active
        logits = logits.masked_fill(mask, float("-inf"))
        if self.training:
            g = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-9, 1 - 1e-9)))
            soft = torch.softmax((logits + g) / self.tau, dim=-1)
            hard = F.one_hot(soft.argmax(-1), self.slots).float()
            z = hard + soft - soft.detach()                                                    # [B, S, K]
        else:
            z = F.one_hot(logits.argmax(-1), self.slots).float()
        # load balancing (Switch Transformer): K x sum_k (share of parts on k) x (mean probability of k); 1 = even.
        # Without it one regime takes almost every part and the others never learn (82 % / 17 % / 0 % ...)
        act = int(self.active)
        prob = torch.softmax(logits, dim=-1)[..., :act].reshape(-1, act)
        used = cnt.reshape(-1) > 0
        if used.any():
            share = F.one_hot(prob[used].argmax(-1), act).float().mean(0)
            self.balance = act * (share * prob[used].mean(0)).sum()
        else:
            self.balance = logits.sum() * 0.0
        zf = torch.gather(z, 1, seg[..., None].expand(-1, -1, self.slots)).to(x.dtype)          # [B, T, K]
        # the embedding carries the (straight-through) gradient to the choice; the adapter is applied only for the
        # regime each frame chose (computing all slots for every frame cost [B, T, slots, d] of memory)
        out = x + zf @ self.emb.to(x.dtype)
        pick = zf.argmax(-1)                                                                   # [B, T]
        add = torch.zeros_like(x)
        for k in torch.unique(pick).tolist():
            sel = pick == k
            xs = x[sel]
            # under autocast the matmuls come out in bf16/fp16 while x (and add) may be float32
            add[sel] = (F.gelu(xs @ self.down[k].to(x.dtype)) @ self.up[k].to(x.dtype)).to(add.dtype)
        return out + add, pick

    @torch.no_grad()
    def grow(self, src: int, noise: float = 0.01) -> Optional[int]:
        k = int(self.active)
        if k >= self.slots:
            return None
        self.emb[k] = self.emb[src] + noise * torch.randn_like(self.emb[src])
        self.down[k] = self.down[src] + noise * torch.randn_like(self.down[src]) / math.sqrt(self.down.shape[1])
        self.up[k] = self.up[src].clone()
        self.score.weight[k] = self.score.weight[src]
        self.score.bias[k] = self.score.bias[src] - math.log(2.0)
        self.score.bias[src] -= math.log(2.0)
        self.active.fill_(k + 1)
        return k


def part_index(section_cols: Optional[torch.Tensor], T: int, B: int, device, block: int = 200) -> torch.Tensor:
    """[B, T] index of the song part of every frame inside the window: a new part where the 'position in the part'
    column of the section features restarts (sections.section_features), else 2 s blocks."""
    if section_cols is None:
        return (torch.arange(T, device=device) // block)[None].expand(B, -1).contiguous()
    from .sections import LABELS
    pos = section_cols[..., len(LABELS) + 9]
    new = torch.zeros(B, T, dtype=torch.long, device=device)
    new[:, 1:] = (pos[:, 1:] < pos[:, :-1] - 1e-6).long()
    return new.cumsum(1)


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
        self.regime = RegimeModule(d, cfg.regimes, cfg.regimes_active, cfg.regime_rank) if cfg.regimes > 0 else None
        self.cause_head = nn.Linear(d, len(CAUSES))
        V = cfg.vocab().size
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(cfg.max_tokens, d)
        self.phase = nn.Embedding(N_PHASE, d)
        self.sr_mlp = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.style_mlp = nn.Sequential(nn.Linear(2 * cfg.style_dim, d), nn.GELU(), nn.Linear(d, d))
        self.null_cond = nn.Parameter(torch.zeros(d))
        self.at_time = nn.Linear(d, d)
        # draft of the map (re-mapping passes): added to the encoder input; zero at the start, so a checkpoint made
        # without it behaves exactly as before and learns to use it
        if cfg.draft:
            from .critic import MAP_DIM
            self.draft_in = nn.Linear(MAP_DIM + 1, d)
            nn.init.zeros_(self.draft_in.weight)
            nn.init.zeros_(self.draft_in.bias)
        self.dec = nn.ModuleList([DecLayer(d, cfg.n_heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.dec_layers)])
        self.dec_norm = nn.LayerNorm(d)
        # every token except the TIME tokens, plus one "an event starts" slot (index time0): the time itself is
        # factorized below, so a step never holds a [batch, tokens, 10240] table
        self.out = nn.Linear(d, V - cfg.window_ms + 1)
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

    def encode(self, feats: torch.Tensor, beat: torch.Tensor, draft: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """feats [B, T, sum(groups)] (the groups' columns in order, see data.group_columns), beat [B, T, beat_dim],
        draft [B, T, MAP_DIM + 1] (picture of a draft map + "a draft is given" flag) or None."""
        x, w = self.selector(self.split_groups(feats, beat))
        if getattr(self, "draft_in", None) is not None:
            if draft is None:
                draft = torch.zeros(*x.shape[:2], self.draft_in.in_features, device=x.device, dtype=x.dtype)
            x = x + self.draft_in(draft.to(x.dtype))
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + sinusoid(x.shape[1], x.shape[2], x.device)[None]
        half = len(self.enc) // 2
        regime = None
        for i, layer in enumerate(self.enc):
            if i == half and self.regime is not None:
                sec = None
                if "section" in self.cfg.groups:
                    a = sum(wd for n, wd in self.cfg.groups.items() if list(self.cfg.groups).index(n) < list(self.cfg.groups).index("section"))
                    sec = feats[..., a:a + self.cfg.groups["section"]]
                x, regime = self.regime(x, part_index(sec, x.shape[1], x.shape[0], x.device))
            x = layer(x)
        x = self.enc_norm(x)
        bal = self.regime.balance if (self.regime is not None and regime is not None and hasattr(self.regime, "balance")) else None
        return {"memory": x, "source_weights": w, "cause_logits": self.cause_head(x), "regime": regime, "regime_balance": bal}

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

    def decode_parts(self, x: torch.Tensor, mem, caches: Optional[List[Dict]] = None, key_pad=None,
                     tok_frame: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Factorized next-token distribution (log-probabilities):
        ``kind`` [B, L, V - window_ms + 1] over the non-time tokens and the "TIME" slot (index time0),
        ``frame`` [B, L, frames] which 10 ms frame (the pointer at the encoder), ``off`` [B, frames, frame_ms] the
        millisecond inside a frame. log p(TIME = 10 f + o) = kind[time0] + frame[f] + off[f, o]."""
        for i, layer in enumerate(self.dec):
            x = layer(x, mem["kv"][i], None if caches is None else caches[i], key_pad=key_pad)
        h = self.dec_norm(x)
        kind = torch.log_softmax(self.out(h).float(), dim=-1)
        v = self.cfg.vocab()
        T = mem["ptr_k"].shape[1]
        n_frames = min(T, v.window_ms // self.cfg.frame_ms)
        ptr = torch.einsum("bld,btd->blt", self.ptr_q(h), mem["ptr_k"][:, :n_frames]).float() / math.sqrt(h.shape[-1])
        ptr = ptr + mem["ptr_b"][:, None, :n_frames].float()                          # [B, L, frames]
        if tok_frame is not None:
            dist = torch.arange(n_frames, device=h.device)[None, None, :] - tok_frame[:, :, None]
            ptr = ptr + self.dist_bias(dist.clamp(-1, MAX_DIST) + 1).squeeze(-1).float()
        return {"kind": kind, "frame": torch.log_softmax(ptr, dim=-1), "off": mem["off"][:, :n_frames]}

    def full_logprobs(self, parts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """[B, L, V] log-probabilities of every token (for sampling a few positions; training uses the parts)."""
        v = self.cfg.vocab()
        kind, fr, off = parts["kind"], parts["frame"], parts["off"]
        t = (kind[..., v.time0:v.time0 + 1].unsqueeze(-1) + fr.unsqueeze(-1) + off[:, None]).flatten(2)
        full = torch.full((*kind.shape[:2], v.window_ms), float("-inf"), device=kind.device, dtype=kind.dtype)
        full[..., : t.shape[-1]] = t
        return torch.cat([kind[..., : v.time0], full, kind[..., v.time0 + 1:]], dim=-1)

    def decode(self, x: torch.Tensor, mem, caches: Optional[List[Dict]] = None, key_pad=None,
               tok_frame: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Log-probabilities of the next token for every position, over the whole vocabulary (generation)."""
        return self.full_logprobs(self.decode_parts(x, mem, caches, key_pad, tok_frame))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enc = self.encode(batch["feats"], batch["beat"], batch.get("draft"))
        B = batch["tokens"].shape[0]
        drop = None
        if self.training:      # always a mask in training, so the null condition is part of the graph (DDP)
            drop = torch.rand(B, device=batch["tokens"].device) < self.cfg.cond_dropout
        cond = self.condition(batch["sr"], batch["style"], batch["style_mask"], drop)
        x = self.embed_tokens(batch["tokens"], batch["phase"], batch["tok_frame"], enc["memory"], cond)
        parts = self.decode_parts(x, self.memory_kv(enc["memory"]), key_pad=batch["tokens"] == 0, tok_frame=batch["tok_frame"])
        return {"parts": parts, **enc}


def v5_losses(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cause_weight: float = 0.2,
              label_smoothing: float = 0.02, window_ms: int = 10240,
              regime_balance: float = 0.01) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Next-token cross-entropy (+ the cause head on event frames). Per-item weights (hard mining) via
    batch['item_weight']; returns (loss, logs) with the per-item token loss in logs['_per_item'] for hard mining."""
    from .tokens import N_SPECIAL

    parts = out["parts"]
    kind = parts["kind"][:, :-1]                                             # [B, L, V_kind] log-probabilities
    fr = parts["frame"][:, :-1]                                              # [B, L, frames]
    off = parts["off"]                                                       # [B, frames, ms]
    tgt = batch["tokens"][:, 1:]
    valid = (tgt != 0).float()
    time0 = N_SPECIAL
    is_t = batch["is_time"][:, 1:] > 0
    # kind index of every target: a TIME token -> the time0 slot; the tokens after the time range shift down
    tk = torch.where(is_t, torch.full_like(tgt, time0), tgt)
    tk = torch.where(tgt >= time0 + window_ms, tgt - window_ms + 1, tk)
    lp_kind = kind.gather(-1, tk.clamp(0, kind.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
    t_ms = torch.where(is_t, tgt - time0, torch.zeros_like(tgt))
    f_idx = (t_ms // off.shape[-1]).clamp(max=fr.shape[-1] - 1)
    o_idx = t_ms % off.shape[-1]
    lp_frame = fr.gather(-1, f_idx.unsqueeze(-1)).squeeze(-1)
    lp_off = off.gather(1, f_idx.unsqueeze(-1).expand(-1, -1, off.shape[-1])).gather(-1, o_idx.unsqueeze(-1)).squeeze(-1)
    nll = -(lp_kind + torch.where(is_t, lp_frame + lp_off, torch.zeros_like(lp_kind)))
    if label_smoothing > 0:                                                  # on the token kind only
        nll = (1 - label_smoothing) * nll - label_smoothing * kind.mean(-1)
    ce = nll * valid
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
    if out.get("regime_balance") is not None and regime_balance > 0:
        total = total + regime_balance * out["regime_balance"].float()
        logs["regime_balance"] = float(out["regime_balance"].detach())
    logs["total"] = float(total.detach())
    out["_per_item"] = per_item.detach()
    return total, logs


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
