"""
HybridParamNet: one-pass (non-autoregressive) network that predicts the parameter track.

    representations ("tokenizers") --stems + modality dropout + gated fusion--> 100 fps features
        -> dilated depthwise conv blocks @ 100 fps  (local transients, millisecond detail)
        -> stride-4 downsample -> RoPE transformer @ 25 fps (+-10 s musical context)
        -> upsample + skip -> conv blocks @ 100 fps
        -> per-frame heads for every parameter of the universal algorithm (+ a chord-size head that
           keeps the decoder from propagating a wrong chord size through relational lane codes)

Conditioning (FiLM in every block): star rating, global style vector (with per-key "unknown"
masks) and a length-agnostic global song summary (attention pooling over 1 s song features).
The network never sees absolute time or song progress, so a 60-minute track is processed exactly
like a 3-minute one (window by window, see inference.py).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .params import FRAME_MS, MAX_MEMBERS, N_LANE_CODES, N_LN_BUCKETS
from .solver import STYLE_DIM
from .audio_features import BEAT_DIM


@dataclass
class HybridModelConfig:
    rep_dims: Dict[str, int]
    coarse_dim: int
    beat_dim: int = BEAT_DIM
    style_dim: int = STYLE_DIM
    d_model: int = 256
    conv_layers: int = 4
    conv_kernel: int = 7
    down_factor: int = 4
    attn_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    out_conv_layers: int = 2
    dropout: float = 0.1
    rep_dropout: float = 0.15
    rep_dropout_by_name: Dict[str, float] = field(default_factory=dict)  # per-representation override, e.g. {"mert95m": 0.5}
    style_dropout: float = 0.3
    style_key_dropout: float = 0.2
    sr_dropout: float = 0.1
    beat_dropout: float = 0.3
    global_dropout: float = 0.2
    frame_ms: int = FRAME_MS
    n_lane_codes: int = N_LANE_CODES
    n_ln_buckets: int = N_LN_BUCKETS
    max_members: int = MAX_MEMBERS
    rope_base: float = 10000.0
    event_prior: float = 0.03
    canvas: bool = False              # masked discrete diffusion: a partially known map of the window as input (canvas.py)
    importance_head: bool = False     # difficulty ladder: per-frame importance + monotone threshold per star rating
    plan: bool = False                # think first, then map: per-bar technique plan as input + a head that predicts it
    error_head: bool = False          # break and fix: P(a visible canvas frame is wrong) - the self-correction critic
    pattern_heads: bool = False       # pattern knowledge: rhythm / shape / hand / flow / copy heads (motif_targets.py)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "HybridModelConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)  # type: ignore[arg-type]


PRESETS: Dict[str, Dict[str, int]] = {
    "tiny": dict(d_model=64, conv_layers=2, attn_layers=2, n_heads=4, out_conv_layers=1),
    "small": dict(d_model=128, conv_layers=3, attn_layers=4, n_heads=4, out_conv_layers=2),
    "base": dict(d_model=256, conv_layers=4, attn_layers=6, n_heads=8, out_conv_layers=2),
    "large": dict(d_model=384, conv_layers=6, attn_layers=10, n_heads=8, out_conv_layers=2),
}


LADDER_BINS = 12


def ladder_threshold(c0: torch.Tensor, c: torch.Tensor, sr: torch.Tensor) -> torch.Tensor:
    """theta(sr): importance a moment needs to be mapped at star rating ``sr``. Monotone DEcreasing in sr (harder maps
    keep less important moments too): c0 - sum_j softplus(c_j) * clamp(sr - j, 0, 1)."""
    j = torch.arange(c.shape[0], device=sr.device, dtype=torch.float32)
    ramp = (sr.float().unsqueeze(-1) - j).clamp(0.0, 1.0)
    return c0 - (F.softplus(c) * ramp).sum(dim=-1)


def build_config(rep_dims: Dict[str, int], coarse_dim: int, preset: str = "base", **overrides: object) -> HybridModelConfig:
    params: Dict[str, object] = dict(PRESETS[preset])
    params.update(overrides)
    return HybridModelConfig(rep_dims=dict(rep_dims), coarse_dim=int(coarse_dim), **params)  # type: ignore[arg-type]


class FiLM(nn.Module):
    def __init__(self, d_cond: int, d: int):
        super().__init__()
        self.proj = nn.Linear(d_cond, 2 * d)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(c).unsqueeze(1).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


class ConvBlock(nn.Module):
    def __init__(self, d: int, d_cond: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.film = FiLM(d_cond, d)
        self.dw = nn.Conv1d(d, d, kernel, padding=dilation * (kernel // 2), dilation=dilation, groups=d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.pw2 = nn.Linear(2 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.film(self.norm(x), c)
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)
        h = self.pw2(F.gelu(self.pw1(h)))
        return x + self.drop(h)


def _rope(q: torch.Tensor, k: torch.Tensor, base: float) -> Tuple[torch.Tensor, torch.Tensor]:
    T, Dh = q.shape[-2], q.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, Dh, 2, device=q.device, dtype=torch.float32) / Dh))
    ang = torch.arange(T, device=q.device, dtype=torch.float32)[:, None] * inv[None, :]
    cos = torch.cos(ang).to(q.dtype)
    sin = torch.sin(ang).to(q.dtype)

    def rot(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    return rot(q), rot(k)


class AttnBlock(nn.Module):
    def __init__(self, d: int, d_cond: int, n_heads: int, ff_mult: int, dropout: float, rope_base: float):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.rope_base = rope_base
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(d)
        self.film1 = FiLM(d_cond, d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.film2 = FiLM(d_cond, d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.film1(self.norm1(x), c)
        q, k, v = self.qkv(h).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k = _rope(q, k, self.rope_base)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        x = x + self.drop(self.ff(self.film2(self.norm2(x), c)))
        return x


class AttnPool(nn.Module):
    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.query.expand(x.shape[0], -1, -1)
        mask = None if valid is None else ~valid.bool()
        out, _ = self.attn(q, x, x, key_padding_mask=mask, need_weights=False)
        return self.norm(out.squeeze(1))


class HybridParamNet(nn.Module):
    def __init__(self, cfg: HybridModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        H = cfg.frame_ms
        self.rep_names = list(cfg.rep_dims.keys())
        self.rep_slices: Dict[str, Tuple[int, int]] = {}
        pos = 0
        for name, dim in cfg.rep_dims.items():
            self.rep_slices[name] = (pos, pos + int(dim))
            pos += int(dim)
        self.input_dim = pos

        self.stems = nn.ModuleDict({name: nn.Sequential(nn.Linear(int(dim), d), nn.LayerNorm(d)) for name, dim in cfg.rep_dims.items()})
        n_reps = len(self.rep_names)
        self.rep_gate = nn.Parameter(torch.full((n_reps,), 1.0 / math.sqrt(max(1, n_reps))))
        self.rep_missing = nn.Parameter(torch.zeros(n_reps, d))
        self.beat_stem = nn.Linear(cfg.beat_dim, d)
        self.in_norm = nn.LayerNorm(d)
        if cfg.canvas:
            from .canvas import N_CANVAS_STATES
            from .params import NUM_LANES
            self.canvas_state = nn.Embedding(N_CANVAS_STATES, d)
            self.canvas_ms = nn.Embedding(H, d)
            self.canvas_ln = nn.Linear(NUM_LANES, d)
            self.canvas_out = nn.Linear(d, d)
            nn.init.zeros_(self.canvas_out.weight)   # starts as the plain network: an older checkpoint is unchanged
            nn.init.zeros_(self.canvas_out.bias)

        if cfg.plan:
            from .motif_targets import N_PLAN, N_PLAN_COPY
            self.plan_family = nn.Embedding(N_PLAN + 1, d)      # last index = unknown
            self.plan_copy = nn.Embedding(N_PLAN_COPY + 1, d)
            self.plan_out = nn.Linear(d, d)
            nn.init.zeros_(self.plan_out.weight)                 # starts as the network without a plan
            nn.init.zeros_(self.plan_out.bias)
            self.plan_head = nn.Linear(d, N_PLAN + N_PLAN_COPY)
        cond_in = 2 + 2 * cfg.style_dim
        self.cond_mlp = nn.Sequential(nn.Linear(cond_in, d), nn.SiLU(), nn.Linear(d, d))
        self.global_in = nn.Sequential(nn.Linear(cfg.coarse_dim, d), nn.GELU(), nn.Linear(d, d))
        self.global_pool = AttnPool(d, n_heads=4 if d % 4 == 0 else 1)
        self.global_out = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.global_null = nn.Parameter(torch.zeros(d))

        self.conv_in = nn.ModuleList([
            ConvBlock(d, d, cfg.conv_kernel, 2 ** (i % 3), cfg.dropout) for i in range(cfg.conv_layers)
        ])
        self.down = nn.Conv1d(d, d, cfg.down_factor, stride=cfg.down_factor)
        self.attn = nn.ModuleList([
            AttnBlock(d, d, cfg.n_heads, cfg.ff_mult, cfg.dropout, cfg.rope_base) for _ in range(cfg.attn_layers)
        ])
        self.up = nn.ConvTranspose1d(d, d, cfg.down_factor, stride=cfg.down_factor)
        self.conv_out = nn.ModuleList([
            ConvBlock(d, d, cfg.conv_kernel, 1, cfg.dropout) for _ in range(cfg.out_conv_layers)
        ])
        self.out_norm = nn.LayerNorm(d)

        M = cfg.max_members
        self.time_head = nn.Linear(d, 1 + H)
        self.lane_head = nn.Linear(d, cfg.n_lane_codes)
        self.chord_head = nn.Linear(d, M)
        self.stagger_head = nn.Linear(d, M * H)
        self.ln_head = nn.Linear(d, M)
        self.ln_bucket_head = nn.Linear(d, M * cfg.n_ln_buckets)
        self.ln_resid_head = nn.Linear(d, M)
        self.style_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.style_dim))
        if cfg.error_head:
            self.error_head = nn.Linear(d, 1)
        if cfg.pattern_heads:
            from .motif_targets import PATTERN_HEADS
            self.pattern_head = nn.Linear(d, sum(PATTERN_HEADS.values()))
        if cfg.importance_head:
            # difficulty ladder: how important a moment is; it is mapped at star rating s when importance > theta(s)
            self.importance_head = nn.Linear(d, 1)
            self.ladder_c0 = nn.Parameter(torch.tensor(3.0))
            self.ladder_c = nn.Parameter(torch.full((LADDER_BINS,), -0.5))

        with torch.no_grad():
            p = cfg.event_prior
            self.time_head.bias.zero_()
            self.time_head.bias[1:] = math.log(p / H) - math.log(1.0 - p)
            self.stagger_head.bias.view(M, H)[:, 0] = 3.0

    # ------------------------------------------------------------------------------------------
    def _condition(
        self,
        sr: torch.Tensor,
        style: torch.Tensor,
        style_mask: torch.Tensor,
        coarse: Optional[torch.Tensor],
        coarse_mask: Optional[torch.Tensor],
        use_global: torch.Tensor,
    ) -> torch.Tensor:
        sr = sr.view(-1, 1).float()
        sr_known = (sr > 0).float()
        cond = torch.cat([sr / 10.0 * sr_known, sr_known, style * style_mask, style_mask], dim=-1)
        c = self.cond_mlp(cond)
        if coarse is not None:
            g = self.global_out(self.global_pool(self.global_in(coarse), coarse_mask))
            g = torch.where(use_global.view(-1, 1) > 0, g, self.global_null.expand_as(g))
        else:
            g = self.global_null.expand(c.shape[0], -1)
        return c + g

    def forward(
        self,
        feats: torch.Tensor,
        beat: torch.Tensor,
        sr: torch.Tensor,
        style: torch.Tensor,
        style_mask: torch.Tensor,
        coarse: Optional[torch.Tensor] = None,
        coarse_mask: Optional[torch.Tensor] = None,
        rep_present: Optional[torch.Tensor] = None,
        canvas: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        plan: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        B, T, _ = feats.shape
        dev = feats.device
        n_reps = len(self.rep_names)
        present = torch.ones(B, n_reps, device=dev) if rep_present is None else rep_present.float().to(dev)
        use_global = torch.ones(B, device=dev)
        sr = sr.view(B, 1).float()
        style_mask = style_mask.float()

        if self.training:  # condition / modality dropout (the inference-time "unknown" states)
            drop_all = torch.rand(B, 1, device=dev) < cfg.style_dropout
            drop_key = torch.rand(B, style_mask.shape[1], device=dev) < cfg.style_key_dropout
            style_mask = style_mask * (~(drop_all | drop_key)).float()
            sr = sr * (torch.rand(B, 1, device=dev) >= cfg.sr_dropout).float()
            beat = beat * (torch.rand(B, 1, 1, device=dev) >= cfg.beat_dropout).float()
            use_global = (torch.rand(B, device=dev) >= cfg.global_dropout).float()
            if n_reps > 1:
                p_drop = torch.tensor([float(cfg.rep_dropout_by_name.get(r, cfg.rep_dropout)) for r in self.rep_names], device=dev)
                keep = (torch.rand(B, n_reps, device=dev) >= p_drop.view(1, -1)).float() * present
                none_left = keep.sum(dim=1) == 0
                keep[none_left] = present[none_left]
                present = keep

        x = self.beat_stem(beat)
        for i, name in enumerate(self.rep_names):
            s0, s1 = self.rep_slices[name]
            h = self.stems[name](feats[..., s0:s1])
            p = present[:, i].view(B, 1, 1)
            x = x + self.rep_gate[i] * (h * p + self.rep_missing[i].view(1, 1, -1) * (1.0 - p))
        if cfg.canvas:
            if canvas is None:
                st = torch.zeros(B, T, dtype=torch.long, device=dev)
                cms = torch.zeros(B, T, dtype=torch.long, device=dev)
                cln = torch.zeros(B, T, self.canvas_ln.in_features, device=dev)
            else:
                st, cms, cln = canvas[0].long().to(dev), canvas[1].long().to(dev), canvas[2].float().to(dev)
            e = self.canvas_state(st) + self.canvas_ms(cms.clamp(0, cfg.frame_ms - 1)) * (st >= 2).unsqueeze(-1).float() \
                + self.canvas_ln(cln)
            x = x + self.canvas_out(e)
        if cfg.plan:
            from .motif_targets import N_PLAN, N_PLAN_COPY
            if plan is None:
                pf = torch.full((B, T), N_PLAN, dtype=torch.long, device=dev)
                pc = torch.full((B, T), N_PLAN_COPY, dtype=torch.long, device=dev)
            else:
                pf, pc = plan[0].long().to(dev).clamp(0, N_PLAN), plan[1].long().to(dev).clamp(0, N_PLAN_COPY)
            x = x + self.plan_out(self.plan_family(pf) + self.plan_copy(pc))
        x = self.in_norm(x)
        c = self._condition(sr, style.float(), style_mask, coarse, coarse_mask, use_global)

        pad = (-T) % cfg.down_factor
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        for blk in self.conv_in:
            x = blk(x, c)
        skip = x
        h = self.down(x.transpose(1, 2)).transpose(1, 2)
        for blk in self.attn:
            h = blk(h, c)
        pooled = h.mean(dim=1)
        u = self.up(h.transpose(1, 2)).transpose(1, 2) + skip
        for blk in self.conv_out:
            u = blk(u, c)
        u = self.out_norm(u)[:, :T]

        M, Hm = cfg.max_members, cfg.frame_ms
        extra = {}
        if cfg.importance_head:
            extra = {"importance": self.importance_head(u).squeeze(-1), "ladder_c0": self.ladder_c0, "ladder_c": self.ladder_c}
        if cfg.pattern_heads:
            extra["pattern_logits"] = self.pattern_head(u)
        if cfg.error_head:
            extra["error_logits"] = self.error_head(u).squeeze(-1)
        if cfg.plan:
            extra["plan_logits"] = self.plan_head(u)
        return {
            **extra,
            "time_logits": self.time_head(u),
            "lane_logits": self.lane_head(u),
            "chord_logits": self.chord_head(u),
            "stagger_logits": self.stagger_head(u).view(B, T, M, Hm),
            "ln_logits": self.ln_head(u),
            "ln_bucket_logits": self.ln_bucket_head(u).view(B, T, M, cfg.n_ln_buckets),
            "ln_resid": torch.sigmoid(self.ln_resid_head(u)),
            "style_pred": self.style_head(pooled),
        }


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
