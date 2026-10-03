"""
1D Diffusion UNet for osu!mania 4K Beatmap Generation (ManiaDiffusionUNet).

Architecture (matching Stable Diffusion 1.5 principles adapted for 1D symbolic beatmaps):
- Input: [B, 12, T] (4 tap channels + 4 LN start channels + 4 LN sustain channels).
- Timestep Conditioning: Sinusoidal Positional Embedding + 2-layer MLP.
- Difficulty Conditioning: Continuous Star Rating (1.0 to 10.0) projection.
- Audio Conditioning: Multi-Head Cross-Attention to rich acoustic tokens [B, T, D_audio] at every resolution.
- Temporal Self-Attention: Global cross-bar attention across beats to ensure motif repetition, hand balance, and drop coherence.
- Output: [B, 12, T] Predicted noise epsilon.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE and nn is not None:
    BaseModule = nn.Module
else:
    class BaseModule:  # type: ignore
        pass


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class CrossAttention1D(BaseModule):
    """
    Multi-Head Cross-Attention between beatmap hidden states and audio conditioning tokens.
    """
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, dim_head: int = 32):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.05),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> transpose to [B, T, C]
        B, C, T = x.shape
        h = self.heads

        q = self.to_q(x.transpose(1, 2))                    # [B, T, inner_dim]
        k = self.to_k(context)                              # [B, T_ctx, inner_dim]
        v = self.to_v(context)                              # [B, T_ctx, inner_dim]

        q = q.view(B, -1, h, q.shape[-1] // h).transpose(1, 2)
        k = k.view(B, -1, h, k.shape[-1] // h).transpose(1, 2)
        v = v.view(B, -1, h, v.shape[-1] // h).transpose(1, 2)

        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(sim, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, h * v.shape[-1])
        out = self.to_out(out).transpose(1, 2)              # Back to [B, C, T]
        return out + x


class SelfAttention1D(BaseModule):
    """
    Temporal Self-Attention across the beatmap timeline to model motif repetition and hand alternations.
    """
    def __init__(self, in_channels: int, heads: int = 8, dim_head: int = 32):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.GroupNorm(8, in_channels)
        self.to_qkv = nn.Conv1d(in_channels, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(inner_dim, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        h = self.heads
        norm_x = self.norm(x)
        qkv = self.to_qkv(norm_x).chunk(3, dim=1)
        q, k, v = [t.view(B, h, -1, T).transpose(2, 3) for t in qkv]

        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(sim, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).contiguous().view(B, -1, T)
        return x + self.to_out(out)


def get_num_groups(channels: int, max_groups: int = 8) -> int:
    """Finds the largest group count <= max_groups that evenly divides channels."""
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResBlock1D(BaseModule):
    """
    1D Residual Block with timestep embedding and difficulty projection injection.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.05,
    ):
        super().__init__()
        g1 = get_num_groups(in_channels, 8)
        self.norm1 = nn.GroupNorm(g1, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),  # Scale and Shift (AdaLN style)
        )

        g2 = get_num_groups(out_channels, 8)
        self.norm2 = nn.GroupNorm(g2, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))

        time_scale_shift = self.time_mlp(time_emb).unsqueeze(-1)  # [B, 2*out_channels, 1]
        scale, shift = time_scale_shift.chunk(2, dim=1)

        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(self.dropout(self.act2(h)))
        return h + self.residual_conv(x)


class Downsample1D(BaseModule):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(BaseModule):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ManiaDiffusionUNet(BaseModule):
    """
    Complete 1D UNet for beatmap diffusion.
    Operates on [B, 12, T] tensors (4 taps + 4 LN starts + 4 LN sustains).
    """
    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 12,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4),
        audio_dim: int = 256,
        time_dim: int = 256,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Timestep + Star Rating Conditioning
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.sr_emb = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial projection from beatmap channels (12)
        self.init_conv = nn.Conv1d(in_channels, base_channels, kernel_size=5, padding=2)

        # Downsampling Stages with skip channel tracking
        self.down_blocks = nn.ModuleList()
        skip_channel_sizes = [base_channels]
        current_ch = base_channels

        for idx, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                res_block = ResBlock1D(current_ch, out_ch, time_dim)
                cross_attn = CrossAttention1D(out_ch, audio_dim)
                self.down_blocks.append(nn.ModuleList([res_block, cross_attn]))
                current_ch = out_ch
                skip_channel_sizes.append(current_ch)

            if idx < len(channel_multipliers) - 1:
                self.down_blocks.append(Downsample1D(current_ch))
                skip_channel_sizes.append(current_ch)

        # Middle Stage (Bottleneck with global beatmap Self-Attention)
        self.mid_block1 = ResBlock1D(current_ch, current_ch, time_dim)
        self.mid_self_attn = SelfAttention1D(current_ch)
        self.mid_cross_attn = CrossAttention1D(current_ch, audio_dim)
        self.mid_block2 = ResBlock1D(current_ch, current_ch, time_dim)

        # Upsampling Stages: pop skip channels in exact reverse order
        self.up_blocks = nn.ModuleList()
        reversed_mults = list(reversed(channel_multipliers))

        for idx, mult in enumerate(reversed_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_in_ch = skip_channel_sizes.pop()
                in_ch = current_ch + skip_in_ch
                res_block = ResBlock1D(in_ch, out_ch, time_dim)
                cross_attn = CrossAttention1D(out_ch, audio_dim)
                self.up_blocks.append(nn.ModuleList([res_block, cross_attn]))
                current_ch = out_ch

            if idx < len(reversed_mults) - 1:
                self.up_blocks.append(Upsample1D(current_ch))

        # Final Output Projection
        final_groups = get_num_groups(current_ch, 8)
        self.final_norm = nn.GroupNorm(final_groups, current_ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv1d(current_ch, out_channels, kernel_size=5, padding=2)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        audio_context: torch.Tensor,
        star_rating: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of UNet:
            x: [B, 12, T] Noisy beatmap latent
            timesteps: [B] Timestep integers (0..999)
            audio_context: [B, T_audio, audio_dim] Multi-tokenizer audio features
            star_rating: [B, 1] Target difficulty
        Returns:
            predicted_noise: [B, 12, T]
        """
        # Embed time and difficulty
        t_emb = self.time_emb(timesteps.float())
        sr_emb = self.sr_emb(star_rating)
        cond_emb = t_emb + sr_emb

        # Initial convolution
        h = self.init_conv(x)
        skip_connections = [h]

        # Downsample
        for block in self.down_blocks:
            if isinstance(block, Downsample1D):
                h = block(h)
                skip_connections.append(h)
            else:
                res_block, cross_attn = block
                h = res_block(h, cond_emb)
                # Resample audio context to match temporal length if needed
                if audio_context.shape[1] != h.shape[2]:
                    ctx_resampled = F.interpolate(
                        audio_context.transpose(1, 2),
                        size=h.shape[2],
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)
                else:
                    ctx_resampled = audio_context
                h = cross_attn(h, ctx_resampled)
                skip_connections.append(h)

        # Middle
        h = self.mid_block1(h, cond_emb)
        h = self.mid_self_attn(h)
        if audio_context.shape[1] != h.shape[2]:
            ctx_mid = F.interpolate(
                audio_context.transpose(1, 2),
                size=h.shape[2],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            ctx_mid = audio_context
        h = self.mid_cross_attn(h, ctx_mid)
        h = self.mid_block2(h, cond_emb)

        # Upsample
        for block in self.up_blocks:
            if isinstance(block, Upsample1D):
                h = block(h)
            else:
                res_block, cross_attn = block
                skip = skip_connections.pop()
                if skip.shape[2] != h.shape[2]:
                    skip = F.interpolate(skip, size=h.shape[2], mode="linear", align_corners=False)
                h = torch.cat([h, skip], dim=1)
                h = res_block(h, cond_emb)
                if audio_context.shape[1] != h.shape[2]:
                    ctx_up = F.interpolate(
                        audio_context.transpose(1, 2),
                        size=h.shape[2],
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)
                else:
                    ctx_up = audio_context
                h = cross_attn(h, ctx_up)

        # Final output
        out = self.final_conv(self.final_act(self.final_norm(h)))
        return out
