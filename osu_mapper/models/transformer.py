"""
ManiaPatternTransformer architecture for osu!mania 4K beatmap generation.

Features:
- Bidirectional Deep Multi-Band Audio Encoder: Processes the full 80-bin frequency spectrum
  with multi-scale temporal convolutions and future lookahead context (sub-bass, mid, highs, melody).
- Multi-Head Audio-to-Rhythm Cross-Attention: Token queries attend directly to multi-band audio keys/values,
  grounding every placed note and chord in the exact acoustic frequencies and transients.
- 5-State Physical Lane Autoregression:
    0: STATE_FREE       (Lane is idle)
    1: STATE_TAP        (Instantaneous circle note hit)
    2: STATE_LN_START   (Long note head press)
    3: STATE_LN_HOLD    (Continuous sustain / key held down)
    4: STATE_LN_RELEASE (Long note release / tail)
  This enables the model to natively distinguish between tap notes, micro-sliders, and long holds
  without relying on post-hoc heuristics or filters.
- Musical Grid & Metric Conditioning: Embeds musical beat fractions (1/1, 1/2, 1/3, 1/4, 1/6, 1/8) and downbeat phase.
- Acoustic Energy Gating: Senses music slowdowns, breaks, drops, and pauses.
- Star Rating (1.0 to 9.9) continuous conditioning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


# Constants for 5 discrete lane states
STATE_FREE = 0
STATE_TAP = 1
STATE_LN_START = 2
STATE_LN_HOLD = 3
STATE_LN_RELEASE = 4


def keys_to_token(k0: int, k1: int, k2: int, k3: int) -> int:
    """Converts 4 binary key states to a discrete token index (0 to 15)."""
    return (int(bool(k0))) | (int(bool(k1)) << 1) | (int(bool(k2)) << 2) | (int(bool(k3)) << 3)


def token_to_keys(token: int) -> Tuple[int, int, int, int]:
    """Decodes a token (0 to 15) into 4 binary key states."""
    return (token & 1, (token >> 1) & 1, (token >> 2) & 1, (token >> 3) & 1)


@dataclass
class TransformerConfig:
    audio_dim: int = 80          # Mel-spectrogram bins across full frequency spectrum (20Hz - 8000Hz)
    d_model: int = 256           # Latent representation dimension
    nhead: int = 8               # Multi-head attention heads
    num_layers: int = 4          # Number of Cross-Attention Transformer layers
    dim_feedforward: int = 512   # FFN expansion dimension
    num_classes: int = 16        # 16 4K chord combinations
    num_lane_states: int = 5     # 5 discrete physical states per lane (Free, Tap, LN_Start, LN_Hold, LN_Release)
    max_seq_len: int = 1024      # Temporal context window length
    dropout: float = 0.1         # Dropout rate
    context_dropout: float = 0.20 # Probability of zeroing/masking history during training
    audio_lookahead_frames: int = 16 # Future temporal lookahead window (~256ms)


if TORCH_AVAILABLE:
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: [B, T, D]
            return x + self.pe[:, :x.size(1)]

    class DeepAudioEncoder(nn.Module):
        """
        Bidirectional Multi-Band Audio Encoder.
        Extracts multi-scale acoustic features across all frequency bands (bass/kicks, mid/synths, highs/cymbals)
        with temporal future lookahead.
        """
        def __init__(self, in_dim: int = 80, d_model: int = 256, dropout: float = 0.1):
            super().__init__()
            self.in_proj = nn.Linear(in_dim, d_model)
            
            # Multi-scale 1D depthwise-separable temporal convolutions for audio transients
            self.conv_block = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.Dropout(dropout),
            )
            
            # Bidirectional Self-Attention encoder for acoustic context & global song dynamics
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.audio_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.norm = nn.LayerNorm(d_model)

        def forward(self, audio_mel: torch.Tensor) -> torch.Tensor:
            # audio_mel: [B, T, in_dim]
            x = self.in_proj(audio_mel)  # [B, T, d_model]
            
            # Temporal convolution
            x_conv = self.conv_block(x.transpose(1, 2)).transpose(1, 2)
            x = self.norm(x + x_conv)
            
            # Bidirectional audio attention (sees future and past acoustic frames)
            audio_encoded = self.audio_transformer(x)
            return audio_encoded

    class CrossAttentionDecoderBlock(nn.Module):
        """
        Transformer Decoder Block with:
        1. Causal Self-Attention (Autoregressive note history & metric context)
        2. Audio Cross-Attention (Queries: rhythmic/lane tokens, Keys/Values: full-spectrum audio features)
        3. Feedforward network
        """
        def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
            self.activation = nn.GELU()

        def forward(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            # 1. Causal Self-Attention
            q1 = tgt
            sa_out, _ = self.self_attn(q1, q1, q1, attn_mask=tgt_mask)
            tgt = self.norm1(tgt + self.dropout1(sa_out))

            # 2. Audio Cross-Attention (Grounding tokens in full acoustic spectrum)
            ca_out, _ = self.cross_attn(query=tgt, key=memory, value=memory)
            tgt = self.norm2(tgt + self.dropout2(ca_out))

            # 3. Position-wise Feedforward
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = self.norm3(tgt + self.dropout3(ff_out))
            return tgt

    class ManiaPatternTransformer(nn.Module):
        """
        PyTorch Transformer for osu!mania 4K pattern generation.
        Conditioned on:
        - Multi-band audio spectrogram via Cross-Attention
        - Star Rating difficulty tensor
        - Musical Grid Metric (beat fraction / downbeat phase)
        - 5-State Physical Lane Autoregression (0=Free, 1=Tap, 2=LN_Start, 3=LN_Hold, 4=LN_Release)
        """
        def __init__(self, config: Optional[TransformerConfig] = None):
            super().__init__()
            self.config = config or TransformerConfig()

            # Bidirectional Deep Multi-Band Audio Encoder
            self.audio_encoder = DeepAudioEncoder(
                in_dim=self.config.audio_dim,
                d_model=self.config.d_model,
                dropout=self.config.dropout,
            )
            self.audio_proj = nn.Linear(self.config.audio_dim, self.config.d_model)

            # Conditioning vector for Star Rating (scalar 1.0 - 9.9 -> d_model)
            self.sr_encoder = nn.Sequential(
                nn.Linear(1, self.config.d_model // 2),
                nn.SiLU(),
                nn.Linear(self.config.d_model // 2, self.config.d_model),
            )

            # Autoregressive Token Embedding (Chords 0-15)
            self.token_embed = nn.Embedding(self.config.num_classes, self.config.d_model)

            # 5-State Physical Lane Projection: 4 lanes * 5 states = 20 one-hot inputs -> d_model
            self.num_lane_states = getattr(self.config, "num_lane_states", 5)
            self.lane_state_proj = nn.Linear(4 * self.num_lane_states, self.config.d_model)

            # Musical Metric Grid Embedding (Beat divisor, Bar position, Downbeat flag) -> d_model
            self.metric_proj = nn.Sequential(
                nn.Linear(3, self.config.d_model // 2),
                nn.GELU(),
                nn.Linear(self.config.d_model // 2, self.config.d_model),
            )

            # Positional encoding
            self.pos_encoder = PositionalEncoding(self.config.d_model, self.config.max_seq_len)

            # Cross-Attention Transformer Decoder layers
            self.decoder_layers = nn.ModuleList([
                CrossAttentionDecoderBlock(
                    d_model=self.config.d_model,
                    nhead=self.config.nhead,
                    dim_feedforward=self.config.dim_feedforward,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.num_layers)
            ])

            # Output heads
            self.chord_head = nn.Linear(self.config.d_model, self.config.num_classes)
            self.ln_head = nn.Linear(self.config.d_model, 4)            # LN trigger probability per column
            self.ln_duration_head = nn.Linear(self.config.d_model, 4)   # LN sustain tail duration in frames

            # Acoustic energy & silence gate (suppresses notes in drops/pauses)
            self.acoustic_gate = nn.Sequential(
                nn.Linear(self.config.d_model, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
            mask = torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
            return mask

        def forward(
            self,
            audio_mel: torch.Tensor,
            star_rating: Optional[torch.Tensor] = None,
            prev_tokens: Optional[torch.Tensor] = None,
            lane_states: Optional[torch.Tensor] = None,
            grid_metrics: Optional[torch.Tensor] = None,
            return_hidden: bool = False,
            context_dropout: Optional[float] = None,
        ):
            """
            Args:
                audio_mel: [Batch, SeqLen, audio_dim] Full multi-band log-mel spectrogram
                star_rating: [Batch, 1] Continuous difficulty scalar (1.0 to 9.9)
                prev_tokens: [Batch, SeqLen] Autoregressive history of chord tokens (0-15)
                lane_states: Optional [Batch, SeqLen, 4] integer states [0..4] or [Batch, SeqLen, 20] float one-hot
                grid_metrics: Optional [Batch, SeqLen, 3] musical metric values (divisor, bar_pos, downbeat)
                return_hidden: bool, if True returns latent representations
                context_dropout: Optional override for context dropout probability
            """
            B, T = audio_mel.shape[0], audio_mel.shape[1]
            
            # 1. Encode full acoustic spectrum with lookahead & multi-band attention
            if audio_mel.shape[-1] == self.config.audio_dim:
                audio_encoded = self.audio_encoder(audio_mel)
            else:
                audio_encoded = self.audio_proj(audio_mel)

            # 2. Context Dropout on tokens
            cd_prob = context_dropout if context_dropout is not None else (self.config.context_dropout if self.training else 0.0)
            if prev_tokens is not None:
                if self.training and cd_prob > 0.0:
                    drop_mask = (torch.rand(B, 1, device=prev_tokens.device) < cd_prob).expand(B, T)
                    effective_tokens = prev_tokens.clone()
                    effective_tokens[drop_mask] = 0
                    tok_feat = self.token_embed(effective_tokens)
                    tok_feat[drop_mask] = 0.0
                else:
                    tok_feat = self.token_embed(prev_tokens)
            else:
                tok_feat = torch.zeros(B, T, self.config.d_model, device=audio_mel.device)

            # 3. Star rating difficulty conditioning
            if star_rating is not None:
                sr_feat = self.sr_encoder(star_rating).unsqueeze(1)
            else:
                sr_feat = torch.zeros(B, 1, self.config.d_model, device=audio_mel.device)

            # 4. 5-State Physical Lane Embedding (Free=0, Tap=1, LN_Start=2, LN_Hold=3, LN_Release=4)
            if lane_states is not None:
                if lane_states.shape[-1] == 4:
                    num_states = getattr(self.config, "num_lane_states", 5)
                    one_hot = F.one_hot(lane_states.long().clamp(0, num_states - 1), num_classes=num_states).float().view(B, T, 4 * num_states)
                    lane_feat = self.lane_state_proj(one_hot)
                elif lane_states.shape[-1] == (4 * getattr(self.config, "num_lane_states", 5)):
                    lane_feat = self.lane_state_proj(lane_states.float())
                else:
                    lane_feat = torch.zeros(B, T, self.config.d_model, device=audio_mel.device)
            else:
                lane_feat = torch.zeros(B, T, self.config.d_model, device=audio_mel.device)

            # 5. Musical Grid Metrics (Beat Divisor, Bar Fraction, Downbeat)
            if grid_metrics is not None and grid_metrics.shape[-1] == 3:
                metric_feat = self.metric_proj(grid_metrics.float())
            else:
                metric_feat = torch.zeros(B, T, self.config.d_model, device=audio_mel.device)

            # 6. Rhythm target queries
            tgt = self.pos_encoder(tok_feat + sr_feat + lane_feat + metric_feat)
            causal_mask = self._generate_causal_mask(T, audio_mel.device)

            # 7. Cross-Attention Decoder Layers
            hidden = tgt
            for layer in self.decoder_layers:
                hidden = layer(hidden, memory=audio_encoded, tgt_mask=causal_mask)

            if return_hidden or (star_rating is None and prev_tokens is None):
                return hidden

            # 8. Predictions
            chord_logits = self.chord_head(hidden)
            ln_logits = self.ln_head(hidden)
            ln_durations = F.relu(self.ln_duration_head(hidden))

            # 9. Acoustic silence gating
            energy_gate = self.acoustic_gate(audio_encoded)
            silence_boost = (1.0 - energy_gate) * 4.0
            chord_logits[:, :, 0] += silence_boost.squeeze(-1)

            return chord_logits, ln_logits
