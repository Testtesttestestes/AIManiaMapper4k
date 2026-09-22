"""
ManiaPatternTransformer architecture for osu!mania 4K beatmap generation.

Features:
- Mel-spectrogram audio feature projection
- Star Rating (1.0 to 9.9) continuous conditioning tensor
- Causal autoregressive self-attention layers
- 16-class chord tokenization (2^4 = 16 discrete 4K key configurations)
- Multi-head Long Note (LN) duration prediction
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


def keys_to_token(k0: int, k1: int, k2: int, k3: int) -> int:
    """Converts 4 binary key states to a discrete token index (0 to 15)."""
    return (int(bool(k0))) | (int(bool(k1)) << 1) | (int(bool(k2)) << 2) | (int(bool(k3)) << 3)


def token_to_keys(token: int) -> Tuple[int, int, int, int]:
    """Decodes a token (0 to 15) into 4 binary key states."""
    k0 = token & 1
    k1 = (token >> 1) & 1
    k2 = (token >> 2) & 1
    k3 = (token >> 3) & 1
    return (k0, k1, k2, k3)


@dataclass
class TransformerConfig:
    audio_dim: int = 80          # Mel-spectrogram bins
    d_model: int = 1024          # Latent representation dimension
    nhead: int = 8               # Multi-head attention heads
    num_layers: int = 6          # Number of Transformer layers
    dim_feedforward: int = 2048  # FFN expansion dimension
    num_classes: int = 16        # 16 4K chord combinations
    max_seq_len: int = 1024      # Temporal context window length
    dropout: float = 0.1         # Dropout rate


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

    class ManiaPatternTransformer(nn.Module):
        """
        PyTorch causal Transformer for osu!mania 4K pattern generation.
        """
        def __init__(self, config: Optional[TransformerConfig] = None):
            super().__init__()
            self.config = config or TransformerConfig()

            # Audio projection (mel-spectrogram -> d_model)
            self.audio_proj = nn.Linear(self.config.audio_dim, self.config.d_model)

            # Conditioning vector for Star Rating (scalar 1.0 - 9.9 -> d_model)
            self.sr_encoder = nn.Sequential(
                nn.Linear(1, self.config.d_model // 2),
                nn.SiLU(),
                nn.Linear(self.config.d_model // 2, self.config.d_model),
            )

            # Token embedding for autoregressive history (previous tokens 0-15)
            self.token_embed = nn.Embedding(self.config.num_classes, self.config.d_model)

            # Positional encoding
            self.pos_encoder = PositionalEncoding(self.config.d_model, self.config.max_seq_len)

            # Causal Transformer Decoder layers
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=self.config.d_model,
                nhead=self.config.nhead,
                dim_feedforward=self.config.dim_feedforward,
                dropout=self.config.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=self.config.num_layers)

            # Output heads
            self.chord_head = nn.Linear(self.config.d_model, self.config.num_classes)
            self.ln_head = nn.Linear(self.config.d_model, 4)  # LN probability per column (4 keys)

        def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
            mask = torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
            return mask

        def forward(
            self,
            audio_mel: torch.Tensor,
            star_rating: Optional[torch.Tensor] = None,
            prev_tokens: Optional[torch.Tensor] = None,
            return_hidden: bool = False,
        ):
            """
            Args:
                audio_mel: [Batch, SeqLen, audio_dim] or [Batch, SeqLen, d_model]
                star_rating: [Batch, 1] Continuous difficulty scalar (optional)
                prev_tokens: [Batch, SeqLen] Previous chord tokens (0-15) (optional)
                return_hidden: bool, if True returns hidden state [B, T, d_model]
            Returns:
                hidden [Batch, SeqLen, d_model] if return_hidden or star_rating is None
                otherwise (chord_logits, ln_logits)
            """
            B, T = audio_mel.shape[0], audio_mel.shape[1]
            if audio_mel.shape[-1] == self.config.audio_dim:
                audio_feat = self.audio_proj(audio_mel)
            elif audio_mel.shape[-1] == self.config.d_model:
                audio_feat = audio_mel
            else:
                audio_feat = self.audio_proj(audio_mel)

            if prev_tokens is not None:
                tok_feat = self.token_embed(prev_tokens)
            else:
                tok_feat = torch.zeros_like(audio_feat)

            if star_rating is not None:
                sr_feat = self.sr_encoder(star_rating).unsqueeze(1)
            else:
                sr_feat = torch.zeros(B, 1, self.config.d_model, device=audio_mel.device)

            # Fuse audio, autoregressive token context, and Star Rating condition
            x = audio_feat + tok_feat + sr_feat
            x = self.pos_encoder(x)

            # Apply causal mask
            mask = self._generate_causal_mask(T, audio_mel.device)
            hidden = self.transformer(x, mask=mask)

            if return_hidden or (star_rating is None and prev_tokens is None):
                return hidden

            chord_logits = self.chord_head(hidden)
            ln_logits = self.ln_head(hidden)
            return chord_logits, ln_logits

else:
    class ManiaPatternTransformer:
        """
        Architecture specification and pure-Python fallback interface
        when PyTorch is not available in the current execution runtime.
        """
        def __init__(self, config: Optional[TransformerConfig] = None):
            self.config = config or TransformerConfig()
            self.is_torch_backed = False

        def get_model_summary(self) -> Dict[str, Any]:
            """Returns theoretical architecture parameters and sizing."""
            return {
                "name": "ManiaPatternTransformer",
                "audio_dim": self.config.audio_dim,
                "d_model": self.config.d_model,
                "nhead": self.config.nhead,
                "num_layers": self.config.num_layers,
                "num_classes": self.config.num_classes,
                "max_seq_len": self.config.max_seq_len,
                "torch_available": False,
                "estimated_parameters": (
                    self.config.audio_dim * self.config.d_model
                    + self.config.num_layers * (4 * self.config.d_model * self.config.d_model + 2 * self.config.d_model * self.config.dim_feedforward)
                    + self.config.d_model * self.config.num_classes
                ),
            }

        def forward(self, *args, **kwargs):
            raise NotImplementedError(
                "PyTorch is not installed in the current environment. "
                "Install PyTorch (`pip install torch`) to run live neural inference."
            )
