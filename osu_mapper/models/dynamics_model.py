"""
Model 2: DynamicsSVModel for Slider Velocity (SV) and timing zone dynamics.

Predicts continuous SV speed curves and timing point acceleration zones
from local audio energy and spectrogram dynamics.
"""

from __future__ import annotations

from typing import Any, Tuple

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


class DynamicsSVModel(BaseModule):
    """
    Temporal ConvNet + Bidirectional GRU for continuous SV prediction.
    Outputs SV multiplier per frame and timing point transition probabilities.
    """
    def __init__(self, audio_dim: int = 80, hidden_dim: int = 128):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to use DynamicsSVModel.")
        super().__init__()
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim

        # Front-end convolutional feature extractor
        self.conv1 = nn.Conv1d(audio_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Temporal sequence modeling
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Heads: SV continuous multiplier & timing change trigger
        self.sv_head = nn.Linear(hidden_dim * 2, 1)
        self.trigger_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, audio_mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_mel: [Batch, SeqLen, audio_dim]
        Returns:
            sv_curve: [Batch, SeqLen, 1] continuous SV multiplier centered around 1.0
            trigger_logits: [Batch, SeqLen, 1] probability logits for timing line placement
        """
        x = audio_mel.transpose(1, 2)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)

        gru_out, _ = self.gru(x)
        # SV centered at 1.0 (range ~ 0.2 to 4.0)
        sv = F.softplus(self.sv_head(gru_out)) + 0.2
        triggers = self.trigger_head(gru_out)
        return sv, triggers
