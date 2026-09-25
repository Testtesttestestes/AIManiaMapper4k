"""
Model 2: DynamicsSVModel for continuous Slider Velocity curves.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicsSVModel(nn.Module):
    def __init__(self, audio_dim: int = 80, hidden_dim: int = 128):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(audio_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.sv_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Maps to [0.5, 2.0] via scale
        )

    def forward(self, audio_mel: torch.Tensor) -> torch.Tensor:
        x = audio_mel.transpose(1, 2)
        x = self.conv_in(x).transpose(1, 2)
        out, _ = self.gru(x)
        raw_sv = self.sv_head(out).squeeze(-1)
        sv = 0.5 + raw_sv * 1.5  # 0.5x to 2.0x
        return sv
