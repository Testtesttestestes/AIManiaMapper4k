"""
Model 1: SyncBPMOffsetModel for rhythm synchronization and tempo estimation.

Analyzes Mel-spectrogram audio representation to predict:
1. Base tempo (BPM) via continuous regression
2. Initial beat offset (Offset in milliseconds)
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SyncBPMOffsetModel(nn.Module):
    """
    1D-CNN + Temporal Pooling network for tempo (BPM) and first downbeat offset regression.
    """
    def __init__(self, audio_dim: int = 80, hidden_dim: int = 256):
        super().__init__()
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim

        # Input projection & 1D convolutions over time
        self.conv_stack = nn.Sequential(
            nn.Conv1d(audio_dim, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(32),
        )

        # Regressor heads
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 32, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
        )

        self.bpm_head = nn.Linear(64, 1)      # Normalized BPM output
        self.offset_head = nn.Linear(64, 1)   # Normalized Offset output (ms)

    def forward(self, audio_mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_mel: [Batch, SeqLen, audio_dim]
        Returns:
            bpm_pred: [Batch, 1] predicted BPM
            offset_pred: [Batch, 1] predicted offset in ms
        """
        # Transpose to [Batch, audio_dim, SeqLen] for Conv1d
        x = audio_mel.transpose(1, 2)
        features = self.conv_stack(x)
        features = features.flatten(1)
        latent = self.fc(features)

        # Unscale predictions to sensible ranges (BPM ~ 60..300, Offset ~ 0..3000ms)
        bpm = F.softplus(self.bpm_head(latent)) * 100.0 + 60.0
        offset = F.softplus(self.offset_head(latent)) * 500.0

        return bpm, offset
