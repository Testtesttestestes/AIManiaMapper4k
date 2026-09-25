"""
Model 1: SyncBPMOffsetModel for tempo and offset estimation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SyncBPMOffsetModel(nn.Module):
    def __init__(self, audio_dim: int = 80, hidden_dim: int = 256):
        super().__init__()
        self.audio_dim = audio_dim
        self.conv_in = nn.Sequential(
            nn.Conv1d(audio_dim, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.bpm_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 301),  # Bins from 30.0 to 330.0 BPM
        )

        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, audio_mel: torch.Tensor):
        x = audio_mel.transpose(1, 2)
        x = self.conv_in(x).transpose(1, 2)
        feat = self.transformer(x)
        pooled = feat.mean(dim=1)

        bpm_logits = self.bpm_head(pooled)
        probs = F.softmax(bpm_logits, dim=-1)
        bpm_bins = torch.linspace(30.0, 330.0, 301, device=audio_mel.device)
        pred_bpm = torch.sum(probs * bpm_bins, dim=-1, keepdim=True)

        pred_offset = self.offset_head(pooled) * 1000.0
        return pred_bpm, pred_offset
