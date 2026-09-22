"""
Model 4: DifficultyEvaluatorModel for Star Rating and Pattern Density validation.

Takes hit object token sequences and audio features to evaluate:
1. Star Rating (SR: 1.0 - 9.9) adherence to user targets (+-0.3 star tolerance)
2. Average and peak Notes Per Second (NPS) density
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifficultyEvaluatorModel(nn.Module):
    """
    Evaluator network assessing beatmap difficulty and pattern density compliance.
    """
    def __init__(self, num_classes: int = 16, token_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.token_embed = nn.Embedding(num_classes, token_dim)
        
        self.conv_net = nn.Sequential(
            nn.Conv1d(token_dim, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 16, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, token_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_seq: [Batch, SeqLen] integer chord tokens (0..15)
        Returns:
            predicted_sr: [Batch, 1] predicted Star Rating (1.0 to 9.9)
        """
        emb = self.token_embed(token_seq).transpose(1, 2)
        feat = self.conv_net(emb).flatten(1)
        sr = F.softplus(self.regressor(feat)) + 1.0
        return sr
