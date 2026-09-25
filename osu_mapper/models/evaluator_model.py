"""
Model 4: DifficultyEvaluatorModel for Star Rating and density validation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DifficultyEvaluatorModel(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.token_embed = nn.Embedding(16, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True, bidirectional=True)
        self.sr_head = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(tokens)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        sr = self.sr_head(pooled).squeeze(-1)
        return sr
