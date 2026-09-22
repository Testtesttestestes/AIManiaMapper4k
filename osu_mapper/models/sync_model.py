"""
Model 1: SyncBPMOffsetModel for rhythm synchronization and tempo estimation.

Enhanced architecture designed for high sample-efficiency on small datasets:
1. Differentiable Autocorrelation / Rhythm Periodicity Layer (Onset Spectral Flux)
2. Dilated 1D Convolutional Backbone with Squeeze-and-Excitation / Temporal Pooling
3. Dual-Head BPM Predictor: 301-bin Classification (60-360 BPM) + Soft-Argmax Expectation + Residual Regression
4. Beat-Periodic Phase Offset Head: Predicts circular downbeat phase within beat period,
   guaranteeing mathematical alignment without phase drift.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableAutocorr(nn.Module):
    """
    Extracts differentiable periodicity features from audio Mel spectrograms.
    Computes spectral flux (onset strength envelope) and its multi-lag autocorrelation,
    explicitly exposing periodic rhythm peaks (tempo) to the network.
    """
    def __init__(self, min_lag: int = 8, max_lag: int = 96):
        super().__init__()
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.num_lags = max_lag - min_lag

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [Batch, Channels=80, Time]
        Returns:
            autocorr: [Batch, num_lags] normalized periodicity feature vector
        """
        # Spectral flux: positive half-wave rectified difference over time
        flux = F.relu(mel[:, :, 1:] - mel[:, :, :-1]).mean(dim=1, keepdim=True)  # [B, 1, T-1]
        flux_norm = flux - flux.mean(dim=-1, keepdim=True)
        var = (flux_norm ** 2).mean(dim=-1, keepdim=True) + 1e-6

        # Fast autocorrelation over candidate lags
        B, _, T = flux_norm.shape
        autocorr_list = []
        for lag in range(self.min_lag, min(self.max_lag, T // 2)):
            corr = (flux_norm[:, 0, :-lag] * flux_norm[:, 0, lag:]).mean(dim=-1, keepdim=True) / var[:, 0]
            autocorr_list.append(corr)

        if not autocorr_list:
            return torch.zeros(B, self.num_lags, device=mel.device)

        ac = torch.cat(autocorr_list, dim=-1)
        if ac.shape[-1] < self.num_lags:
            pad = torch.zeros(B, self.num_lags - ac.shape[-1], device=mel.device)
            ac = torch.cat([ac, pad], dim=-1)
        return ac[:, :self.num_lags]


class SyncBPMOffsetModel(nn.Module):
    """
    Advanced Tempo (BPM) and Downbeat Offset synchronization network.
    Combines 1D-CNN temporal features with differentiable autocorrelation tempogram
    and soft-argmax classification to achieve sub-BPM precision even on small datasets.
    """
    def __init__(self, audio_dim: int = 80, hidden_dim: int = 256):
        super().__init__()
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim

        # 1. Differentiable Autocorrelation feature extractor
        self.autocorr = DifferentiableAutocorr(min_lag=8, max_lag=96)  # 88 lags (~40 to 450 BPM)
        self.autocorr_proj = nn.Sequential(
            nn.Linear(88, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        # 2. 1D Convolutional Backbone with Dilations for multi-scale rhythm capture
        self.conv_in = nn.Sequential(
            nn.Conv1d(audio_dim, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        self.res_block1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(128),
        )

        self.conv_mid = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )

        self.res_block2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=3, dilation=4, padding=4),
            nn.BatchNorm1d(256),
        )

        self.conv_out = nn.Sequential(
            nn.Conv1d(256, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(32),
        )

        # 3. Dense latent projection fusing CNN temporal summary + Autocorrelation vector
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 32 + 64, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        # 4. BPM Prediction: 301-bin Classification (60..360 BPM) + Soft-Argmax + Residual
        self.num_bins = 301
        self.bpm_min = 60.0
        self.bpm_max = 360.0
        self.register_buffer("bpm_bins", torch.linspace(self.bpm_min, self.bpm_max, self.num_bins))

        self.bpm_classifier = nn.Linear(128, self.num_bins)
        self.bpm_residual = nn.Linear(128, 1)

        # 5. Offset Prediction: Circular phase in [0, 1) + measure position
        self.offset_phase_head = nn.Linear(128, 2)  # cos(phi), sin(phi)
        self.offset_bar_head = nn.Linear(128, 4)    # beat 0, 1, 2, 3 in 4/4 bar
        self.offset_direct = nn.Linear(128, 1)      # auxiliary direct scalar head

    def forward(
        self,
        audio_mel: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_mel: [Batch, SeqLen, audio_dim] or [Batch, audio_dim, SeqLen]
            return_logits: If True, returns (bpm, offset, bpm_logits) for classification loss
        Returns:
            bpm: [Batch, 1] predicted tempo in BPM
            offset: [Batch, 1] predicted downbeat offset in ms
        """
        if audio_mel.dim() == 2:
            audio_mel = audio_mel.unsqueeze(0)

        # Ensure shape [Batch, audio_dim, SeqLen]
        if audio_mel.shape[1] != self.audio_dim and audio_mel.shape[2] == self.audio_dim:
            x = audio_mel.transpose(1, 2)
        else:
            x = audio_mel

        # 1. Differentiable Autocorrelation
        ac_feat = self.autocorr(x)
        ac_emb = self.autocorr_proj(ac_feat)  # [B, 64]

        # 2. Convolutional Backbone
        h = self.conv_in(x)
        h = F.gelu(h + self.res_block1(h))
        h = self.conv_mid(h)
        h = F.gelu(h + self.res_block2(h))
        h = self.conv_out(h)
        h = h.flatten(1)  # [B, hidden_dim * 32]

        # 3. Fuse with periodicity embedding
        fused = torch.cat([h, ac_emb], dim=-1)
        latent = self.fc(fused)

        # 4. Sharp Peak-Centered BPM Prediction
        bpm_logits = self.bpm_classifier(latent)  # [B, 301]

        # Find peak bin index (mode of distribution)
        peak_idx = bpm_logits.argmax(dim=-1, keepdim=True)  # [B, 1]

        # Focus expectation strictly on local neighborhood [-3, +3] around the top peak
        # This completely prevents distant uniform tails from dragging the prediction towards the middle (210 BPM)
        offsets = torch.arange(-3, 4, device=audio_mel.device).view(1, 7)  # [1, 7]
        local_bins_idx = torch.clamp(peak_idx + offsets, 0, self.num_bins - 1)  # [B, 7]

        local_logits = torch.gather(bpm_logits, dim=-1, index=local_bins_idx)  # [B, 7]
        local_probs = F.softmax(local_logits / 0.2, dim=-1)  # sharp temperature
        local_bpm_values = self.bpm_bins[local_bins_idx]  # [B, 7]
        expected_bpm = (local_probs * local_bpm_values).sum(dim=-1, keepdim=True)  # [B, 1]

        residual = torch.tanh(self.bpm_residual(latent)) * 0.5
        pred_bpm = expected_bpm + residual
        pred_bpm = torch.clamp(pred_bpm, min=50.0, max=360.0)

        # 5. Beat-Periodic Offset computation
        phase_raw = self.offset_phase_head(latent)
        phase_cos = phase_raw[:, 0:1]
        phase_sin = phase_raw[:, 1:2]
        # Normalized continuous phase in [0, 1)
        phi = (torch.atan2(phase_sin, phase_cos) / (2.0 * math.pi) + 0.5) % 1.0

        bar_probs = F.softmax(self.offset_bar_head(latent), dim=-1)
        beat_indices = torch.tensor([0.0, 1.0, 2.0, 3.0], device=audio_mel.device).unsqueeze(0)
        expected_beat = (bar_probs * beat_indices).sum(dim=-1, keepdim=True)

        beat_len_ms = 60000.0 / (pred_bpm + 1e-5)
        # Downbeat offset anchored within first bar
        periodic_offset = (expected_beat + phi) * beat_len_ms

        direct_offset = F.softplus(self.offset_direct(latent)) * 250.0
        # Hybrid blend: 70% periodic beat-snapped + 30% direct macro offset
        pred_offset = 0.75 * periodic_offset + 0.25 * direct_offset
        pred_offset = torch.clamp(pred_offset, min=0.0)

        if return_logits:
            return pred_bpm, pred_offset, bpm_logits
        return pred_bpm, pred_offset
