"""
Acoustic Onset-Driven Algorithmic Scaffold Generator.
Fully vectorized PyTorch implementation for ultra-fast GPU training and inference (0 Python loops).
"""

from typing import Any, Optional

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


def compute_acoustic_onset_scaffold(
    audio_mel: Any,
    grid_metrics: Optional[Any],
    star_rating: Any,
    silence_threshold: float = 0.20,
) -> Any:
    """
    Generates an acoustic rhythm scaffold directly from audio spectral features and grid metrics.
    100% Vectorized in PyTorch (zero CPU-GPU synchronization bottlenecks).

    Args:
        audio_mel: [B, T, 80] Log-mel spectrogram
        grid_metrics: Optional [B, T, 8] Grid metrics (divisor_fraction, bar_phase, is_downbeat, etc.)
        star_rating: [B, 1] or [B] Star rating difficulty tensor
        silence_threshold: Relative energy threshold below which no notes are placed

    Returns:
        scaffold_tokens: [B, T] Integer tensor of discrete 4K chord tokens (0-15)
    """
    if not TORCH_AVAILABLE or torch is None:
        return None

    B, T, D = audio_mel.shape
    device = audio_mel.device

    # 1. Compute spectral energy and spectral flux in linear power domain
    linear_mel = audio_mel.exp()
    frame_energy = linear_mel.mean(dim=-1)                                 # [B, T]
    mean_energy = frame_energy.mean(dim=-1, keepdim=True).clamp(min=1e-4)   # [B, 1]
    rel_energy = frame_energy / mean_energy                                # [B, T]

    # Spectral flux (half-wave rectified difference between consecutive frames)
    diff = torch.zeros_like(linear_mel)
    diff[:, 1:] = (linear_mel[:, 1:] - linear_mel[:, :-1]).clamp(min=0.0)
    flux = diff.mean(dim=-1)                                              # [B, T]
    mean_flux = flux.mean(dim=-1, keepdim=True).clamp(min=1e-4)           # [B, 1]
    rel_flux = flux / mean_flux                                            # [B, T]

    # Multi-band energy (Low: kick/bass 0..20)
    low_band = linear_mel[:, :, :20].mean(dim=-1)                         # [B, T]
    low_flux = torch.zeros_like(low_band)
    low_flux[:, 1:] = (low_band[:, 1:] - low_band[:, :-1]).clamp(min=0.0)
    rel_low_flux = low_flux / (low_flux.mean(dim=-1, keepdim=True).clamp(min=1e-4))

    # Star rating & onset sensitivity
    sr_tensor = star_rating.view(B, 1) if star_rating.ndim == 1 else star_rating[:, :1]  # [B, 1]
    onset_sens = (1.8 - (sr_tensor * 0.15)).clamp(min=0.5)                                # [B, 1]

    # Active mask based on audio transients
    not_silent = (rel_energy >= silence_threshold)                                        # [B, T]
    has_strong_onset = (rel_flux > onset_sens) | (rel_low_flux > (onset_sens * 1.2))

    if grid_metrics is not None:
        is_downbeat = (grid_metrics[:, :, 2] > 0.5)                                       # [B, T]
        div_f = grid_metrics[:, :, 0]                                                     # [B, T]
        is_beat = (div_f < 0.05) | (div_f > 0.95)
        has_rhythm_pulse = (is_downbeat & (rel_energy > 0.4)) | (is_beat & (rel_energy > 0.6) & (rel_flux > 0.8))
    else:
        is_downbeat = torch.zeros(B, T, dtype=torch.bool, device=device)
        has_rhythm_pulse = torch.zeros(B, T, dtype=torch.bool, device=device)

    active_mask = not_silent & (has_strong_onset | has_rhythm_pulse)                      # [B, T]

    # Vectorized chord / lane assignment
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    lane_idx = (t_idx * 3 + (rel_low_flux > 1.2).long()) % 4                              # [B, T]
    single_token = (1 << lane_idx)                                                         # [B, T]

    # Chords on downbeats / heavy kicks
    t_even = ((t_idx % 2) == 0)
    chord_token = torch.where(t_even, torch.full_like(single_token, 9), torch.full_like(single_token, 6)) # 9=1001, 6=0110
    triple_chord = torch.full_like(single_token, 11)                                      # 11=1011

    is_heavy = (is_downbeat | (rel_low_flux > 1.8)) & (rel_energy > 0.8) & (sr_tensor >= 2.0)
    is_triple = is_heavy & (sr_tensor >= 5.0) & (rel_low_flux > 2.5)

    base_tokens = torch.where(is_heavy, chord_token, single_token)
    base_tokens = torch.where(is_triple, triple_chord, base_tokens)

    scaffold = torch.where(active_mask, base_tokens, torch.zeros_like(base_tokens))
    return scaffold
