"""
Acoustic Onset-Driven Algorithmic Scaffold Generator.
Computes deterministic rhythm scaffolds strictly from audio spectral flux,
energy transients, and timing grid metrics without any ground-truth leakage.
"""

from typing import Any, Optional, Tuple

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
    scaffold = torch.zeros(B, T, dtype=torch.int64, device=device)

    # 1. Compute spectral energy and spectral flux (transient attacks)
    # Exponentiate log-mel to linear power domain
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

    # Multi-band energy (Low: kick/bass 0..20, Mid: snare/vocal 20..60, High: cymbals/hihats 60..80)
    low_band = linear_mel[:, :, :20].mean(dim=-1)                         # [B, T]
    low_flux = torch.zeros_like(low_band)
    low_flux[:, 1:] = (low_band[:, 1:] - low_band[:, :-1]).clamp(min=0.0)
    rel_low_flux = low_flux / (low_flux.mean(dim=-1, keepdim=True).clamp(min=1e-4))

    for b in range(B):
        sr = float(star_rating[b].item()) if star_rating.ndim > 1 else float(star_rating[b])

        # Difficulty parameters
        max_chord = 1 if sr < 2.0 else (2 if sr < 4.0 else (3 if sr < 6.0 else 4))
        onset_sens = max(0.5, 1.8 - (sr * 0.15))   # Lower threshold for higher SR

        gm = grid_metrics[b] if grid_metrics is not None else None
        
        for t in range(T):
            e_val = float(rel_energy[b, t].item())
            f_val = float(rel_flux[b, t].item())
            lf_val = float(rel_low_flux[b, t].item())

            # Silence Gating: zero notes during quiet / break / intro sections
            if e_val < silence_threshold:
                continue

            is_downbeat = False
            is_beat = False
            is_half = False

            if gm is not None:
                is_downbeat = bool(gm[t, 2].item() > 0.5)
                div_f = float(gm[t, 0].item())
                is_beat = (div_f < 0.05) or (div_f > 0.95)
                is_half = abs(div_f - 0.5) < 0.05

            # Onset detection condition: strong audio transient or downbeat with audio presence
            has_strong_onset = (f_val > onset_sens) or (lf_val > onset_sens * 1.2)
            has_rhythm_pulse = (is_downbeat and e_val > 0.4) or (is_beat and e_val > 0.6 and f_val > 0.8)

            if not (has_strong_onset or has_rhythm_pulse):
                continue

            # Determine lane & chord pattern based on musical intensity
            lane_idx = (t * 3 + (1 if lf_val > 1.2 else 0)) % 4
            
            # Chord size selection
            if (is_downbeat or lf_val > 1.8) and max_chord >= 2 and e_val > 0.8:
                # 2-note chord (e.g. columns 0 and 3 or 1 and 2)
                if t % 2 == 0:
                    chord_tok = 9   # 1001 (cols 0, 3)
                else:
                    chord_tok = 6   # 0110 (cols 1, 2)
                if max_chord >= 3 and sr >= 5.0 and lf_val > 2.5:
                    chord_tok = 11  # 1011 (cols 0, 1, 3)
            else:
                # Single note
                chord_tok = 1 << lane_idx

            scaffold[b, t] = chord_tok

    return scaffold
