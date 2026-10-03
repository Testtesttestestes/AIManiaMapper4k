"""
Diffusion-based Pattern Generator for osu!mania 4K (ManiaDiffusion).

Generates beatmaps non-autoregressively using 1D Latent/Piano-roll Diffusion (like Stable Diffusion 1.5):
- Synthesizes entire measures and phrases simultaneously in parallel.
- Eliminates exposure bias, autoregressive drift, empty 5-second gaps, and single-lane spam.
- User-selectable denoising steps (e.g. 20, 30, 50) and Classifier-Free Guidance (CFG scale, e.g. 3.0).
- Immediate rosu-pp Star Rating and difficulty attribute validation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

from osu_mapper.dataset.feature_extractor import compute_multiscale_pyramid_features
from osu_mapper.models.mania_diffusion import (
    ManiaDiffusion,
    beatmap_tensor_to_hit_objects,
)
from osu_mapper.pipeline.pattern_generator import (
    ManiaHitObject,
    solve_sliders_and_collisions,
)
from osu_mapper.pipeline.timing import (
    MusicalGridTick,
    TimingPoint,
    generate_musical_grid_ticks_from_timing_points,
)


def generate_patterns_with_diffusion(
    checkpoint_path: Union[str, Path],
    audio_mel: torch.Tensor,
    bpm: float,
    offset_ms: int,
    duration_ms: int,
    star_rating: float = 3.5,
    timing_points: Optional[List[TimingPoint]] = None,
    num_inference_steps: int = 30,
    cfg_scale: float = 3.0,
    chunk_size: int = 256,
    overlap: int = 32,
    device: Optional[str] = None,
) -> List[ManiaHitObject]:
    """
    Executes fast DDIM diffusion sampling to generate a complete osu!mania 4K beatmap.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for ManiaDiffusion inference.")

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n🌊 [ManiaDiffusion] Initializing 1D Latent Beatmap Diffusion (Device: {dev})...")
    print(f"  • Target Star Rating: {star_rating:.2f}*")
    print(f"  • DDIM Denoising Steps: {num_inference_steps} (matching Stable Diffusion 1.5)")
    print(f"  • Classifier-Free Guidance (CFG Scale): {cfg_scale:.1f}")

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        # Fallback to model_3 checkpoint or default path if available
        alt_paths = [
            Path("checkpoints/mania_diffusion_best.pt"),
            Path("checkpoints/mania_diffusion_latest.pt"),
            Path("checkpoints/model_3_transformer_best.pt"),
        ]
        found = False
        for alt in alt_paths:
            if alt.exists():
                ckpt_file = alt
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Neither {checkpoint_path} nor any fallback diffusion checkpoint was found.")

    # Load weights
    print(f"  • Loading checkpoint: {ckpt_file}...")
    ckpt_data = torch.load(ckpt_file, map_location=dev, weights_only=False)

    state_dict = None
    for k in ("model_state", "model_state_dict", "state_dict"):
        if isinstance(ckpt_data, dict) and k in ckpt_data and isinstance(ckpt_data[k], dict):
            state_dict = ckpt_data[k]
            break
    if state_dict is None and isinstance(ckpt_data, dict) and all(isinstance(v, torch.Tensor) for v in ckpt_data.values()):
        state_dict = ckpt_data

    # Auto-detect base channels from checkpoint
    base_channels = 64
    if state_dict is not None:
        for k, v in state_dict.items():
            if k.endswith("init_conv.weight") and hasattr(v, "shape") and len(v.shape) >= 1:
                base_channels = int(v.shape[0])
                break

    print(f"  • UNet Base Channels: {base_channels} (auto-configured from checkpoint)")
    model = ManiaDiffusion(unet_base_channels=base_channels).to(dev)
    model.eval()

    if state_dict is not None:
        # Check if loading dedicated ManiaDiffusion weights
        if any(k.startswith("unet.") for k in state_dict.keys()):
            model.load_state_dict(state_dict, strict=False)
            print("  ✓ Successfully loaded native ManiaDiffusion UNet weights.")
        else:
            # Transfer audio encoder from Model 3 checkpoint
            model.load_from_model3_checkpoint(ckpt_file)
            print("  ✓ Initialized ManiaDiffusion with transferred Model 3 acoustic representations.")

    # 1. Generate musical grid ticks aligned to the song's BPM and timing points
    ticks = generate_musical_grid_ticks_from_timing_points(
        timing_points=timing_points if timing_points else [],
        duration_ms=duration_ms,
        default_bpm=bpm,
        default_offset_ms=float(offset_ms),
        subdiv_binary=8,
        subdiv_ternary=6,
    )
    ticks = [t for t in ticks if 0 <= t.time_ms <= duration_ms]
    num_ticks = len(ticks)
    if num_ticks == 0:
        return []

    beat_len_ms = 60000.0 / max(30.0, bpm)
    frame_ms = 16.0

    # 2. Extract multi-scale spectrogram and grid metrics at each tick
    tick_times_arr = [float(t.time_ms) for t in ticks]
    tick_mel_list = []
    grid_metrics_list = []
    num_frames = audio_mel.shape[0]

    for i, t_obj in enumerate(ticks):
        f_idx = max(0, min(num_frames - 1, int(round(t_obj.time_ms / frame_ms))))
        tick_mel_list.append(audio_mel[f_idx])

        progress = float(i / max(1, num_ticks))
        time_to_end = float((num_ticks - i) / max(1, num_ticks))
        grid_metrics_list.append([
            float(t_obj.divisor_fraction),
            float(t_obj.bar_phase),
            1.0 if t_obj.is_downbeat else 0.0,
            progress,
            time_to_end,
            1.0,
            float(t_obj.meter),
            1.0 if t_obj.is_ternary else 0.0,
        ])

    all_tick_mel = torch.stack(tick_mel_list).to(dev)                             # [num_ticks, 80]
    all_grid_metrics = torch.tensor(grid_metrics_list, dtype=torch.float32, device=dev) # [num_ticks, 8]

    # 3. Windowed Diffusion Sampling across the entire song
    # If the song fits in a single chunk (<=512 ticks), run in one shot;
    # otherwise, slice into overlapping chunks and blend with triangular window.
    step_hop = chunk_size - overlap
    full_denoised_tensor = torch.full((12, num_ticks), -1.0, device=dev, dtype=torch.float32)
    weight_accum = torch.zeros(num_ticks, device=dev, dtype=torch.float32)

    total_chunks = max(1, math.ceil(max(0, num_ticks - chunk_size) / step_hop) + 1)
    print(f"  • Generating {total_chunks} beatmap phrase chunks with cross-window blending...")

    for chunk_idx in range(total_chunks):
        c_start = chunk_idx * step_hop
        c_end = min(num_ticks, c_start + chunk_size)
        actual_len = c_end - c_start

        if actual_len <= 0:
            break

        chunk_mel = all_tick_mel[c_start:c_end].unsqueeze(0)        # [1, actual_len, 80]
        chunk_grid = all_grid_metrics[c_start:c_end].unsqueeze(0)  # [1, actual_len, 8]

        # Pad to chunk_size if last chunk is shorter
        if actual_len < chunk_size:
            pad_len = chunk_size - actual_len
            chunk_mel = F.pad(chunk_mel, (0, 0, 0, pad_len), value=0.0)
            chunk_grid = F.pad(chunk_grid, (0, 0, 0, pad_len), value=0.0)

        # Run DDIM Denoising
        def ddim_progress(step: int, total_steps: int):
            if step % 5 == 0 or step == total_steps:
                pct = int((step / total_steps) * 100)
                bar = "█" * (pct // 10) + "░" * (10 - (pct // 10))
                print(f"\r    [Chunk {chunk_idx + 1}/{total_chunks}] DDIM Step {step:2d}/{total_steps} |{bar}| {pct}%", end="", flush=True)

        denoised = model.sample(
            audio_mel=chunk_mel,
            grid_metrics=chunk_grid,
            star_rating=star_rating,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            progress_callback=ddim_progress,
        )[0, :, :actual_len]  # [12, actual_len]
        print("")

        # Triangular blend window over overlap region
        blend_w = torch.ones(actual_len, device=dev, dtype=torch.float32)
        if overlap > 0 and chunk_idx > 0 and actual_len > overlap:
            blend_w[:overlap] = torch.linspace(0.0, 1.0, overlap, device=dev)
        if overlap > 0 and chunk_idx < total_chunks - 1 and actual_len > overlap:
            blend_w[-overlap:] = torch.linspace(1.0, 0.0, overlap, device=dev)

        full_denoised_tensor[:, c_start:c_end] = torch.where(
            weight_accum[c_start:c_end] == 0,
            denoised,
            (full_denoised_tensor[:, c_start:c_end] * weight_accum[c_start:c_end] + denoised * blend_w) / (weight_accum[c_start:c_end] + blend_w),
        )
        weight_accum[c_start:c_end] += blend_w

    # 4. Decode 12-channel continuous diffusion tensor into discrete ManiaHitObjects
    # Fast bisection calibration on decoded threshold to match target_sr accurately
    from osu_mapper.dataset.difficulty_calculator import calculate_osu_mania_star_rating

    # Human-ergonomic minimum gap per lane scaled with target SR
    min_lane_gap = max(40.0, 110.0 - max(0.0, star_rating - 2.0) * 14.0)

    # Search for the optimal activation threshold in [0.05, 0.70]
    low_t = 0.05
    high_t = 0.70
    best_solved: List[ManiaHitObject] = []
    best_delta = 999.0
    best_thresh = 0.25

    for _ in range(5):
        mid_t = (low_t + high_t) / 2.0
        candidate_raw = beatmap_tensor_to_hit_objects(
            denoised_tensor=full_denoised_tensor,
            ticks=ticks,
            duration_ms=float(duration_ms),
            beat_len_ms=beat_len_ms,
            tap_threshold=mid_t,
            ln_threshold=max(0.12, mid_t * 0.8),
            min_lane_gap_ms=min_lane_gap,
        )
        candidate_solved = solve_sliders_and_collisions(
            candidate_raw,
            beat_len_ms=beat_len_ms,
            min_ln_ms=max(110.0, beat_len_ms * 0.35),
        )

        curr_sr = calculate_osu_mania_star_rating(
            hit_objects=[(o.column, o.start_time, o.end_time, o.is_ln) for o in candidate_solved],
            keys=4,
        )
        delta = abs(curr_sr - star_rating)
        if delta < best_delta:
            best_delta = delta
            best_solved = candidate_solved
            best_thresh = mid_t

        if curr_sr > star_rating:
            # Over-dense -> raise threshold to keep only more prominent neural notes
            low_t = mid_t
        else:
            # Under-dense -> lower threshold
            high_t = mid_t

    solved_objects = best_solved if best_solved else candidate_solved

    total_objs = len(solved_objects)
    final_lns = sum(1 for o in solved_objects if o.is_ln)
    final_taps = total_objs - final_lns
    ln_pct = (final_lns / max(1, total_objs)) * 100.0
    duration_sec = max(0.001, duration_ms / 1000.0)
    avg_nps = total_objs / duration_sec

    print(f"\n  ✓ [ManiaDiffusion] Generation complete: {total_objs} hit objects ({final_taps} taps, {final_lns} sliders/LNs [{ln_pct:.1f}%], {avg_nps:.1f} NPS | calibrated thresh {best_thresh:.2f}).")
    return solved_objects
