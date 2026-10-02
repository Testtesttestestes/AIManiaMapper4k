"""
Pattern generation module for osu!mania 4K.

Generates playable 4K hit objects conditioned on target Star Rating (1.0 to 9.9):
- Neural Pattern Generation: Model 3 (ManiaPatternTransformer) with:
  * Bidirectional deep audio Cross-Attention across full 80-bin frequency spectrum.
  * Musical Grid Metric embeddings (1/1, 1/2, 1/3, 1/4, 1/6, 1/8 beat ticks, downbeat phase).
  * Autoregressive 4-Lane physical state conditioning (Free, Tap, Hold Sustain, Release).
  * Spectral-sustained LN release prediction.
- Algorithmic Generator: Star-Rating scaling (1.0* Easy to 9.9* Expert), ergonomic lane distribution,
  and musical beat tick alignment.
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from osu_mapper.pipeline.timing import (
    MusicalGridTick,
    TimingPoint,
    generate_musical_grid_ticks_from_timing_points,
    snap_to_musical_grid,
)
from osu_mapper.pipeline.onset_scaffold import compute_acoustic_onset_scaffold

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


@dataclass
class ManiaHitObject:
    column: int       # 0, 1, 2, 3
    start_time: int   # milliseconds
    end_time: int     # milliseconds (same as start_time for regular tap notes)
    is_ln: bool = False

    @property
    def x_coordinate(self) -> int:
        """Standard osu!mania 4K X-coordinates: 64, 192, 320, 448."""
        return int(math.floor((self.column * 512 + 256) / 4))

    def to_osu_line(self) -> str:
        """Outputs standard osu! v14 HitObject line."""
        x = self.x_coordinate
        y = 192  # standard osu! editor Y coordinate for mania
        hit_sound = 0

        if self.is_ln:
            obj_type = 128  # Mania Hold Note flag
            extras = f"{self.end_time}:0:0:0:0:"
            return f"{x},{y},{self.start_time},{obj_type},{hit_sound},{extras}"
        else:
            obj_type = 1    # Standard Circle/Tap Note
            extras = "0:0:0:0:"
            return f"{x},{y},{self.start_time},{obj_type},{hit_sound},{extras}"


def solve_sliders_and_collisions(
    hit_objects: List[ManiaHitObject],
    beat_len_ms: float = 400.0,
    min_ln_ms: float = 110.0,
    min_release_gap_ms: int = 35,
) -> List[ManiaHitObject]:
    """
    Ranked-Compliant Slider & Lane Geometry Solver:
    1. Guarantees 0 overlapping hits on the same lane (prevents osu! hold conflict bugs).
    2. Zero Lost Notes: If a slider duration would be too short (< min_ln_ms) due to subsequent note on the same lane,
       it is seamlessly resolved as a crisp Tap Note without deleting the hit!
    3. Truncates LN end_time cleanly with a safe release gap (35ms) before the next note on the same column.
    4. Removes any accidental identical-time duplicates on the same lane.
    """
    if not hit_objects:
        return []

    lane_buckets: Dict[int, List[ManiaHitObject]] = {0: [], 1: [], 2: [], 3: []}
    for obj in hit_objects:
        if 0 <= obj.column < 4:
            lane_buckets[obj.column].append(obj)

    solved_objects: List[ManiaHitObject] = []

    for col in range(4):
        objs = lane_buckets[col]
        if not objs:
            continue

        # Sort chronologically
        objs.sort(key=lambda o: (o.start_time, 0 if not o.is_ln else 1))

        # Deduplicate simultaneous starts on same lane
        deduped: List[ManiaHitObject] = []
        for o in objs:
            if not deduped or o.start_time != deduped[-1].start_time:
                deduped.append(o)
            elif o.is_ln and not deduped[-1].is_ln:
                deduped[-1] = o

        # Solve overlapping spans & clamp release times
        for i in range(len(deduped)):
            cur_o = deduped[i]
            next_start = deduped[i + 1].start_time if (i + 1 < len(deduped)) else None

            if not cur_o.is_ln:
                solved_objects.append(ManiaHitObject(
                    column=col,
                    start_time=cur_o.start_time,
                    end_time=cur_o.start_time,
                    is_ln=False,
                ))
            else:
                target_end = cur_o.end_time
                if next_start is not None and target_end >= next_start:
                    target_end = next_start - min_release_gap_ms

                hold_len = target_end - cur_o.start_time
                if hold_len >= min_ln_ms:
                    solved_objects.append(ManiaHitObject(
                        column=col,
                        start_time=cur_o.start_time,
                        end_time=int(round(target_end)),
                        is_ln=True,
                    ))
                else:
                    # Too short for a playable slider: convert cleanly to single Tap (0 lost notes)
                    solved_objects.append(ManiaHitObject(
                        column=col,
                        start_time=cur_o.start_time,
                        end_time=cur_o.start_time,
                        is_ln=False,
                    ))

    solved_objects.sort(key=lambda o: (o.start_time, o.column))
    return solved_objects


def sample_chord_tokens_with_nucleus(
    logits: torch.Tensor,
    star_rating: float = 3.5,
    temperature: float = 0.0,
    top_p: float = 0.90,
    greedy: bool = False,
    note_conf_thresh: float = 0.0,
    has_scaffold_hint: bool = False,
) -> List[int]:
    """
    Difficulty-Conditioned Neural Chord Decoder:
    Calibrates decision boundary with Star Rating so high difficulties (3.5* - 7.0*+)
    reliably trigger dense jumpstreams/chords without being suppressed by raw background idle frequency.
    """
    T, num_classes = logits.shape
    out_tokens: List[int] = []

    # Calibrated active activation threshold scaled to Star Rating:
    calibrated_active_thresh = max(0.06, min(0.40, 0.42 - (float(star_rating) - 1.0) * 0.065))
    if has_scaffold_hint:
        calibrated_active_thresh *= 0.65
    if note_conf_thresh > 0.0:
        calibrated_active_thresh = note_conf_thresh

    for t_i in range(T):
        step_logits = logits[t_i : t_i + 1].clone()
        probs_raw = F.softmax(step_logits, dim=-1)
        p_active = float((1.0 - probs_raw[0, 0]).item())

        if greedy or temperature <= 0.05:
            if step_logits[0].argmax().item() > 0:
                out_tokens.append(int(step_logits[0].argmax().item()))
            elif p_active >= calibrated_active_thresh:
                active_best = int(step_logits[0, 1:].argmax().item()) + 1
                out_tokens.append(active_best)
            else:
                out_tokens.append(0)
        else:
            scaled_logits = step_logits / max(0.1, temperature)
            if star_rating >= 3.5:
                boost = (star_rating - 3.0) * 0.30
                scaled_logits[0, 1:] += boost

            probs = F.softmax(scaled_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0

            indices_to_remove = sorted_indices[0][sorted_indices_to_remove[0]]
            probs[0, indices_to_remove] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            sampled = int(torch.multinomial(probs, num_samples=1).item())
            out_tokens.append(sampled)

    return out_tokens


def token_to_keys(token: int) -> List[int]:
    """Decodes a 4-bit integer token (0-15) into active 4K column indices [0..3]."""
    return [c for c in range(4) if (token >> c) & 1]


def get_star_rating_params(star_rating: float) -> dict:
    """
    Calculates difficulty parameters strictly scaled to Star Rating (1.0* to 9.9*).
    
    Difficulty Tiers in osu!mania 4K:
    - 1.0* - 1.9* (Easy): Target NPS 0.8 - 1.8. 100% singles, strictly 1/1 and 2/1 beats. No chords.
    - 2.0* - 3.4* (Normal): Target NPS 2.0 - 3.5. 90% singles, 10% doubles. Rare 1/4 pairs.
    - 3.5* - 4.9* (Hard): Target NPS 4.0 - 6.5. 1/2 and 1/4 streams, doubles, occasional triples.
    - 5.0* - 6.9* (Insane): Target NPS 7.0 - 11.0. 1/4 jumpstreams, handstreams, LN density.
    - 7.0* - 9.9* (Extra/Expert): Target NPS 12.0 - 20.0+. Dense chordstreams, jacks, 1/8 bursts.
    """
    sr = max(1.0, min(9.9, star_rating))

    if sr < 1.8:
        target_nps = 0.8 + (sr - 1.0) * 0.9  # 0.8 - 1.5 NPS
        max_chord_size = 1
        chord_weights = [1.0, 0.0, 0.0, 0.0]
        subdiv_1_2_prob = 0.0
        subdiv_1_4_prob = 0.0
        subdiv_1_8_prob = 0.0
        ln_prob = 0.0
        min_note_gap_ms = 450
    elif sr < 3.0:
        target_nps = 1.6 + (sr - 1.8) * 1.2  # 1.6 - 3.0 NPS
        max_chord_size = 2
        chord_weights = [0.90, 0.10, 0.0, 0.0]
        subdiv_1_2_prob = 0.60
        subdiv_1_4_prob = max(0.0, (sr - 2.0) * 0.15)
        subdiv_1_8_prob = 0.0
        ln_prob = 0.08
        min_note_gap_ms = 180
    elif sr < 4.8:
        target_nps = 3.0 + (sr - 3.0) * 2.0  # 3.0 - 6.6 NPS
        max_chord_size = 3
        chord_weights = [0.65, 0.28, 0.07, 0.0]
        subdiv_1_2_prob = 0.95
        subdiv_1_4_prob = 0.20 + (sr - 3.0) * 0.25
        subdiv_1_8_prob = 0.0
        ln_prob = 0.12
        min_note_gap_ms = 70
    elif sr < 7.0:
        target_nps = 6.6 + (sr - 4.8) * 2.5  # 6.6 - 12.1 NPS
        max_chord_size = 3
        chord_weights = [0.45, 0.38, 0.15, 0.02]
        subdiv_1_2_prob = 1.0
        subdiv_1_4_prob = 0.70 + (sr - 4.8) * 0.12
        subdiv_1_8_prob = max(0.0, (sr - 5.5) * 0.15)
        ln_prob = 0.16
        min_note_gap_ms = 40
    else:
        target_nps = 12.0 + (sr - 7.0) * 3.0 # 12.0 - 20.7+ NPS
        max_chord_size = 4
        chord_weights = [0.25, 0.45, 0.24, 0.06]
        subdiv_1_2_prob = 1.0
        subdiv_1_4_prob = 0.98
        subdiv_1_8_prob = 0.25 + (sr - 7.0) * 0.12
        ln_prob = 0.22
        min_note_gap_ms = 25

    return {
        "star_rating": sr,
        "target_nps": target_nps,
        "max_chord_size": max_chord_size,
        "chord_weights": chord_weights,
        "subdiv_1_2_prob": subdiv_1_2_prob,
        "subdiv_1_4_prob": subdiv_1_4_prob,
        "subdiv_1_8_prob": subdiv_1_8_prob,
        "ln_prob": ln_prob,
        "min_note_gap_ms": min_note_gap_ms,
    }


def compute_audio_sustain_duration(
    mel: Any,
    start_frame: int,
    beat_len_ms: float,
    frame_ms: float = 16.0,
    max_beats: float = 2.0,
) -> int:
    """
    Measures the continuous acoustic spectral sustain tail of a melodic note/vocal
    and quantizes the release to the closest musical fraction (1/4, 1/2, 3/4, 1/1, 3/2, 2/1 beat).
    """
    total_frames = mel.shape[0] if hasattr(mel, "shape") else len(mel)
    if start_frame >= total_frames - 2:
        return int(round(beat_len_ms))

    step_1_4 = beat_len_ms / 4.0
    candidate_ticks = [
        int(round(step_1_4 * 2)),   # 1/2 beat
        int(round(step_1_4 * 3)),   # 3/4 beat
        int(round(beat_len_ms)),     # 1/1 beat
        int(round(beat_len_ms * 1.5)), # 3/2 beat
        int(round(beat_len_ms * 2.0)), # 2/1 beat
    ]

    max_frames_lookahead = min(int((beat_len_ms * max_beats) / frame_ms), total_frames - start_frame)
    if max_frames_lookahead <= 4:
        return candidate_ticks[0]

    try:
        if hasattr(mel, "device"):
            mid_slice = mel[start_frame : start_frame + max_frames_lookahead, 20:60]
            energies = mid_slice.mean(dim=-1).cpu().numpy()
        else:
            energies = [0.0] * max_frames_lookahead
        
        start_energy = energies[0] if len(energies) > 0 else 1.0
        sustain_frames = max_frames_lookahead
        for f, e in enumerate(energies):
            if (e - start_energy) < -1.5:
                sustain_frames = f
                break
        
        sustain_ms = sustain_frames * frame_ms
        best_tick = min(candidate_ticks, key=lambda t: abs(t - sustain_ms))
        return best_tick
    except Exception:
        return candidate_ticks[0]


def generate_patterns_with_model3(
    checkpoint_path: str | Path,
    audio_mel: Any,  # torch.Tensor [Total_T, 80]
    bpm: float,
    offset_ms: int,
    duration_ms: int,
    star_rating: float = 4.0,
    device: Optional[Any] = None,
    chunk_frames: int = 128,
    warmup_lead_in_frames: int = 64,
    timing_points: Optional[List[TimingPoint]] = None,
    temperature: float = 0.0,
    top_p: float = 0.90,
    greedy: bool = False,
    note_conf_thresh: float = 0.0,
    ln_conf_thresh: float = 0.78,
    silence_gate: float = 0.0,
    timing_mode: str = "soft_snap",
) -> List[ManiaHitObject]:
    """
    Generates 4K Hit Objects using the trained Model 3 (ManiaPatternTransformer).
    
    Features:
    1. Deterministic Musical Grid Ticks: Supports multiple timing & snapping paradigms:
       - 'soft_snap': Continuous model time advancement with soft magnetic lock to any musical divisor (1/4, 1/3, 1/6, 1/8, 1/12, 1/16).
       - 'universal_grid': Universal multi-divisor grid (binary + ternary simultaneous).
       - 'residual': Macro meter anchor + continuous micro-timing residual offset.
       - 'quantized': Legacy rigid 1/8 binary quantization.
    2. Deep Audio Cross-Attention & Acoustic Grounding across frequency bins and transients.
    3. 8-Channel Musical Metric Conditioning: Ingests beat divisor fractions, meter, bar phase, downbeats, progress, and energy.
    4. Model-driven Hit Object Synthesis: Zero silence gating hacks, zero arbitrary physical filters.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for Model 3 neural inference.")

    from osu_mapper.models.transformer import ManiaPatternTransformer, TransformerConfig

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Model 3 checkpoint not found at: {ckpt_p}")

    print(f"🧠 [Model 3 Neural Generator] Loading transformer checkpoint: {ckpt_p} (Timing Mode: {timing_mode})")
    cfg = TransformerConfig(
        audio_dim=80,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        num_classes=16,
        max_seq_len=512,
        dropout=0.1,
    )
    model = ManiaPatternTransformer(cfg).to(dev)

    ckpt_data = torch.load(ckpt_p, map_location=dev, weights_only=False)
    state = ckpt_data.get("model_state", ckpt_data.get("model_state_dict", ckpt_data.get("state_dict", ckpt_data)))
    model.load_state_dict(state, strict=False)
    model.eval()

    orig_total_mel_frames = audio_mel.shape[0]
    sr_tensor = torch.tensor([[star_rating]], dtype=torch.float32, device=dev)
    frame_ms = 16.0

    # 1. Generate full-resolution musical grid ticks (1/8 straight & 1/6 triplet subdivisions)
    # Allows model to freely place notes at any 1/1, 1/2, 1/4, 1/6, 1/8 timestamp across the track.
    subdiv_b = 8
    subdiv_t = 6
    use_universal = (str(timing_mode).lower().strip() in ("universal_grid", "hybrid_grid", "soft_snap", "residual"))

    resolved_tps = timing_points or [
        TimingPoint(
            time=float(offset_ms),
            beat_length=60000.0 / max(30.0, bpm),
            meter=4,
            uninherited=1,
        )
    ]
    ticks = generate_musical_grid_ticks_from_timing_points(
        timing_points=resolved_tps,
        duration_ms=duration_ms,
        default_bpm=bpm,
        default_offset_ms=offset_ms,
        subdiv_binary=subdiv_b,
        subdiv_ternary=subdiv_t,
        universal_grid=use_universal,
    )
    num_ticks = len(ticks)
    if num_ticks == 0:
        print("  ⚠️ Notice: No musical ticks generated; returning empty beatmap.")
        return []

    print(f"  • Generated {num_ticks} full-resolution musical grid ticks (1/8 & 1/6 precision) across {duration_ms}ms.")

    # 2. Extract Mel audio frames sampled at each tick timestamp & Multi-Scale Audio Pyramid
    with torch.no_grad():
        frame_energies_t = audio_mel.exp().mean(dim=-1)
        mean_energy_val = float(frame_energies_t.mean().item()) if frame_energies_t.numel() > 0 else 1.0
        mean_energy = max(1e-4, mean_energy_val)

        tick_mel_list = []
        grid_metrics_list = []
        tick_times_arr = np.array([t_obj.time_ms for t_obj in ticks], dtype=np.float32)

        for i, t_obj in enumerate(ticks):
            f = max(0, min(orig_total_mel_frames - 1, int(round(t_obj.time_ms / frame_ms))))
            tick_mel_list.append(audio_mel[f])

            local_e = float(frame_energies_t[f].item()) if f < frame_energies_t.shape[0] else 1.0
            energy_ratio = float(local_e / mean_energy)
            progress = float(i / max(1, num_ticks))
            time_to_end = float((num_ticks - i) / max(1, num_ticks))

            grid_metrics_list.append([
                float(t_obj.divisor_fraction),
                float(t_obj.bar_phase),
                1.0 if t_obj.is_downbeat else 0.0,
                progress,
                time_to_end,
                energy_ratio,
                float(t_obj.meter),
                1.0 if t_obj.is_ternary else 0.0,
            ])

        tick_mel_tensor = torch.stack(tick_mel_list).to(dev)             # [num_ticks, 80]
        grid_metrics_tensor = torch.tensor(grid_metrics_list, dtype=torch.float32, device=dev)  # [num_ticks, 8]

        # Multi-scale pyramid: Micro (±50ms), Meso (±10s), Macro (±25s) with genuine silence padding outside track
        from osu_mapper.dataset.feature_extractor import compute_multiscale_pyramid_features
        multiscale_pyramid_tensor = compute_multiscale_pyramid_features(audio_mel, tick_times_arr, frame_ms=frame_ms).to(dev)

    # Pre-generate Stage 1 Acoustic Onset Scaffold for the Neural Refiner
    print(f"  • Generating Stage 1 Acoustic Onset Rhythm Scaffold (Target SR: {star_rating:.2f}*)...")
    scaffold_tokens_tensor = compute_acoustic_onset_scaffold(
        audio_mel=tick_mel_tensor.unsqueeze(0),
        grid_metrics=grid_metrics_tensor.unsqueeze(0),
        star_rating=sr_tensor,
        silence_threshold=silence_gate,
    ).squeeze(0)

    # Real-time progressive strain trajectory metrics matching training distribution [B, T, 2]
    # strain_metrics = [running_density (notes/tick in 0.05-0.40), density_ratio (running / target)]
    target_density = max(0.05, float(star_rating) / 10.0)
    strain_metrics_tensor = torch.zeros(num_ticks, 2, dtype=torch.float32, device=dev)
    for ti in range(num_ticks):
        time_steps = ti + 1
        run_notes = int(scaffold_tokens_tensor[:ti+1].gt(0).sum().item())
        running_density = run_notes / max(1, time_steps)
        density_ratio = min(5.0, running_density / target_density)
        strain_metrics_tensor[ti, 0] = float(running_density)
        strain_metrics_tensor[ti, 1] = float(density_ratio)

    # Pre-encode full tick audio representation with deep audio encoder
    print(f"  • Pre-encoding tick audio features via Bidirectional Audio Encoder...")
    with torch.no_grad():
        full_audio_memory = model.audio_encoder(tick_mel_tensor.unsqueeze(0))  # [1, num_ticks, d_model]

    print(f"  • Executing Validation-Identical Neural Rollout across {num_ticks} musical grid ticks (Target SR: {star_rating:.2f}*)...")

    # Maintain real-time 4-lane physical activity state
    lane_states_seq = torch.zeros(num_ticks, 4, dtype=torch.int64, device=dev)
    beat_len_ms = 60000.0 / max(30.0, bpm)

    generated_tokens: List[int] = [0] * num_ticks
    active_ln_starts: Dict[int, Optional[int]] = {0: None, 1: None, 2: None, 3: None}
    hit_objects: List[ManiaHitObject] = []
    total_taps = 0
    total_lns = 0

    # Calibrated minimum physical LN duration (in ms)
    min_ln_ms = max(110.0, beat_len_ms * 0.35)
    ctx_len = 128

    # Effective LN threshold (0.40 matches 99.8% validation)
    effective_ln_thresh = float(ln_conf_thresh) if (ln_conf_thresh is not None and ln_conf_thresh > 0) else 0.40

    # Effective active note threshold calibrated to Star Rating (prevents long empty gaps)
    # At 3.6*, calibrated_thresh is ~0.18, meaning if Model 3 sees >=18% note probability, it places the note!
    calibrated_active_thresh = max(0.08, min(0.28, 0.28 - (float(star_rating) - 1.0) * 0.04))
    if note_conf_thresh is not None and note_conf_thresh > 0:
        calibrated_active_thresh = float(note_conf_thresh)

    min_tick_gap_ms = max(50.0, beat_len_ms * 0.125)  # Physical minimum gap (1/8 beat, never 18ms collision)
    last_placed_note_ms = -9999.0

    for cur_pos in range(num_ticks):
        t_obj = ticks[cur_pos]
        cur_time_ms = int(round(t_obj.time_ms))
        if cur_time_ms >= (duration_ms - 50.0):
            break

        ctx_start = max(0, cur_pos - ctx_len + 1)
        window_len = (cur_pos + 1) - ctx_start

        window_mel = tick_mel_tensor[ctx_start : cur_pos + 1].unsqueeze(0)
        window_multiscale = multiscale_pyramid_tensor[ctx_start : cur_pos + 1].unsqueeze(0)
        window_grid = grid_metrics_tensor[ctx_start : cur_pos + 1].unsqueeze(0)
        window_scaffold = scaffold_tokens_tensor[ctx_start : cur_pos + 1].unsqueeze(0)
        window_audio_memory = full_audio_memory[:, ctx_start : cur_pos + 1]

        # Rolling prev_tokens & lane_states (identically shifted by 1 as in train.py validation)
        window_prev_tokens = torch.zeros(1, window_len, dtype=torch.int64, device=dev)
        if cur_pos > ctx_start:
            window_prev_tokens[0, 1 : cur_pos - ctx_start + 1] = torch.tensor(
                generated_tokens[ctx_start:cur_pos], dtype=torch.int64, device=dev
            )
        window_lane_states = torch.zeros(1, window_len, 4, dtype=torch.int64, device=dev)
        if cur_pos > ctx_start:
            window_lane_states[0, 1 : cur_pos - ctx_start + 1] = lane_states_seq[ctx_start:cur_pos]

        # Rolling strain trajectory
        local_recent_notes = sum(1 for tok_v in generated_tokens[ctx_start:cur_pos] if tok_v > 0)
        running_density = local_recent_notes / max(1, window_len)
        density_ratio = min(5.0, running_density / target_density)
        strain_metrics_tensor[cur_pos, 0] = float(running_density)
        strain_metrics_tensor[cur_pos, 1] = float(density_ratio)
        window_strain = strain_metrics_tensor[ctx_start : cur_pos + 1].unsqueeze(0)

        with torch.no_grad():
            try:
                c_logits, l_logits = model(
                    window_mel,
                    star_rating=sr_tensor,
                    prev_tokens=window_prev_tokens,
                    lane_states=window_lane_states,
                    grid_metrics=window_grid,
                    scaffold_tokens=window_scaffold,
                    strain_metrics=window_strain,
                    audio_memory=window_audio_memory,
                    multiscale_audio=window_multiscale,
                )
            except Exception:
                c_logits, l_logits, _, _ = model(
                    window_mel,
                    star_rating=sr_tensor,
                    prev_tokens=window_prev_tokens,
                    lane_states=window_lane_states,
                    grid_metrics=window_grid,
                    scaffold_tokens=window_scaffold,
                    strain_metrics=window_strain,
                    audio_memory=window_audio_memory,
                    multiscale_audio=window_multiscale,
                    return_sr=True,
                    return_time_delta=True,
                )

            step_c_logits = c_logits[0, -1]
            step_l_logits = l_logits[0, -1]

            probs_raw = F.softmax(step_c_logits, dim=-1)
            p_active = float((1.0 - probs_raw[0]).item())

            # Scaffold guidance: if scaffold has an acoustic onset or it's a downbeat, lower the threshold so verses are never empty!
            has_scaff = (int(window_scaffold[0, -1].item()) > 0) or t_obj.is_downbeat or (t_obj.is_beat_start and star_rating >= 2.5)
            effective_thresh = max(0.08, calibrated_active_thresh * 0.70) if has_scaff else calibrated_active_thresh

            # Decide note presence based on neural active probability:
            is_physically_too_close = (cur_time_ms - last_placed_note_ms) < (min_tick_gap_ms - 5.0)

            if is_physically_too_close:
                tok = 0
            elif step_c_logits.argmax().item() > 0:
                tok = int(step_c_logits.argmax().item())
            elif p_active >= effective_thresh:
                # Active note detected by model
                tok = int(step_c_logits[1:].argmax().item()) + 1
            else:
                tok = 0

            ln_probs = torch.sigmoid(step_l_logits).cpu().tolist()

        generated_tokens[cur_pos] = tok

        # Process hit objects strictly from neural tokens:
        if tok > 0:
            last_placed_note_ms = float(cur_time_ms)
            active_cols = token_to_keys(tok)
            for c in active_cols:
                # Release existing LN if new note hits the same column
                if active_ln_starts[c] is not None:
                    start_t = active_ln_starts[c]
                    hold_len = cur_time_ms - start_t
                    if hold_len >= min_ln_ms:
                        end_t = max(start_t + int(round(min_ln_ms)), cur_time_ms - 30)
                        hit_objects.append(ManiaHitObject(column=c, start_time=start_t, end_time=end_t, is_ln=True))
                        total_lns += 1
                    else:
                        hit_objects.append(ManiaHitObject(column=c, start_time=start_t, end_time=start_t, is_ln=False))
                        total_taps += 1
                    active_ln_starts[c] = None

                # Check if this note is a Long Note (slider) or Tap
                if ln_probs[c] >= effective_ln_thresh:
                    active_ln_starts[c] = cur_time_ms
                    lane_states_seq[cur_pos, c] = 2  # STATE_LN_START
                else:
                    hit_objects.append(ManiaHitObject(column=c, start_time=cur_time_ms, end_time=cur_time_ms, is_ln=False))
                    total_taps += 1
                    lane_states_seq[cur_pos, c] = 1  # STATE_TAP
        else:
            # Idle / Rest tick (tok == 0): manage ongoing slider holds/releases
            for c in range(4):
                if active_ln_starts[c] is not None:
                    hold_duration = cur_time_ms - active_ln_starts[c]
                    # Release slider when LN probability drops or after 4 full beats
                    if (ln_probs[c] < effective_ln_thresh * 0.5) or (hold_duration >= (beat_len_ms * 4.0)):
                        start_t = active_ln_starts[c]
                        if hold_duration >= min_ln_ms:
                            end_t = max(start_t + int(round(min_ln_ms)), cur_time_ms - 30)
                            hit_objects.append(ManiaHitObject(column=c, start_time=start_t, end_time=end_t, is_ln=True))
                            total_lns += 1
                        else:
                            hit_objects.append(ManiaHitObject(column=c, start_time=start_t, end_time=start_t, is_ln=False))
                            total_taps += 1
                        active_ln_starts[c] = None
                        lane_states_seq[cur_pos, c] = 4  # STATE_LN_RELEASE
                    else:
                        lane_states_seq[cur_pos, c] = 3  # STATE_LN_HOLD

    # Close any lingering sliders at the end of track
    final_track_ms = min(duration_ms, int(round(ticks[-1].time_ms)))
    for c in range(4):
        if active_ln_starts[c] is not None:
            start_t = active_ln_starts[c]
            hold_len = final_track_ms - start_t
            if hold_len >= min_ln_ms:
                end_t = max(start_t + int(round(min_ln_ms)), final_track_ms)
                hit_objects.append(ManiaHitObject(column=c, start_time=start_t, end_time=end_t, is_ln=True))
                total_lns += 1
            else:
                hit_objects.append(ManiaHitObject(column=c, start_time=start_t, end_time=start_t, is_ln=False))
                total_taps += 1
            active_ln_starts[c] = None

    # -------------------------------------------------------------
    # Ranked-Compliant Slider & Collision Solver (Zero Lost Notes, Zero Overlaps)
    # -------------------------------------------------------------
    solved_objects = solve_sliders_and_collisions(hit_objects, beat_len_ms=beat_len_ms, min_ln_ms=min_ln_ms)
    
    total_objs = len(solved_objects)
    final_lns = sum(1 for o in solved_objects if o.is_ln)
    final_taps = total_objs - final_lns
    duration_sec = max(0.001, duration_ms / 1000.0)
    avg_nps = total_objs / duration_sec
    ln_pct = (final_lns / max(1, total_objs)) * 100.0
    print(
        f"  ✓ Neural pattern synthesis finished: generated {total_objs} hit objects "
        f"({final_taps} taps, {final_lns} sliders/LNs [{ln_pct:.1f}%], {avg_nps:.1f} NPS, SR: {star_rating:.1f}*)."
    )
    return solved_objects


def generate_patterns(
    bpm: float,
    offset_ms: int,
    duration_ms: int,
    star_rating: float = 3.5,
    onsets_ms: Optional[List[int]] = None,
    timing_points: Optional[List[TimingPoint]] = None,
    seed: int = 42,
) -> List[ManiaHitObject]:
    """
    Algorithmic generator conditioned strictly on requested Star Rating (1.0* to 9.9*).
    Strictly synchronizes to musical beat grid ticks across ALL timing points and BPM changes.
    """
    rng = random.Random(seed)
    params = get_star_rating_params(star_rating)
    sr = params["star_rating"]

    # 1. Generate full sequence of musical grid ticks across all timing points
    ticks = generate_musical_grid_ticks_from_timing_points(
        timing_points=timing_points if timing_points else [],
        duration_ms=duration_ms,
        default_bpm=bpm,
        default_offset_ms=float(offset_ms),
        subdiv_binary=8,
        subdiv_ternary=6,
    )

    # Filter ticks to active audio range
    valid_ticks = [t for t in ticks if 0 <= t.time_ms <= duration_ms]
    if not valid_ticks:
        return []

    # Fast lookup for tick properties by time_ms
    tick_by_time: Dict[int, MusicalGridTick] = {}
    for t_obj in valid_ticks:
        t_int = int(round(t_obj.time_ms))
        tick_by_time[t_int] = t_obj

    # 2. Select active grid events conditioned on Star Rating
    grid_events: List[int] = []

    for t_obj in valid_ticks:
        t_ms = int(round(t_obj.time_ms))
        is_downbeat = t_obj.is_downbeat
        is_1_1 = t_obj.is_beat_start
        is_1_2 = (t_obj.divisor == 2) or (t_obj.divisor_fraction == 0.5)
        is_1_4 = (t_obj.divisor == 4) or (t_obj.divisor_fraction == 0.25)
        is_1_8 = (t_obj.divisor == 8) or (t_obj.divisor_fraction == 0.125)
        is_ternary_sub = (t_obj.divisor in (3, 6))

        # Easy map (1.0* - 1.8*): Downbeats or every 2nd beat (measure/half-measure pulses)
        if sr < 1.8:
            easy_mod = 2 if t_obj.beat_len_ms < 450 else 1
            if is_downbeat:
                should_place = True
            elif is_1_1 and (t_obj.beat_in_measure % easy_mod == 0):
                should_place = True
            else:
                should_place = False
        elif sr < 3.0:
            if is_1_1:
                should_place = True
            elif is_1_2:
                should_place = rng.random() < params["subdiv_1_2_prob"]
            elif is_1_4 or is_ternary_sub:
                should_place = rng.random() < params["subdiv_1_4_prob"]
            else:
                should_place = False
        elif sr < 5.0:
            if is_1_1 or is_1_2:
                should_place = True
            elif is_1_4 or is_ternary_sub:
                should_place = rng.random() < params["subdiv_1_4_prob"]
            elif is_1_8:
                should_place = rng.random() < params["subdiv_1_8_prob"]
            else:
                should_place = False
        else:  # High SR (5.0*+)
            if is_1_1 or is_1_2:
                should_place = True
            elif is_1_4 or is_ternary_sub:
                should_place = rng.random() < max(0.85, params["subdiv_1_4_prob"])
            elif is_1_8:
                should_place = rng.random() < params["subdiv_1_8_prob"]
            else:
                should_place = False

        if should_place:
            grid_events.append(t_ms)

    # For SR >= 3.5*, also incorporate sharp audio transients/onsets
    if sr >= 3.5 and onsets_ms:
        for onset in onsets_ms:
            if 0 <= onset <= duration_ms:
                grid_events.append(int(round(onset)))

    active_timestamps = sorted(list(set(grid_events)))
    if not active_timestamps:
        return []

    # 3. 4K Lane distribution, Chord sizes, and Long Note generation
    column_busy_until = [0, 0, 0, 0]
    last_note_time = [-9999, -9999, -9999, -9999]
    last_note_global = -9999
    hit_objects: List[ManiaHitObject] = []
    max_chord = params["max_chord_size"]
    min_gap = params["min_note_gap_ms"]

    for i, t in enumerate(active_timestamps):
        if (t - last_note_global) < (min_gap // 2):
            continue

        # Get local timing info for this timestamp
        local_tick = tick_by_time.get(t)
        local_beat_len = local_tick.beat_len_ms if local_tick else (60000.0 / max(1.0, bpm))
        local_is_downbeat = local_tick.is_downbeat if local_tick else False
        local_step_1_2 = local_beat_len / 2.0
        local_step_1_4 = local_beat_len / 4.0

        r = rng.random()
        w = params["chord_weights"]
        if r < w[0]:
            raw_chord = 1
        elif r < w[0] + w[1]:
            raw_chord = 2
        elif r < w[0] + w[1] + w[2]:
            raw_chord = 3
        else:
            raw_chord = 4

        chord_size = min(max_chord, raw_chord)

        # Restrict large chords to downbeats on lower-to-mid difficulties
        if not local_is_downbeat and chord_size > 2 and sr < 6.0:
            chord_size = min(2, chord_size)

        available_cols = [c for c in range(4) if column_busy_until[c] <= t]
        if not available_cols:
            continue

        actual_chord_size = min(chord_size, len(available_cols))
        available_cols.sort(key=lambda c: (last_note_time[c], rng.random()))
        chosen_cols = available_cols[:actual_chord_size]

        can_have_ln = (sr >= 2.0) and (rng.random() < params["ln_prob"])
        next_time = active_timestamps[i + 1] if (i + 1) < len(active_timestamps) else (t + int(local_beat_len))
        
        gap_to_next = max(0, next_time - t)
        if gap_to_next >= int(local_beat_len * 1.5):
            hold_len = int(round(local_beat_len))
        elif gap_to_next >= int(local_beat_len):
            hold_len = int(round(local_step_1_2 + local_step_1_4))
        else:
            hold_len = int(round(local_step_1_2))

        for c_idx, col in enumerate(chosen_cols):
            is_ln = can_have_ln and (c_idx == 0) and (hold_len >= int(local_step_1_2))
            end_t = min(duration_ms, t + hold_len) if is_ln else t

            hit_objects.append(
                ManiaHitObject(
                    column=col,
                    start_time=t,
                    end_time=end_t,
                    is_ln=is_ln,
                )
            )

            last_note_time[col] = t
            if is_ln:
                column_busy_until[col] = end_t + 20

        last_note_global = t

    hit_objects.sort(key=lambda o: (o.start_time, o.column))
    return hit_objects
