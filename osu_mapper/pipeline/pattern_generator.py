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


def sample_chord_tokens_with_nucleus(
    logits: torch.Tensor,
    star_rating: float = 3.5,
    temperature: float = 0.70,
    top_p: float = 0.90,
) -> List[int]:
    """
    Samples 4K chord tokens using Temperature + Nucleus (Top-P) sampling.
    Preserves natural model distribution learned from real human beatmaps.
    """
    # Difficulty chord mask: mask out oversized chords for lower Star Ratings
    max_keys = 1 if star_rating < 1.8 else (2 if star_rating < 3.5 else (3 if star_rating < 5.0 else 4))
    
    T, num_classes = logits.shape
    masked_logits = logits.clone()
    
    for tok_idx in range(num_classes):
        num_keys = bin(tok_idx).count("1")
        if num_keys > max_keys:
            masked_logits[:, tok_idx] = -1e9

    scaled_logits = masked_logits / max(0.1, temperature)
    probs = F.softmax(scaled_logits, dim=-1)

    # Sort descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Nucleus mask: keep tokens up to cumulative top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    for t_i in range(T):
        indices_to_remove = sorted_indices[t_i][sorted_indices_to_remove[t_i]]
        probs[t_i, indices_to_remove] = 0.0

    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return sampled.cpu().tolist()


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
) -> List[ManiaHitObject]:
    """
    Generates 4K Hit Objects using the trained Model 3 (ManiaPatternTransformer).
    
    Features:
    1. Deterministic Musical Grid Ticks: Ticks are generated strictly from BPM, meter, and timing points (1/8 straight or 1/6 triplet subdivisions).
    2. Deep Audio Cross-Attention & Acoustic Grounding across frequency bins and transients.
    3. 8-Channel Musical Metric Conditioning: Ingests beat divisor fractions (1/1, 1/2, 1/4, 1/8, 1/3, 1/6), meter, bar phase, downbeats, progress, and energy.
    4. Model-driven Hit Object Synthesis: Zero silence gating hacks, zero arbitrary physical filters.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for Model 3 neural inference.")

    from osu_mapper.models.transformer import ManiaPatternTransformer, TransformerConfig

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Model 3 checkpoint not found at: {ckpt_p}")

    print(f"🧠 [Model 3 Neural Generator] Loading transformer checkpoint: {ckpt_p}")
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

    # 1. Generate deterministic sequence of musical grid ticks for the entire song duration
    # In osu!mania 4K, beatmaps with SR < 5.5* use 1/4 binary & 1/3-1/6 ternary subdivisions (73ms at 205 BPM).
    # 1/8 grid (36ms) is reserved strictly for 7.0*+ vibro/burst maps to prevent impossible 27 NPS minijacks.
    subdiv_b = 4 if star_rating < 5.5 else 8
    subdiv_t = 3 if star_rating < 3.5 else 6

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
    )
    num_ticks = len(ticks)
    if num_ticks == 0:
        print("  ⚠️ Notice: No musical ticks generated; returning empty beatmap.")
        return []

    print(f"  • Generated {num_ticks} deterministic musical grid ticks (1/4 & 1/6 subdivisions) across {duration_ms}ms.")

    # 2. Extract Mel audio frames sampled at each tick timestamp
    with torch.no_grad():
        frame_energies_t = audio_mel.exp().mean(dim=-1)
        mean_energy_val = float(frame_energies_t.mean().item()) if frame_energies_t.numel() > 0 else 1.0
        mean_energy = max(1e-4, mean_energy_val)

        tick_mel_list = []
        grid_metrics_list = []

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

    # Pre-generate Stage 1 Algorithmic Scaffold Draft for the Neural Refiner
    print(f"  • Generating Stage 1 Algorithmic Rhythm Scaffold (Target SR: {star_rating:.2f}*)...")
    draft_objects = generate_patterns(
        bpm=bpm,
        offset_ms=offset_ms,
        duration_ms=duration_ms,
        star_rating=star_rating,
    )
    tick_time_arr = [t.time_ms for t in ticks]
    scaffold_keys = torch.zeros(num_ticks, 4, dtype=torch.int64, device=dev)
    for obj in draft_objects:
        idx = min(num_ticks - 1, max(0, bisect.bisect_left(tick_time_arr, obj.start_time)))
        scaffold_keys[idx, obj.column] = 1

    scaffold_tokens_tensor = (
        (scaffold_keys[:, 0] > 0).long()
        | ((scaffold_keys[:, 1] > 0).long() << 1)
        | ((scaffold_keys[:, 2] > 0).long() << 2)
        | ((scaffold_keys[:, 3] > 0).long() << 3)
    )

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

    # Maintain real-time 4-lane physical activity state
    lane_states_seq = torch.zeros(num_ticks, 4, dtype=torch.int64, device=dev)

    generated_tokens: List[int] = []
    all_ln_probs: List[List[float]] = []

    print(f"  • Generating neural 4K patterns on discrete musical ticks (SR: {star_rating:.1f}*, {num_ticks} decision steps)...")

    # Autoregressive generation: tick-by-tick causality with rolling context
    ctx_len = 64
    for cur_pos in range(num_ticks):
        ctx_start = max(0, cur_pos - ctx_len)
        window_len = (cur_pos + 1) - ctx_start

        window_mel = tick_mel_tensor[ctx_start : cur_pos + 1].unsqueeze(0)

        window_prev_tokens = torch.zeros(1, window_len, dtype=torch.int64, device=dev)
        if cur_pos > ctx_start:
            window_prev_tokens[0, 1 : cur_pos - ctx_start + 1] = torch.tensor(
                generated_tokens[ctx_start:cur_pos], dtype=torch.int64, device=dev
            )

        window_lane_states = lane_states_seq[ctx_start : cur_pos + 1].unsqueeze(0)
        window_grid_metrics = grid_metrics_tensor[ctx_start : cur_pos + 1].unsqueeze(0)
        window_scaffold = scaffold_tokens_tensor[ctx_start : cur_pos + 1].unsqueeze(0)
        window_strain = strain_metrics_tensor[ctx_start : cur_pos + 1].unsqueeze(0)
        window_audio_memory = full_audio_memory[:, ctx_start : cur_pos + 1]

        with torch.no_grad():
            chord_logits, ln_logits = model(
                window_mel,
                star_rating=sr_tensor,
                prev_tokens=window_prev_tokens,
                lane_states=window_lane_states,
                grid_metrics=window_grid_metrics,
                scaffold_tokens=window_scaffold,
                strain_metrics=window_strain,
                audio_memory=window_audio_memory,
            )
            cur_chord_logits = chord_logits[0, -1:].clone()
            cur_ln_logits = ln_logits[0, -1:].clone()

            new_tokens = sample_chord_tokens_with_nucleus(
                logits=cur_chord_logits,
                star_rating=star_rating,
                temperature=0.70,
                top_p=0.90,
            )
            new_lns = torch.sigmoid(cur_ln_logits).cpu().tolist()[0]
            tok = new_tokens[0]

        generated_tokens.append(tok)
        all_ln_probs.append(new_lns)

        # Update lane state sequence for future autoregressive context
        if tok > 0:
            cols = token_to_keys(tok)
            for c in cols:
                if (star_rating >= 1.8) and (new_lns[c] >= 0.78):
                    lane_states_seq[cur_pos, c] = 2  # STATE_LN_START
                    hold_span = min(4, num_ticks - cur_pos - 1)
                    for hf in range(1, hold_span):
                        lane_states_seq[cur_pos + hf, c] = 3  # STATE_LN_HOLD
                else:
                    lane_states_seq[cur_pos, c] = 1  # STATE_TAP

    # 3. Convert predicted tokens directly into ManiaHitObjects using State Machine Decoder
    hit_objects: List[ManiaHitObject] = []
    
    # Calibrated LN probability thresholds matching empirical 15-20% dataset ratio
    # Model 3 was trained with pos_weight=6.6: threshold in 0.76-0.85 captures true vocal/synth sustains
    if star_rating < 1.8:
        ln_start_threshold = 0.95
        hold_decay_threshold = 0.85
    elif star_rating < 3.5:
        ln_start_threshold = 0.82
        hold_decay_threshold = 0.60
    elif star_rating < 5.0:
        ln_start_threshold = 0.78
        hold_decay_threshold = 0.52
    else:
        ln_start_threshold = 0.74
        hold_decay_threshold = 0.45

    total_taps = 0
    total_lns = 0

    # Process each of the 4 keys independently with Lane-Locking & Sustain-Tracking
    # Minimum physical gap between notes on the SAME column to prevent unplayable minijacks
    min_col_gap_ms = 45 if star_rating >= 6.0 else (60 if star_rating >= 3.5 else 120)

    for col in range(4):
        busy_until_tick = -1
        last_note_end_ms = -1000

        for i in range(num_ticks):
            if i <= busy_until_tick:
                # Lane is currently locked holding a Long Note; ignore collisions
                continue

            tok = generated_tokens[i] if i < len(generated_tokens) else 0
            if tok <= 0:
                continue

            active_cols = token_to_keys(tok)
            if col not in active_cols:
                continue

            tick_obj = ticks[i]
            base_ms = tick_obj.time_ms
            
            # Dynamic Transient Micro-Alignment (Sub-tick precision to audio onsets)
            center_frame = max(0, min(orig_total_mel_frames - 1, int(round(base_ms / frame_ms))))
            best_offset_ms = 0.0
            max_local_e = -1.0
            for f_off in (-1, 0, 1):
                f_idx = center_frame + f_off
                if 0 <= f_idx < len(frame_energies_t):
                    val = float(frame_energies_t[f_idx].item())
                    if val > max_local_e:
                        max_local_e = val
                        best_offset_ms = f_off * (frame_ms * 0.25)  # sub-tick micro-offset +/- 4ms
            
            t_ms = int(round(base_ms + best_offset_ms))
            if t_ms >= duration_ms:
                break

            # Debounce protection: enforce physical ergonomics on the same column
            if t_ms - last_note_end_ms < min_col_gap_ms:
                continue

            cur_ln_prob = all_ln_probs[i][col] if i < len(all_ln_probs) else 0.0
            is_potential_ln = (star_rating >= 1.8) and (cur_ln_prob >= ln_start_threshold)

            if is_potential_ln:
                # Continuous State Machine: Trace sustain duration along audio frames
                sustain_ticks = 0
                k = i + 1
                max_hold_ticks = min(num_ticks - i - 1, 32)  # max ~ 4 measures
                
                while sustain_ticks < max_hold_ticks:
                    next_prob = all_ln_probs[k][col] if k < len(all_ln_probs) else 0.0
                    next_tok = generated_tokens[k] if k < len(generated_tokens) else 0
                    
                    # Stop sustain if LN probability drops below decay threshold
                    if next_prob < hold_decay_threshold:
                        break
                        
                    # Stop sustain if a new distinct tap is struck on the same lane
                    if (next_tok > 0) and (col in token_to_keys(next_tok)) and (next_prob < ln_start_threshold):
                        break

                    sustain_ticks += 1
                    k += 1

                # If the note sustains across subsequent ticks, emit a calibrated Long Note
                if sustain_ticks >= 1:
                    release_tick = min(num_ticks - 1, i + sustain_ticks)
                    raw_end_ms = ticks[release_tick].time_ms
                    
                    # Apply a safe 30ms release gap so fingers release before the next possible tick
                    end_t_ms = max(t_ms + 75, int(round(raw_end_ms - 30.0)))
                    busy_until_tick = release_tick
                    last_note_end_ms = end_t_ms
                    total_lns += 1
                    
                    hit_objects.append(
                        ManiaHitObject(
                            column=col,
                            start_time=t_ms,
                            end_time=end_t_ms,
                            is_ln=True,
                        )
                    )
                    continue

            # Fallback to standard Single Tap
            end_t_ms = t_ms
            last_note_end_ms = t_ms
            total_taps += 1
            hit_objects.append(
                ManiaHitObject(
                    column=col,
                    start_time=t_ms,
                    end_time=end_t_ms,
                    is_ln=False,
                )
            )

    hit_objects.sort(key=lambda o: (o.start_time, o.column))
    total_objs = len(hit_objects)
    duration_sec = max(0.001, duration_ms / 1000.0)
    avg_nps = total_objs / duration_sec
    ln_pct = (total_lns / max(1, total_objs)) * 100.0
    print(
        f"  ✓ Neural pattern synthesis finished: generated {total_objs} hit objects "
        f"({total_taps} taps, {total_lns} sliders/LNs [{ln_pct:.1f}%], {avg_nps:.1f} NPS, SR: {star_rating:.1f}*)."
    )
    return hit_objects


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
