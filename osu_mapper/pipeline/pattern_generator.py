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


def hit_objects_to_grid_tokens(
    hit_objects: List[ManiaHitObject],
    ticks: List[MusicalGridTick],
    duration_ms: int,
    device: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projects hit objects back onto discrete musical grid ticks.
    Returns:
        (tokens_tensor [num_ticks], lane_states_tensor [num_ticks, 4])
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for hit_objects_to_grid_tokens.")

    num_ticks = len(ticks)
    tokens = torch.zeros(num_ticks, dtype=torch.int64, device=device)
    lane_states = torch.zeros(num_ticks, 4, dtype=torch.int64, device=device)
    tick_times = [float(t.time_ms) for t in ticks]

    if not hit_objects or num_ticks == 0:
        return tokens, lane_states

    for obj in hit_objects:
        if not (0 <= obj.column < 4):
            continue
        idx = bisect.bisect_left(tick_times, float(obj.start_time))
        best_i = idx
        if idx > 0 and (idx >= len(tick_times) or abs(tick_times[idx - 1] - obj.start_time) < abs(tick_times[idx] - obj.start_time)):
            best_i = idx - 1

        if 0 <= best_i < num_ticks and abs(tick_times[best_i] - obj.start_time) <= 45.0:
            tokens[best_i] = int(tokens[best_i].item()) | (1 << obj.column)
            lane_states[best_i, obj.column] = 2 if obj.is_ln else 1  # STATE_LN_START or STATE_TAP

            if obj.is_ln and obj.end_time > obj.start_time:
                end_idx = bisect.bisect_left(tick_times, float(obj.end_time))
                for h_i in range(best_i + 1, min(num_ticks, end_idx)):
                    lane_states[h_i, obj.column] = 3  # STATE_LN_HOLD
                if 0 <= end_idx < num_ticks:
                    lane_states[end_idx, obj.column] = 4  # STATE_LN_RELEASE

    return tokens, lane_states


def _run_single_neural_pass(
    model: Any,
    dev: Any,
    ticks: List[MusicalGridTick],
    tick_times_list: List[float],
    tick_mel_tensor: torch.Tensor,
    grid_metrics_tensor: torch.Tensor,
    grid_metrics_list: List[List[float]],
    multiscale_pyramid_tensor: torch.Tensor,
    full_audio_memory: torch.Tensor,
    initial_scaffold_tokens: torch.Tensor,
    star_rating: float,
    bpm: float,
    offset_ms: int,
    duration_ms: int,
    timing_mode: str,
    temperature: float,
    top_p: float,
    greedy: bool,
    note_conf_thresh: float,
    ln_conf_thresh: float,
    silence_gate: float,
    pass_index: int = 1,
    is_refinement_pass: bool = False,
) -> Tuple[List[ManiaHitObject], torch.Tensor, torch.Tensor]:
    """
    Executes a single neural rollout pass (either initial draft generation or iterative self-refinement).
    """
    num_ticks = len(ticks)
    sr_tensor = torch.tensor([[star_rating]], dtype=torch.float32, device=dev)
    beat_len_ms = 60000.0 / max(30.0, bpm)
    min_ln_ms = max(110.0, beat_len_ms * 0.35)
    effective_silence_gate = max(0.12, float(silence_gate))

    target_density = max(0.05, float(star_rating) / 10.0)
    strain_metrics_tensor = torch.zeros(num_ticks, 2, dtype=torch.float32, device=dev)
    for ti in range(num_ticks):
        time_steps = ti + 1
        run_notes = int(initial_scaffold_tokens[:ti+1].gt(0).sum().item())
        running_density = run_notes / max(1, time_steps)
        density_ratio = min(5.0, running_density / target_density)
        strain_metrics_tensor[ti, 0] = float(running_density)
        strain_metrics_tensor[ti, 1] = float(density_ratio)

    lane_states_seq = torch.zeros(num_ticks, 4, dtype=torch.int64, device=dev)
    generated_tokens: List[int] = [0] * num_ticks
    all_ln_probs: List[List[float]] = [[0.0, 0.0, 0.0, 0.0] for _ in range(num_ticks)]
    active_notes_history: List[int] = []

    active_ln_starts: Dict[int, Optional[int]] = {0: None, 1: None, 2: None, 3: None}
    hit_objects: List[ManiaHitObject] = []
    total_taps = 0
    total_lns = 0

    ctx_len = 64
    target_start_ms = float(offset_ms) if offset_ms is not None else 0.0
    cur_pos = max(0, bisect.bisect_left(tick_times_list, target_start_ms))
    consecutive_silence_ticks = 0

    # For refinement passes, anneal temperature slightly for greater precision and coherence
    pass_temp = max(0.10, temperature * (0.85 ** (pass_index - 1))) if not greedy else 0.0

    while cur_pos < num_ticks:
        cur_time_ms = int(round(ticks[cur_pos].time_ms))
        if cur_time_ms >= (duration_ms - 100.0):
            break

        ctx_start = max(0, cur_pos - ctx_len)
        window_len = (cur_pos + 1) - ctx_start

        # Acoustic Silence & Break Gating
        cur_energy_ratio = grid_metrics_list[cur_pos][5]
        if cur_energy_ratio < effective_silence_gate:
            consecutive_silence_ticks += 1
            for c in range(4):
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
                    lane_states_seq[cur_pos, c] = 4  # STATE_LN_RELEASE

            generated_tokens[cur_pos] = 0
            all_ln_probs[cur_pos] = [0.0, 0.0, 0.0, 0.0]

            if consecutive_silence_ticks > 48 and (cur_time_ms > duration_ms * 0.6):
                break

            cur_pos = max(cur_pos + 1, bisect.bisect_left(tick_times_list, cur_time_ms + beat_len_ms / 2.0))
            continue
        else:
            consecutive_silence_ticks = 0

        window_mel = tick_mel_tensor[ctx_start : cur_pos + 1].unsqueeze(0)
        window_prev_tokens = torch.zeros(1, window_len, dtype=torch.int64, device=dev)
        if cur_pos > ctx_start:
            window_prev_tokens[0, 1 : cur_pos - ctx_start + 1] = torch.tensor(
                generated_tokens[ctx_start:cur_pos], dtype=torch.int64, device=dev
            )

        local_recent_notes = sum(1 for tok_v in generated_tokens[ctx_start:cur_pos] if tok_v > 0)
        running_density = local_recent_notes / max(1, window_len)
        density_ratio = min(5.0, running_density / target_density)
        strain_metrics_tensor[cur_pos, 0] = float(running_density)
        strain_metrics_tensor[cur_pos, 1] = float(density_ratio)

        window_multiscale = multiscale_pyramid_tensor[ctx_start : cur_pos + 1].unsqueeze(0)
        window_lane_states = lane_states_seq[ctx_start : cur_pos + 1].unsqueeze(0)
        window_grid_metrics = grid_metrics_tensor[ctx_start : cur_pos + 1].unsqueeze(0)
        window_scaffold = initial_scaffold_tokens[ctx_start : cur_pos + 1].unsqueeze(0)
        window_strain = strain_metrics_tensor[ctx_start : cur_pos + 1].unsqueeze(0)
        window_audio_memory = full_audio_memory[:, ctx_start : cur_pos + 1]

        with torch.no_grad():
            try:
                chord_logits, ln_logits, _, pred_time_deltas = model(
                    window_mel,
                    star_rating=sr_tensor,
                    prev_tokens=window_prev_tokens,
                    lane_states=window_lane_states,
                    grid_metrics=window_grid_metrics,
                    scaffold_tokens=window_scaffold,
                    strain_metrics=window_strain,
                    audio_memory=window_audio_memory,
                    multiscale_audio=window_multiscale,
                    return_sr=True,
                    return_time_delta=True,
                )
                pred_dt_ms = float(pred_time_deltas[0, -1, 0].item())
            except Exception:
                chord_logits, ln_logits = model(
                    window_mel,
                    star_rating=sr_tensor,
                    prev_tokens=window_prev_tokens,
                    lane_states=window_lane_states,
                    grid_metrics=window_grid_metrics,
                    scaffold_tokens=window_scaffold,
                    strain_metrics=window_strain,
                    audio_memory=window_audio_memory,
                    multiscale_audio=window_multiscale,
                )
                pred_dt_ms = beat_len_ms / (4.0 if star_rating >= 4.0 else 2.0)

            cur_chord_logits = chord_logits[0, -1].clone()
            cur_ln_logits = ln_logits[0, -1].clone()

            # Calibrate note activation: Allow token 0 (rest) naturally so model doesn't spam notes on every tick!
            raw_probs = F.softmax(cur_chord_logits, dim=-1)
            p_rest = float(raw_probs[0].item())
            p_active = 1.0 - p_rest

            # Adaptive threshold for placing a note based on Star Rating & acoustic energy
            target_nps = max(1.0, min(16.0, 1.2 + (star_rating - 1.0) * 1.3))
            expected_note_prob = min(0.85, max(0.12, target_nps / (1000.0 / max(50.0, beat_len_ms / 4.0))))
            active_thresh = max(0.08, min(0.55, 0.50 - (star_rating - 2.5) * 0.05))

            if is_refinement_pass:
                prev_draft_tok = int(initial_scaffold_tokens[cur_pos].item())
                # If previous pass had a note here, support it if audio has energy, but PRUNE if audio is quiet
                if prev_draft_tok > 0:
                    if cur_energy_ratio < 0.40:
                        # Prune noise/debris in quiet passages
                        cur_chord_logits[0] += 3.0
                    else:
                        cur_chord_logits[prev_draft_tok] += 1.5

            active_logits = cur_chord_logits.clone()
            is_downbeat = ticks[cur_pos].is_downbeat or (ticks[cur_pos].divisor == 1)
            prev_tok = active_notes_history[-1] if active_notes_history else 0
            prev_k = bin(prev_tok).count("1")

            # Apply ergonomic chord rules to active classes (1..15)
            for t_i in range(1, 16):
                k_cnt = bin(t_i).count("1")
                if star_rating < 2.0 and k_cnt > 1:
                    active_logits[t_i] = -1e9
                elif star_rating < 3.5 and k_cnt > 2:
                    active_logits[t_i] = -1e9
                elif star_rating < 5.0 and k_cnt > 2 and not is_downbeat:
                    active_logits[t_i] = -1e9
                elif k_cnt == 4 and not (is_downbeat and star_rating >= 6.0):
                    active_logits[t_i] = -1e9

                if star_rating >= 5.0 and k_cnt >= 2:
                    active_logits[t_i] += min(8.0, (float(star_rating) - 4.0) * 0.6)

                if star_rating < 7.0 and prev_k >= 2 and k_cnt >= 2 and not is_downbeat:
                    active_logits[t_i] -= 6.0

            # Discourage active taps on held LN lanes
            for c in range(4):
                if active_ln_starts[c] is not None:
                    for t_cand in range(1, 16):
                        if (t_cand >> c) & 1:
                            active_logits[t_cand] -= 10.0

            # Anti-Trill & Anti-Jack limiters
            if len(active_notes_history) >= 4:
                if (active_notes_history[-1] == active_notes_history[-3]) and (active_notes_history[-2] == active_notes_history[-4]):
                    active_logits[active_notes_history[-1]] -= 8.0
                    active_logits[active_notes_history[-2]] -= 8.0

            if len(active_notes_history) >= 2 and star_rating < 8.0:
                if active_notes_history[-1] == active_notes_history[-2] and active_notes_history[-1] > 0:
                    active_logits[active_notes_history[-1]] -= 10.0

            eff_temp = 0.30 if (pass_temp <= 0.05 and not greedy) else max(0.1, pass_temp)
            if greedy or pass_temp <= 0.05:
                tok = int(active_logits.argmax().item())
            else:
                scaled = active_logits / eff_temp
                probs = F.softmax(scaled, dim=-1)
                tok = int(torch.multinomial(probs, 1).item())

            new_lns = torch.sigmoid(cur_ln_logits).cpu().tolist()

        generated_tokens[cur_pos] = tok
        all_ln_probs[cur_pos] = new_lns
        if tok > 0:
            active_notes_history.append(tok)

        # LN tracking state machine
        new_cols = set(token_to_keys(tok))
        effective_ln_thresh = max(0.30, min(0.60, float(ln_conf_thresh) if (0.0 < ln_conf_thresh < 0.70) else 0.38))
        for c in range(4):
            if active_ln_starts[c] is not None:
                is_held = (new_lns[c] >= 0.28) and (c not in new_cols) and ((cur_time_ms - active_ln_starts[c]) < int(round(beat_len_ms * 3.0)))
                if is_held:
                    lane_states_seq[cur_pos, c] = 3  # STATE_LN_HOLD
                else:
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
                    lane_states_seq[cur_pos, c] = 4  # STATE_LN_RELEASE

        for c in new_cols:
            if active_ln_starts[c] is None:
                if (star_rating >= 1.5) and (new_lns[c] >= effective_ln_thresh):
                    active_ln_starts[c] = cur_time_ms
                    lane_states_seq[cur_pos, c] = 2  # STATE_LN_START
                else:
                    hit_objects.append(
                        ManiaHitObject(
                            column=c,
                            start_time=cur_time_ms,
                            end_time=cur_time_ms,
                            is_ln=False,
                        )
                    )
                    total_taps += 1
                    lane_states_seq[cur_pos, c] = 1  # STATE_TAP
            else:
                start_t = active_ln_starts[c]
                hold_len = cur_time_ms - start_t
                if hold_len >= min_ln_ms:
                    end_t = max(start_t + int(round(min_ln_ms)), cur_time_ms - 30)
                    hit_objects.append(ManiaHitObject(column=c, start_time=start_t, end_time=end_t, is_ln=True))
                    total_lns += 1
                else:
                    hit_objects.append(ManiaHitObject(column=c, start_time=start_t, end_time=start_t, is_ln=False))
                    total_taps += 1

                hit_objects.append(ManiaHitObject(column=c, start_time=cur_time_ms, end_time=cur_time_ms, is_ln=False))
                total_taps += 1
                active_ln_starts[c] = None
                lane_states_seq[cur_pos, c] = 1

        # Timing advancement
        timing_mode_norm = str(timing_mode).lower().strip()
        raw_beats = pred_dt_ms / beat_len_ms
        has_dense_drop = (cur_energy_ratio >= 0.85) and (star_rating >= 3.2)
        is_calm_section = (cur_energy_ratio < 0.60)

        if timing_mode_norm in ("soft_snap", "soft", "magnetic", "free_snap"):
            raw_jump_ms = max(beat_len_ms * 0.0625, float(pred_dt_ms))
            raw_target_ms = ticks[cur_pos].time_ms + raw_jump_ms
            nearest_idx = bisect.bisect_left(tick_times_list, raw_target_ms)
            cand_indices = []
            if nearest_idx > 0:
                cand_indices.append(nearest_idx - 1)
            if nearest_idx < len(tick_times_list):
                cand_indices.append(nearest_idx)
            if nearest_idx + 1 < len(tick_times_list):
                cand_indices.append(nearest_idx + 1)

            best_idx = cand_indices[0] if cand_indices else cur_pos + 1
            best_dist = abs(tick_times_list[best_idx] - raw_target_ms)
            for c_i in cand_indices[1:]:
                dist_cand = abs(tick_times_list[c_i] - raw_target_ms)
                if dist_cand < best_dist:
                    best_dist = dist_cand
                    best_idx = c_i

            target_next_pos = max(cur_pos + 1, best_idx)

        elif timing_mode_norm in ("universal_grid", "hybrid_grid", "universal"):
            if star_rating < 2.5:
                cand_divs = [1.0, 0.5, 0.3333]
            elif star_rating < 4.0:
                cand_divs = [0.5, 0.3333, 0.25, 0.1667]
            elif star_rating < 6.0:
                cand_divs = [0.3333, 0.25, 0.1667, 0.125] if has_dense_drop else [0.5, 0.3333, 0.25]
            elif star_rating < 9.0:
                cand_divs = [0.25, 0.1667, 0.125, 0.0833]
            else:
                cand_divs = [0.125, 0.0833, 0.0625]

            quantized_fraction = min(cand_divs, key=lambda d: abs(d - raw_beats))
            quantized_jump_ms = quantized_fraction * beat_len_ms
            target_time_ms = ticks[cur_pos].time_ms + quantized_jump_ms
            next_pos = bisect.bisect_left(tick_times_list, target_time_ms - 3.0)
            target_next_pos = max(cur_pos + 1, next_pos)

        elif timing_mode_norm in ("residual", "anchor_residual", "micro"):
            macro_divs = [1.0, 0.5, 0.3333, 0.25, 0.1667, 0.125]
            anchor_frac = min(macro_divs, key=lambda d: abs(d - raw_beats))
            anchor_jump_ms = anchor_frac * beat_len_ms
            residual_ms = max(-25.0, min(25.0, float(pred_dt_ms) - anchor_jump_ms))
            effective_jump_ms = max(beat_len_ms * 0.0625, anchor_jump_ms + residual_ms)
            target_time_ms = ticks[cur_pos].time_ms + effective_jump_ms
            next_pos = bisect.bisect_left(tick_times_list, target_time_ms - 4.0)
            target_next_pos = max(cur_pos + 1, next_pos)

        else:
            if star_rating < 2.5:
                cand_divs = [1.0, 2.0] if is_calm_section else [0.5, 1.0]
            elif star_rating < 4.0:
                cand_divs = [0.25, 0.5] if has_dense_drop else ([0.5, 1.0] if is_calm_section else [0.25, 0.5, 1.0])
            elif star_rating < 6.0:
                cand_divs = [0.25] if has_dense_drop else ([0.5, 1.0] if is_calm_section else [0.25, 0.5])
            elif star_rating < 9.0:
                cand_divs = [0.125, 0.25] if has_dense_drop else [0.25]
            elif star_rating < 13.0:
                cand_divs = [0.125] if has_dense_drop else [0.125, 0.25]
            else:
                cand_divs = [0.0833, 0.125]

            quantized_fraction = min(cand_divs, key=lambda d: abs(d - raw_beats))
            quantized_jump_ms = quantized_fraction * beat_len_ms
            target_time_ms = ticks[cur_pos].time_ms + quantized_jump_ms
            next_pos = bisect.bisect_left(tick_times_list, target_time_ms - 5.0)
            target_next_pos = max(cur_pos + 1, next_pos)

        # Anti-overshoot check
        if target_next_pos > cur_pos + 1:
            for intermediate_i in range(cur_pos + 1, target_next_pos):
                scaff_val = int(initial_scaffold_tokens[intermediate_i].item())
                inter_energy = grid_metrics_list[intermediate_i][5]
                is_db = ticks[intermediate_i].is_downbeat
                div_val = ticks[intermediate_i].divisor

                should_intercept = False
                if scaff_val > 0 and inter_energy >= effective_silence_gate:
                    if is_db:
                        should_intercept = True
                    elif star_rating >= 3.0 and div_val in (1, 2, 4):
                        should_intercept = True
                    elif star_rating >= 6.0:
                        should_intercept = True

                if should_intercept:
                    target_next_pos = intermediate_i
                    break

        cur_pos = target_next_pos

    # Close trailing sliders
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

    # Acoustic Critic & Refinement Inpainting Pass
    existing_note_times = set(o.start_time for o in hit_objects)
    missed_ticks_to_inpaint: List[int] = []

    for k in range(num_ticks):
        t_k = int(round(ticks[k].time_ms))
        if t_k >= duration_ms - 100:
            break

        has_nearby_note = any((t_k + dt_chk) in existing_note_times for dt_chk in (-35, -20, 0, 20, 35))
        if has_nearby_note:
            continue

        scaff_val = int(initial_scaffold_tokens[k].item())
        k_energy = grid_metrics_list[k][5]
        is_db = ticks[k].is_downbeat
        div_val = ticks[k].divisor

        is_missed_onset = False
        if k_energy >= max(0.55, effective_silence_gate):
            if is_db:
                is_missed_onset = True
            elif scaff_val > 0 and star_rating >= 3.0 and div_val in (1, 2, 4):
                is_missed_onset = True
            elif scaff_val > 0 and star_rating >= 6.0:
                is_missed_onset = True

        if is_missed_onset:
            missed_ticks_to_inpaint.append(k)

    if missed_ticks_to_inpaint:
        with torch.no_grad():
            for k in missed_ticks_to_inpaint:
                t_k = int(round(ticks[k].time_ms))
                ctx_start = max(0, k - ctx_len)
                w_len = (k + 1) - ctx_start

                w_mel = tick_mel_tensor[ctx_start : k + 1].unsqueeze(0)
                w_multi = multiscale_pyramid_tensor[ctx_start : k + 1].unsqueeze(0)
                w_lane = lane_states_seq[ctx_start : k + 1].unsqueeze(0)
                w_grid = grid_metrics_tensor[ctx_start : k + 1].unsqueeze(0)
                w_scaff = initial_scaffold_tokens[ctx_start : k + 1].unsqueeze(0)
                w_strain = strain_metrics_tensor[ctx_start : k + 1].unsqueeze(0)
                w_audio = full_audio_memory[:, ctx_start : k + 1]

                w_prev = torch.zeros(1, w_len, dtype=torch.int64, device=dev)
                if k > ctx_start:
                    w_prev[0, 1 : k - ctx_start + 1] = torch.tensor(
                        generated_tokens[ctx_start:k], dtype=torch.int64, device=dev
                    )

                try:
                    c_logits, l_logits, _, _ = model(
                        w_mel,
                        star_rating=sr_tensor,
                        prev_tokens=w_prev,
                        lane_states=w_lane,
                        grid_metrics=w_grid,
                        scaffold_tokens=w_scaff,
                        strain_metrics=w_strain,
                        audio_memory=w_audio,
                        multiscale_audio=w_multi,
                        return_sr=True,
                        return_time_delta=True,
                    )
                except Exception:
                    c_logits, l_logits = model(
                        w_mel,
                        star_rating=sr_tensor,
                        prev_tokens=w_prev,
                        lane_states=w_lane,
                        grid_metrics=w_grid,
                        scaffold_tokens=w_scaff,
                        strain_metrics=w_strain,
                        audio_memory=w_audio,
                        multiscale_audio=w_multi,
                    )

                cur_c_logits = c_logits[0, -1].clone()
                cur_c_logits[0] = -1e9

                for col_idx in range(4):
                    is_lane_busy = any(
                        (o.column == col_idx and o.is_ln and o.start_time < t_k < o.end_time)
                        for o in hit_objects
                    )
                    if is_lane_busy:
                        for cand_tok in range(1, 16):
                            if (cand_tok >> col_idx) & 1:
                                cur_c_logits[cand_tok] = -1e9

                infill_tok = int(cur_c_logits.argmax().item())
                if infill_tok > 0:
                    generated_tokens[k] = infill_tok
                    infill_cols = token_to_keys(infill_tok)
                    infill_ln_probs = torch.sigmoid(l_logits[0, -1]).cpu().tolist()
                    all_ln_probs[k] = infill_ln_probs

                    for c in infill_cols:
                        if (star_rating >= 2.5) and (infill_ln_probs[c] >= 0.50):
                            hit_objects.append(ManiaHitObject(
                                column=c,
                                start_time=t_k,
                                end_time=t_k + int(round(min_ln_ms)),
                                is_ln=True,
                            ))
                            total_lns += 1
                        else:
                            hit_objects.append(ManiaHitObject(
                                column=c,
                                start_time=t_k,
                                end_time=t_k,
                                is_ln=False,
                            ))
                            total_taps += 1
                    existing_note_times.add(t_k)

    solved_objects = solve_sliders_and_collisions(hit_objects, beat_len_ms=beat_len_ms, min_ln_ms=min_ln_ms)
    out_tokens_tensor = torch.tensor(generated_tokens, dtype=torch.int64, device=dev)
    return solved_objects, out_tokens_tensor, lane_states_seq


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
    num_passes: int = 2,
    initial_draft_objects: Optional[List[ManiaHitObject]] = None,
) -> List[ManiaHitObject]:
    """
    Generates 4K Hit Objects using trained Model 3 (ManiaPatternTransformer) with Multi-Pass Iterative Refinement.
    
    Multi-Pass Workflow:
    - Pass 1 (Draft Generation): Generates the initial layout from the acoustic onset scaffold or algorithmic draft.
    - Pass 2..N (Neural Self-Refinement & Revision):
      The model is fed its OWN previously generated beatmap as a draft scaffold.
      In each iteration, the model evaluates and sculpts the map:
      * Preserves strong rhythmic hits aligned with music energy.
      * Prunes noise/hallucinations in quiet/silent passages.
      * Fills in missed transients and downbeats.
      * Adjusts LN sustain tails to match acoustic decay.
      * Calibrates chord density to match target Star Rating.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for Model 3 neural inference.")

    from osu_mapper.models.transformer import ManiaPatternTransformer, TransformerConfig

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Model 3 checkpoint not found at: {ckpt_p}")

    total_passes = max(1, int(num_passes))
    pass_label = f"{total_passes}-Pass Iterative Refinement" if total_passes > 1 else "Single-Pass Generation"
    print(f"🧠 [Model 3 Neural Generator] Loading transformer checkpoint: {ckpt_p} (Mode: {timing_mode}, {pass_label})")

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
    frame_ms = 16.0

    # 1. Generate full-resolution musical grid ticks
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

    print(f"  • Generated {num_ticks} full-resolution musical grid ticks across {duration_ms}ms.")

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

        from osu_mapper.dataset.feature_extractor import compute_multiscale_pyramid_features
        multiscale_pyramid_tensor = compute_multiscale_pyramid_features(audio_mel, tick_times_arr, frame_ms=frame_ms).to(dev)

        # Pre-encode full tick audio representation with deep audio encoder
        print(f"  • Pre-encoding tick audio features via Bidirectional Audio Encoder...")
        full_audio_memory = model.audio_encoder(tick_mel_tensor.unsqueeze(0))

    tick_times_list = [float(t_obj.time_ms) for t_obj in ticks]
    sr_tensor = torch.tensor([[star_rating]], dtype=torch.float32, device=dev)

    # Prepare initial scaffold for Pass 1
    if initial_draft_objects is not None and len(initial_draft_objects) > 0:
        print(f"  • Using user-provided draft beatmap ({len(initial_draft_objects)} objects) as Pass 1 scaffold...")
        current_scaffold_tokens, _ = hit_objects_to_grid_tokens(initial_draft_objects, ticks, duration_ms, device=dev)
    else:
        print(f"  • Generating Stage 1 Acoustic Onset Rhythm Scaffold (Target SR: {star_rating:.2f}*)...")
        current_scaffold_tokens = compute_acoustic_onset_scaffold(
            audio_mel=tick_mel_tensor.unsqueeze(0),
            grid_metrics=grid_metrics_tensor.unsqueeze(0),
            star_rating=sr_tensor,
            silence_threshold=silence_gate,
        ).squeeze(0)

    current_objects: List[ManiaHitObject] = []

    # Multi-Pass Execution Loop
    for p in range(1, total_passes + 1):
        is_refinement = (p > 1)
        pass_name = f"Pass {p}/{total_passes} (Initial Draft)" if p == 1 else f"Pass {p}/{total_passes} (Neural Self-Refinement)"
        print(f"\n  🔄 [Multi-Pass Iteration {p}/{total_passes}] Executing {pass_name} (Target SR: {star_rating:.2f}*)...")

        pass_objects, pass_tokens, pass_lanes = _run_single_neural_pass(
            model=model,
            dev=dev,
            ticks=ticks,
            tick_times_list=tick_times_list,
            tick_mel_tensor=tick_mel_tensor,
            grid_metrics_tensor=grid_metrics_tensor,
            grid_metrics_list=grid_metrics_list,
            multiscale_pyramid_tensor=multiscale_pyramid_tensor,
            full_audio_memory=full_audio_memory,
            initial_scaffold_tokens=current_scaffold_tokens,
            star_rating=star_rating,
            bpm=bpm,
            offset_ms=offset_ms,
            duration_ms=duration_ms,
            timing_mode=timing_mode,
            temperature=temperature,
            top_p=top_p,
            greedy=greedy,
            note_conf_thresh=note_conf_thresh,
            ln_conf_thresh=ln_conf_thresh,
            silence_gate=silence_gate,
            pass_index=p,
            is_refinement_pass=is_refinement,
        )

        # Log pass results and compare with previous pass
        total_objs = len(pass_objects)
        final_lns = sum(1 for o in pass_objects if o.is_ln)
        final_taps = total_objs - final_lns
        duration_sec = max(0.001, duration_ms / 1000.0)
        avg_nps = total_objs / duration_sec
        ln_pct = (final_lns / max(1, total_objs)) * 100.0

        if is_refinement:
            # Calculate changes relative to previous pass
            prev_times = set((o.column, o.start_time) for o in current_objects)
            cur_times = set((o.column, o.start_time) for o in pass_objects)
            added_notes = len(cur_times - prev_times)
            removed_notes = len(prev_times - cur_times)
            retained_notes = len(cur_times & prev_times)
            print(
                f"  ✓ [{pass_name}] Completed: {total_objs} objects ({final_taps} taps, {final_lns} LNs [{ln_pct:.1f}%], {avg_nps:.1f} NPS) | "
                f"Changes: +{added_notes} added, -{removed_notes} pruned, {retained_notes} confirmed."
            )
        else:
            print(
                f"  ✓ [{pass_name}] Completed: {total_objs} objects ({final_taps} taps, {final_lns} LNs [{ln_pct:.1f}%], {avg_nps:.1f} NPS)."
            )

        current_objects = pass_objects
        # Set up the scaffold for next pass from the model's own current output!
        current_scaffold_tokens, _ = hit_objects_to_grid_tokens(current_objects, ticks, duration_ms, device=dev)

    print(f"\n  🎉 [Multi-Pass Complete] Finished {total_passes} generation pass(es) -> Final result: {len(current_objects)} ranked-solved objects.")
    return current_objects


def refine_patterns_with_model3(
    checkpoint_path: str | Path,
    audio_mel: Any,
    existing_objects: List[ManiaHitObject],
    bpm: float,
    offset_ms: int,
    duration_ms: int,
    star_rating: float = 4.0,
    passes: int = 1,
    timing_points: Optional[List[TimingPoint]] = None,
    timing_mode: str = "soft_snap",
    temperature: float = 0.15,
) -> List[ManiaHitObject]:
    """
    Refines an existing beatmap (or algorithmic draft) using Model 3 over N refinement iterations.
    """
    return generate_patterns_with_model3(
        checkpoint_path=checkpoint_path,
        audio_mel=audio_mel,
        bpm=bpm,
        offset_ms=offset_ms,
        duration_ms=duration_ms,
        star_rating=star_rating,
        timing_points=timing_points,
        timing_mode=timing_mode,
        temperature=temperature,
        num_passes=passes,
        initial_draft_objects=existing_objects,
    )


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
