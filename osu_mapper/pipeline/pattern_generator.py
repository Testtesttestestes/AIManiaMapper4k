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

from osu_mapper.pipeline.timing import snap_to_musical_grid

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
) -> List[ManiaHitObject]:
    """
    Generates 4K Hit Objects using the trained Model 3 (ManiaPatternTransformer).
    
    Features:
    1. Deep Audio Cross-Attention: attends directly to all 80 audio frequency bins.
    2. Musical Metric Grid Conditioning: Ingests beat divisors (1/1, 1/2, 1/3, 1/4, 1/8) & downbeat phase.
    3. Autoregressive 4-Lane State Dynamics: Ingests real-time Free/Tap/Hold/Release state of each key.
    4. Audio-Sustained Musical LN Duration: Releases hold notes on exact musical ticks aligned with audio energy.
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

    orig_total_frames = audio_mel.shape[0]
    sr_tensor = torch.tensor([[star_rating]], dtype=torch.float32, device=dev)
    beat_len_ms = 60000.0 / max(1.0, bpm)
    bar_len_ms = beat_len_ms * 4.0
    frame_ms = 16.0

    sr_params = get_star_rating_params(star_rating)
    max_allowed_chord = sr_params["max_chord_size"]
    min_gap = sr_params["min_note_gap_ms"]

    # 1. Lead-in silence padding for warm start
    lead_in = torch.zeros(warmup_lead_in_frames, audio_mel.shape[-1], dtype=audio_mel.dtype, device=audio_mel.device)
    padded_mel = torch.cat([lead_in, audio_mel], dim=0)
    total_padded_frames = padded_mel.shape[0]

    # Precalculate global audio energy envelope and relative energy ratio per frame
    with torch.no_grad():
        frame_energies_t = padded_mel.exp().mean(dim=-1)  # [T]
        valid_energies = frame_energies_t[warmup_lead_in_frames:] if total_padded_frames > warmup_lead_in_frames else frame_energies_t
        mean_energy_val = float(valid_energies.mean().item()) if valid_energies.numel() > 0 else 1.0
        mean_energy = max(1e-4, mean_energy_val)
        frame_energies = frame_energies_t.cpu().tolist()

    # Precalculate 6D Musical Grid & Macro-Temporal Song Structure Metrics across all padded frames
    grid_metrics_np = []
    for f in range(total_padded_frames):
        orig_f = f - warmup_lead_in_frames
        t_ms = orig_f * frame_ms
        rel_ms = t_ms - offset_ms

        bar_phase = (rel_ms % bar_len_ms) / bar_len_ms if bar_len_ms > 0 else 0.0
        dist_downbeat = abs(rel_ms % bar_len_ms)
        is_downbeat = 1.0 if (dist_downbeat < 16.0 or abs(dist_downbeat - bar_len_ms) < 16.0) else 0.0

        best_div = 0.0
        min_delta = 999.0
        for div in (1, 2, 4, 3, 6, 8):
            step = beat_len_ms / float(div)
            cand_dist = abs(rel_ms % step)
            dist = min(cand_dist, abs(cand_dist - step))
            if dist < min_delta and dist <= (frame_ms * 0.75):
                min_delta = dist
                best_div = float(div)

        progress = float(max(0, orig_f) / max(1, orig_total_frames))
        time_to_end = float(max(0, orig_total_frames - orig_f) / max(1, orig_total_frames))
        local_e = float(frame_energies[f])
        energy_ratio = float(local_e / mean_energy)

        grid_metrics_np.append([best_div, float(bar_phase), is_downbeat, progress, time_to_end, energy_ratio])

    padded_grid_metrics = torch.tensor(grid_metrics_np, dtype=torch.float32, device=dev)

    # Pre-encode the ENTIRE full-song audio spectrum in one global forward pass
    print(f"  • Pre-encoding entire audio track with Bidirectional Cross-Attention & Spectral Differentials...")
    with torch.no_grad():
        full_audio_memory = model.audio_encoder(padded_mel.unsqueeze(0).to(dev))  # [1, total_padded_frames, d_model]

    # Maintain real-time 4-lane physical activity state
    lane_states_seq = torch.zeros(total_padded_frames, 4, dtype=torch.int64, device=dev)

    hit_objects: List[ManiaHitObject] = []
    last_note_any_lane = -9999
    generated_tokens: List[int] = [0] * warmup_lead_in_frames
    all_ln_probs: List[List[float]] = [[0.0, 0.0, 0.0, 0.0]] * warmup_lead_in_frames

    print(f"  • Generating neural 4K patterns with Full-Song Global Audio Context (SR: {star_rating:.1f}*, {orig_total_frames} frames)...")

    # Step through sequence with sliding window context
    step_size = chunk_frames
    for cur_pos in range(warmup_lead_in_frames, total_padded_frames, step_size):
        end_pos = min(total_padded_frames, cur_pos + step_size)

        ctx_start = max(0, cur_pos - 128)
        window_len = end_pos - ctx_start
        window_mel = padded_mel[ctx_start:end_pos].unsqueeze(0).to(dev)

        ctx_tokens = generated_tokens[ctx_start:cur_pos]
        window_prev_tokens = torch.zeros(1, window_len, dtype=torch.int64, device=dev)
        if ctx_tokens:
            window_prev_tokens[0, :len(ctx_tokens)] = torch.tensor(ctx_tokens, dtype=torch.int64, device=dev)

        window_lane_states = lane_states_seq[ctx_start:end_pos].unsqueeze(0)
        window_grid_metrics = padded_grid_metrics[ctx_start:end_pos].unsqueeze(0)
        window_audio_memory = full_audio_memory[:, ctx_start:end_pos]

        with torch.no_grad():
            chord_logits, ln_logits = model(
                window_mel,
                star_rating=sr_tensor,
                prev_tokens=window_prev_tokens,
                lane_states=window_lane_states,
                grid_metrics=window_grid_metrics,
                audio_memory=window_audio_memory,
            )
            offset_in_window = cur_pos - ctx_start
            new_chord_logits = chord_logits[0, offset_in_window:]
            new_ln_logits = ln_logits[0, offset_in_window:]

            if max_allowed_chord < 4:
                for tok_idx in range(16):
                    keys = token_to_keys(tok_idx)
                    if len(keys) > max_allowed_chord:
                        new_chord_logits[:, tok_idx] -= 25.0

            new_tokens = new_chord_logits.argmax(dim=-1).cpu().tolist()
            new_lns = torch.sigmoid(new_ln_logits).cpu().tolist()

        generated_tokens.extend(new_tokens)
        all_ln_probs.extend(new_lns)

        # Update lane state sequence for future context using the 5 physical states
        for local_idx, (tok, lns) in enumerate(zip(new_tokens, new_lns)):
            global_f = cur_pos + local_idx
            if global_f >= total_padded_frames:
                break
            if tok > 0:
                cols = token_to_keys(tok)
                for c in cols:
                    if (star_rating >= 2.0) and (lns[c] > 0.45):
                        lane_states_seq[global_f, c] = 2  # STATE_LN_START
                        # Sustain propagation across hold length
                        hold_frames = min(int(beat_len_ms / frame_ms), total_padded_frames - global_f - 1)
                        for hf in range(1, hold_frames):
                            lane_states_seq[global_f + hf, c] = 3  # STATE_LN_HOLD
                        if (global_f + hold_frames) < total_padded_frames:
                            lane_states_seq[global_f + hold_frames, c] = 4  # STATE_LN_RELEASE
                    else:
                        lane_states_seq[global_f, c] = 1  # STATE_TAP (instantaneous hit, free on next frame)

    # 2. Convert predicted tokens into ManiaHitObjects with Musical Snapping & Audio-Sustained LNs
    for f_idx in range(orig_total_frames):
        padded_f_idx = f_idx + warmup_lead_in_frames
        tok = generated_tokens[padded_f_idx]
        if tok <= 0:
            continue

        raw_t_ms = int(f_idx * 16.0)
        if raw_t_ms >= duration_ms:
            break

        # Snap to closest musical beat subdivision tick
        t_ms = snap_to_musical_grid(raw_t_ms, bpm=bpm, offset_ms=offset_ms, max_snap_dist_ms=24.0)

        if t_ms < (offset_ms - 20):
            continue

        if star_rating < 2.5 and (t_ms - last_note_any_lane) < min_gap:
            continue

        active_cols = token_to_keys(tok)
        if len(active_cols) > max_allowed_chord:
            active_cols = active_cols[:max_allowed_chord]

        cur_ln_prob = all_ln_probs[padded_f_idx] if padded_f_idx < len(all_ln_probs) else [0.0] * 4
        placed_any = False

        for col in active_cols:
            is_ln = (star_rating >= 2.0) and (cur_ln_prob[col] > 0.45)
            if is_ln:
                hold_duration = compute_audio_sustain_duration(audio_mel, f_idx, beat_len_ms)
                end_t_ms = snap_to_musical_grid(t_ms + hold_duration, bpm=bpm, offset_ms=offset_ms, max_snap_dist_ms=24.0)
                end_t_ms = min(duration_ms, max(t_ms + int(beat_len_ms / 2.0), end_t_ms))
            else:
                end_t_ms = t_ms

            hit_objects.append(
                ManiaHitObject(
                    column=col,
                    start_time=t_ms,
                    end_time=end_t_ms,
                    is_ln=is_ln,
                )
            )
            placed_any = True

        if placed_any:
            last_note_any_lane = t_ms

    hit_objects.sort(key=lambda o: (o.start_time, o.column))
    print(f"  ✓ Neural pattern synthesis finished: generated {len(hit_objects)} hit objects (SR: {star_rating:.1f}*).")
    return hit_objects


def generate_patterns(
    bpm: float,
    offset_ms: int,
    duration_ms: int,
    star_rating: float = 3.5,
    onsets_ms: Optional[List[int]] = None,
    seed: int = 42,
) -> List[ManiaHitObject]:
    """
    Algorithmic generator conditioned strictly on requested Star Rating (1.0* to 9.9*).
    Strictly synchronizes to musical beat grid ticks without floating micro-jitter.
    """
    rng = random.Random(seed)
    params = get_star_rating_params(star_rating)
    beat_length = 60000.0 / max(1.0, bpm)
    step_1_4 = beat_length / 4.0
    step_1_2 = beat_length / 2.0

    onset_set: List[int] = sorted(onsets_ms) if onsets_ms else []

    # Determine rhythm grid steps based on Star Rating
    grid_events: List[int] = []
    num_1_4_steps = int((duration_ms - offset_ms) / step_1_4)
    easy_step_mod = 8 if beat_length < 450 else 4

    for step_idx in range(max(0, num_1_4_steps)):
        t_grid = int(round(offset_ms + step_idx * step_1_4))
        if t_grid >= duration_ms:
            break

        is_1_1 = (step_idx % 4 == 0)
        is_1_2 = (step_idx % 2 == 0)
        is_easy_beat = (step_idx % easy_step_mod == 0)

        should_place = False
        if params["star_rating"] < 1.8:
            should_place = is_easy_beat
        elif params["star_rating"] < 3.0:
            if is_1_1:
                should_place = True
            elif is_1_2:
                should_place = rng.random() < params["subdiv_1_2_prob"]
            else:
                should_place = rng.random() < params["subdiv_1_4_prob"]
        else:
            if is_1_1 or is_1_2:
                should_place = True
            else:
                should_place = rng.random() < params["subdiv_1_4_prob"]

        if should_place:
            grid_events.append(t_grid)

    # For moderate to high SR (>= 3.5*), also capture isolated sharp audio transients
    if params["star_rating"] >= 3.5 and onset_set:
        for onset in onset_set:
            if offset_ms <= onset < duration_ms:
                grid_events.append(onset)

    active_timestamps = sorted(list(set(grid_events)))

    column_busy_until = [0, 0, 0, 0]
    last_note_time = [-9999, -9999, -9999, -9999]
    last_note_global = -9999
    hit_objects: List[ManiaHitObject] = []
    max_chord = params["max_chord_size"]
    min_gap = params["min_note_gap_ms"]

    for i, t in enumerate(active_timestamps):
        if (t - last_note_global) < (min_gap // 2):
            continue

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

        dist_to_beat = abs((t - offset_ms) % beat_length)
        is_downbeat = dist_to_beat < 15.0 or abs(dist_to_beat - beat_length) < 15.0
        if not is_downbeat and chord_size > 2 and params["star_rating"] < 6.0:
            chord_size = min(2, chord_size)

        available_cols = [c for c in range(4) if column_busy_until[c] <= t]
        if not available_cols:
            continue

        actual_chord_size = min(chord_size, len(available_cols))
        available_cols.sort(key=lambda c: (last_note_time[c], rng.random()))
        chosen_cols = available_cols[:actual_chord_size]

        can_have_ln = (params["star_rating"] >= 2.0) and (rng.random() < params["ln_prob"])
        next_time = active_timestamps[i + 1] if (i + 1) < len(active_timestamps) else (t + int(beat_length))
        
        gap_to_next = max(0, next_time - t)
        if gap_to_next >= int(beat_length * 1.5):
            hold_len = int(round(beat_length))
        elif gap_to_next >= int(beat_length):
            hold_len = int(round(step_1_2 + step_1_4))
        else:
            hold_len = int(round(step_1_2))

        for c_idx, col in enumerate(chosen_cols):
            is_ln = can_have_ln and (c_idx == 0) and (hold_len >= int(step_1_2))
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
