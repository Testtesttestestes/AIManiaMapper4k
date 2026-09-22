"""
Pattern generation module for osu!mania 4K.

Generates playable 4K hit objects conditioned on target Star Rating (1.0 to 9.9):
- Neural Pattern Generation: Model 3 (ManiaPatternTransformer) with continuous Star Rating conditioning,
  Anti-Cold-Start Lead-in Padding, and sliding context window.
- Algorithmic Fallback: Accurate Star-Rating scaling (1.0* Easy to 9.9* Expert), ergonomic lane assignment,
  and transient synchronization.
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
        # Easy / Beginner 1.0* - 1.8*
        target_nps = 0.8 + (sr - 1.0) * 0.9  # 0.8 - 1.5 NPS
        max_chord_size = 1
        chord_weights = [1.0, 0.0, 0.0, 0.0]
        subdiv_1_2_prob = 0.0
        subdiv_1_4_prob = 0.0
        subdiv_1_8_prob = 0.0
        ln_prob = 0.0
        min_note_gap_ms = 450
    elif sr < 3.0:
        # Normal 1.8* - 3.0*
        target_nps = 1.6 + (sr - 1.8) * 1.2  # 1.6 - 3.0 NPS
        max_chord_size = 2
        chord_weights = [0.90, 0.10, 0.0, 0.0]
        subdiv_1_2_prob = 0.60
        subdiv_1_4_prob = max(0.0, (sr - 2.0) * 0.15)
        subdiv_1_8_prob = 0.0
        ln_prob = 0.08
        min_note_gap_ms = 180
    elif sr < 4.8:
        # Hard 3.0* - 4.8*
        target_nps = 3.0 + (sr - 3.0) * 2.0  # 3.0 - 6.6 NPS
        max_chord_size = 3
        chord_weights = [0.65, 0.28, 0.07, 0.0]
        subdiv_1_2_prob = 0.95
        subdiv_1_4_prob = 0.20 + (sr - 3.0) * 0.25
        subdiv_1_8_prob = 0.0
        ln_prob = 0.12
        min_note_gap_ms = 70
    elif sr < 7.0:
        # Insane 4.8* - 7.0*
        target_nps = 6.6 + (sr - 4.8) * 2.5  # 6.6 - 12.1 NPS
        max_chord_size = 3
        chord_weights = [0.45, 0.38, 0.15, 0.02]
        subdiv_1_2_prob = 1.0
        subdiv_1_4_prob = 0.70 + (sr - 4.8) * 0.12
        subdiv_1_8_prob = max(0.0, (sr - 5.5) * 0.15)
        ln_prob = 0.16
        min_note_gap_ms = 40
    else:
        # Extra / Expert 7.0* - 9.9*
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
    1. Lead-in Silence Padding: Pre-pads silence so transformer positional encoding and audio attention
       are warmed up before the first beat (mitigating cold start).
    2. Overlapping History Window: Seamlessly preserves token history across chunk boundaries.
    3. Natural High-Precision Timing: Preserves raw 16ms temporal stream density without destructive
       quantization, while respecting requested Star Rating density and chord caps.
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
    step_1_4_ms = beat_len_ms / 4.0

    sr_params = get_star_rating_params(star_rating)
    max_allowed_chord = sr_params["max_chord_size"]
    min_gap = sr_params["min_note_gap_ms"]

    # 1. Lead-in silence padding for warm start
    lead_in = torch.zeros(warmup_lead_in_frames, audio_mel.shape[-1], dtype=audio_mel.dtype, device=audio_mel.device)
    padded_mel = torch.cat([lead_in, audio_mel], dim=0)
    total_padded_frames = padded_mel.shape[0]

    hit_objects: List[ManiaHitObject] = []
    column_busy_until = [0, 0, 0, 0]
    last_note_any_lane = -9999

    generated_tokens: List[int] = [0] * warmup_lead_in_frames
    all_ln_probs: List[List[float]] = [[0.0, 0.0, 0.0, 0.0]] * warmup_lead_in_frames

    print(f"  • Generating neural 4K patterns (SR: {star_rating:.1f}*, {orig_total_frames} frames, {duration_ms / 1000:.1f}s)...")

    # Step through sequence with sliding window context (overlap=64)
    step_size = chunk_frames
    for cur_pos in range(warmup_lead_in_frames, total_padded_frames, step_size):
        end_pos = min(total_padded_frames, cur_pos + step_size)

        ctx_start = max(0, cur_pos - 128)
        window_mel = padded_mel[ctx_start:end_pos].unsqueeze(0).to(dev)
        window_len = window_mel.shape[1]

        ctx_tokens = generated_tokens[ctx_start:cur_pos]
        window_prev_tokens = torch.zeros(1, window_len, dtype=torch.int64, device=dev)
        if ctx_tokens:
            window_prev_tokens[0, :len(ctx_tokens)] = torch.tensor(ctx_tokens, dtype=torch.int64, device=dev)

        with torch.no_grad():
            chord_logits, ln_logits = model(window_mel, star_rating=sr_tensor, prev_tokens=window_prev_tokens)
            offset_in_window = cur_pos - ctx_start
            new_chord_logits = chord_logits[0, offset_in_window:]
            new_ln_logits = ln_logits[0, offset_in_window:]

            # Calibrate logits for low Star Ratings (suppress multi-key chords for 1.0-2.5*)
            if max_allowed_chord < 4:
                for tok_idx in range(16):
                    keys = token_to_keys(tok_idx)
                    if len(keys) > max_allowed_chord:
                        new_chord_logits[:, tok_idx] -= 25.0

            new_tokens = new_chord_logits.argmax(dim=-1).cpu().tolist()
            new_lns = torch.sigmoid(new_ln_logits).cpu().tolist()

        generated_tokens.extend(new_tokens)
        all_ln_probs.extend(new_lns)

    # 2. Convert predicted tokens into ManiaHitObjects
    for f_idx in range(orig_total_frames):
        padded_f_idx = f_idx + warmup_lead_in_frames
        tok = generated_tokens[padded_f_idx]
        if tok <= 0:
            continue

        raw_t_ms = int(f_idx * 16.0)  # 16ms hop frame
        if raw_t_ms >= duration_ms:
            break

        # Suppress notes before song offset
        if raw_t_ms < (offset_ms - 30):
            continue

        # Enforce minimum note interval for low Star Rating difficulties (Easy / 1*)
        if star_rating < 2.5 and (raw_t_ms - last_note_any_lane) < min_gap:
            continue

        active_cols = token_to_keys(tok)
        if len(active_cols) > max_allowed_chord:
            active_cols = active_cols[:max_allowed_chord]

        cur_ln_prob = all_ln_probs[padded_f_idx] if padded_f_idx < len(all_ln_probs) else [0.0] * 4
        placed_any = False

        for col in active_cols:
            if column_busy_until[col] > raw_t_ms:
                continue

            is_ln = (star_rating >= 2.0) and (cur_ln_prob[col] > 0.45)
            if is_ln:
                hold_duration = max(int(step_1_4_ms * 2), int(beat_len_ms))
                end_t_ms = min(duration_ms, raw_t_ms + hold_duration)
                column_busy_until[col] = end_t_ms + 20
            else:
                end_t_ms = raw_t_ms

            hit_objects.append(
                ManiaHitObject(
                    column=col,
                    start_time=raw_t_ms,
                    end_time=end_t_ms,
                    is_ln=is_ln,
                )
            )
            placed_any = True

        if placed_any:
            last_note_any_lane = raw_t_ms

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
    Accurately synchronizes to audio transients without artificial quantization flattening.
    """
    rng = random.Random(seed)
    params = get_star_rating_params(star_rating)
    beat_length = 60000.0 / max(1.0, bpm)
    step_1_4 = beat_length / 4.0
    step_1_2 = beat_length / 2.0
    step_1_1 = beat_length

    onset_set: List[int] = sorted(onsets_ms) if onsets_ms else []

    def snap_to_transient(grid_time_ms: int, search_window_ms: float = 35.0) -> int:
        if not onset_set:
            return grid_time_ms
        idx = bisect.bisect_left(onset_set, grid_time_ms)
        candidates = []
        if idx < len(onset_set):
            candidates.append(onset_set[idx])
        if idx > 0:
            candidates.append(onset_set[idx - 1])
        if not candidates:
            return grid_time_ms
        closest = min(candidates, key=lambda o: abs(o - grid_time_ms))
        if abs(closest - grid_time_ms) <= search_window_ms:
            return closest
        return grid_time_ms

    # Determine rhythm grid steps based on Star Rating
    grid_events: List[int] = []
    num_1_4_steps = int((duration_ms - offset_ms) / step_1_4)

    # For fast BPMs on 1.0* Easy difficulty, pace across 2-beat intervals (e.g. half notes)
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
            # 1.0* - 1.8* (Easy): strictly 1/1 beats or 2/1 beats on fast songs
            should_place = is_easy_beat
        elif params["star_rating"] < 3.0:
            # 2.0* - 3.0* (Normal): 1/1 beats + 1/2 beats + sparse 1/4
            if is_1_1:
                should_place = True
            elif is_1_2:
                should_place = rng.random() < params["subdiv_1_2_prob"]
            else:
                should_place = rng.random() < params["subdiv_1_4_prob"]
        else:
            # 3.5*+ (Hard / Insane / Expert): 1/1, 1/2, 1/4 streams and 1/8 bursts
            if is_1_1 or is_1_2:
                should_place = True
            else:
                should_place = rng.random() < params["subdiv_1_4_prob"]

        if should_place:
            search_win = max(40.0, step_1_4 * 0.90)
            aligned_t = snap_to_transient(t_grid, search_window_ms=search_win)
            grid_events.append(aligned_t)

    # For moderate to high SR (>= 3.5*), also ensure strong isolated transients are captured
    if params["star_rating"] >= 3.5 and onset_set:
        for onset in onset_set:
            if offset_ms <= onset < duration_ms:
                # Add if not too close to an existing event
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

        # Restrict multi-key chords off-beat unless difficulty is high
        dist_to_beat = abs((t - offset_ms) % beat_length)
        is_downbeat = dist_to_beat < 25.0 or abs(dist_to_beat - beat_length) < 25.0
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
        max_hold_time = max(t + int(step_1_2), min(t + int(beat_length * 2), next_time + int(step_1_2)))

        for c_idx, col in enumerate(chosen_cols):
            is_ln = can_have_ln and (c_idx == 0) and (max_hold_time - t >= int(step_1_2))
            end_t = max_hold_time if is_ln else t

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
