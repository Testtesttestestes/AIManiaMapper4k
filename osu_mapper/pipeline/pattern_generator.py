"""
Pattern generation module for osu!mania 4K.

Generates playable 4K hit objects conditioned on target Star Rating (1.0 to 9.9):
- Neural Pattern Generation: Using Model 3 (ManiaPatternTransformer autoregressive model)
- Algorithmic Fallback: Conditioned on musical transients, density (NPS), chord types, LN ratios, and ergonomic anti-finger-trap constraints.
"""

from __future__ import annotations

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
    """Calculates algorithmic parameters based on requested Star Rating (1.0 to 9.9)."""
    sr = max(1.0, min(9.9, star_rating))

    target_nps = 1.2 + (sr ** 1.35) * 1.05
    subdiv_1_4_prob = max(0.0, min(0.95, (sr - 2.0) / 4.5))
    subdiv_1_8_prob = max(0.0, min(0.40, (sr - 6.5) / 3.5))

    if sr < 2.5:
        chord_weights = [0.95, 0.05, 0.0, 0.0]
    elif sr < 4.0:
        chord_weights = [0.75, 0.25, 0.0, 0.0]
    elif sr < 6.0:
        chord_weights = [0.55, 0.35, 0.10, 0.0]
    elif sr < 7.5:
        chord_weights = [0.40, 0.40, 0.18, 0.02]
    else:
        chord_weights = [0.25, 0.45, 0.24, 0.06]

    ln_prob = 0.08 + (sr / 9.9) * 0.15

    return {
        "star_rating": sr,
        "target_nps": target_nps,
        "subdiv_1_4_prob": subdiv_1_4_prob,
        "subdiv_1_8_prob": subdiv_1_8_prob,
        "chord_weights": chord_weights,
        "ln_prob": ln_prob,
    }


def generate_patterns_with_model3(
    checkpoint_path: str | Path,
    audio_mel: Any,  # torch.Tensor [Total_T, 80]
    bpm: float,
    offset_ms: int,
    duration_ms: int,
    star_rating: float = 4.0,
    device: Optional[Any] = None,
    chunk_frames: int = 256,
) -> List[ManiaHitObject]:
    """
    Generates 4K Hit Objects using the trained Model 3 (ManiaPatternTransformer).
    Autoregressively synthesizes tokens across the entire song duration.
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
    model.load_state_dict(state)
    model.eval()

    total_frames = audio_mel.shape[0]
    sr_tensor = torch.tensor([[star_rating]], dtype=torch.float32, device=dev)
    beat_len_ms = 60000.0 / max(1.0, bpm)
    step_1_4_ms = beat_len_ms / 4.0

    hit_objects: List[ManiaHitObject] = []
    column_busy_until = [0, 0, 0, 0]
    prev_tok_val = 0

    print(f"  • Generating neural 4K patterns for {total_frames} frames ({duration_ms / 1000:.1f}s)...")

    # Process in chunks of chunk_frames with autoregressive seeding
    for start_f in range(0, total_frames, chunk_frames):
        end_f = min(total_frames, start_f + chunk_frames)
        cur_chunk_len = end_f - start_f
        chunk_mel = audio_mel[start_f:end_f].unsqueeze(0).to(dev)

        # Autoregressive prev_tokens buffer
        prev_tokens = torch.zeros(1, cur_chunk_len, dtype=torch.int64, device=dev)
        prev_tokens[0, 0] = prev_tok_val

        with torch.no_grad():
            chord_logits, ln_logits = model(chunk_mel, star_rating=sr_tensor, prev_tokens=prev_tokens)
            pred_tokens = chord_logits.argmax(dim=-1).squeeze(0).cpu().tolist()
            ln_probs = torch.sigmoid(ln_logits).squeeze(0).cpu().tolist()

        if pred_tokens:
            prev_tok_val = pred_tokens[-1]

        # Convert frame predictions to HitObjects
        for idx in range(cur_chunk_len):
            tok = pred_tokens[idx]
            if tok <= 0:
                continue

            frame_idx = start_f + idx
            raw_t_ms = int(frame_idx * 16.0)  # 16ms hop length
            if raw_t_ms >= duration_ms:
                break

            # Snap to nearest 1/4 beat grid
            grid_step = round((raw_t_ms - offset_ms) / step_1_4_ms)
            snapped_t_ms = int(round(offset_ms + grid_step * step_1_4_ms))
            if abs(snapped_t_ms - raw_t_ms) <= 28:
                t_ms = max(0, snapped_t_ms)
            else:
                t_ms = max(0, raw_t_ms)

            active_cols = token_to_keys(tok)
            cur_ln_probs = ln_probs[idx] if idx < len(ln_probs) else [0.0] * 4

            for col in active_cols:
                if column_busy_until[col] > t_ms:
                    continue  # column held by active LN

                # Check if LN is active (probability > 0.45)
                is_ln = cur_ln_probs[col] > 0.45
                if is_ln:
                    hold_duration = max(int(step_1_4_ms * 2), int(beat_len_ms))
                    end_t_ms = min(duration_ms, t_ms + hold_duration)
                    column_busy_until[col] = end_t_ms + 20
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

    hit_objects.sort(key=lambda o: (o.start_time, o.column))
    print(f"  ✓ Neural pattern synthesis finished: generated {len(hit_objects)} hit objects.")
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
    Algorithmic generator fallback for 4K hit objects.
    """
    rng = random.Random(seed)
    params = get_star_rating_params(star_rating)
    beat_length = 60000.0 / max(1.0, bpm)
    step_1_4 = beat_length / 4.0
    step_1_2 = beat_length / 2.0

    onset_set: List[int] = sorted(onsets_ms) if onsets_ms else []

    def snap_to_transient(grid_time_ms: int, search_window_ms: float = 30.0) -> int:
        if not onset_set:
            return grid_time_ms
        import bisect
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

    active_set = set()
    for onset in onset_set:
        if 0 <= onset < duration_ms:
            active_set.add(onset)

    num_steps = int((duration_ms - offset_ms) / step_1_4)
    for step_idx in range(max(0, num_steps)):
        t_grid = int(round(offset_ms + step_idx * step_1_4))
        if t_grid >= duration_ms:
            break

        is_1_1 = (step_idx % 4 == 0)
        is_1_2 = (step_idx % 2 == 0)

        should_place = False
        if is_1_1:
            should_place = True
        elif is_1_2:
            should_place = rng.random() < max(0.5, params["star_rating"] / 5.0)
        else:
            should_place = rng.random() < params["subdiv_1_4_prob"]

        if should_place:
            aligned_t = snap_to_transient(t_grid, search_window_ms=max(35.0, step_1_4 * 0.75))
            active_set.add(aligned_t)

    active_timestamps = sorted(list(active_set))

    column_busy_until = [0, 0, 0, 0]
    last_note_time = [-9999, -9999, -9999, -9999]
    hit_objects: List[ManiaHitObject] = []

    for i, t in enumerate(active_timestamps):
        r = rng.random()
        w = params["chord_weights"]
        if r < w[0]:
            chord_size = 1
        elif r < w[0] + w[1]:
            chord_size = 2
        elif r < w[0] + w[1] + w[2]:
            chord_size = 3
        else:
            chord_size = 4

        dist_to_beat = abs((t - offset_ms) % beat_length)
        is_downbeat = dist_to_beat < 15.0 or abs(dist_to_beat - beat_length) < 15.0
        if not is_downbeat and chord_size > 2:
            chord_size = 2

        available_cols = [c for c in range(4) if column_busy_until[c] <= t]
        if not available_cols:
            continue

        actual_chord_size = min(chord_size, len(available_cols))
        available_cols.sort(key=lambda c: (last_note_time[c], rng.random()))
        chosen_cols = available_cols[:actual_chord_size]

        can_have_ln = rng.random() < params["ln_prob"]
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

    hit_objects.sort(key=lambda o: (o.start_time, o.column))
    return hit_objects
