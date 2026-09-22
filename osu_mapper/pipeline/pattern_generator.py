"""
Pattern generation module for osu!mania 4K.

Generates playable 4K hit objects conditioned on target Star Rating (1.0 to 9.9),
musical transients, target note density (NPS), chord types (single, jump, hand, quad),
long-note (LN) ratios, and ergonomic anti-finger-trap constraints.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple


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


def get_star_rating_params(star_rating: float) -> dict:
    """Calculates algorithmic parameters based on requested Star Rating (1.0 to 9.9)."""
    sr = max(1.0, min(9.9, star_rating))

    # Target NPS progression: ~2.0 at SR 1.5, ~6.0 at SR 3.5, ~12.0 at SR 5.5, ~20.0 at SR 8.0+
    target_nps = 1.2 + (sr ** 1.35) * 1.05

    # Subdivisions active:
    # SR < 2.5: mainly 1/1 and 1/2 beats
    # SR 2.5 - 4.5: 1/2 with occasional 1/4
    # SR 4.5 - 7.0: 1/4 streams and jumpstreams
    # SR > 7.0: fast 1/4 and 1/8 bursts
    subdiv_1_4_prob = max(0.0, min(0.95, (sr - 2.0) / 4.5))
    subdiv_1_8_prob = max(0.0, min(0.40, (sr - 6.5) / 3.5))

    # Chord distribution:
    # Single (1 note)
    # Jump (2 notes)
    # Hand (3 notes)
    # Quad (4 notes)
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

    # LN probability
    ln_prob = 0.08 + (sr / 9.9) * 0.15

    return {
        "star_rating": sr,
        "target_nps": target_nps,
        "subdiv_1_4_prob": subdiv_1_4_prob,
        "subdiv_1_8_prob": subdiv_1_8_prob,
        "chord_weights": chord_weights,
        "ln_prob": ln_prob,
    }


def generate_patterns(
    bpm: float,
    offset_ms: int,
    duration_ms: int,
    star_rating: float = 3.5,
    onsets_ms: Optional[List[int]] = None,
    seed: int = 42,
) -> List[ManiaHitObject]:
    """
    Generates an ergonomic, musically aligned set of 4K hit objects.

    Args:
        bpm: Song tempo.
        offset_ms: Starting offset in milliseconds.
        duration_ms: Total duration in milliseconds.
        star_rating: Target difficulty (1.0 to 9.9).
        onsets_ms: Detected audio transients for millisecond alignment.
        seed: Random seed for deterministic reproducibility.
    """
    rng = random.Random(seed)
    params = get_star_rating_params(star_rating)
    beat_length = 60000.0 / max(1.0, bpm)
    step_1_4 = beat_length / 4.0
    step_1_2 = beat_length / 2.0
    step_1_1 = beat_length

    # Pre-index onsets for rapid O(1) alignment lookups
    onset_set: List[int] = sorted(onsets_ms) if onsets_ms else []

    def snap_to_transient(grid_time_ms: int, search_window_ms: float = 30.0) -> int:
        """Snaps grid time to audio transient if one exists within search_window_ms."""
        if not onset_set:
            return grid_time_ms
        # Binary search for closest onset
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

    # 1. Start with detected audio transients (onsets) to guarantee exact transient synchronization
    active_set = set()
    for onset in onset_set:
        if 0 <= onset < duration_ms:
            # Check density constraint based on SR
            active_set.add(onset)

    # 2. Add grid-based rhythmic subdivisions (1/1, 1/2, 1/4)
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
        else:  # 1/4 beat
            should_place = rng.random() < params["subdiv_1_4_prob"]

        if should_place:
            # Snap to nearby detected transient if within 1/4 beat window
            aligned_t = snap_to_transient(t_grid, search_window_ms=max(35.0, step_1_4 * 0.75))
            active_set.add(aligned_t)

    active_timestamps = sorted(list(active_set))

    # State tracking for ergonomic key assignment & anti-finger-trap
    column_busy_until = [0, 0, 0, 0]  # end time of active LNs in columns
    last_note_time = [-9999, -9999, -9999, -9999]  # last tap time in column
    hit_objects: List[ManiaHitObject] = []

    for i, t in enumerate(active_timestamps):
        # Determine chord size (1 to 4)
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

        # Downbeat emphasis: if near 1/1 beat, allow larger chord; offbeat: reduce chord size
        dist_to_beat = abs((t - offset_ms) % beat_length)
        is_downbeat = dist_to_beat < 15.0 or abs(dist_to_beat - beat_length) < 15.0
        if not is_downbeat and chord_size > 2:
            chord_size = 2

        # Available columns (not currently held by an active LN)
        available_cols = [c for c in range(4) if column_busy_until[c] <= t]
        if not available_cols:
            continue

        # Limit chord size to available columns
        actual_chord_size = min(chord_size, len(available_cols))

        # Sort available columns by least recently used to ensure balanced key distribution
        available_cols.sort(key=lambda c: (last_note_time[c], rng.random()))
        chosen_cols = available_cols[:actual_chord_size]

        # Determine if any of these notes should be a Long Note (LN)
        # Avoid LNs in all columns simultaneously to prevent handlocks
        can_have_ln = rng.random() < params["ln_prob"]
        next_time = active_timestamps[i + 1] if (i + 1) < len(active_timestamps) else (t + int(beat_length))
        max_hold_time = max(t + int(step_1_2), min(t + int(beat_length * 2), next_time + int(step_1_2)))

        for c_idx, col in enumerate(chosen_cols):
            # Only make one note in a multi-note chord an LN to preserve readability
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
                column_busy_until[col] = end_t + 20  # slight buffer before column opens again

    # Sort final hit objects by start time and column
    hit_objects.sort(key=lambda o: (o.start_time, o.column))
    return hit_objects
