"""
Timing points and Slider Velocity (SV) dynamics generator for osu!mania 4K.

Generates:
- Uninherited Timing Points (Red lines): fundamental tempo and time signature
- Inherited Timing Points (Green lines): dynamic SV speed adjustments responding to RMS audio intensity
- Musical Grid Snapper: snaps notes to exact fractional beat ticks (1/1, 1/2, 1/3, 1/4, 1/6, 1/8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TimingPoint:
    time: float
    beat_length: float
    meter: int = 4
    sample_set: int = 1
    sample_index: int = 0
    volume: int = 70
    uninherited: int = 1  # 1 for Red line (BPM), 0 for Green line (SV)
    effects: int = 0

    def __init__(
        self,
        time: float = 0.0,
        beat_length: float = 500.0,
        meter: int = 4,
        sample_set: int = 1,
        sample_index: int = 0,
        volume: int = 70,
        uninherited: int | bool = 1,
        effects: int = 0,
        time_ms: Optional[float] = None,
        slider_velocity: Optional[float] = None,
    ):
        self.time = float(time_ms if time_ms is not None else time)
        self.beat_length = float(beat_length)
        self.meter = int(meter)
        self.sample_set = int(sample_set)
        self.sample_index = int(sample_index)
        self.volume = int(volume)
        self.uninherited = 1 if (uninherited is True or uninherited == 1) else 0
        self.effects = int(effects)

    def to_osu_line(self) -> str:
        """Formats the timing point into standard osu! v14 line syntax."""
        return (
            f"{int(round(self.time))},"
            f"{self.beat_length:.4f},"
            f"{self.meter},"
            f"{self.sample_set},"
            f"{self.sample_index},"
            f"{self.volume},"
            f"{self.uninherited},"
            f"{self.effects}"
        )


def snap_to_musical_grid(
    raw_time_ms: int | float,
    bpm: float,
    offset_ms: int | float,
    divisors: Tuple[int, ...] = (1, 2, 4, 3, 6, 8),
    max_snap_dist_ms: float = 24.0,
) -> int:
    """
    Snaps a continuous or frame-based timestamp strictly to the closest musical beat divisor tick.
    Prevents micro-phase jitter (early/late notes) during rhythm game playback.
    """
    beat_len = 60000.0 / max(1.0, bpm)
    rel_time = raw_time_ms - offset_ms

    best_tick = raw_time_ms
    min_dist = float("inf")

    for div in divisors:
        step = beat_len / float(div)
        k = round(rel_time / step)
        cand_time = offset_ms + k * step
        dist = abs(cand_time - raw_time_ms)
        if dist < min_dist:
            min_dist = dist
            best_tick = cand_time

    if min_dist <= max_snap_dist_ms:
        return int(round(best_tick))
    return int(round(raw_time_ms))


def generate_timing_points(
    bpm: float,
    offset_ms: int,
    duration_ms: int,
    rms_profile: Optional[List[Tuple[int, float]]] = None,
    sv_dynamic_intensity: float = 0.25,
) -> List[TimingPoint]:
    """
    Constructs uninherited BPM point and inherited SV timing points throughout the song.
    """
    beat_length = 60000.0 / max(1.0, bpm)
    points: List[TimingPoint] = []

    # 1. Uninherited base timing point (Red Line)
    base_point = TimingPoint(
        time=float(offset_ms),
        beat_length=beat_length,
        meter=4,
        sample_set=1,
        sample_index=0,
        volume=80,
        uninherited=1,
        effects=0,
    )
    points.append(base_point)

    # 2. Inherited SV timing points (Green Lines) across bars
    measure_length_ms = beat_length * 4.0
    if rms_profile and duration_ms > measure_length_ms:
        current_time = offset_ms
        prev_sv = 1.0

        while current_time < duration_ms:
            m_end = current_time + measure_length_ms
            relevant_rms = [rms for (t, rms) in rms_profile if current_time <= t < m_end]
            avg_rms = (sum(relevant_rms) / len(relevant_rms)) if relevant_rms else 0.5

            sv_mult = 1.0 + (avg_rms - 0.5) * (2.0 * sv_dynamic_intensity)
            sv_mult = max(0.75, min(1.40, round(sv_mult, 2)))

            if abs(sv_mult - prev_sv) >= 0.05 or current_time == offset_ms:
                sv_beat_length = -100.0 / sv_mult
                vol = max(40, min(100, int(60 + avg_rms * 40)))
                kiai = 1 if avg_rms > 0.8 else 0

                points.append(
                    TimingPoint(
                        time=float(current_time),
                        beat_length=sv_beat_length,
                        meter=4,
                        sample_set=1,
                        sample_index=0,
                        volume=vol,
                        uninherited=0,
                        effects=kiai,
                    )
                )
                prev_sv = sv_mult

            current_time += int(round(measure_length_ms * 2))

    return points
