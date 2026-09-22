"""
Timing points and Slider Velocity (SV) dynamics generator for osu!mania 4K.

Generates:
- Uninherited Timing Points (Red lines): fundamental tempo and time signature
- Inherited Timing Points (Green lines): dynamic SV speed adjustments responding to RMS audio intensity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


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


def generate_timing_points(
    bpm: float,
    offset_ms: int,
    duration_ms: int,
    rms_profile: List[Tuple[int, float]] = None,
    sv_dynamic_intensity: float = 0.25,
) -> List[TimingPoint]:
    """
    Constructs uninherited BPM point and inherited SV timing points throughout the song.
    
    Args:
        bpm: Tempo in beats per minute.
        offset_ms: Millisecond of the first beat.
        duration_ms: Song duration in milliseconds.
        rms_profile: List of (time_ms, normalized_rms) tuples.
        sv_dynamic_intensity: Depth of SV fluctuation (0.0 = static 1.0x SV, 0.3 = 0.85x to 1.3x).
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
    # In mania, 1 measure = 4 beats. We place SV changes every 2 to 4 measures or at significant RMS shifts.
    measure_length_ms = beat_length * 4.0
    if rms_profile and duration_ms > measure_length_ms:
        # Group RMS values per measure
        current_time = offset_ms
        prev_sv = 1.0

        while current_time < duration_ms:
            # Find average RMS in the upcoming measure
            m_end = current_time + measure_length_ms
            relevant_rms = [rms for (t, rms) in rms_profile if current_time <= t < m_end]
            avg_rms = (sum(relevant_rms) / len(relevant_rms)) if relevant_rms else 0.5

            # Map normalized RMS (0.0 - 1.0) to SV multiplier (e.g. 0.85x to 1.25x)
            # base = 1.0
            sv_mult = 1.0 + (avg_rms - 0.5) * (2.0 * sv_dynamic_intensity)
            sv_mult = max(0.75, min(1.40, round(sv_mult, 2)))

            # If SV changed substantially from previous, emit inherited point
            if abs(sv_mult - prev_sv) >= 0.05 or current_time == offset_ms:
                # osu! format: beatLength = -100.0 / sv_multiplier
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

            current_time += int(round(measure_length_ms * 2))  # adjust every 2 measures

    return points
