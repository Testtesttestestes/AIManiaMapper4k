"""
Timing points, musical grid tick generator, and Slider Velocity (SV) dynamics for osu!mania 4K.
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
        self.meter = int(meter) if int(meter) > 0 else 4
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


@dataclass
class MusicalGridTick:
    time_ms: float
    bpm: float
    beat_len_ms: float
    meter: int                    # e.g. 4 for 4/4, 3 for 3/4, 6 for 6/8
    divisor: int                  # 1, 2, 4, 8 (or 3, 6 for triplet meter)
    divisor_fraction: float       # 1.0, 0.5, 0.25, 0.125 (or 0.333, 0.166)
    subdivision_index: int        # 0 .. subdivisions_per_beat - 1
    subdivisions_per_beat: int    # 8 for binary meter (1/8 steps), 6 for ternary meter (1/6 steps)
    beat_in_measure: int          # 0 .. meter - 1
    bar_phase: float              # 0.0 .. 1.0 within measure
    is_downbeat: bool             # True if start of measure (beat 0, tick 0)
    is_beat_start: bool           # True if start of 1/1 beat (tick 0)
    is_ternary: bool              # True if section is triplet / compound (3/4, 6/8, 12/8)


def generate_musical_grid_ticks_from_timing_points(
    timing_points: List[TimingPoint | Tuple[int, float, bool] | Tuple[int, float, int, bool]],
    duration_ms: int | float,
    default_bpm: float = 120.0,
    default_offset_ms: float = 0.0,
    subdiv_binary: int = 8,   # 8 subdivisions per beat -> 1/1, 1/2, 1/4, 1/8 ticks
    subdiv_ternary: int = 6,  # 6 subdivisions per beat -> 1/1, 1/3, 1/6 ticks
) -> List[MusicalGridTick]:
    """
    Constructs a deterministic sequence of musical grid decision points strictly aligned
    to all BPM changes, offsets, meters (4/4, 3/4, 6/8, etc.), and fractional beat ticks.
    """
    uninherited_tps: List[Tuple[float, float, int]] = []
    
    for tp in timing_points:
        if isinstance(tp, TimingPoint):
            if tp.uninherited:
                b_len = max(50.0, float(tp.beat_length))
                m = tp.meter if tp.meter > 0 else 4
                uninherited_tps.append((float(tp.time), b_len, m))
        elif isinstance(tp, (tuple, list)):
            if len(tp) >= 3:
                t_ms = float(tp[0])
                b_len = float(tp[1])
                if len(tp) >= 4:
                    meter = int(tp[2]) if int(tp[2]) > 0 else 4
                    uninh = bool(tp[3])
                else:
                    meter = 4
                    uninh = bool(tp[2])
                if uninh and b_len > 0:
                    uninherited_tps.append((t_ms, b_len, meter))

    if not uninherited_tps:
        base_b_len = 60000.0 / max(30.0, default_bpm)
        uninherited_tps = [(float(default_offset_ms), base_b_len, 4)]
    else:
        uninherited_tps.sort(key=lambda x: x[0])

    ticks: List[MusicalGridTick] = []
    max_duration = float(duration_ms)

    for idx, (tp_time, tp_beat_len, tp_meter) in enumerate(uninherited_tps):
        next_tp_time = uninherited_tps[idx + 1][0] if idx + 1 < len(uninherited_tps) else max_duration + (tp_beat_len * 4.0)
        section_end_ms = min(max_duration + (tp_beat_len * 2.0), next_tp_time)

        bpm = 60000.0 / tp_beat_len
        is_ternary = (tp_meter % 3 == 0)
        subdivs = subdiv_ternary if is_ternary else subdiv_binary
        step_ms = tp_beat_len / float(subdivs)
        bar_len_ms = tp_beat_len * float(tp_meter)

        if idx == 0:
            start_t = tp_time
            while start_t > 0:
                start_t -= bar_len_ms
            current_t = start_t
        else:
            current_t = tp_time

        while current_t < section_end_ms:
            if current_t >= 0:
                rel_t = current_t - tp_time
                total_steps = round(rel_t / step_ms)
                
                steps_per_bar = subdivs * tp_meter
                step_in_bar = total_steps % steps_per_bar
                if step_in_bar < 0:
                    step_in_bar += steps_per_bar

                beat_in_measure = step_in_bar // subdivs
                subdiv_idx = step_in_bar % subdivs

                if is_ternary:
                    if subdiv_idx == 0:
                        div = 1
                        div_frac = 1.0
                    elif subdiv_idx in (2, 4):
                        div = 3
                        div_frac = 1.0 / 3.0
                    else:
                        div = 6
                        div_frac = 1.0 / 6.0
                else:
                    if subdiv_idx == 0:
                        div = 1
                        div_frac = 1.0
                    elif subdiv_idx == 4:
                        div = 2
                        div_frac = 0.5
                    elif subdiv_idx in (2, 6):
                        div = 4
                        div_frac = 0.25
                    else:
                        div = 8
                        div_frac = 0.125

                bar_phase = float(step_in_bar) / float(steps_per_bar)
                is_downbeat = (step_in_bar == 0)
                is_beat_start = (subdiv_idx == 0)

                if not ticks or abs(current_t - ticks[-1].time_ms) > (step_ms * 0.4):
                    ticks.append(
                        MusicalGridTick(
                            time_ms=round(current_t, 2),
                            bpm=round(bpm, 2),
                            beat_len_ms=round(tp_beat_len, 2),
                            meter=tp_meter,
                            divisor=div,
                            divisor_fraction=round(div_frac, 4),
                            subdivision_index=subdiv_idx,
                            subdivisions_per_beat=subdivs,
                            beat_in_measure=beat_in_measure,
                            bar_phase=round(bar_phase, 4),
                            is_downbeat=is_downbeat,
                            is_beat_start=is_beat_start,
                            is_ternary=is_ternary,
                        )
                    )

            current_t += step_ms

    ticks.sort(key=lambda t: t.time_ms)
    return ticks


def snap_to_musical_grid(
    raw_time_ms: int | float,
    bpm: float,
    offset_ms: int | float,
    divisors: Tuple[int, ...] = (1, 2, 4, 3, 6, 8),
    max_snap_dist_ms: float = 24.0,
) -> int:
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
    beat_length = 60000.0 / max(1.0, bpm)
    points: List[TimingPoint] = []

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
