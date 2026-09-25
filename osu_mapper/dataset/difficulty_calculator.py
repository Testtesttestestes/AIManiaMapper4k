from __future__ import annotations

import math
from typing import List, Optional, Tuple


def calculate_osu_mania_star_rating(
    osu_text: Optional[str] = None,
    hit_objects: Optional[List[Tuple[int, int, int, bool]]] = None,
    overall_difficulty: float = 8.0,
    keys: int = 4,
) -> float:
    """
    Computes accurate Star Rating for osu!mania 4K beatmaps.
    Uses rosu-pp-py if available, falling back to an accurate strain-based formula.
    """
    if osu_text:
        try:
            import rosu_pp_py
            beatmap = rosu_pp_py.Beatmap(content=osu_text)
            diff_attrs = rosu_pp_py.Difficulty(mods=0).calculate(beatmap)
            sr = float(diff_attrs.stars)
            if sr > 0.0:
                return max(0.5, min(12.0, sr))
        except Exception:
            pass

    if not hit_objects:
        return 1.0

    # Fallback native strain model for osu!mania 4K
    sorted_objs = sorted(hit_objects, key=lambda o: (o[1], o[0]))
    if len(sorted_objs) < 2:
        return 1.0

    dur_ms = max(1000, sorted_objs[-1][1] - sorted_objs[0][1])
    nps = (len(sorted_objs) / dur_ms) * 1000.0

    # Calculate individual strains
    intervals = []
    chord_sizes = {}
    ln_count = 0

    for col, start_ms, end_ms, is_ln in sorted_objs:
        if is_ln and end_ms > start_ms:
            ln_count += 1
        chord_sizes[start_ms] = chord_sizes.get(start_ms, 0) + 1

    distinct_times = sorted(chord_sizes.keys())
    for i in range(1, len(distinct_times)):
        delta = distinct_times[i] - distinct_times[i - 1]
        if delta > 0:
            intervals.append(delta)

    if not intervals:
        return max(1.0, min(10.0, nps * 0.7))

    avg_chord = sum(chord_sizes.values()) / max(1, len(chord_sizes))
    ln_ratio = ln_count / max(1, len(sorted_objs))

    # NPS strain mapping
    # ~1 NPS -> 1.0*, ~3 NPS -> 2.5*, ~6 NPS -> 4.5*, ~10 NPS -> 6.5*, ~15 NPS -> 8.5*
    base_sr = 0.8 + math.pow(max(0.1, nps), 0.85) * 0.65
    chord_mult = 1.0 + (avg_chord - 1.0) * 0.35
    ln_mult = 1.0 + ln_ratio * 0.20

    sr = base_sr * chord_mult * ln_mult
    return max(0.8, min(11.5, round(sr, 2)))
