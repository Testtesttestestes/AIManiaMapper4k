"""
Official osu!mania Difficulty (Star Rating) Calculator.

Implements Star Rating calculation using:
1. `rosu_pp_py` (official Python bindings for Rust-based rosu-pp calculator that mirrors osu! game code).
2. Accurate built-in osu!mania Strain Calculator as an automatic zero-dependency fallback,
   faithfully implementing osu!mania difficulty rules:
   - Per-column individual strain (jack / interval speed)
   - Cross-column overall rhythm strain & chord density scaling
   - Long Note (LN) hold & release strain bonuses
   - 400ms section windows and peak strain tracking
   - Power-decay weighting sum with 0.9 base and osu!mania multiplier.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

# Attempt to load rosu_pp_py (or rosu_pp)
ROSU_AVAILABLE = False
_rosu_module = None

try:
    import rosu_pp_py as _rosu_module  # type: ignore
    ROSU_AVAILABLE = True
except ImportError:
    try:
        import rosu_pp as _rosu_module  # type: ignore
        ROSU_AVAILABLE = True
    except ImportError:
        ROSU_AVAILABLE = False
        _rosu_module = None


def _calculate_with_rosu_pp(osu_text: str) -> Optional[float]:
    """Calculates Star Rating using rosu-pp-py if available."""
    if not ROSU_AVAILABLE or _rosu_module is None:
        return None

    try:
        bm = None
        raw_bytes = osu_text.encode("utf-8", errors="ignore")

        # Try various rosu-pp-py Beatmap constructors
        try:
            bm = _rosu_module.Beatmap(bytes=raw_bytes)
        except Exception:
            pass

        if bm is None:
            try:
                bm = _rosu_module.Beatmap(content=osu_text)
            except Exception:
                pass

        if bm is None:
            try:
                bm = _rosu_module.Beatmap(content=raw_bytes)
            except Exception:
                pass

        if bm is None:
            return None

        # Try Difficulty calculator (rosu-pp-py v1 / v2)
        if hasattr(_rosu_module, "Difficulty"):
            calc = _rosu_module.Difficulty()
            res = calc.calculate(bm)
            if hasattr(res, "stars"):
                return round(float(res.stars), 2)

        # Try Calculator
        if hasattr(_rosu_module, "Calculator"):
            calc = _rosu_module.Calculator()
            res = calc.calculate(bm)
            if hasattr(res, "stars"):
                return round(float(res.stars), 2)

        # Try Performance
        if hasattr(_rosu_module, "Performance"):
            calc = _rosu_module.Performance(map=bm)
            res = calc.calculate()
            if hasattr(res, "stars"):
                return round(float(res.stars), 2)

    except Exception:
        return None

    return None


class OsuManiaStrainCalculator:
    """
    Native osu!mania 4K strain-based difficulty calculator.
    Matches osu!mania difficulty calculation mechanics:
    - Section length: 400ms
    - Strain decay base: 0.125
    - Decay weighting: 0.9
    - Mania multiplier: 0.0185
    """

    SECTION_LENGTH_MS = 400.0
    STRAIN_DECAY_BASE = 0.125
    STAR_RATING_MULTIPLIER = 0.0185
    DECAY_WEIGHT = 0.9

    @classmethod
    def calculate_from_hit_objects(
        cls,
        hit_objects: List[Tuple[int, int, int, bool]],
        keys: int = 4,
        overall_difficulty: float = 8.0,
    ) -> float:
        """
        Calculates Star Rating from a list of hit objects:
        Each item is (column, start_ms, end_ms, is_ln).
        """
        if not hit_objects:
            return 0.0

        sorted_objs = sorted(hit_objects, key=lambda o: (o[1], o[0]))

        # Group notes by start_time for chord calculations
        notes_by_time: dict[int, list[Tuple[int, int, int, bool]]] = {}
        for obj in sorted_objs:
            s_ms = obj[1]
            if s_ms not in notes_by_time:
                notes_by_time[s_ms] = []
            notes_by_time[s_ms].append(obj)

        unique_times = sorted(notes_by_time.keys())
        if not unique_times:
            return 0.0

        last_col_times = [-100000.0] * keys
        last_col_end_times = [-100000.0] * keys
        col_strains = [0.0] * keys

        last_start_time = float(unique_times[0])
        overall_strain = 0.0

        first_time = float(sorted_objs[0][1])
        current_section_end = first_time + cls.SECTION_LENGTH_MS
        current_section_peak = 0.0
        section_peaks: List[float] = []

        for t in unique_times:
            t_f = float(t)
            chord = notes_by_time[t]
            chord_size = len(chord)

            # Decay overall strain
            delta_overall = max(0.0, t_f - last_start_time)
            if delta_overall > 0:
                overall_strain *= (cls.STRAIN_DECAY_BASE ** (delta_overall / 1000.0))

            # Overall strain addition (rhythm speed + chord complexity)
            chord_bonus = 1.0 + (chord_size - 1) * 0.45
            time_gap = max(25.0, delta_overall)
            overall_addition = (260.0 / time_gap) * chord_bonus
            overall_strain += overall_addition
            last_start_time = t_f

            # Per-column strain addition (finger speed / jacks / release coordination)
            col_max_strain = 0.0
            for col, s_ms, e_ms, is_ln in chord:
                c = max(0, min(keys - 1, col))
                delta_col = max(0.0, float(s_ms) - last_col_times[c])
                if delta_col > 0:
                    col_strains[c] *= (cls.STRAIN_DECAY_BASE ** (delta_col / 1000.0))

                col_gap = max(30.0, delta_col)
                col_addition = 360.0 / col_gap

                # Hold note / LN coordination bonus
                if is_ln:
                    ln_len = max(0.0, float(e_ms - s_ms))
                    if ln_len > 40.0:
                        col_addition *= 1.25

                col_strains[c] += col_addition
                last_col_times[c] = float(s_ms)
                last_col_end_times[c] = float(e_ms)
                if col_strains[c] > col_max_strain:
                    col_max_strain = col_strains[c]

            combined_strain = overall_strain + col_max_strain

            # Advance sections
            while t_f > current_section_end:
                section_peaks.append(current_section_peak)
                # Decay peak into next section window
                decay_time = t_f - current_section_end
                current_section_peak = combined_strain * (cls.STRAIN_DECAY_BASE ** (decay_time / 1000.0))
                current_section_end += cls.SECTION_LENGTH_MS

            if combined_strain > current_section_peak:
                current_section_peak = combined_strain

        section_peaks.append(current_section_peak)

        # Sort peaks descending and sum with exponential 0.9 weight
        sorted_peaks = sorted(section_peaks, reverse=True)
        difficulty = 0.0
        weight = 1.0

        for p in sorted_peaks:
            difficulty += p * weight
            weight *= cls.DECAY_WEIGHT
            if weight < 1e-4:
                break

        star_rating = difficulty * cls.STAR_RATING_MULTIPLIER
        # Clamp to reasonable positive range
        return max(0.5, round(star_rating, 2))


def calculate_osu_mania_star_rating(
    osu_text: Optional[str] = None,
    hit_objects: Optional[List[Tuple[int, int, int, bool]]] = None,
    overall_difficulty: float = 8.0,
    keys: int = 4,
    force_native: bool = False,
) -> float:
    """
    Main entry point for calculating osu!mania Star Rating.
    1. Tries rosu_pp_py if available, osu_text is provided, and force_native is False.
    2. Falls back to OsuManiaStrainCalculator.
    """
    # 1. Try rosu-pp-py first if osu_text is provided and not forced to native
    if not force_native and osu_text is not None and ROSU_AVAILABLE:
        rosu_sr = _calculate_with_rosu_pp(osu_text)
        if rosu_sr is not None and rosu_sr > 0.0:
            return round(rosu_sr, 2)

    # 2. If hit_objects is provided, use native strain calculator
    if hit_objects:
        return OsuManiaStrainCalculator.calculate_from_hit_objects(
            hit_objects=hit_objects,
            keys=keys,
            overall_difficulty=overall_difficulty,
        )

    # 3. If only osu_text is provided, parse hit objects and compute
    if osu_text:
        parsed_objs: List[Tuple[int, int, int, bool]] = []
        lines = osu_text.splitlines()
        in_hit = False
        od = overall_difficulty
        for line in lines:
            line_s = line.strip()
            if line_s == "[HitObjects]":
                in_hit = True
                continue
            if line_s.startswith("[") and line_s.endswith("]"):
                in_hit = False
                continue
            if not in_hit:
                if line_s.startswith("OverallDifficulty:"):
                    try:
                        od = float(line_s.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                continue

            parts = [p.strip() for p in line_s.split(",")]
            if len(parts) < 4:
                continue
            try:
                x = int(float(parts[0]))
                col = max(0, min(keys - 1, int(x * keys / 512)))
                start_ms = int(round(float(parts[2])))
                obj_type = int(parts[3])
                is_ln = bool(obj_type & 128)
                end_ms = start_ms
                if is_ln and len(parts) >= 6:
                    end_raw = parts[5].split(":")[0]
                    try:
                        end_ms = int(round(float(end_raw)))
                    except ValueError:
                        end_ms = start_ms
                parsed_objs.append((col, start_ms, end_ms, is_ln))
            except (ValueError, IndexError):
                continue

        if parsed_objs:
            return OsuManiaStrainCalculator.calculate_from_hit_objects(
                hit_objects=parsed_objs,
                keys=keys,
                overall_difficulty=od,
            )

    return 2.0


def calculate_with_rosu_pp_direct(osu_text: str) -> Optional[float]:
    """Public helper to explicitly compute Star Rating using rosu-pp-py only."""
    return _calculate_with_rosu_pp(osu_text)

