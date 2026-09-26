#!/usr/bin/env python3
"""
Dataset Compatibility & Rhythm Feasibility Auditor for osu!mania 4K AI Mapper.

Audits raw .osz / .osu beatmaps or dataset sources to determine:
1. Quantization & Grid Compatibility:
   - Does the discrete tick generator (1/4, 1/8, 1/3, 1/6) capture notes losslessly?
   - Quantization residuals (ms delta between real notes and discrete ticks).
   - Tick collision rate (notes collapsed into the same tick slot = training data loss).
2. Polyrhythm & Meter Switching:
   - Binary (4/4, 1/4, 1/8) vs Ternary (1/3, 1/6, 12/8) vs Irregular (1/5, 1/7, 1/12, 1/16).
   - Multi-BPM maps and variable meter frequency.
3. Long Note (LN) Anatomy & Gaps:
   - Zero-gap LN chains (tail ending at exact ms where next note starts in same lane).
   - Overlaps (note inside active LN on same lane).
   - LN length distribution.
4. Lane Distribution & Hand Balance:
   - Left hand (Lane 0, 1) vs Right hand (Lane 2, 3) balance.
5. Overall Feasibility Score:
   - Identifies which maps confuse the Transformer and provides concrete remediation steps.

Usage:
    python audit_dataset_compatibility.py
    python audit_dataset_compatibility.py --dir "sample beatmaps"
    python audit_dataset_compatibility.py --map "sample beatmaps/mymap.osz"
    python audit_dataset_compatibility.py --out report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from osu_mapper.dataset.feature_extractor import parse_osu_for_training
from osu_mapper.dataset.source_resolver import resolve_beatmap_sources
from osu_mapper.pipeline.timing import (
    MusicalGridTick,
    generate_musical_grid_ticks_from_timing_points,
)


@dataclass
class MapAuditResult:
    map_name: str
    osz_name: str
    star_rating: float
    bpm: float
    timing_points_count: int
    is_multi_bpm: bool
    total_notes: int
    ln_count: int
    ln_ratio_pct: float
    
    # Quantization metrics
    avg_quantization_error_ms: float
    max_quantization_error_ms: float
    notes_exact_grid_pct: float     # Error <= 2ms
    notes_acceptable_pct: float     # Error <= 10ms
    notes_off_grid_pct: float       # Error > 15ms (out of standard grid)
    
    # Tick collisions (loss of notes due to same-tick quantization)
    tick_collisions_count: int
    tick_collision_rate_pct: float
    
    # Rhythm divisor breakdown
    binary_notes_pct: float         # 1/1, 1/2, 1/4, 1/8
    triplet_notes_pct: float        # 1/3, 1/6, 1/12
    irregular_notes_pct: float      # 1/5, 1/7, unaligned freehand
    
    # LN & Pattern physical anomalies
    zero_gap_ln_count: int          # Next note in same lane starts <= 5ms after LN end
    overlap_note_count: int         # Note starts while LN is still active in same lane
    lane_distribution: List[int]    # [Col0, Col1, Col2, Col3]
    lane_balance_ratio: float       # Left (0,1) vs Right (2,3) ratio
    
    # Feasibility grading
    compatibility_score: float      # 0 to 100
    compatibility_grade: str        # 'EXCELLENT', 'GOOD', 'COMPLEX', 'PROBLEMATIC'
    warnings: List[str]


def determine_note_divisor(time_ms: float, bpm: float, offset_ms: float) -> Tuple[str, float]:
    """Identifies the closest musical divisor (1/1, 1/2, 1/3, 1/4, 1/6, 1/8, etc.) for a timestamp."""
    beat_len = 60000.0 / max(20.0, bpm)
    delta_from_offset = time_ms - offset_ms
    beat_fraction = (delta_from_offset / beat_len) % 1.0
    if beat_fraction < 0:
        beat_fraction += 1.0
    
    # Standard candidate fractions
    candidates = [
        ("1/1", 0.0),
        ("1/1", 1.0),
        ("1/2", 0.5),
        ("1/4", 0.25),
        ("1/4", 0.75),
        ("1/8", 0.125),
        ("1/8", 0.375),
        ("1/8", 0.625),
        ("1/8", 0.875),
        ("1/3", 1.0 / 3.0),
        ("1/3", 2.0 / 3.0),
        ("1/6", 1.0 / 6.0),
        ("1/6", 5.0 / 6.0),
        ("1/12", 1.0 / 12.0),
        ("1/12", 5.0 / 12.0),
        ("1/12", 7.0 / 12.0),
        ("1/12", 11.0 / 12.0),
        ("1/16", 1.0 / 16.0),
    ]
    
    best_div = "irregular"
    best_dist_ms = 99999.0
    
    for div_name, target_frac in candidates:
        frac_diff = abs(beat_fraction - target_frac)
        frac_diff = min(frac_diff, 1.0 - frac_diff)
        dist_ms = frac_diff * beat_len
        if dist_ms < best_dist_ms:
            best_dist_ms = dist_ms
            best_div = div_name
            
    if best_dist_ms > 12.0:
        best_div = "irregular"
        
    return best_div, best_dist_ms


def audit_beatmap_content(content: str, map_name: str, osz_name: str) -> Optional[MapAuditResult]:
    details = parse_osu_for_training(content, map_name)
    if not details or len(details.hit_objects) == 0:
        return None
        
    hit_objs = sorted(details.hit_objects, key=lambda o: (o[1], o[0]))
    total_notes = len(hit_objs)
    ln_count = sum(1 for o in hit_objs if o[3] and o[2] > o[1])
    ln_ratio = (ln_count / total_notes) * 100.0 if total_notes > 0 else 0.0
    
    min_t = hit_objs[0][1]
    max_t = max(o[2] for o in hit_objs)
    duration_ms = max(5000.0, float(max_t - min_t + 2000.0))
    
    # 1. Generate standard ticks as used by current generator
    ticks = generate_musical_grid_ticks_from_timing_points(
        timing_points=details.timing_points,
        duration_ms=duration_ms,
        default_bpm=details.bpm,
        default_offset_ms=details.offset_ms,
        subdiv_binary=8,
        subdiv_ternary=6,
    )
    
    if len(ticks) == 0:
        return None
        
    tick_times = [t.time_ms for t in ticks]
    
    # 2. Measure Quantization Errors & Collisions
    quant_errors: List[float] = []
    occupied_slots: Dict[Tuple[int, int], int] = {}  # (tick_idx, col) -> count
    collisions = 0
    
    exact_count = 0
    acceptable_count = 0
    off_grid_count = 0
    
    binary_div_count = 0
    triplet_div_count = 0
    irregular_div_count = 0
    
    lane_counts = [0, 0, 0, 0]
    
    import bisect

    for col, start_ms, end_ms, is_ln in hit_objs:
        if 0 <= col < 4:
            lane_counts[col] += 1
            
        # Find closest tick using binary search
        idx = bisect.bisect_left(tick_times, start_ms)
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(tick_times):
            candidates.append(idx)
        if idx + 1 < len(tick_times):
            candidates.append(idx + 1)
            
        best_tick_idx = candidates[0]
        best_err = abs(tick_times[best_tick_idx] - start_ms)
        for cand in candidates[1:]:
            err = abs(tick_times[cand] - start_ms)
            if err < best_err:
                best_err = err
                best_tick_idx = cand
                
        err_ms = float(best_err)
        quant_errors.append(err_ms)
        
        if err_ms <= 2.0:
            exact_count += 1
        elif err_ms <= 10.0:
            acceptable_count += 1
        else:
            off_grid_count += 1
            
        # Check collision on the same tick slot
        slot_key = (best_tick_idx, col)
        if slot_key in occupied_slots:
            collisions += 1
            occupied_slots[slot_key] += 1
        else:
            occupied_slots[slot_key] = 1
            
        # Divisor categorization
        div_name, _ = determine_note_divisor(start_ms, details.bpm, details.offset_ms)
        if div_name in ("1/1", "1/2", "1/4", "1/8", "1/16"):
            binary_div_count += 1
        elif div_name in ("1/3", "1/6", "1/12"):
            triplet_div_count += 1
        else:
            irregular_div_count += 1

    # 3. Analyze LN zero-gaps & overlaps
    zero_gaps = 0
    overlaps = 0
    lane_objects: Dict[int, List[Tuple[int, int, bool]]] = {0: [], 1: [], 2: [], 3: []}
    for col, s_ms, e_ms, is_ln in hit_objs:
        if 0 <= col < 4:
            lane_objects[col].append((s_ms, e_ms, is_ln))
            
    for col in range(4):
        objs = lane_objects[col]
        for i in range(len(objs) - 1):
            cur_s, cur_e, cur_is_ln = objs[i]
            next_s, next_e, next_is_ln = objs[i + 1]
            
            if cur_is_ln and cur_e > cur_s:
                # Check overlap (next note starts while current is still held)
                if next_s < cur_e - 10:
                    overlaps += 1
                # Check zero-gap (next note starts exactly or within 5ms of release)
                elif abs(next_s - cur_e) <= 5:
                    zero_gaps += 1

    # 4. Lane balance
    left_hand = lane_counts[0] + lane_counts[1]
    right_hand = lane_counts[2] + lane_counts[3]
    lane_balance = (left_hand / max(1, right_hand)) if right_hand > 0 else 99.0
    
    # 5. Timing Points & Multi-BPM
    uninherited_tps = [
        tp for tp in details.timing_points
        if (getattr(tp, 'uninherited', 1) in (1, True)) or (isinstance(tp, (list, tuple)) and (len(tp) < 3 or bool(tp[2])))
    ]
    is_multi_bpm = len(uninherited_tps) > 1
    
    # 6. Feasibility Score Calculation
    score = 100.0
    warnings: List[str] = []
    
    # Quantization error penalty
    off_grid_pct = (off_grid_count / total_notes) * 100.0
    score -= off_grid_pct * 1.5
    if off_grid_pct > 10.0:
        warnings.append(f"High off-grid rate ({off_grid_pct:.1f}% notes > 10ms from grid)")
        
    # Collision penalty
    collision_pct = (collisions / total_notes) * 100.0
    score -= collision_pct * 3.0
    if collisions > 0:
        warnings.append(f"Tick collisions ({collisions} notes [{collision_pct:.1f}%] collapse onto same tick)")
        
    # Polyrhythm warning
    triplet_pct = (triplet_div_count / total_notes) * 100.0
    irregular_pct = (irregular_div_count / total_notes) * 100.0
    if triplet_pct > 20.0:
        warnings.append(f"Heavy Triplet/Ternary meter ({triplet_pct:.1f}% 1/3 or 1/6 notes)")
    if irregular_pct > 15.0:
        score -= irregular_pct * 0.8
        warnings.append(f"Irregular swing / unaligned rhythm ({irregular_pct:.1f}% irregular notes)")
        
    # Overlap and zero-gap warnings
    if overlaps > 0:
        score -= min(15.0, overlaps * 2.0)
        warnings.append(f"Physical Overlaps in dataset ({overlaps} notes placed inside active LN)")
    if zero_gaps > (total_notes * 0.05):
        warnings.append(f"Excessive Zero-Gap LNs ({zero_gaps} release-to-press transitions without pause)")
        
    # Multi-BPM warning
    if is_multi_bpm:
        warnings.append(f"Multi-BPM map ({len(uninherited_tps)} tempo changes)")
        
    # Lane skew warning
    if lane_balance > 1.8 or lane_balance < 0.55:
        warnings.append(f"Heavy Lane Skew (Left/Right balance ratio = {lane_balance:.2f})")
        
    score = max(0.0, min(100.0, score))
    
    if score >= 90.0:
        grade = "EXCELLENT"
    elif score >= 75.0:
        grade = "GOOD"
    elif score >= 55.0:
        grade = "COMPLEX"
    else:
        grade = "PROBLEMATIC"
        
    return MapAuditResult(
        map_name=details.version,
        osz_name=osz_name,
        star_rating=details.star_rating,
        bpm=details.bpm,
        timing_points_count=len(details.timing_points),
        is_multi_bpm=is_multi_bpm,
        total_notes=total_notes,
        ln_count=ln_count,
        ln_ratio_pct=round(ln_ratio, 1),
        avg_quantization_error_ms=round(float(sum(quant_errors) / max(1, len(quant_errors))), 2),
        max_quantization_error_ms=round(float(max(quant_errors) if quant_errors else 0.0), 2),
        notes_exact_grid_pct=round((exact_count / total_notes) * 100.0, 1),
        notes_acceptable_pct=round(((exact_count + acceptable_count) / total_notes) * 100.0, 1),
        notes_off_grid_pct=round(off_grid_pct, 1),
        tick_collisions_count=collisions,
        tick_collision_rate_pct=round(collision_pct, 2),
        binary_notes_pct=round((binary_div_count / total_notes) * 100.0, 1),
        triplet_notes_pct=round(triplet_pct, 1),
        irregular_notes_pct=round(irregular_pct, 1),
        zero_gap_ln_count=zero_gaps,
        overlap_note_count=overlaps,
        lane_distribution=lane_counts,
        lane_balance_ratio=round(lane_balance, 2),
        compatibility_score=round(score, 1),
        compatibility_grade=grade,
        warnings=warnings,
    )


def audit_all_dataset(source_dir: str | Path) -> List[MapAuditResult]:
    source_dir = Path(source_dir)
    osz_files, song_folders = resolve_beatmap_sources(source_dir)
    results: List[MapAuditResult] = []

    # 1. Parse from OSZ files
    for osz in osz_files:
        try:
            with zipfile.ZipFile(osz, "r") as zf:
                for name in zf.namelist():
                    if name.lower().endswith(".osu"):
                        try:
                            content = zf.read(name).decode("utf-8-sig")
                        except UnicodeDecodeError:
                            content = zf.read(name).decode("latin-1", errors="ignore")
                        res = audit_beatmap_content(content, name, osz.name)
                        if res:
                            results.append(res)
        except Exception as e:
            print(f"[Warn] Failed reading {osz.name}: {e}")

    # 2. Parse from uncompressed song folders
    for folder in song_folders:
        for osu_file in folder.glob("*.osu"):
            try:
                try:
                    content = osu_file.read_text(encoding="utf-8-sig")
                except UnicodeDecodeError:
                    content = osu_file.read_text(encoding="latin-1", errors="ignore")
                res = audit_beatmap_content(content, osu_file.name, folder.name)
                if res:
                    results.append(res)
            except Exception as e:
                print(f"[Warn] Failed reading {osu_file.name}: {e}")

    return results


def print_audit_report(results: List[MapAuditResult]) -> None:
    if not results:
        print("\n[!] No valid osu!mania 4K beatmaps found to audit.")
        return

    total_maps = len(results)
    total_notes = sum(r.total_notes for r in results)
    total_lns = sum(r.ln_count for r in results)
    avg_ln_pct = sum(r.ln_ratio_pct for r in results) / total_maps
    
    avg_quant_err = sum(r.avg_quantization_error_ms for r in results) / total_maps
    avg_on_grid = sum(r.notes_acceptable_pct for r in results) / total_maps
    avg_off_grid = sum(r.notes_off_grid_pct for r in results) / total_maps
    total_collisions = sum(r.tick_collisions_count for r in results)
    avg_collision_rate = sum(r.tick_collision_rate_pct for r in results) / total_maps
    
    binary_maps = sum(1 for r in results if r.triplet_notes_pct < 5.0 and r.irregular_notes_pct < 5.0)
    triplet_maps = sum(1 for r in results if r.triplet_notes_pct >= 15.0)
    multi_bpm_maps = sum(1 for r in results if r.is_multi_bpm)
    total_zero_gaps = sum(r.zero_gap_ln_count for r in results)
    total_overlaps = sum(r.overlap_note_count for r in results)
    
    grade_counts = {
        "EXCELLENT": sum(1 for r in results if r.compatibility_grade == "EXCELLENT"),
        "GOOD": sum(1 for r in results if r.compatibility_grade == "GOOD"),
        "COMPLEX": sum(1 for r in results if r.compatibility_grade == "COMPLEX"),
        "PROBLEMATIC": sum(1 for r in results if r.compatibility_grade == "PROBLEMATIC"),
    }

    print("\n" + "=" * 78)
    print(" 📊 OSU!MANIA 4K DATASET COMPATIBILITY & GENERATOR FEASIBILITY AUDIT")
    print("=" * 78)
    print(f"Total Maps Analyzed:     {total_maps:,}")
    print(f"Total Notes Analyzed:    {total_notes:,} ({total_lns:,} Long Notes | {avg_ln_pct:.1f}% LN ratio)")
    print("-" * 78)
    print("🎯 1. DISCRETE GRID QUANTIZATION & ACCURACY")
    print(f"  • Notes aligned to grid (<=10ms):  {avg_on_grid:.1f}%")
    print(f"  • Notes falling off-grid (>10ms): {avg_off_grid:.1f}%")
    print(f"  • Mean Quantization Delta:        {avg_quant_err:.2f} ms")
    print(f"  • Tick Collisions (Data loss):    {total_collisions:,} notes ({avg_collision_rate:.2f}% avg per map)")
    if total_collisions > 0:
        print(f"    ⚠️ Warning: {total_collisions} notes collapsed into the same tick slot during training preparation!")
    else:
        print("    ✅ Clean: Zero tick collisions detected across the dataset.")

    print("-" * 78)
    print("🎵 2. RHYTHMIC METER & POLYRHYTHMS")
    print(f"  • Pure Binary (4/4, 1/4, 1/8):    {binary_maps} maps ({(binary_maps / total_maps) * 100.0:.1f}%)")
    print(f"  • Triplet/Ternary (1/3, 1/6):     {triplet_maps} maps ({(triplet_maps / total_maps) * 100.0:.1f}%)")
    print(f"  • Multi-BPM / Tempo Shifts:       {multi_bpm_maps} maps ({(multi_bpm_maps / total_maps) * 100.0:.1f}%)")

    print("-" * 78)
    print("🎹 3. LONG NOTE (LN) BEHAVIOR & PHYSICAL RESTRAINTS")
    print(f"  • Zero-Gap LN Transitions:        {total_zero_gaps:,} occurrences (Release -> Press in <=5ms)")
    print(f"  • Physical Note Overlaps:         {total_overlaps:,} occurrences (Notes inside active LN in dataset)")
    if total_zero_gaps > 0:
        print(f"    ℹ️ Insight: The training dataset contains {total_zero_gaps} continuous LN chains.")
        print("               This explains why the model learns to string LNs tail-to-head to boost strain!")

    print("-" * 78)
    print("🏆 4. GENERATOR COMPATIBILITY BREAKDOWN")
    print(f"  • 🟢 EXCELLENT (Flawlessly reproducible):  {grade_counts['EXCELLENT']} maps ({(grade_counts['EXCELLENT'] / total_maps) * 100.0:.1f}%)")
    print(f"  • 🟡 GOOD (Minor quantization delta):      {grade_counts['GOOD']} maps ({(grade_counts['GOOD'] / total_maps) * 100.0:.1f}%)")
    print(f"  • 🟠 COMPLEX (Heavy triplets/Multi-BPM):   {grade_counts['COMPLEX']} maps ({(grade_counts['COMPLEX'] / total_maps) * 100.0:.1f}%)")
    print(f"  • 🔴 PROBLEMATIC (High collisions/off-grid): {grade_counts['PROBLEMATIC']} maps ({(grade_counts['PROBLEMATIC'] / total_maps) * 100.0:.1f}%)")

    print("=" * 78)
    print("📋 SAMPLE BEATMAP DETAILS (Top 8 audited)")
    print(f"{'Map Title / Diff':<32} {'SR':<5} {'BPM':<6} {'OnGrid%':<9} {'Coll':<6} {'Grade':<11} {'Key Warning'}")
    print("-" * 78)
    for r in results[:8]:
        name_short = (r.map_name[:30] + "..") if len(r.map_name) > 30 else r.map_name
        warn_str = r.warnings[0] if r.warnings else "None (Clean)"
        print(f"{name_short:<32} {r.star_rating:<5.2f} {r.bpm:<6.1f} {r.notes_acceptable_pct:<9.1f} {r.tick_collisions_count:<6} {r.compatibility_grade:<11} {warn_str[:25]}")
    
    if len(results) > 8:
        print(f"... and {len(results) - 8} more beatmaps.")

    print("=" * 78)
    print("💡 ACTIONABLE ARCHITECTURAL RECOMMENDATIONS:")
    print("-" * 78)
    
    if avg_off_grid > 5.0 or total_collisions > 0:
        print("1. [Quantization]: Increase subdivision resolution from 8 (1/8) to dynamic tick snapping,")
        print("   or add sub-tick micro-offsets (del_ms) to hit object outputs.")
    else:
        print("1. [Quantization]: Standard 1/8 binary + 1/6 ternary grid ticks are mathematically sufficient for 95%+ notes.")

    if total_zero_gaps > 100:
        print("2. [Zero-Gap LNs]: The dataset itself teaches the neural network zero-gap transitions.")
        print("   -> Apply a minimum Release Gap (e.g. min 40ms buffer) in pattern_generator.py sanitizer.")
        print("   -> Lower ln_threshold from 0.28 to 0.45-0.55 during inference decoding.")

    if total_overlaps > 0:
        print("3. [Sanitizer Overlap Masking]: Implement active lane lock during generation so that a key cannot")
        print("   receive new hit events while an LN is active on that column.")

    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Audit dataset compatibility and rhythm feasibility for osu!mania 4K AI Mapper.")
    parser.add_argument("--dir", "--sample-dir", default="sample beatmaps", help="Directory containing .osz beatmaps or song folders.")
    parser.add_argument("--map", default=None, help="Path to a single .osz or .osu file to audit.")
    parser.add_argument("--out", default="dataset_audit_report.json", help="Path to save JSON audit report.")
    args = parser.parse_args()

    results: List[MapAuditResult] = []

    if args.map:
        map_path = Path(args.map)
        if not map_path.exists():
            print(f"[Error] Map file not found: {map_path}")
            sys.exit(1)
            
        if map_path.suffix.lower() == ".osu":
            try:
                content = map_path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                content = map_path.read_text(encoding="latin-1", errors="ignore")
            r = audit_beatmap_content(content, map_path.name, map_path.parent.name)
            if r:
                results.append(r)
        elif map_path.suffix.lower() == ".osz":
            with zipfile.ZipFile(map_path, "r") as zf:
                for name in zf.namelist():
                    if name.lower().endswith(".osu"):
                        try:
                            content = zf.read(name).decode("utf-8-sig")
                        except UnicodeDecodeError:
                            content = zf.read(name).decode("latin-1", errors="ignore")
                        r = audit_beatmap_content(content, name, map_path.name)
                        if r:
                            results.append(r)
    else:
        sample_dir = Path(args.dir)
        if not sample_dir.exists():
            # Try alternate standard dirs
            for alt in ["sample_beatmaps", "dataset_processed", "."]:
                if Path(alt).is_dir():
                    sample_dir = Path(alt)
                    break
        print(f"[Dataset Audit] Scanning beatmaps in: '{sample_dir.resolve()}' ...")
        results = audit_all_dataset(sample_dir)

    print_audit_report(results)

    # Save to JSON
    if results:
        out_path = Path(args.out)
        data = [asdict(r) for r in results]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Report] Full audit details written to '{out_path.resolve()}'")


if __name__ == "__main__":
    main()
