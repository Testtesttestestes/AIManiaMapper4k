#!/usr/bin/env python3
"""
CLI Tool to Inspect & Calculate exact osu!mania Star Rating using official `rosu-pp-py`.

Usage:
    python compare_difficulty.py --map "path/to/beatmap.osz"
    python compare_difficulty.py --map "path/to/difficulty.osu"
    python compare_difficulty.py --dir "path/to/folder_with_maps"
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from typing import List, Optional, Tuple

from osu_mapper.dataset.difficulty_calculator import (
    calculate_osu_mania_star_rating,
    ensure_rosu_pp_installed,
)


def parse_osu_metadata(osu_text: str) -> dict:
    """Parses basic metadata from .osu file content."""
    meta = {
        "title": "Unknown Title",
        "artist": "Unknown Artist",
        "version": "Normal",
        "creator": "Unknown",
        "mode": 3,  # 3 = osu!mania
        "keys": 4,
        "od": 8.0,
        "notes_count": 0,
        "ln_count": 0,
        "total_objects": 0,
    }

    in_hit_objects = False
    for line in osu_text.splitlines():
        line_str = line.strip()
        if not line_str or line_str.startswith("//"):
            continue

        if line_str == "[HitObjects]":
            in_hit_objects = True
            continue
        elif line_str.startswith("[") and line_str.endswith("]"):
            in_hit_objects = False
            continue

        if not in_hit_objects:
            if line_str.startswith("Title:"):
                meta["title"] = line_str.split(":", 1)[1].strip()
            elif line_str.startswith("Artist:"):
                meta["artist"] = line_str.split(":", 1)[1].strip()
            elif line_str.startswith("Version:"):
                meta["version"] = line_str.split(":", 1)[1].strip()
            elif line_str.startswith("Creator:"):
                meta["creator"] = line_str.split(":", 1)[1].strip()
            elif line_str.startswith("Mode:"):
                try:
                    meta["mode"] = int(line_str.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line_str.startswith("CircleSize:"):
                try:
                    meta["keys"] = int(float(line_str.split(":", 1)[1].strip()))
                except ValueError:
                    pass
            elif line_str.startswith("OverallDifficulty:"):
                try:
                    meta["od"] = float(line_str.split(":", 1)[1].strip())
                except ValueError:
                    pass
        else:
            parts = [p.strip() for p in line_str.split(",")]
            if len(parts) >= 4:
                try:
                    obj_type = int(parts[3])
                    is_ln = bool(obj_type & 128)
                    meta["total_objects"] += 1
                    if is_ln:
                        meta["ln_count"] += 1
                    else:
                        meta["notes_count"] += 1
                except (ValueError, IndexError):
                    pass

    return meta


def process_osu_content(
    osu_text: str, filename: str
) -> Tuple[dict, float]:
    """Computes SR using official rosu-pp."""
    meta = parse_osu_metadata(osu_text)
    meta["filename"] = filename

    # Calculate strictly with rosu-pp
    rosu_sr = calculate_osu_mania_star_rating(
        osu_text=osu_text,
        keys=meta["keys"],
        overall_difficulty=meta["od"],
    )

    return meta, rosu_sr


def inspect_map_file(file_path: str):
    """Processes a single .osz, .osu, or folder of beatmaps."""
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: '{file_path}'")
        return

    # Ensure rosu-pp is active
    rosu = ensure_rosu_pp_installed()
    if rosu is None:
        print("❌ Error: `rosu-pp-py` could not be loaded. Please run 'pip install rosu-pp-py'")
        return

    results = []

    if os.path.isdir(file_path):
        for root, _, files in os.walk(file_path):
            for f in files:
                full = os.path.join(root, f)
                if f.lower().endswith(".osz"):
                    try:
                        with zipfile.ZipFile(full, "r") as z:
                            for name in z.namelist():
                                if name.lower().endswith(".osu"):
                                    content = z.read(name).decode("utf-8", errors="ignore")
                                    results.append(process_osu_content(content, f"{f} -> {name}"))
                    except Exception as e:
                        print(f"⚠️ Failed to read osz '{full}': {e}")
                elif f.lower().endswith(".osu"):
                    try:
                        with open(full, "r", encoding="utf-8", errors="ignore") as f_obj:
                            content = f_obj.read()
                            results.append(process_osu_content(content, f))
                    except Exception as e:
                        print(f"⚠️ Failed to read osu '{full}': {e}")

    elif file_path.lower().endswith(".osz"):
        try:
            with zipfile.ZipFile(file_path, "r") as z:
                osu_files = [n for n in z.namelist() if n.lower().endswith(".osu")]
                if not osu_files:
                    print(f"⚠️ No .osu difficulty files found inside '{file_path}'")
                    return
                for name in osu_files:
                    content = z.read(name).decode("utf-8", errors="ignore")
                    results.append(process_osu_content(content, name))
        except Exception as e:
            print(f"❌ Failed to extract .osz file '{file_path}': {e}")
            return

    elif file_path.lower().endswith(".osu"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                results.append(process_osu_content(content, os.path.basename(file_path)))
        except Exception as e:
            print(f"❌ Failed to read .osu file '{file_path}': {e}")
            return
    else:
        print(f"❌ Unsupported file extension for '{file_path}'. Expected .osz or .osu")
        return

    if not results:
        print("⚠️ No valid osu! difficulties found.")
        return

    # Print Table
    print("\n" + "=" * 80)
    print(" ⭐ OSU!MANIA OFFICIAL DIFFICULTY (ROSU-PP RATING) ⭐")
    print("=" * 80)
    print("✅ Engine: `rosu-pp-py` (Official Rust Calculator for osu!)")
    print("-" * 80)

    header = f"{'Difficulty (Version)':<28} | {'Keys':<4} | {'Notes':<6} | {'LNs':<6} | {'OD':<4} | {'Star Rating':<10}"
    print(header)
    print("-" * 80)

    for meta, rosu_sr in results:
        diff_name = meta["version"][:27]
        keys = f"{meta['keys']}K"
        notes = str(meta["notes_count"])
        lns = str(meta["ln_count"])
        od = f"{meta['od']:.1f}"
        sr_str = f"{rosu_sr:.2f}*"

        row = f"{diff_name:<28} | {keys:<4} | {notes:<6} | {lns:<6} | {od:<4} | {sr_str:<10}"
        print(row)

    print("=" * 80)
    print(f"Total difficulties calculated: {len(results)} (All values exact 1:1 with osu! engine)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate official osu!mania Star Rating using rosu-pp-py."
    )
    parser.add_argument(
        "--map",
        "--template",
        "--osz",
        "--osu",
        dest="map_path",
        type=str,
        help="Path to .osz package or .osu beatmap file.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Path to folder containing multiple .osz or .osu files.",
    )

    args = parser.parse_args()

    target = args.map_path or args.dir
    if not target:
        if os.path.exists("output_beatmaps"):
            target = "output_beatmaps"
        else:
            parser.print_help()
            sys.exit(1)

    inspect_map_file(target)


if __name__ == "__main__":
    main()
