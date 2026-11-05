#!/usr/bin/env python3
"""
Learns the mapping-pattern model (osu_mapper/hybrid/patterns.py) from real maps.

    python learn_patterns.py --data dataset_hybrid                 # maps of the prepared dataset (recommended)
    python learn_patterns.py --input_dir "sample beatmaps"         # straight from .osz archives, no audio needed
    python learn_patterns.py --data dataset_hybrid --out checkpoints/hybrid/patterns.npz

Takes about a second per song on an SSD (no GPU, one CPU core, a few hundred MB of RAM); progress is printed every
20 songs. --no_audio skips the audio part and needs no feature files at all. The composer (generate_composer.py) loads
checkpoints/hybrid/patterns.npz automatically. Re-run it whenever the dataset grows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

from osu_mapper.hybrid.patterns import PatternModel, default_pattern_path


def from_dataset(model: PatternModel, data: Path, include_derived: bool, use_audio: bool = True) -> int:
    """Counts every map of a prepared dataset. Maps are processed song by song, so the audio features of a song
    (tens of MB on disk) are read once, not once per difficulty."""
    import numpy as np

    from osu_mapper.hybrid.params import ParamTrack
    from osu_mapper.hybrid.renderer import render_track

    index = json.loads((data / "index.json").read_text(encoding="utf-8"))
    maps = [m for m in index["maps"]
            if not (m.get("derived") and not include_derived) and m.get("roundtrip_exact", True) and float(m.get("sr", 0.0)) > 0]
    by_song = {}
    for m in maps:
        by_song.setdefault(m["song_id"], []).append(m)
    slices = None
    if use_audio:
        from osu_mapper.hybrid.audio_features import FeatureConfig
        from osu_mapper.hybrid.composer import analyze_song, segment_sections
        slices = FeatureConfig.from_dict(index["feature_config"]).slices()
    print(f"[learn] {len(maps)} maps of {len(by_song)} songs" + ("" if use_audio else " (no audio: section contrast not fitted)"), flush=True)
    t0 = time.time()
    n = 0
    for k, (sid, song_maps) in enumerate(by_song.items(), 1):
        song = index["songs"].get(sid, {})
        feats = None
        sections_by_timing = {}
        if use_audio and song:
            path = data / song["features"][0]
            if path.exists():
                feats = np.load(path)  # one sequential read per song (memory-mapping re-read it from disk for every map)
        for m in song_maps:
            track = ParamTrack.load_npz(data / m["track"])
            notes = render_track(track)
            sr = float(m["sr"])
            model.add_map(notes, track.timing_points, sr)
            if feats is not None:
                key = tuple((round(t.time, 1), round(t.beat_len, 4)) for t in track.timing_points if t.uninherited)
                if key not in sections_by_timing:
                    an = analyze_song(feats, slices, track.timing_points, int(song["duration_ms"]))
                    sections_by_timing[key] = segment_sections(an)
                model.add_sections(notes, sections_by_timing[key], sr)
            n += 1
        if k == 1 or k % 20 == 0 or k == len(by_song):
            el = time.time() - t0
            eta = el / k * (len(by_song) - k)
            print(f"  {k}/{len(by_song)} songs, {n} maps, {el:.0f}s elapsed, ~{eta:.0f}s left", flush=True)
    return n


def from_archives(model: PatternModel, folder: Path) -> int:
    from osu_mapper.dataset.source_resolver import resolve_beatmap_sources
    from osu_mapper.hybrid.beatmap import parse_osu_mania
    from osu_mapper.hybrid.inference import star_rating_of
    from osu_mapper.hybrid.sources import _decode_text, is_lfs_pointer

    osz_files, folders = resolve_beatmap_sources(folder)
    n = 0
    seen = set()
    texts = []
    print(f"[learn] reading {len(osz_files)} archives and {len(folders)} folders", flush=True)
    for k, osz in enumerate(osz_files, 1):
        if k % 50 == 0:
            print(f"  read {k}/{len(osz_files)} archives", flush=True)
        if is_lfs_pointer(osz):
            continue
        try:
            with zipfile.ZipFile(osz) as z:
                texts += [(f"{Path(osz).name}::{e}", _decode_text(z.read(e))) for e in z.namelist() if e.lower().endswith(".osu")]
        except (zipfile.BadZipFile, OSError) as err:
            print(f"  [error] {osz}: {err}")
    for f in folders:
        texts += [(str(p), _decode_text(p.read_bytes())) for p in Path(f).glob("*.osu")]
    t0 = time.time()
    for k, (name, text) in enumerate(texts, 1):
        if k % 200 == 0:
            el = time.time() - t0
            print(f"  {k}/{len(texts)} difficulties, {n} 4K maps counted, ~{el / k * (len(texts) - k):.0f}s left", flush=True)
        m = parse_osu_mania(text)
        if m is None or len(m.notes) < 20:
            continue
        key = tuple(sorted(m.canonical_notes()))
        if key in seen:  # duplicate re-uploads would count twice
            continue
        seen.add(key)
        sr = star_rating_of(m.notes, m.timing_points, m.overall_difficulty)
        model.add_map(m.notes, m.timing_points, sr)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="prepared dataset folder (prepare_hybrid_dataset.py)")
    ap.add_argument("--input_dir", default=None, help="folder with .osz archives / song folders")
    ap.add_argument("--out", default=str(default_pattern_path()))
    ap.add_argument("--include_derived", action="store_true", help="also count thinned (derived) maps of the dataset")
    ap.add_argument("--no_audio", action="store_true",
                    help="skip the audio part (how section density follows the music); much faster, the composer then "
                         "matches the density variation of real maps instead")
    args = ap.parse_args()
    print("[learn] starting (CPU only, no GPU needed)", flush=True)
    if not args.data and not args.input_dir:
        sys.exit("give --data dataset_hybrid or --input_dir \"sample beatmaps\"")
    t0 = time.time()
    model = PatternModel.empty()
    n = 0
    if args.data:
        n += from_dataset(model, Path(args.data), args.include_derived, use_audio=not args.no_audio)
    if args.input_dir:
        n += from_archives(model, Path(args.input_dir))
    if n == 0:
        sys.exit("no maps found")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(model.summary())
    print(f"[done] {n} maps -> {out} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
