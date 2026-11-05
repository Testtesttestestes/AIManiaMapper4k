#!/usr/bin/env python3
"""
Learns the mapping-pattern model (osu_mapper/hybrid/patterns.py) from real maps.

    python learn_patterns.py --data dataset_hybrid                 # maps of the prepared dataset (recommended)
    python learn_patterns.py --input_dir "sample beatmaps"         # straight from .osz archives, no audio needed
    python learn_patterns.py --data dataset_hybrid --out checkpoints/hybrid/patterns.npz

Takes seconds to minutes (no GPU). The composer (generate_composer.py) loads
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


def from_dataset(model: PatternModel, data: Path, include_derived: bool) -> int:
    from osu_mapper.hybrid.params import ParamTrack
    from osu_mapper.hybrid.renderer import render_track

    index = json.loads((data / "index.json").read_text(encoding="utf-8"))
    sections = song_sections(data, index)
    n = 0
    for m in index["maps"]:
        if (m.get("derived") and not include_derived) or not m.get("roundtrip_exact", True):
            continue
        sr = float(m.get("sr", 0.0))
        if sr <= 0:
            continue
        track = ParamTrack.load_npz(data / m["track"])
        notes = render_track(track)
        model.add_map(notes, track.timing_points, sr)
        if m["song_id"] in sections:
            model.add_sections(notes, sections[m["song_id"]](track.timing_points), sr)
        n += 1
    return n


def song_sections(data: Path, index):
    """Per song: a function timing points -> composer sections (audio intensity), from the stored features."""
    import numpy as np

    from osu_mapper.hybrid.audio_features import FeatureConfig
    from osu_mapper.hybrid.composer import analyze_song, segment_sections

    slices = FeatureConfig.from_dict(index["feature_config"]).slices()
    out = {}
    for sid, song in index["songs"].items():
        path = data / song["features"][0]
        if not path.exists():
            continue

        def make(tps, path=path, song=song):
            feats = np.load(path, mmap_mode="r")
            return segment_sections(analyze_song(feats, slices, tps, int(song["duration_ms"])))
        out[sid] = make
    return out


def from_archives(model: PatternModel, folder: Path) -> int:
    from osu_mapper.dataset.source_resolver import resolve_beatmap_sources
    from osu_mapper.hybrid.beatmap import parse_osu_mania
    from osu_mapper.hybrid.inference import star_rating_of
    from osu_mapper.hybrid.sources import _decode_text, is_lfs_pointer

    osz_files, folders = resolve_beatmap_sources(folder)
    n = 0
    seen = set()
    texts = []
    for osz in osz_files:
        if is_lfs_pointer(osz):
            continue
        try:
            with zipfile.ZipFile(osz) as z:
                texts += [(f"{Path(osz).name}::{e}", _decode_text(z.read(e))) for e in z.namelist() if e.lower().endswith(".osu")]
        except (zipfile.BadZipFile, OSError) as err:
            print(f"  [error] {osz}: {err}")
    for f in folders:
        texts += [(str(p), _decode_text(p.read_bytes())) for p in Path(f).glob("*.osu")]
    for name, text in texts:
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
    args = ap.parse_args()
    if not args.data and not args.input_dir:
        sys.exit("give --data dataset_hybrid or --input_dir \"sample beatmaps\"")
    t0 = time.time()
    model = PatternModel.empty()
    n = 0
    if args.data:
        n += from_dataset(model, Path(args.data), args.include_derived)
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
