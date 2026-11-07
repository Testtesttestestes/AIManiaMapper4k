#!/usr/bin/env python3
"""
Learns the mapping-pattern model (osu_mapper/hybrid/patterns.py) from real maps.

    python learn_patterns.py --data dataset_hybrid                 # maps of the prepared dataset (recommended)
    python learn_patterns.py --input_dir "sample beatmaps"         # straight from .osz archives, no audio needed
    python learn_patterns.py --data dataset_hybrid --out checkpoints/hybrid/patterns.npz

About a second per song per CPU core (no GPU), songs are counted on --workers processes in parallel; progress is
printed as songs finish. --no_audio skips the audio part and needs no feature files at all. The composer (generate_composer.py) loads
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


def _count_songs(job) -> "tuple":
    """Worker: a fresh PatternModel counted over a chunk of songs (summed by the main process)."""
    import os

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    import numpy as np

    from osu_mapper.hybrid.params import ParamTrack
    from osu_mapper.hybrid.renderer import render_track

    data, songs, slices, use_audio = job
    data = Path(data)
    model = PatternModel.empty()
    n = 0
    if use_audio:
        from osu_mapper.hybrid.composer import analyze_song, segment_sections
    for song, song_maps in songs:
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
    return model, n, len(songs)


def from_dataset(model: PatternModel, data: Path, include_derived: bool, use_audio: bool = True, workers: int = 1) -> int:
    """Counts every map of a prepared dataset, song by song (the audio features of a song are read once), on
    ``workers`` processes in parallel."""
    print(f"[learn] reading {data / 'index.json'}", flush=True)
    index = json.loads((data / "index.json").read_text(encoding="utf-8"))
    maps = [m for m in index["maps"]
            if not (m.get("derived") and not include_derived) and m.get("roundtrip_exact", True) and float(m.get("sr", 0.0)) > 0]
    by_song = {}
    for m in maps:
        by_song.setdefault(m["song_id"], []).append(m)
    slices = None
    if use_audio:
        slices = {k: tuple(v) for k, v in (index.get("feature_slices") or {}).items()}
        if not slices:
            from osu_mapper.hybrid.audio_features import FeatureConfig
            slices = FeatureConfig.from_dict(index["feature_config"]).slices()
    songs = [(index["songs"].get(sid, {}), ms) for sid, ms in by_song.items()]
    workers = max(1, min(int(workers), len(songs)))
    print(f"[learn] {len(maps)} maps of {len(songs)} songs on {workers} process(es)"
          + ("" if use_audio else " (no audio: section contrast not fitted)"), flush=True)
    n_chunks = 1 if workers == 1 else min(len(songs), workers * 4)
    chunks = [songs[i::n_chunks] for i in range(n_chunks)]
    jobs = [(str(data), c, slices, use_audio) for c in chunks if c]
    t0 = time.time()
    n = done = 0

    def report(k_songs: int) -> None:
        el = time.time() - t0
        eta = el / max(1, k_songs) * (len(songs) - k_songs)
        print(f"  {k_songs}/{len(songs)} songs, {n} maps, {el:.0f}s elapsed, ~{eta:.0f}s left", flush=True)

    if workers == 1:
        for c in jobs:
            for one in c[1]:  # song by song for a fine-grained progress line
                part, k, _ = _count_songs((c[0], [one], c[2], c[3]))
                model.merge(part)
                n += k
                done += 1
                if done == 1 or done % 20 == 0 or done == len(songs):
                    report(done)
        return n
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_count_songs, j) for j in jobs]
        for f in as_completed(futures):
            part, k, ns = f.result()
            model.merge(part)
            n += k
            done += ns
            report(done)
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


def update_rhythm_bpm(model: PatternModel, folder: Path) -> int:
    """(Re)counts only the tempo-aware rhythm table of an existing model from .osu files (folders of .osu or .osz)."""
    import numpy as np

    from osu_mapper.hybrid.beatmap import parse_osu_mania
    from osu_mapper.hybrid.inference import star_rating_of
    from osu_mapper.hybrid.patterns import chord_events, div_class, sr_bin

    model.rhythm_hits_bpm = None
    model.rhythm_beats_bpm = None
    texts = [p.read_bytes().decode("utf-8", errors="replace") for p in sorted(folder.rglob("*.osu"))]
    for osz in sorted(folder.rglob("*.osz")):
        try:
            with zipfile.ZipFile(osz) as z:
                texts += [z.read(e).decode("utf-8", errors="replace") for e in z.namelist() if e.lower().endswith(".osu")]
        except (zipfile.BadZipFile, OSError):
            continue
    seen, n = set(), 0
    for k, text in enumerate(texts, 1):
        m = parse_osu_mania(text)
        if m is None or len(m.notes) < 20:
            continue
        key = hash(tuple(sorted(m.canonical_notes())))
        if key in seen:
            continue
        seen.add(key)
        times, _, _ = chord_events(m.notes)
        if times.size < 8:
            continue
        reds = sorted((tp for tp in m.timing_points if tp.uninherited and tp.beat_len > 0), key=lambda r: r.time)
        cls = np.array([div_class(float(t), reds) for t in times], dtype=np.int64)
        model.add_rhythm_bpm(times, cls, m.timing_points, sr_bin(star_rating_of(m.notes, m.timing_points, m.overall_difficulty)))
        n += 1
        if k % 500 == 0:
            print(f"  {k}/{len(texts)} difficulties", flush=True)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="prepared dataset folder (prepare_hybrid_dataset.py)")
    ap.add_argument("--input_dir", default=None, help="folder with .osz archives / song folders")
    ap.add_argument("--out", default=str(default_pattern_path()))
    ap.add_argument("--include_derived", action="store_true", help="also count thinned (derived) maps of the dataset")
    import os
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)),
                    help="parallel processes for --data (default: CPU cores - 1, at most 16; each needs ~0.5 GB of RAM)")
    ap.add_argument("--no_audio", action="store_true",
                    help="skip the audio part (how section density follows the music); much faster, the composer then "
                         "matches the density variation of real maps instead")
    ap.add_argument("--update_rhythm_bpm", default=None, metavar="MODEL",
                    help="only (re)count the tempo-aware rhythm table of an existing model from --input_dir "
                         "(.osz or .osu files) and save it to --out")
    args = ap.parse_args()
    print("[learn] starting (CPU only, no GPU needed)", flush=True)
    if args.update_rhythm_bpm:
        if not args.input_dir:
            sys.exit("--update_rhythm_bpm needs --input_dir")
        model = PatternModel.load(args.update_rhythm_bpm)
        n = update_rhythm_bpm(model, Path(args.input_dir))
        model.save(args.out)
        print(f"[done] tempo-aware rhythm of {n} maps -> {args.out}")
        return
    if not args.data and not args.input_dir:
        sys.exit("give --data dataset_hybrid or --input_dir \"sample beatmaps\"")
    t0 = time.time()
    model = PatternModel.empty()
    n = 0
    if args.data:
        n += from_dataset(model, Path(args.data), args.include_derived, use_audio=not args.no_audio, workers=args.workers)
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
