#!/usr/bin/env python3
"""
Builds the training set of the hybrid mapper.

For every song (audio file) of every .osz archive / song folder:
  1. decodes the audio once with FFmpeg (24 kHz mono) and computes all audio representations,
     optionally also for degraded quality variants (--quality_variants mp3_48k lowpass_6k);
  2. for every 4K difficulty: solves the exact parameter track, re-renders it and checks that the
     result is bit-identical to the .osu (maps failing this are reported and excluded);
  3. stores star rating (rosu-pp) and the global style vector.

    python prepare_hybrid_dataset.py --input_dir "sample beatmaps" --output_dir dataset_hybrid
    python prepare_hybrid_dataset.py --input_dir "F:\\mania4k dataset" --quality_variants mp3_48k

Re-running is incremental: songs whose features already exist are not decoded again.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from osu_mapper.hybrid.audio_features import QUALITY_VARIANTS, FeatureConfig, coarse_features, compute_stacked_features, degrade_audio, load_audio
from osu_mapper.hybrid.dataset import INDEX_VERSION
from osu_mapper.hybrid.params import PARAM_VERSION
from osu_mapper.hybrid.renderer import render_track
from osu_mapper.hybrid.solver import STYLE_KEYS, SolveStats, compute_style, solve_map
from osu_mapper.hybrid.sources import iter_song_groups, stable_id


def star_rating(osu_text: str, m) -> float:
    try:
        from osu_mapper.dataset.difficulty_calculator import calculate_osu_mania_star_rating
        hits = [(n.col, n.start, n.end, n.ln) for n in m.notes]
        return float(calculate_osu_mania_star_rating(osu_text=osu_text, hit_objects=hits, overall_difficulty=m.overall_difficulty, keys=4))
    except Exception:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare the hybrid (neural -> parametric) osu!mania 4K dataset")
    ap.add_argument("--input_dir", "-i", default="sample beatmaps")
    ap.add_argument("--output_dir", "-o", default="dataset_hybrid")
    ap.add_argument("--ffmpeg", default=None, help="path to ffmpeg(.exe)")
    ap.add_argument("--quality_variants", nargs="*", default=[], choices=sorted(QUALITY_VARIANTS), help="extra degraded-audio feature copies")
    ap.add_argument("--max_songs", type=int, default=0, help="stop after N songs (0 = all)")
    ap.add_argument("--overwrite", action="store_true", help="recompute features even if they exist")
    ap.add_argument("--device", default="auto", help="device for STFT features and neural encoders: auto (GPU if available) / cpu / cuda / cuda:1")
    ap.add_argument("--external", nargs="*", default=None, metavar="NAME",
                    help="neural encoders: '--external' alone = all of them (encodec24 mert95m demucs_stems), "
                         "or list a subset, e.g. --external encodec24 mert95m")
    args = ap.parse_args()
    if args.external is not None and len(args.external) == 0:
        from osu_mapper.hybrid.external_reps import EXTERNAL_REGISTRY
        args.external = sorted(EXTERNAL_REGISTRY)
    args.external = args.external or []
    if args.external:
        from osu_mapper.hybrid.external_reps import preflight_external
        preflight_external(args.external, device=args.device)

    out = Path(args.output_dir)
    (out / "features").mkdir(parents=True, exist_ok=True)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    fcfg = FeatureConfig(external=tuple(args.external))
    index_path = out / "index.json"
    old_songs: Dict[str, Any] = {}
    if index_path.exists() and not args.overwrite:
        try:
            old = json.loads(index_path.read_text(encoding="utf-8"))
            if old.get("feature_config") == fcfg.to_dict():
                old_songs = old.get("songs", {})
        except Exception:
            old_songs = {}

    songs: Dict[str, Any] = {}
    maps: List[Dict[str, Any]] = []
    totals = {"songs": 0, "maps": 0, "roundtrip_fail": 0, "overflow_notes": 0, "notes": 0, "audio_hours": 0.0}
    t_start = time.time()

    def write_index() -> None:
        data = {
            "version": INDEX_VERSION,
            "param_version": PARAM_VERSION,
            "feature_config": fcfg.to_dict(),
            "feature_slices": {k: list(v) for k, v in fcfg.slices().items()},
            "style_keys": list(STYLE_KEYS),
            "songs": songs,
            "maps": maps,
            "totals": totals,
        }
        tmp = index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(index_path)

    for group in iter_song_groups(args.input_dir):
        if args.max_songs and totals["songs"] >= args.max_songs:
            break
        sid = group.song_id
        label = f"{Path(group.source).name} :: {group.audio_name}"
        feat_file = out / "features" / f"{sid}.npy"
        coarse_file = out / "features" / f"{sid}.coarse.npy"
        variant_files = [out / "features" / f"{sid}.{v}.npy" for v in args.quality_variants]
        need = args.overwrite or sid not in old_songs or not feat_file.exists() or not coarse_file.exists() or not all(p.exists() for p in variant_files)
        try:
            if need:
                t0 = time.time()
                raw = group.read_audio()
                wave = load_audio(raw, fcfg.sample_rate, ext=group.audio_ext, ffmpeg_path=args.ffmpeg)
                if wave.size < fcfg.sample_rate:
                    print(f"[skip] {label}: audio shorter than 1 s")
                    continue
                stacked = compute_stacked_features(wave, fcfg, device=args.device)
                np.save(feat_file, stacked.astype(np.float16))
                np.save(coarse_file, coarse_features(stacked).astype(np.float16))
                for vname, vfile in zip(args.quality_variants, variant_files):
                    vwave = degrade_audio(wave, fcfg.sample_rate, vname, args.ffmpeg)
                    np.save(vfile, compute_stacked_features(vwave, fcfg, device=args.device).astype(np.float16))
                num_frames = int(stacked.shape[0])
                duration_ms = int(round(wave.size * 1000.0 / fcfg.sample_rate))
                print(f"[song] {label}: {duration_ms / 1000:.1f}s, features {stacked.shape} in {time.time() - t0:.1f}s")
            else:
                num_frames = int(old_songs[sid]["num_frames"])
                duration_ms = int(old_songs[sid]["duration_ms"])
                print(f"[song] {label}: cached features")
        except Exception as err:
            print(f"[error] {label}: {err}")
            continue

        songs[sid] = {
            "source": group.source,
            "audio": group.audio_name,
            "features": [f"features/{sid}.npy"] + [f"features/{sid}.{v}.npy" for v in args.quality_variants],
            "coarse": f"features/{sid}.coarse.npy",
            "num_frames": num_frames,
            "duration_ms": duration_ms,
        }
        totals["songs"] += 1
        totals["audio_hours"] += duration_ms / 3.6e6

        for osu_name, osu_text, m in group.maps:
            mid = stable_id(group.source, osu_name)
            stats = SolveStats()
            track = solve_map(m, num_frames=num_frames, stats=stats)
            exact = render_track(track) == sorted(m.notes)
            sr = star_rating(osu_text, m)
            style, known = compute_style(track, m.notes, sr, m.overall_difficulty)
            track.save_npz(out / "maps" / f"{mid}.npz")
            maps.append({
                "map_id": mid,
                "song_id": sid,
                "track": f"maps/{mid}.npz",
                "source": group.source,
                "osu_file": osu_name,
                "version": m.metadata.get("Version", ""),
                "creator": m.metadata.get("Creator", ""),
                "sr": round(sr, 3),
                "od": m.overall_difficulty,
                "style": [round(float(x), 5) for x in style],
                "style_mask": [float(x) for x in known],
                "notes": len(m.notes),
                "events": stats.events,
                "overflow": stats.overflow,
                "overflow_reasons": stats.overflow_reasons,
                "overlaps_in_ln": stats.overlaps_in_ln,
                "roundtrip_exact": bool(exact),
            })
            totals["maps"] += 1
            totals["notes"] += len(m.notes)
            totals["overflow_notes"] += stats.overflow
            if not exact:
                totals["roundtrip_fail"] += 1
                print(f"  [ROUNDTRIP FAIL] {osu_name} - excluded from training (please report)")
            print(f"  - {m.metadata.get('Version', osu_name)}: SR {sr:.2f}, {len(m.notes)} notes, {stats.events} events, "
                  f"overflow {stats.overflow}, exact={'yes' if exact else 'NO'}")
        if totals["songs"] % 20 == 0:
            write_index()

    write_index()
    print("\n" + "=" * 70)
    print(f"songs {totals['songs']} | maps {totals['maps']} | audio {totals['audio_hours']:.2f} h | notes {totals['notes']:,}")
    print(f"exact roundtrip failures: {totals['roundtrip_fail']} | notes in overflow (not expressible densely): {totals['overflow_notes']}")
    print(f"index: {index_path} ({time.time() - t_start:.0f}s)")
    if totals["maps"] == 0:
        sys.exit("No 4K mania maps found. Check --input_dir or 'sample beatmaps/sources.txt'.")


if __name__ == "__main__":
    main()
