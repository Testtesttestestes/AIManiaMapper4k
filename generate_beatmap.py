#!/usr/bin/env python3
"""
CLI entry point for generating playable osu!mania 4K beatmaps using trained AI models.

Features:
1. Map any song directly from the dataset (or 'sample beatmaps/' .osz archives):
   python generate_beatmap.py --song 1 --sr 4.5
   python generate_beatmap.py --list_songs

2. Map from a user-created template .osz / .osu file (with pre-set BPM, timing points, red/green lines):
   python generate_beatmap.py --template "path/to/my_template.osz" --sr 4.0

3. Map from any standalone audio file (.mp3, .wav, .ogg, .flac) with manual BPM/Offset or Model 1 Sync:
   python generate_beatmap.py "path/to/song.mp3" --bpm 175 --offset 1200 --sr 5.2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

from osu_mapper.dataset.feature_extractor import (
    find_ffmpeg,
    parse_osu_for_training,
)
from osu_mapper.pipeline import generate_beatmap_pipeline
from osu_mapper.pipeline.audio import create_synthetic_click_wav
from osu_mapper.pipeline.timing import TimingPoint


def list_available_dataset_songs() -> List[dict]:
    """Finds all available sample beatmaps and returns song info."""
    manifest_p = Path("dataset_manifest/manifest.json")
    songs = []
    if manifest_p.exists():
        try:
            with open(manifest_p, "r", encoding="utf-8") as f:
                data = json.load(f)
                for idx, item in enumerate(data.get("beatmaps", [])):
                    songs.append({
                        "index": idx + 1,
                        "title": item.get("title", "Unknown Title"),
                        "artist": item.get("artist", "Unknown Artist"),
                        "bpm": item.get("bpm", 0.0),
                        "offset_ms": item.get("offset_ms", 0),
                        "source_osz": item.get("source_osz", ""),
                        "osu_file": item.get("osu_file_name", ""),
                    })
        except Exception:
            pass

    if not songs:
        # Fallback to scanning sample beatmaps
        osz_files = sorted(glob.glob("sample beatmaps/*.osz"))
        for idx, osz in enumerate(osz_files):
            songs.append({
                "index": idx + 1,
                "title": Path(osz).stem,
                "artist": "Sample Beatmap",
                "bpm": 0.0,
                "offset_ms": 0,
                "source_osz": osz,
                "osu_file": "",
            })
    return songs


def extract_template_osz(osz_path: str | Path, work_dir: Path) -> Tuple[str, dict, List[TimingPoint]]:
    """
    Extracts audio and timing points from a template .osz archive.
    """
    osz_p = Path(osz_path)
    if not osz_p.exists():
        raise FileNotFoundError(f"Template .osz file not found: {osz_p}")

    extracted_dir = work_dir / "template_extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(osz_p, "r") as zf:
        zf.extractall(extracted_dir)

    osu_files = list(extracted_dir.glob("*.osu"))
    if not osu_files:
        raise ValueError(f"No .osu difficulty files found inside '{osz_p.name}'.")

    # Pick first osu file or one matching 4K
    osu_target = osu_files[0]
    parsed_meta = {}
    extracted_timing_points: List[TimingPoint] = []

    osu_text = osu_target.read_text(encoding="utf-8", errors="ignore")
    parsed = parse_osu_for_training(osu_text, osu_target.name)

    audio_file_name = None
    if parsed:
        audio_file_name = parsed.audio_filename
        parsed_meta["bpm"] = parsed.bpm
        parsed_meta["offset_ms"] = parsed.offset_ms
        parsed_meta["version"] = parsed.version

        for tp in parsed.timing_points:
            t_ms, val, uninh = tp
            if uninh:
                extracted_timing_points.append(TimingPoint(
                    time_ms=t_ms,
                    beat_length=val,
                    slider_velocity=1.0,
                    uninherited=True,
                ))
            else:
                sv = max(0.1, min(10.0, -100.0 / val)) if val < 0 else 1.0
                extracted_timing_points.append(TimingPoint(
                    time_ms=t_ms,
                    beat_length=-100.0 / sv,
                    slider_velocity=sv,
                    uninherited=False,
                ))

    # Parse Title & Artist from [Metadata]
    for line in osu_text.splitlines():
        if line.startswith("Title:"):
            parsed_meta["title"] = line.split(":", 1)[1].strip()
        elif line.startswith("Artist:"):
            parsed_meta["artist"] = line.split(":", 1)[1].strip()
        elif line.startswith("AudioFilename:") and not audio_file_name:
            audio_file_name = line.split(":", 1)[1].strip()

    # Locate audio inside extracted folder
    audio_path = None
    if audio_file_name and (extracted_dir / audio_file_name).exists():
        audio_path = str((extracted_dir / audio_file_name).resolve())
    else:
        # Search for any audio
        for ext in ("*.mp3", "*.ogg", "*.wav", "*.flac"):
            cands = list(extracted_dir.glob(ext))
            if cands:
                audio_path = str(cands[0].resolve())
                break

    if not audio_path:
        raise FileNotFoundError(f"Could not locate audio file in template '{osz_p.name}'.")

    return audio_path, parsed_meta, extracted_timing_points


def convert_audio_to_wav(audio_input: str, output_wav: Path) -> str:
    """Converts MP3/OGG/FLAC to standard WAV using FFmpeg or returns WAV if already WAV."""
    p = Path(audio_input)
    if p.suffix.lower() == ".wav":
        return str(p.resolve())

    ffmpeg_bin = find_ffmpeg()
    if not ffmpeg_bin:
        # Try raw decode if available
        return audio_input

    import subprocess
    cmd = [ffmpeg_bin, "-y", "-v", "error", "-i", str(p.resolve()), "-ar", "44100", "-ac", "2", str(output_wav.resolve())]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode == 0 and output_wav.exists() and output_wav.stat().st_size > 0:
        return str(output_wav.resolve())
    return audio_input


def main():
    parser = argparse.ArgumentParser(
        description="osu!mania 4K AI Beatmap Generator (Model 3 Transformer Pattern Synthesizer)"
    )
    parser.add_argument("audio", nargs="?", default=None, help="Path to audio (.wav, .mp3, .ogg) OR template .osz / .osu file")
    parser.add_argument("--template", "--osz", "--osu", type=str, default=None, help="Path to template .osz / .osu file with pre-set BPM/timing points")
    parser.add_argument("--song", type=int, default=None, help="Map a specific song from dataset by index number (1..N)")
    parser.add_argument("--list_songs", action="store_true", help="List all available songs from dataset and exit")
    parser.add_argument("--sr", "--star-rating", type=float, default=3.5, dest="star_rating", help="Target difficulty Star Rating (1.0 to 9.9)")
    parser.add_argument("--title", type=str, default=None, help="Song title override")
    parser.add_argument("--artist", type=str, default=None, help="Song artist override")
    parser.add_argument("--version", type=str, default=None, help="Difficulty name (default: 'AI 4K [X.X*]')")
    parser.add_argument("--creator", type=str, default="AIManiaMapper", help="Beatmap creator name")
    parser.add_argument("--bpm", type=float, default=None, help="Manual BPM override (takes precedence over AI sync)")
    parser.add_argument("--offset", type=int, default=None, help="Manual offset in ms override (takes precedence over AI sync)")
    parser.add_argument("--model3_checkpoint", "--checkpoint", "-c", type=str, default=None, help="Path to Model 3 (ManiaPatternTransformer) .pt weights")
    parser.add_argument("--sync_checkpoint", type=str, default=None, help="Path to Model 1 (SyncBPMOffsetModel) .pt weights")
    parser.add_argument("--output", "-o", type=str, default="output_beatmaps/generated_map.osz", help="Output .osz file path")
    parser.add_argument("--synthetic", action="store_true", help="Generate a synthetic test track if no audio provided")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds for synthetic audio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")

    args = parser.parse_args()

    # 1. List songs mode
    songs = list_available_dataset_songs()
    if args.list_songs:
        print("\n" + "=" * 75)
        print(" 🎵 AVAILABLE DATASET SONGS FOR MAPPING")
        print("=" * 75)
        for s in songs:
            print(f" #{s['index']:<3} | {s['title'][:32]:<32} | {s['artist'][:20]:<20} | BPM: {s['bpm']}")
        print("=" * 75)
        print("To map any of these songs with Model 3:")
        print("  python generate_beatmap.py --song 1 --sr 4.5\n")
        return

    # Create output directory
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    work_dir = Path(tempfile.mkdtemp(prefix="osu_mapper_gen_"))

    template_timing_points: Optional[List[TimingPoint]] = None
    template_meta = {}
    audio_path = args.audio

    # 2. Select song from dataset by index
    if args.song is not None:
        matched = [s for s in songs if s["index"] == args.song]
        if not matched:
            print(f"❌ [Error] Song #{args.song} not found in dataset. Use --list_songs to see all options.")
            sys.exit(1)
        target_song = matched[0]
        print(f"🎯 [Dataset Selection] Selected #{target_song['index']}: {target_song['title']} - {target_song['artist']}")
        if target_song["source_osz"] and Path(target_song["source_osz"]).exists():
            args.template = target_song["source_osz"]
        else:
            print(f"❌ [Error] Source archive '{target_song['source_osz']}' not found.")
            sys.exit(1)

    # 3. Handle template .osz / .osu if provided (or if positional audio ends with .osz/.osu)
    template_target = args.template
    if not template_target and audio_path and (audio_path.lower().endswith(".osz") or audio_path.lower().endswith(".osu")):
        template_target = audio_path
        audio_path = None

    if template_target:
        print(f"📦 [Template Loader] Extracting template and timing points from: {template_target}")
        if template_target.lower().endswith(".osz"):
            audio_path, template_meta, template_timing_points = extract_template_osz(template_target, work_dir)
        elif template_target.lower().endswith(".osu"):
            osu_p = Path(template_target)
            osu_text = osu_p.read_text(encoding="utf-8", errors="ignore")
            parsed = parse_osu_for_training(osu_text, osu_p.name)
            if parsed:
                template_meta["bpm"] = parsed.bpm
                template_meta["offset_ms"] = parsed.offset_ms
                template_meta["version"] = parsed.version
                audio_cand = osu_p.parent / parsed.audio_filename
                if audio_cand.exists():
                    audio_path = str(audio_cand.resolve())

    # 4. Synthetic fallback if no audio provided
    if not audio_path or args.synthetic:
        synth_path = "output_beatmaps/synthetic_test.wav"
        print(f"[Generator] Synthesizing rhythmic test audio ({args.duration}s) to {synth_path}...")
        create_synthetic_click_wav(synth_path, duration_sec=args.duration, bpm=args.bpm or 140.0)
        audio_path = synth_path

    if not os.path.exists(audio_path):
        print(f"❌ [Error] Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # Convert to WAV if needed for accurate waveform analysis
    wav_target = work_dir / "converted_audio.wav"
    effective_wav = convert_audio_to_wav(audio_path, wav_target)

    # Resolve metadata
    final_title = args.title or template_meta.get("title", Path(audio_path).stem)
    final_artist = args.artist or template_meta.get("artist", "AI Artist")
    final_bpm = args.bpm if args.bpm is not None else template_meta.get("bpm")
    final_offset = args.offset if args.offset is not None else template_meta.get("offset_ms")

    print("\n" + "=" * 65)
    print(" 🎮 OSU!MANIA 4K AI BEATMAP GENERATOR (MODEL 3 TRANSFORMER)")
    print("=" * 65)
    print(f"  • Audio Track:    {Path(audio_path).name}")
    print(f"  • Song Title:     {final_title}")
    print(f"  • Artist:         {final_artist}")
    print(f"  • Target SR:      {args.star_rating:.1f}*")
    print(f"  • BPM:            {f'{final_bpm:.2f} (Preset/Manual)' if final_bpm else 'Auto Sync Inference'}")
    print(f"  • Offset:         {f'{final_offset} ms (Preset/Manual)' if final_offset is not None else 'Auto Sync Inference'}")
    print(f"  • Model 3 Weights:{args.model3_checkpoint or 'Auto (checkpoints/model_3_transformer_best.pt)'}")
    print(f"  • Output Package: {args.output}")
    print("=" * 65 + "\n")

    try:
        osu_str, osz_path, analysis = generate_beatmap_pipeline(
            wav_path=effective_wav,
            output_osz_path=args.output,
            star_rating=args.star_rating,
            title=final_title,
            artist=final_artist,
            version=args.version,
            creator=args.creator,
            manual_bpm=final_bpm,
            manual_offset=final_offset,
            sync_model_checkpoint=args.sync_checkpoint,
            pattern_model_checkpoint=args.model3_checkpoint,
            existing_timing_points=template_timing_points,
            seed=args.seed,
        )

        print("\n" + "=" * 65)
        print(" 🎉 BEATMAP GENERATION SUCCEEDED!")
        print("=" * 65)
        print(f"  • Final BPM:         {analysis.bpm:.2f}")
        print(f"  • Final Offset:      {analysis.offset_ms} ms")
        print(f"  • Track Duration:    {analysis.duration_sec:.2f}s")
        print(f"  • Package Created:   {osz_path}")
        print(f"  • Ready to play in:  osu! (Just double-click or drag into osu!)")
        print("=" * 65 + "\n")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
