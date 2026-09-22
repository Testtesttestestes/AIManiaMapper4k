#!/usr/bin/env python3
"""
CLI entry point for generating osu!mania 4K beatmaps from WAV audio.

Usage examples:
    python3 generate_beatmap.py audio.wav --sr 4.5 --output "output_beatmaps/my_map.osz"
    python3 generate_beatmap.py --synthetic --duration 30 --sr 3.8
"""

import argparse
import os
import sys
from pathlib import Path

from osu_mapper.pipeline import generate_beatmap_pipeline
from osu_mapper.pipeline.audio import create_synthetic_click_wav


def main():
    parser = argparse.ArgumentParser(description="AI Beatmap Generator for osu!mania 4K")
    parser.add_argument("audio", nargs="?", default=None, help="Path to input .wav audio file")
    parser.add_argument("--sr", "--star-rating", type=float, default=3.5, dest="star_rating", help="Target difficulty Star Rating (1.0 to 9.9)")
    parser.add_argument("--title", type=str, default="AI Generated Track", help="Song title")
    parser.add_argument("--artist", type=str, default="AI Mania Mapper", help="Song artist")
    parser.add_argument("--version", type=str, default=None, help="Difficulty name (default: 'AI 4K [X.X*]')")
    parser.add_argument("--creator", type=str, default="AIManiaMapper", help="Beatmap creator name")
    parser.add_argument("--bpm", type=float, default=None, help="Manual BPM override")
    parser.add_argument("--offset", type=int, default=None, help="Manual offset in ms override")
    parser.add_argument("--sync_checkpoint", "--sync_model", type=str, default=None, help="Path to Model 1 (SyncBPMOffsetModel) .pt checkpoint")
    parser.add_argument("--output", "-o", type=str, default="output_beatmaps/generated_map.osz", help="Output .osz file path")
    parser.add_argument("--synthetic", action="store_true", help="Generate a synthetic rhythm track if no audio provided")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds for synthetic audio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")

    args = parser.parse_args()

    audio_path = args.audio
    if not audio_path or args.synthetic:
        synth_path = "output_beatmaps/synthetic_test.wav"
        print(f"[Generator] Synthesizing rhythmic test audio ({args.duration}s) to {synth_path}...")
        create_synthetic_click_wav(synth_path, duration_sec=args.duration, bpm=args.bpm or 140.0)
        audio_path = synth_path

    if not os.path.exists(audio_path):
        print(f"[Error] Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"==================================================")
    print(f"osu!mania 4K AI Beatmap Generator")
    print(f"Audio:           {audio_path}")
    print(f"Star Rating:     {args.star_rating:.1f}*")
    print(f"Sync Checkpoint: {args.sync_checkpoint or 'Auto-detected'}")
    print(f"Output:          {args.output}")
    print(f"==================================================")

    osu_str, osz_path, analysis = generate_beatmap_pipeline(
        wav_path=audio_path,
        output_osz_path=args.output,
        star_rating=args.star_rating,
        title=args.title,
        artist=args.artist,
        version=args.version,
        creator=args.creator,
        manual_bpm=args.bpm,
        manual_offset=args.offset,
        sync_model_checkpoint=args.sync_checkpoint,
        seed=args.seed,
    )

    print(f"\n[Success] Beatmap generation complete!")
    print(f"Detected BPM:     {analysis.bpm}")
    print(f"Detected Offset:  {analysis.offset_ms} ms")
    print(f"Audio Duration:   {analysis.duration_sec:.2f} s")
    print(f"Detected Onsets:  {len(analysis.onsets_ms)}")
    print(f"Generated .osz:   {osz_path}")
    print(f"Total .osu lines: {len(osu_str.splitlines())}")


if __name__ == "__main__":
    main()
