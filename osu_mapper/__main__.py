#!/usr/bin/env python3
"""
CLI entry point for osu_mapper package:
    python -m osu_mapper extract <input_dir> <output_dir>
    python -m osu_mapper generate <audio.wav> [options]
"""

import argparse
import sys
from pathlib import Path

from osu_mapper.dataset.extractor import extract_dataset_from_directory
from osu_mapper.pipeline import generate_beatmap_pipeline


def run_extract(args):
    sample_dir = args.input_dir or "sample beatmaps"
    output_dir = args.output_dir or "dataset_manifest"
    maps = extract_dataset_from_directory(sample_dir, output_dir)
    print(f"извлечено {len(maps)} сложностей mania 4K.")
    return 0


def run_generate(args):
    osu_str, osz_path, analysis = generate_beatmap_pipeline(
        wav_path=args.audio,
        output_osz_path=args.output,
        star_rating=args.sr,
        title=args.title,
        artist=args.artist,
        version=args.version,
    )
    print(f"Успешно сгенерирована карта: {osz_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(prog="python -m osu_mapper", description="osu!mania 4K AI mapper and dataset tools")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    # extract subcommand
    p_extract = subparsers.add_parser("extract", help="Extract mania 4K beatmap metadata and manifests from .osz files")
    p_extract.add_argument("input_dir", nargs="?", default="sample beatmaps", help="Directory containing .osz files")
    p_extract.add_argument("output_dir", nargs="?", default="dataset_manifest", help="Target output directory for manifests")
    p_extract.set_defaults(func=run_extract)

    # generate subcommand
    p_gen = subparsers.add_parser("generate", help="Generate osu!mania 4K beatmap from audio")
    p_gen.add_argument("audio", help="Input .wav file")
    p_gen.add_argument("--sr", type=float, default=4.0, help="Target star rating (1.0 to 9.9)")
    p_gen.add_argument("--title", type=str, default="AI Generated Track", help="Beatmap title")
    p_gen.add_argument("--artist", type=str, default="AI Mania Mapper", help="Beatmap artist")
    p_gen.add_argument("--version", type=str, default="AI 4K Normal", help="Difficulty name")
    p_gen.add_argument("-o", "--output", type=str, default="output_beatmaps/generated_map.osz", help="Output .osz path")
    p_gen.set_defaults(func=run_generate)

    parsed = parser.parse_args()
    if not parsed.command:
        parser.print_help()
        sys.exit(1)

    sys.exit(parsed.func(parsed))


if __name__ == "__main__":
    main()
