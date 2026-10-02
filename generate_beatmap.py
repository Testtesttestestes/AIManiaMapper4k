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

from osu_mapper.dataset.source_resolver import resolve_beatmap_sources
from osu_mapper.dataset.feature_extractor import (
    find_ffmpeg,
    parse_osu_for_training,
)
from osu_mapper.pipeline import generate_beatmap_pipeline
from osu_mapper.pipeline.audio import create_synthetic_click_wav
from osu_mapper.pipeline.timing import TimingPoint, parse_timing_point_line


def list_available_dataset_songs() -> List[dict]:
    """Finds all available sample beatmaps from manifest, osz archives, and text pointer files."""
    manifest_p = Path("dataset_manifest/manifest.json")
    songs = []
    seen_sources = set()

    if manifest_p.exists():
        try:
            with open(manifest_p, "r", encoding="utf-8") as f:
                data = json.load(f)
                for idx, item in enumerate(data.get("beatmaps", [])):
                    src = item.get("source_osz", "")
                    if src and Path(src).exists():
                        seen_sources.add(str(Path(src).resolve()))
                    songs.append({
                        "index": len(songs) + 1,
                        "title": item.get("title", "Unknown Title"),
                        "artist": item.get("artist", "Unknown Artist"),
                        "bpm": item.get("bpm", 0.0),
                        "offset_ms": item.get("offset_ms", 0),
                        "source_osz": src,
                        "osu_file": item.get("osu_file_name", ""),
                    })
        except Exception:
            pass

    # Resolve from sample beatmaps directory and pointer text files
    resolved_osz, resolved_folders = resolve_beatmap_sources("sample beatmaps")

    # Process discovered .osz files
    for osz_path in resolved_osz:
        if str(osz_path.resolve()) in seen_sources:
            continue
        seen_sources.add(str(osz_path.resolve()))
        title = osz_path.stem
        artist = "Unknown Artist"
        bpm = 0.0
        offset_ms = 0
        try:
            with zipfile.ZipFile(osz_path, "r") as zf:
                osu_names = [n for n in zf.namelist() if n.lower().endswith(".osu")]
                if osu_names:
                    try:
                        c = zf.read(osu_names[0]).decode("utf-8", errors="ignore")
                    except UnicodeDecodeError:
                        c = zf.read(osu_names[0]).decode("latin-1", errors="ignore")
                    meta, tps = parse_template_osu_text(c)
                    if meta.get("title"):
                        title = meta["title"]
                    if meta.get("artist"):
                        artist = meta["artist"]
                    if meta.get("bpm"):
                        bpm = meta["bpm"]
                    if meta.get("offset_ms"):
                        offset_ms = meta["offset_ms"]
        except Exception:
            pass

        songs.append({
            "index": len(songs) + 1,
            "title": title,
            "artist": artist,
            "bpm": bpm,
            "offset_ms": offset_ms,
            "source_osz": str(osz_path.resolve()),
            "osu_file": "",
        })

    # Process discovered song folders
    for folder_path in resolved_folders:
        if str(folder_path.resolve()) in seen_sources:
            continue
        seen_sources.add(str(folder_path.resolve()))
        title = folder_path.name
        artist = "Unknown Artist"
        bpm = 0.0
        offset_ms = 0
        osu_files = list(folder_path.glob("*.osu"))
        if osu_files:
            try:
                c = osu_files[0].read_text(encoding="utf-8", errors="ignore")
                meta, tps = parse_template_osu_text(c)
                if meta.get("title"):
                    title = meta["title"]
                if meta.get("artist"):
                    artist = meta["artist"]
                if meta.get("bpm"):
                    bpm = meta["bpm"]
                if meta.get("offset_ms"):
                    offset_ms = meta["offset_ms"]
            except Exception:
                pass

        songs.append({
            "index": len(songs) + 1,
            "title": title,
            "artist": artist,
            "bpm": bpm,
            "offset_ms": offset_ms,
            "source_osz": str(folder_path.resolve()),
            "osu_file": str(osu_files[0].resolve()) if osu_files else "",
        })

    return songs


def parse_template_osu_text(osu_text: str) -> Tuple[dict, List[TimingPoint]]:
    """
    Parses metadata and timing points from a template .osu file.
    Works even on empty/blank templates with 0 hit objects.
    """
    meta: dict = {}
    timing_points: List[TimingPoint] = []
    
    current_section = ""
    timing_lines: List[str] = []
    
    for line in osu_text.splitlines():
        line_str = line.strip()
        if not line_str or line_str.startswith("//"):
            continue
        if line_str.startswith("[") and line_str.endswith("]"):
            current_section = line_str[1:-1].strip()
            continue
        
        if current_section == "General" and ":" in line_str:
            k, v = line_str.split(":", 1)
            meta[k.strip()] = v.strip()
            if k.strip() == "AudioFilename":
                meta["audio_filename"] = v.strip()
        elif current_section == "Metadata" and ":" in line_str:
            k, v = line_str.split(":", 1)
            k_clean = k.strip().lower()
            if k_clean == "title":
                meta["title"] = v.strip()
            elif k_clean == "artist":
                meta["artist"] = v.strip()
            elif k_clean == "creator":
                meta["creator"] = v.strip()
            elif k_clean == "version":
                meta["version"] = v.strip()
        elif current_section == "Difficulty" and ":" in line_str:
            k, v = line_str.split(":", 1)
            meta[k.strip()] = v.strip()
        elif current_section == "TimingPoints":
            timing_lines.append(line_str)

    # Parse all timing points
    primary_bpm = None
    primary_offset = None

    for t_str in timing_lines:
        tp = parse_timing_point_line(t_str)
        if tp is None:
            continue
        timing_points.append(tp)

        # First uninherited line defines base BPM and song offset
        if tp.uninherited == 1 and tp.beat_length > 0 and primary_bpm is None:
            primary_bpm = round(60000.0 / tp.beat_length, 2)
            primary_offset = int(round(tp.time))

    if primary_bpm is not None:
        meta["bpm"] = primary_bpm
    if primary_offset is not None:
        meta["offset_ms"] = primary_offset

    return meta, timing_points


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

    # Pick first osu file (or find best match)
    osu_target = osu_files[0]
    osu_text = osu_target.read_text(encoding="utf-8", errors="ignore")
    parsed_meta, extracted_timing_points = parse_template_osu_text(osu_text)

    audio_file_name = parsed_meta.get("audio_filename") or parsed_meta.get("AudioFilename")

    # Locate audio inside extracted folder
    audio_path = None
    if audio_file_name and (extracted_dir / audio_file_name).exists():
        audio_path = str((extracted_dir / audio_file_name).resolve())
    else:
        # Search for any audio in archive
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
    parser.add_argument(
        "--sr", "--star-rating", "--diffs", "--difficulties",
        type=str,
        nargs="*",
        default=["3.5"],
        dest="star_rating",
        help="Target Star Rating(s). Accepts single value, space-separated, or comma-separated list (e.g. --sr '2.0, 2.75, 3.5, 4.25, 5.0' or --sr 2.0 2.75 3.5)",
    )
    parser.add_argument("--title", type=str, default=None, help="Song title override")
    parser.add_argument("--artist", type=str, default=None, help="Song artist override")
    parser.add_argument("--version", type=str, default=None, help="Difficulty name (default: 'AI 4K [X.X*]')")
    parser.add_argument("--creator", type=str, default="AIManiaMapper", help="Beatmap creator name")
    parser.add_argument("--bpm", type=float, default=None, help="Manual BPM override (takes precedence over AI sync)")
    parser.add_argument("--offset", type=int, default=None, help="Manual offset in ms override (takes precedence over AI sync)")
    parser.add_argument("--mode", type=str, choices=["auto", "neural", "algorithmic", "algo"], default="auto", help="Pattern generation mode: 'algorithmic' / 'algo' (musical rhythm synthesis), 'neural' (Model 3 Transformer), or 'auto'")
    parser.add_argument("--algo", "--algorithmic", action="store_true", help="Use algorithmic pattern synthesis directly (musical beat snapping & ergonomic chordstreams)")
    parser.add_argument("--neural", action="store_true", help="Force neural pattern generation via Model 3 Transformer (fails with instructions if weights missing)")
    parser.add_argument("--model3_checkpoint", "--checkpoint", "-c", type=str, default=None, help="Path to Model 3 (ManiaPatternTransformer) .pt weights")
    parser.add_argument("--sync_checkpoint", type=str, default=None, help="Path to Model 1 (SyncBPMOffsetModel) .pt weights")
    parser.add_argument("--output", "-o", type=str, default="output_beatmaps/generated_map.osz", help="Output .osz file path")
    parser.add_argument("--synthetic", action="store_true", help="Generate a synthetic test track if no audio provided")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds for synthetic audio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--temperature", "--temp", type=float, default=0.0, help="Sampling temperature: 0.0=Greedy ArgMax (matches 91.4% validation, default), >0.0=stochastic entropy")
    parser.add_argument("--top_p", "--top-p", type=float, default=0.90, help="Nucleus Top-P cumulative probability threshold (default: 0.90)")
    parser.add_argument("--greedy", action="store_true", default=True, help="Force deterministic ArgMax decoding (default: True, mirrors validation)")
    parser.add_argument("--conf", "--confidence", type=float, default=0.0, help="Active Note Confidence Threshold P(active) in [0.0..1.0] (default: 0.0, unconstrained raw neural logits).")
    parser.add_argument("--silence", "--silence_gate", type=float, default=0.0, help="Acoustic silence noise gate threshold (default: 0.0, zero artificial gating).")
    parser.add_argument("--ln_conf", "--ln-thresh", "--ln", type=float, default=0.40, help="Long Note (slider) activation threshold: 0.40 (matches 99.8% validation), 0.20=dense LN, 0.70=very few LN")
    parser.add_argument(
        "--timing_mode",
        type=str,
        choices=["soft_snap", "universal_grid", "residual", "quantized"],
        default="soft_snap",
        help="Timing advancement & snapping paradigm: "
             "'soft_snap' (continuous model step with gentle magnetic lock to any divisor, ideal for vocal rubato), "
             "'universal_grid' (universal multi-divisor discrete grid: 1/1, 1/2, 1/3, 1/4, 1/6, 1/8, 1/12, 1/16 without 1/8 restriction), "
             "'residual' (macro meter anchor + continuous micro-timing residual offset), "
             "'quantized' (legacy rigid 1/8 binary quantization baseline). Default: soft_snap.",
    )

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
            parsed_meta, template_timing_points = parse_template_osu_text(osu_text)
            template_meta.update(parsed_meta)
            audio_cand_name = parsed_meta.get("audio_filename") or parsed_meta.get("AudioFilename")
            if audio_cand_name:
                audio_cand = osu_p.parent / audio_cand_name
                if audio_cand.exists():
                    audio_path = str(audio_cand.resolve())
        elif Path(template_target).is_dir():
            folder_p = Path(template_target)
            osu_files = list(folder_p.glob("*.osu"))
            if osu_files:
                osu_text = osu_files[0].read_text(encoding="utf-8", errors="ignore")
                parsed_meta, template_timing_points = parse_template_osu_text(osu_text)
                template_meta.update(parsed_meta)
                audio_cand_name = parsed_meta.get("audio_filename") or parsed_meta.get("AudioFilename")
                if audio_cand_name and (folder_p / audio_cand_name).exists():
                    audio_path = str((folder_p / audio_cand_name).resolve())
                else:
                    for ext in ("*.mp3", "*.ogg", "*.wav", "*.flac"):
                        cands = list(folder_p.glob(ext))
                        if cands:
                            audio_path = str(cands[0].resolve())
                            break

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

    # Parse Star Rating(s): accepts comma-separated strings, space-separated lists, or single floats
    raw_sr_items = args.star_rating if isinstance(args.star_rating, list) else [args.star_rating]
    sr_list = []
    try:
        for item in raw_sr_items:
            for sub in str(item).replace(",", " ").split():
                sub_clean = sub.strip()
                if sub_clean:
                    sr_list.append(round(float(sub_clean), 2))
        if not sr_list:
            sr_list = [3.50]
    except ValueError as val_err:
        print(f"❌ [Error] Invalid --sr value '{args.star_rating}'. Please provide numbers like --sr \"2.0, 2.75, 3.5, 4.25\": {val_err}")
        sys.exit(1)

    # Resolve metadata
    final_title = args.title or template_meta.get("title", Path(audio_path).stem)
    final_artist = args.artist or template_meta.get("artist", "AI Artist")
    final_bpm = args.bpm if args.bpm is not None else template_meta.get("bpm")
    final_offset = args.offset if args.offset is not None else template_meta.get("offset_ms")

    sr_display = ", ".join(f"{s:.2f}*" for s in sr_list)

    # Determine pattern generation mode
    force_neural = bool(args.neural or (args.mode == "neural"))
    prefer_algorithmic = bool(args.algo or (args.mode in ("algorithmic", "algo")))

    mode_display = "Neural (Model 3 Transformer)" if force_neural else ("Algorithmic (Heuristic)" if prefer_algorithmic else "Auto (Neural if weights exist)")

    print("\n" + "=" * 65)
    print(" 🎮 OSU!MANIA 4K AI BEATMAP GENERATOR (MODEL 3 TRANSFORMER)")
    print("=" * 65)
    print(f"  • Audio Track:    {Path(audio_path).name}")
    print(f"  • Song Title:     {final_title}")
    print(f"  • Artist:         {final_artist}")
    print(f"  • Difficulties:   {len(sr_list)} diff(s) -> [{sr_display}]")
    print(f"  • Generation Mode:{mode_display}")
    print(f"  • Timing / Snap:  {args.timing_mode.upper()} ({'Continuous + Magnetic Snap' if args.timing_mode == 'soft_snap' else ('Universal Multi-Divisor Grid' if args.timing_mode == 'universal_grid' else ('Macro Anchor + Residual Offset' if args.timing_mode == 'residual' else 'Legacy 1/8 Binary Snapping'))})")
    print(f"  • BPM:            {f'{final_bpm:.2f} (Preset/Manual)' if final_bpm else 'Auto Sync Inference'}")
    print(f"  • Offset:         {f'{final_offset} ms (Preset/Manual)' if final_offset is not None else 'Auto Sync Inference'}")
    print(f"  • Model 3 Weights:{args.model3_checkpoint or 'Auto (checkpoints/model_3_transformer_best.pt)'}")
    print(f"  • Output Package: {args.output}")
    print("=" * 65 + "\n")

    try:
        osu_contents, osz_path, analysis = generate_beatmap_pipeline(
            wav_path=effective_wav,
            output_osz_path=args.output,
            star_rating=sr_list,
            title=final_title,
            artist=final_artist,
            version=args.version,
            creator=args.creator,
            manual_bpm=final_bpm,
            manual_offset=final_offset,
            sync_model_checkpoint=args.sync_checkpoint,
            pattern_model_checkpoint=args.model3_checkpoint,
            existing_timing_points=template_timing_points,
            force_neural=force_neural,
            prefer_algorithmic=prefer_algorithmic,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            greedy=args.greedy or (args.temperature <= 0.05),
            note_conf_thresh=args.conf,
            ln_conf_thresh=args.ln_conf,
            silence_gate=args.silence,
            timing_mode=args.timing_mode,
        )

        print("\n" + "=" * 65)
        print(" 🎉 BEATMAP PACKAGE GENERATION SUCCEEDED!")
        print("=" * 65)
        print(f"  • Difficulties:      {len(osu_contents)} generated [{sr_display}]")
        print(f"  • Final BPM:         {analysis.bpm:.2f}")
        print(f"  • Final Offset:      {analysis.offset_ms} ms")
        print(f"  • Track Duration:    {analysis.duration_sec:.2f}s")
        print(f"  • Package Created:   {osz_path}")
        print(f"  • Ready to play in:  osu! (Single .osz contains all difficulties)")
        print("-" * 65)
        print(" 📊 ROSU-PP VALIDATED DIFFICULTY SUMMARY:")
        for idx, (osu_text, target_s) in enumerate(zip(osu_contents, sr_list)):
            from osu_mapper.dataset.difficulty_calculator import calculate_osu_mania_star_rating
            v_sr = calculate_osu_mania_star_rating(osu_text=osu_text, keys=4)
            d_val = v_sr - target_s
            diff_label = f"Diff {idx + 1}"
            print(f"    • {diff_label:<8} | Target: {target_s:5.2f}* -> rosu Validated: {v_sr:5.2f}* (Delta: {d_val:+5.2f}*)")
        print("=" * 65 + "\n")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
