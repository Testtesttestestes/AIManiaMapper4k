#!/usr/bin/env python3
"""
CLI script to prepare and tokenize training dataset from .osz beatmap archives.

Usage:
    python3 prepare_dataset.py [--input_dir "sample beatmaps"] [--output_dir "dataset_processed"] [--chunk_size 256] [--hop_size 128]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from osu_mapper.dataset.feature_extractor import (
    compute_mel_spectrogram,
    extract_audio_waveform,
    find_ffmpeg,
    parse_osu_for_training,
    quantize_beatmap_to_grid,
)


def process_archives(
    input_dir: str | Path,
    output_dir: str | Path,
    chunk_size: int = 256,
    hop_size: int = 128,
    target_sr: int = 16000,
    n_mels: int = 80,
    ffmpeg_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parses all .osz beatmaps in input_dir and saves tokenized training chunks into output_dir.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if torch is None:
        print("\n" + "=" * 76)
        print("❌ ОШИБКА: Библиотека PyTorch (torch) не установлена!")
        print("=" * 76)
        print("Для токенизации и создания обучающих тензоров требуется PyTorch.")
        print("Установите PyTorch командой:")
        print("  pip install torch numpy\n")
        print("=" * 76 + "\n")
        return {"total_archives": 0, "total_chunks": 0, "error": "torch_not_found"}

    # Pre-flight check for FFmpeg (required for mp3/ogg decoding)
    resolved_ffmpeg = find_ffmpeg(ffmpeg_path)
    if not resolved_ffmpeg:
        print("\n" + "=" * 76)
        print("❌ ОШИБКА: Утилита FFmpeg не найдена на вашей системе!")
        print("=" * 76)
        print("Для извлечения звуковых дорожек из .osz (.mp3 / .ogg) необходим FFmpeg.")
        print("При его отсутствии Windows выдает ошибку:")
        print("  [WinError 2] Не удается найти указанный файл\n")
        print("КАК ИСПРАВИТЬ (выберите ОДИН из вариантов):")
        print(" 1) [РЕКОМЕНДУЕТСЯ] Положите 'ffmpeg.exe' в папку проекта:")
        print(f"    -> {Path('ffmpeg/ffmpeg.exe').resolve()}")
        print("    (Скрипт автоматически найдет его там!)")
        print(" 2) Укажите путь к ffmpeg через аргумент:")
        print('    python prepare_dataset.py --ffmpeg "C:\\путь\\к\\ffmpeg.exe"')
        print(" 3) Задайте переменную окружения в PowerShell:")
        print('    $env:FFMPEG_PATH = "C:\\путь\\к\\ffmpeg.exe"')
        print(" 4) Либо добавьте папку с ffmpeg.exe в системный PATH Windows.")
        print("=" * 76 + "\n")
        return {"total_archives": 0, "total_chunks": 0, "error": "ffmpeg_not_found"}

    print(f"[Preprocessor] Using FFmpeg binary: {resolved_ffmpeg}")

    osz_files = sorted(list(in_path.glob("*.osz")))
    if not osz_files:
        print(f"[Warning] No .osz archives found in '{in_path}'.")
        return {"total_archives": 0, "total_chunks": 0}

    print(f"[Preprocessor] Found {len(osz_files)} .osz archives in '{in_path}'.")
    chunks_list: List[Dict[str, torch.Tensor]] = []
    metadata_records: List[Dict[str, Any]] = []

    total_diffs = 0
    total_audio_sec = 0.0

    for osz in osz_files:
        print(f"\nProcessing '{osz.name}'...")
        try:
            with zipfile.ZipFile(osz, "r") as z:
                # Find primary audio entry (.mp3, .ogg, .wav): largest file or matching AudioFilename
                audio_entry = None
                # First inspect .osu files to find AudioFilename
                for n in z.namelist():
                    if n.lower().endswith(".osu"):
                        raw = z.read(n).decode("utf-8-sig", errors="ignore")
                        for line in raw.splitlines():
                            if line.startswith("AudioFilename:"):
                                cand = line.split(":", 1)[1].strip()
                                # find case-insensitive in zip
                                for z_name in z.namelist():
                                    if z_name.lower() == cand.lower():
                                        audio_entry = z_name
                                        break
                                break
                    if audio_entry:
                        break

                # Fallback: select largest audio file in archive (>100KB)
                if not audio_entry:
                    audio_candidates = [
                        info for info in z.infolist()
                        if info.filename.lower().endswith((".mp3", ".ogg", ".wav"))
                    ]
                    if audio_candidates:
                        audio_candidates.sort(key=lambda info: info.file_size, reverse=True)
                        audio_entry = audio_candidates[0].filename

                if not audio_entry:
                    print(f"  [Skip] No valid audio file found in '{osz.name}'.")
                    continue

                audio_bytes = z.read(audio_entry)
                ext = Path(audio_entry).suffix.lower()
                waveform = extract_audio_waveform(
                    audio_bytes,
                    file_ext=ext,
                    target_sr=target_sr,
                    ffmpeg_path=resolved_ffmpeg,
                )
                if waveform is None or waveform.shape[0] < target_sr:
                    print(f"  [Skip] Failed to decode audio '{audio_entry}' in '{osz.name}'.")
                    continue

                audio_dur = waveform.shape[0] / target_sr
                total_audio_sec += audio_dur
                mel = compute_mel_spectrogram(waveform, sr=target_sr, n_mels=n_mels)
                num_frames = mel.shape[0]

                # Parse .osu files (Mode 3, CircleSize 4)
                osu_entries = [n for n in z.namelist() if n.lower().endswith(".osu")]
                matched_diffs = 0

                for osu_n in osu_entries:
                    osu_raw = z.read(osu_n)
                    try:
                        osu_text = osu_raw.decode("utf-8-sig")
                    except UnicodeDecodeError:
                        osu_text = osu_raw.decode("latin-1", errors="ignore")

                    details = parse_osu_for_training(osu_text, osu_n)
                    if not details:
                        continue

                    matched_diffs += 1
                    total_diffs += 1
                    tokens, ln_active, sv_curve = quantize_beatmap_to_grid(details, num_frames=num_frames)

                    # Slice song into training windows [chunk_size]
                    chunk_count = 0
                    for start_idx in range(0, num_frames - chunk_size + 1, hop_size):
                        end_idx = start_idx + chunk_size
                        chunk_mel = mel[start_idx:end_idx]            # [chunk_size, n_mels]
                        chunk_tok = tokens[start_idx:end_idx]          # [chunk_size]
                        chunk_ln = ln_active[start_idx:end_idx]        # [chunk_size, 4]
                        chunk_sv = sv_curve[start_idx:end_idx]         # [chunk_size]

                        chunks_list.append({
                            "audio_mel": chunk_mel,
                            "tokens": chunk_tok,
                            "ln_mask": chunk_ln,
                            "sv_curve": chunk_sv,
                            "star_rating": torch.tensor([details.star_rating], dtype=torch.float32),
                            "bpm": torch.tensor([details.bpm], dtype=torch.float32),
                            "offset_ms": torch.tensor([details.offset_ms], dtype=torch.float32),
                        })
                        chunk_count += 1

                    print(f"  - Parsed {details.version}: SR={details.star_rating:.1f}, BPM={details.bpm}, Chunks={chunk_count}")

                    metadata_records.append({
                        "archive": osz.name,
                        "osu_file": osu_n,
                        "version": details.version,
                        "bpm": details.bpm,
                        "offset_ms": details.offset_ms,
                        "star_rating": details.star_rating,
                        "total_hit_objects": len(details.hit_objects),
                        "chunks_extracted": chunk_count,
                    })

        except Exception as err:
            print(f"  [Error] Failed processing archive '{osz.name}': {err}")

    # Stack or save dataset
    dataset_file = out_path / "train_data.pt"
    meta_file = out_path / "metadata.json"

    print(f"\n[Packaging] Saving {len(chunks_list)} training chunks to '{dataset_file}'...")
    torch.save(chunks_list, dataset_file)

    summary = {
        "total_archives": len(osz_files),
        "total_diffs": total_diffs,
        "total_audio_hours": round(total_audio_sec / 3600.0, 3),
        "total_chunks": len(chunks_list),
        "chunk_size": chunk_size,
        "hop_size": hop_size,
        "n_mels": n_mels,
        "sample_rate": target_sr,
        "items": metadata_records,
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Dataset preparation complete:")
    print(f"  - Total archives processed: {len(osz_files)}")
    print(f"  - Valid 4K mania diffs:    {total_diffs}")
    print(f"  - Training chunks created: {len(chunks_list)}")
    print(f"  - Output binary tensor:    {dataset_file} ({dataset_file.stat().st_size / (1024 * 1024):.1f} MB)")
    print(f"  - Metadata manifest:       {meta_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from .osz beatmap archives.")
    parser.add_argument("--input_dir", "-i", type=str, default="sample beatmaps", help="Directory with .osz beatmaps")
    parser.add_argument("--output_dir", "-o", type=str, default="dataset_processed", help="Output directory for processed tensors")
    parser.add_argument("--chunk_size", type=int, default=256, help="Frames per sequence chunk (256 frames = ~4.09s)")
    parser.add_argument("--hop_size", type=int, default=128, help="Stride between consecutive chunks (128 frames = ~2.04s)")
    parser.add_argument("--ffmpeg", "--ffmpeg_path", type=str, default=None, help="Explicit path to ffmpeg executable (e.g. ffmpeg/ffmpeg.exe)")
    args = parser.parse_args()

    summary = process_archives(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        hop_size=args.hop_size,
        ffmpeg_path=args.ffmpeg,
    )
    if summary.get("error") == "ffmpeg_not_found":
        sys.exit(1)


if __name__ == "__main__":
    main()
