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
    slice_beatmap_into_musical_chunks,
)
from osu_mapper.dataset.source_resolver import resolve_beatmap_sources


def process_archives(
    input_dir: str | Path,
    output_dir: str | Path,
    chunk_size: int = 256,
    hop_size: int = 128,
    target_sr: int = 16000,
    n_mels: int = 80,
    ffmpeg_path: Optional[str] = None,
    align_mode: str = "beat",
    bars_per_step: int = 2,
    timing_source: str = "map",
    sync_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parses all .osz beatmaps in input_dir and saves tokenized training chunks into output_dir.
    - timing_source="map" (Default): Takes exact BPM, Offset, and SV directly from the original .osu map (Ground Truth),
      ensuring Model 2 and Model 3 train on clean, uncorrupted reference data without error propagation from Model 1.
    - timing_source="model1": Runs Model 1 (SyncBPMOffsetModel) inference on the audio to predict BPM and Offset before slicing.
    - align_mode="beat": Slices audio strictly synchronized to BPM and offset (musical measures / squares).
    - align_mode="frame": Fixed sliding window advancing by hop_size frames.
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

    # Initialize Model 1 if cascaded timing is explicitly requested
    sync_model = None
    if timing_source == "model1":
        from osu_mapper.models.sync_model import SyncBPMOffsetModel
        print(f"[Timing Strategy] Cascaded Mode: Initializing Model 1 (SyncBPMOffsetModel)...")
        sync_model = SyncBPMOffsetModel(audio_dim=n_mels, hidden_dim=256)
        ckpt_p = Path(sync_checkpoint) if sync_checkpoint else Path("checkpoints/model_1_sync_best.pt")
        if ckpt_p.exists():
            print(f"[Timing Strategy] Loading Model 1 checkpoint: '{ckpt_p}'")
            ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                sync_model.load_state_dict(ckpt["model_state"])
            elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                sync_model.load_state_dict(ckpt["model_state_dict"])
            sync_model.eval()
        else:
            print(f"[Timing Strategy Warning] Model 1 checkpoint '{ckpt_p}' not found; using untrained/heuristic fallback.")
    else:
        print("[Timing Strategy] Data Source: ORIGINAL MAP GROUND TRUTH (BPM, Offset & SV directly from .osu)")

    osz_files, song_folders = resolve_beatmap_sources(in_path)
    total_sources = len(osz_files) + len(song_folders)
    if total_sources == 0:
        print(f"[Warning] No .osz archives or song folders found for '{in_path}'.")
        print("💡 Hint: You can place .osz files in 'sample beatmaps/' or create 'sample beatmaps/sources.txt' with full paths to external map folders (e.g. D:\\osu\\Songs).")
        return {"total_archives": 0, "total_chunks": 0}

    print(f"[Preprocessor] Found {len(osz_files)} .osz archives and {len(song_folders)} song directories.")
    chunks_list: List[Dict[str, torch.Tensor]] = []
    song_sync_list: List[Dict[str, Any]] = []
    metadata_records: List[Dict[str, Any]] = []

    total_diffs = 0
    total_audio_sec = 0.0

    for osz in osz_files:
        print(f"\nProcessing '{osz.name}'...")
        try:
            with zipfile.ZipFile(osz, "r") as z:
                # Cache decoded audio representations per audio file within archive
                audio_cache: Dict[str, Tuple[torch.Tensor, int, float]] = {}

                def get_or_decode_audio(target_audio_name: str) -> Optional[Tuple[torch.Tensor, int, float]]:
                    clean_name = target_audio_name.strip().lower()
                    if clean_name in audio_cache:
                        return audio_cache[clean_name]

                    # Match case-insensitively in archive
                    matched_entry = None
                    for z_name in z.namelist():
                        if z_name.lower() == clean_name or Path(z_name).name.lower() == clean_name:
                            matched_entry = z_name
                            break

                    if not matched_entry:
                        # Fallback: largest audio file in archive
                        audio_candidates = [
                            info for info in z.infolist()
                            if info.filename.lower().endswith((".mp3", ".ogg", ".wav"))
                        ]
                        if audio_candidates:
                            audio_candidates.sort(key=lambda info: info.file_size, reverse=True)
                            matched_entry = audio_candidates[0].filename

                    if not matched_entry:
                        return None

                    try:
                        audio_bytes = z.read(matched_entry)
                        ext = Path(matched_entry).suffix.lower()
                        waveform = extract_audio_waveform(
                            audio_bytes,
                            file_ext=ext,
                            target_sr=target_sr,
                            ffmpeg_path=resolved_ffmpeg,
                        )
                        if waveform is None or waveform.shape[0] < target_sr:
                            return None

                        dur = waveform.shape[0] / target_sr
                        audio_mel = compute_mel_spectrogram(waveform, sr=target_sr, n_mels=n_mels)
                        frames = audio_mel.shape[0]
                        res = (audio_mel, frames, dur)
                        audio_cache[clean_name] = res
                        audio_cache[matched_entry.lower()] = res
                        return res
                    except Exception as dec_err:
                        print(f"  [Audio Warning] Failed to decode '{matched_entry}': {dec_err}")
                        return None

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

                    # Fetch exact audio spectrogram for this specific difficulty
                    audio_res = get_or_decode_audio(details.audio_filename)
                    if audio_res is None:
                        print(f"  [Skip {details.version}] No valid audio found matching '{details.audio_filename}' in '{osz.name}'.")
                        continue

                    mel, num_frames, audio_dur = audio_res
                    total_audio_sec += (audio_dur / len(osu_entries))  # approximate

                    # Model 1 prediction override if cascaded mode is enabled
                    timing_override = None
                    if sync_model is not None:
                        with torch.no_grad():
                            pred_b, pred_o = sync_model(mel[:1024].unsqueeze(0))
                            pred_b_val = float(pred_b.squeeze().item())
                            pred_o_val = float(pred_o.squeeze().item())
                            timing_override = (pred_b_val, pred_o_val)

                    matched_diffs += 1
                    total_diffs += 1
                    tokens, ln_active, sv_curve, lane_states, grid_metrics, ticks = quantize_beatmap_to_grid(details, num_frames=num_frames, audio_mel=mel)

                    # Slice song into beat-synchronized or frame-based training windows
                    chunks = slice_beatmap_into_musical_chunks(
                        mel=mel,
                        tokens=tokens,
                        ln_active=ln_active,
                        sv_curve=sv_curve,
                        details=details,
                        chunk_size=chunk_size,
                        hop_size=hop_size,
                        align_mode=align_mode,
                        bars_per_step=bars_per_step,
                        timing_source=timing_source,
                        timing_override=timing_override,
                        lane_states=lane_states,
                        grid_metrics=grid_metrics,
                        ticks=ticks,
                    )
                    chunks_list.extend(chunks)
                    chunk_count = len(chunks)

                    source_label = "GroundTruth .osu" if timing_source == "map" else "Model 1 Sync"
                    print(f"  - Parsed {details.version}: SR={details.star_rating:.2f}*, BPM={details.bpm}, Offset={details.offset_ms}ms, Audio='{details.audio_filename}', Chunks={chunk_count}")

                    metadata_records.append({
                        "archive": osz.name,
                        "osu_file": osu_n,
                        "version": details.version,
                        "audio_filename": details.audio_filename,
                        "bpm": details.bpm,
                        "offset_ms": details.offset_ms,
                        "timing_source": timing_source,
                        "star_rating": details.star_rating,
                        "total_hit_objects": len(details.hit_objects),
                        "chunks_extracted": chunk_count,
                    })

                    # Record sync sample for Model 1
                    if len(song_sync_list) < 5000:
                        song_sync_list.append({
                            "audio_mel": mel[:1024].clone(),
                            "bpm": torch.tensor([details.bpm], dtype=torch.float32),
                            "offset_ms": torch.tensor([details.offset_ms], dtype=torch.float32),
                            "archive": osz.name,
                        })

        except Exception as err:
            print(f"  [Error] Failed processing archive '{osz.name}': {err}")

    # Also process uncompressed song folders (e.g. osu! Songs directory)
    for sdir in song_folders:
        print(f"\nProcessing folder '{sdir.name}'...")
        try:
            audio_cache: Dict[str, Tuple[torch.Tensor, int, float]] = {}

            def get_or_decode_folder_audio(target_audio_name: str) -> Optional[Tuple[torch.Tensor, int, float]]:
                clean_name = target_audio_name.strip().lower()
                if clean_name in audio_cache:
                    return audio_cache[clean_name]

                matched_file = None
                for f in sdir.iterdir():
                    if f.is_file() and f.name.lower() == clean_name:
                        matched_file = f
                        break

                if not matched_file:
                    audio_candidates = [
                        f for f in sdir.iterdir()
                        if f.is_file() and f.suffix.lower() in (".mp3", ".ogg", ".wav")
                    ]
                    if audio_candidates:
                        audio_candidates.sort(key=lambda f: f.stat().st_size, reverse=True)
                        matched_file = audio_candidates[0]

                if not matched_file:
                    return None

                try:
                    audio_bytes = matched_file.read_bytes()
                    ext = matched_file.suffix.lower()
                    waveform = extract_audio_waveform(
                        audio_bytes,
                        file_ext=ext,
                        target_sr=target_sr,
                        ffmpeg_path=resolved_ffmpeg,
                    )
                    if waveform is None or waveform.shape[0] < target_sr:
                        return None

                    dur = waveform.shape[0] / target_sr
                    audio_mel = compute_mel_spectrogram(waveform, sr=target_sr, n_mels=n_mels)
                    frames = audio_mel.shape[0]
                    res = (audio_mel, frames, dur)
                    audio_cache[clean_name] = res
                    audio_cache[matched_file.name.lower()] = res
                    return res
                except Exception as dec_err:
                    print(f"  [Audio Warning] Failed to decode '{matched_file.name}': {dec_err}")
                    return None

            osu_files = sorted(list(sdir.glob("*.osu")))
            for osu_path in osu_files:
                try:
                    with open(osu_path, "rb") as f_obj:
                        osu_raw = f_obj.read()
                    try:
                        osu_text = osu_raw.decode("utf-8-sig")
                    except UnicodeDecodeError:
                        osu_text = osu_raw.decode("latin-1", errors="ignore")

                    details = parse_osu_for_training(osu_text, osu_path.name)
                    if not details:
                        continue

                    audio_res = get_or_decode_folder_audio(details.audio_filename)
                    if audio_res is None:
                        continue

                    mel, num_frames, audio_dur = audio_res
                    total_audio_sec += (audio_dur / max(1, len(osu_files)))

                    timing_override = None
                    if sync_model is not None:
                        with torch.no_grad():
                            pred_b, pred_o = sync_model(mel[:1024].unsqueeze(0))
                            pred_b_val = float(pred_b.squeeze().item())
                            pred_o_val = float(pred_o.squeeze().item())
                            timing_override = (pred_b_val, pred_o_val)

                    matched_diffs += 1
                    total_diffs += 1
                    tokens, ln_active, sv_curve, lane_states, grid_metrics, ticks = quantize_beatmap_to_grid(details, num_frames=num_frames, audio_mel=mel)

                    chunks = slice_beatmap_into_musical_chunks(
                        mel=mel,
                        tokens=tokens,
                        ln_active=ln_active,
                        sv_curve=sv_curve,
                        details=details,
                        chunk_size=chunk_size,
                        hop_size=hop_size,
                        align_mode=align_mode,
                        bars_per_step=bars_per_step,
                        timing_source=timing_source,
                        timing_override=timing_override,
                        lane_states=lane_states,
                        grid_metrics=grid_metrics,
                        ticks=ticks,
                    )
                    chunks_list.extend(chunks)
                    chunk_count = len(chunks)

                    source_label = "GroundTruth .osu" if timing_source == "map" else "Model 1 Sync"
                    print(f"  - Parsed {details.version}: SR={details.star_rating:.2f}*, BPM={details.bpm}, Offset={details.offset_ms}ms, Audio='{details.audio_filename}', Chunks={chunk_count}")

                    metadata_records.append({
                        "archive": sdir.name,
                        "osu_file": osu_path.name,
                        "version": details.version,
                        "audio_filename": details.audio_filename,
                        "bpm": details.bpm,
                        "offset_ms": details.offset_ms,
                        "timing_source": timing_source,
                        "star_rating": details.star_rating,
                        "total_hit_objects": len(details.hit_objects),
                        "chunks_extracted": chunk_count,
                    })

                    if len(song_sync_list) < 5000:
                        song_sync_list.append({
                            "audio_mel": mel[:1024].clone(),
                            "bpm": torch.tensor([details.bpm], dtype=torch.float32),
                            "offset_ms": torch.tensor([details.offset_ms], dtype=torch.float32),
                            "archive": sdir.name,
                        })
                except Exception as parse_err:
                    print(f"  [Warning] Failed parsing '{osu_path.name}': {parse_err}")

        except Exception as err:
            print(f"  [Error] Failed processing folder '{sdir.name}': {err}")

    # Stack or save dataset
    dataset_file = out_path / "train_data.pt"
    song_sync_file = out_path / "song_sync_data.pt"
    meta_file = out_path / "metadata.json"

    print(f"\n[Packaging] Saving {len(chunks_list)} training chunks to '{dataset_file}' (Timing Source: {timing_source.upper()})...")
    torch.save(chunks_list, dataset_file)

    if song_sync_list:
        print(f"[Packaging] Saving {len(song_sync_list)} song-level sync profiles to '{song_sync_file}'...")
        torch.save(song_sync_list, song_sync_file)

    summary = {
        "total_archives": len(osz_files),
        "total_diffs": total_diffs,
        "total_audio_hours": round(total_audio_sec / 3600.0, 3),
        "total_chunks": len(chunks_list),
        "timing_source": timing_source,
        "align_mode": align_mode,
        "bars_per_step": bars_per_step if align_mode == "beat" else None,
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
    print(f"  - Timing source strategy:  {timing_source.upper()} ({'Original Map Ground Truth' if timing_source == 'map' else 'Cascaded Model 1'})")
    print(f"  - Training chunks created: {len(chunks_list)} (align_mode='{align_mode}')")
    print(f"  - Output binary tensor:    {dataset_file} ({dataset_file.stat().st_size / (1024 * 1024):.1f} MB)")
    print(f"  - Metadata manifest:       {meta_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from .osz beatmap archives.")
    parser.add_argument("--input_dir", "-i", type=str, default="sample beatmaps", help="Directory with .osz beatmaps")
    parser.add_argument("--output_dir", "-o", type=str, default="dataset_processed", help="Output directory for processed tensors")
    parser.add_argument("--chunk_size", type=int, default=256, help="Frames per sequence chunk (256 frames = ~4.09s)")
    parser.add_argument("--hop_size", type=int, default=128, help="Stride between consecutive chunks in frame mode (128 frames = ~2.04s)")
    parser.add_argument("--align_mode", type=str, choices=["beat", "frame"], default="beat", help="Chunking alignment mode: 'beat' (BPM/offset musical measures/squares) or 'frame' (unaligned)")
    parser.add_argument("--bars_per_step", type=int, default=2, help="Number of musical bars per step in 'beat' mode (default 2 bars)")
    parser.add_argument("--timing_source", "-t", type=str, choices=["map", "model1"], default="map", help="Timing source strategy for Models 2 and 3: 'map' (Original beatmap ground truth) or 'model1' (Cascaded Model 1 predictions)")
    parser.add_argument("--sync_checkpoint", type=str, default=None, help="Path to Model 1 sync checkpoint when --timing_source=model1 is used")
    parser.add_argument("--ground_truth", "--use_ground_truth", action="store_true", help="Explicitly force taking ground truth data directly from beatmaps")
    parser.add_argument("--ffmpeg", "--ffmpeg_path", type=str, default=None, help="Explicit path to ffmpeg executable (e.g. ffmpeg/ffmpeg.exe)")
    args = parser.parse_args()

    timing_source = "map" if args.ground_truth else args.timing_source

    summary = process_archives(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        hop_size=args.hop_size,
        ffmpeg_path=args.ffmpeg,
        align_mode=args.align_mode,
        bars_per_step=args.bars_per_step,
        timing_source=timing_source,
        sync_checkpoint=args.sync_checkpoint,
    )
    if summary.get("error") == "ffmpeg_not_found":
        sys.exit(1)


if __name__ == "__main__":
    main()
