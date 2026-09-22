"""
End-to-end beatmap generation pipeline for osu!mania 4K.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

from .audio import AudioRhythmResult, analyze_audio_rhythm, predict_bpm_offset_with_model
from .packager import BeatmapMetadata, BeatmapPackager
from .pattern_generator import (
    ManiaHitObject,
    generate_patterns,
    generate_patterns_with_model3,
)
from .timing import TimingPoint, generate_timing_points


def generate_beatmap_pipeline(
    wav_path: str,
    output_osz_path: str,
    star_rating: float = 3.5,
    title: str = "AI Mania Map",
    artist: str = "AI Artist",
    version: Optional[str] = None,
    creator: str = "AIManiaMapper",
    manual_bpm: Optional[float] = None,
    manual_offset: Optional[int] = None,
    sync_model_checkpoint: Optional[str] = None,
    dynamics_model_checkpoint: Optional[str] = None,
    pattern_model_checkpoint: Optional[str] = None,
    existing_timing_points: Optional[List[TimingPoint]] = None,
    seed: int = 42,
    device: str = "auto",
) -> Tuple[str, str, AudioRhythmResult]:
    """
    Executes the full generation pipeline:
    1. Step 1 (Model 1 / Sync or Manual): BPM and offset synchronization
    2. Step 2 (Model 2 / Dynamics or Pre-set): SV & timing zones
    3. Step 3 (Model 3 / Transformer Neural Patterns): 4K Hit Object pattern synthesis using trained weights
    4. Step 4 (Packaging): Serialization to .osu v14 and bundling into .osz archive.

    Returns:
        (osu_content_str, generated_osz_path, audio_analysis_result)
    """
    audio_analysis = analyze_audio_rhythm(wav_path)

    effective_bpm = audio_analysis.bpm
    effective_offset = audio_analysis.offset_ms

    # Step 1: Use Model 1 (SyncBPMOffsetModel) if checkpoint provided or exists in checkpoints/
    checkpoint_to_try = sync_model_checkpoint
    if checkpoint_to_try is None:
        for cand in (Path("checkpoints/model_1_sync_best.pt"), Path("checkpoints/model_1_sync_last.pt")):
            if cand.exists():
                checkpoint_to_try = str(cand)
                break

    if checkpoint_to_try and Path(checkpoint_to_try).exists() and (manual_bpm is None or manual_offset is None):
        try:
            model_bpm, model_offset = predict_bpm_offset_with_model(
                wav_path=wav_path,
                checkpoint_path=checkpoint_to_try,
                device=device,
            )
            print(f"[Model 1: Sync] Inferred BPM={model_bpm:.2f}, Offset={model_offset}ms")
            if manual_bpm is None:
                effective_bpm = model_bpm
            if manual_offset is None:
                effective_offset = model_offset
        except Exception as ex:
            print(f"[Model 1: Sync] Note: using audio analysis fallback: {ex}")

    # Manual user overrides always take absolute precedence if explicitly given
    if manual_bpm is not None:
        effective_bpm = manual_bpm
        print(f"[Timing] Using manual/template BPM: {effective_bpm:.2f}")
    if manual_offset is not None:
        effective_offset = manual_offset
        print(f"[Timing] Using manual/template Offset: {effective_offset} ms")

    if version is None:
        version = f"AI 4K [{star_rating:.1f}*]"

    # Step 2: SV & timing points
    if existing_timing_points is not None and len(existing_timing_points) > 0:
        print(f"[Timing] Preserving {len(existing_timing_points)} user-defined timing points from template.")
        timing_points = existing_timing_points
    else:
        timing_points = generate_timing_points(
            bpm=effective_bpm,
            offset_ms=effective_offset,
            duration_ms=audio_analysis.duration_ms,
            rms_profile=audio_analysis.rms_profile,
        )

    # Step 3: Model 3 (ManiaPatternTransformer) Neural Pattern Synthesis
    m3_ckpt = pattern_model_checkpoint
    if m3_ckpt is None:
        for cand in (Path("checkpoints/model_3_transformer_best.pt"), Path("checkpoints/model_3_transformer_last.pt")):
            if cand.exists():
                m3_ckpt = str(cand)
                break

    hit_objects: List[ManiaHitObject] = []
    used_neural = False

    if m3_ckpt and Path(m3_ckpt).exists():
        try:
            from osu_mapper.dataset.feature_extractor import compute_mel_spectrogram, extract_audio_waveform
            audio_wav = extract_audio_waveform(wav_path, target_sr=16000)
            mel = compute_mel_spectrogram(audio_wav, sr=16000, n_mels=80)
            hit_objects = generate_patterns_with_model3(
                checkpoint_path=m3_ckpt,
                audio_mel=mel,
                bpm=effective_bpm,
                offset_ms=effective_offset,
                duration_ms=audio_analysis.duration_ms,
                star_rating=star_rating,
            )
            used_neural = True
        except Exception as err:
            print(f"⚠️ [Model 3 Warning] Neural pattern synthesis failed ({err}), falling back to algorithmic generation.")

    if not used_neural or not hit_objects:
        print(f"ℹ️ [Pattern Generator] Using algorithmic pattern synthesis (Conditioned on SR {star_rating:.1f}*)...")
        hit_objects = generate_patterns(
            bpm=effective_bpm,
            offset_ms=effective_offset,
            duration_ms=audio_analysis.duration_ms,
            star_rating=star_rating,
            onsets_ms=audio_analysis.onsets_ms,
            seed=seed,
        )

    # Set up metadata & packager
    meta = BeatmapMetadata(
        title=title,
        artist=artist,
        creator=creator,
        version=version,
        audio_filename=Path(wav_path).name,
        overall_difficulty=min(10.0, max(4.0, 4.0 + star_rating * 0.6)),
        hp_drain=min(10.0, max(5.0, 5.0 + star_rating * 0.45)),
    )
    packager = BeatmapPackager(meta)
    osu_content = packager.generate_osu_content(timing_points, hit_objects)

    # Build .osz package
    final_osz = packager.package_osz(
        osu_content=osu_content,
        audio_source_path=wav_path,
        output_osz_path=output_osz_path,
    )

    return osu_content, str(final_osz), audio_analysis
