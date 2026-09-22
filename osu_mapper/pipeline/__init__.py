"""
End-to-end beatmap generation pipeline for osu!mania 4K.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from .audio import AudioRhythmResult, analyze_audio_rhythm, predict_bpm_offset_with_model
from .packager import BeatmapMetadata, BeatmapPackager
from .pattern_generator import ManiaHitObject, generate_patterns
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
    seed: int = 42,
    device: str = "auto",
) -> Tuple[str, str, AudioRhythmResult]:
    """
    Executes the full pipeline:
    1. Step 1 (Model 1 / Rhythm): Predict BPM and beat offset (from neural model or audio analysis)
    2. Step 2 (Model 2 / SV): Timing points & SV dynamics generation based on timing grid
    3. Step 3 (Model 3 / Patterns): Pattern generation conditioned on Star Rating and timing grid
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
        default_ckpt = Path("checkpoints/model_1_sync_best.pt")
        if default_ckpt.exists():
            checkpoint_to_try = str(default_ckpt)

    if checkpoint_to_try and Path(checkpoint_to_try).exists():
        try:
            model_bpm, model_offset = predict_bpm_offset_with_model(
                wav_path=wav_path,
                checkpoint_path=checkpoint_to_try,
                device=device,
            )
            print(f"[Model 1: Sync] Model predicted BPM={model_bpm:.2f}, Offset={model_offset}ms")
            effective_bpm = model_bpm
            effective_offset = model_offset
        except Exception as ex:
            print(f"[Model 1: Sync] Note: using audio analysis fallback: {ex}")

    # Manual user overrides always take precedence if explicitly given
    if manual_bpm is not None:
        effective_bpm = manual_bpm
    if manual_offset is not None:
        effective_offset = manual_offset

    if version is None:
        version = f"AI 4K [{star_rating:.1f}*]"

    # Step 2: Generate SV & timing points aligned to Model 1's BPM and Offset
    timing_points = generate_timing_points(
        bpm=effective_bpm,
        offset_ms=effective_offset,
        duration_ms=audio_analysis.duration_ms,
        rms_profile=audio_analysis.rms_profile,
    )

    # Step 3: Generate hit objects conditioned on star rating and beat-aligned grid
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
