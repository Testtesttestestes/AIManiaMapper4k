"""
End-to-end beatmap generation pipeline for osu!mania 4K.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from .audio import AudioRhythmResult, analyze_audio_rhythm
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
    seed: int = 42,
) -> Tuple[str, str, AudioRhythmResult]:
    """
    Executes the full pipeline:
    1. Audio analysis (transients, BPM, offset)
    2. Timing points & SV dynamics generation
    3. Pattern generation conditioned on Star Rating (1.0 to 9.9)
    4. Serialization to .osu v14 and bundling into .osz archive.

    Returns:
        (osu_content_str, generated_osz_path, audio_analysis_result)
    """
    audio_analysis = analyze_audio_rhythm(wav_path)

    effective_bpm = manual_bpm if manual_bpm is not None else audio_analysis.bpm
    effective_offset = manual_offset if manual_offset is not None else audio_analysis.offset_ms

    if version is None:
        version = f"AI 4K [{star_rating:.1f}*]"

    # Generate SV & timing points
    timing_points = generate_timing_points(
        bpm=effective_bpm,
        offset_ms=effective_offset,
        duration_ms=audio_analysis.duration_ms,
        rms_profile=audio_analysis.rms_profile,
    )

    # Generate hit objects conditioned on star rating
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
