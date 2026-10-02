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
from osu_mapper.dataset.difficulty_calculator import calculate_osu_mania_star_rating


def generate_beatmap_pipeline(
    wav_path: str,
    output_osz_path: str,
    star_rating: float | List[float] = 3.5,
    title: str = "AI Mania Map",
    artist: str = "AI Artist",
    version: Optional[str | List[str]] = None,
    creator: str = "AIManiaMapper",
    manual_bpm: Optional[float] = None,
    manual_offset: Optional[int] = None,
    sync_model_checkpoint: Optional[str] = None,
    dynamics_model_checkpoint: Optional[str] = None,
    pattern_model_checkpoint: Optional[str] = None,
    existing_timing_points: Optional[List[TimingPoint]] = None,
    force_neural: bool = False,
    prefer_algorithmic: bool = False,
    seed: int = 42,
    device: str = "auto",
    temperature: float = 0.0,
    top_p: float = 0.90,
    greedy: bool = False,
    note_conf_thresh: float = 0.0,
    ln_conf_thresh: float = 0.78,
    silence_gate: float = 0.0,
    timing_mode: str = "soft_snap",
) -> Tuple[List[str], str, AudioRhythmResult]:
    """
    Executes the full generation pipeline:
    1. Step 1 (Model 1 / Sync or Manual): BPM and offset synchronization
    2. Step 2 (Model 2 / Dynamics or Pre-set): SV & timing zones
    3. Step 3 (Model 3 / Transformer Neural Patterns): 4K Hit Object pattern synthesis using trained weights
    4. Step 4 (Packaging): Serialization to .osu v14 and bundling into a single .osz archive.

    Supports generating MULTIPLE difficulties in one call by passing a list of star_ratings:
        star_rating=[2.0, 2.75, 3.5, 4.25, 5.0]

    Returns:
        (list_of_osu_content_strings, generated_osz_path, audio_analysis_result)
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

    # Update audio_analysis with effective synchronized values
    audio_analysis.bpm = round(effective_bpm, 2)
    audio_analysis.offset_ms = int(round(effective_offset))

    # Normalize star ratings list
    if isinstance(star_rating, (int, float)):
        sr_list = [float(star_rating)]
    else:
        sr_list = [float(s) for s in star_rating]

    # Normalize versions list
    if version is None:
        version_list = [f"AI 4K [{sr:.2f}*]" for sr in sr_list]
    elif isinstance(version, str):
        if len(sr_list) == 1:
            version_list = [version]
        else:
            version_list = [f"{version} [{sr:.2f}*]" for sr in sr_list]
    else:
        version_list = list(version)
        while len(version_list) < len(sr_list):
            missing_idx = len(version_list)
            version_list.append(f"AI 4K [{sr_list[missing_idx]:.2f}*]")

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
    if m3_ckpt is None and not prefer_algorithmic:
        for cand in (Path("checkpoints/model_3_transformer_best.pt"), Path("checkpoints/model_3_transformer_last.pt")):
            if cand.exists():
                m3_ckpt = str(cand)
                break

    if force_neural and (m3_ckpt is None or not Path(m3_ckpt).exists()):
        raise FileNotFoundError(
            f"Neural generation was requested (--neural), but Model 3 checkpoint was not found!\n"
            f"Expected file: checkpoints/model_3_transformer_best.pt (or specify with -c <path>).\n"
            f"To train Model 3 weights first, run:\n"
            f"  python train.py --model transformer --epochs 15"
        )

    # Pre-extract spectrogram once for all difficulties
    mel = None
    if m3_ckpt and Path(m3_ckpt).exists() and not prefer_algorithmic:
        try:
            from osu_mapper.dataset.feature_extractor import compute_mel_spectrogram, extract_audio_waveform
            audio_wav = extract_audio_waveform(wav_path, target_sr=16000)
            if audio_wav is not None:
                mel = compute_mel_spectrogram(audio_wav, sr=16000, n_mels=80)
        except Exception as ex:
            if force_neural:
                raise RuntimeError(f"Failed to extract audio spectrogram for neural generation: {ex}")
            print(f"[Mel Extraction] Note: {ex}")

    diff_packages: List[Tuple[BeatmapMetadata, str]] = []
    osu_contents: List[str] = []

    for idx, (sr, diff_name) in enumerate(zip(sr_list, version_list)):
        print(f"\n--- [Difficulty {idx + 1}/{len(sr_list)}] Generating '{diff_name}' (Target SR: {sr:.2f}*) ---")
        hit_objects: List[ManiaHitObject] = []
        used_neural = False

        if m3_ckpt and Path(m3_ckpt).exists() and mel is not None and not prefer_algorithmic:
            try:
                hit_objects = generate_patterns_with_model3(
                    checkpoint_path=m3_ckpt,
                    audio_mel=mel,
                    bpm=effective_bpm,
                    offset_ms=effective_offset,
                    duration_ms=audio_analysis.duration_ms,
                    star_rating=sr,
                    timing_points=timing_points,
                    temperature=temperature,
                    top_p=top_p,
                    greedy=greedy,
                    note_conf_thresh=note_conf_thresh,
                    ln_conf_thresh=ln_conf_thresh,
                    silence_gate=silence_gate,
                    timing_mode=timing_mode,
                )
                used_neural = True
                print(f"✨ [Model 3 Transformer] Successfully generated {len(hit_objects)} neural hit objects.")
            except Exception as err:
                if force_neural:
                    raise RuntimeError(f"Neural pattern generation failed: {err}")
                print(f"⚠️ [Model 3 Warning] Neural pattern synthesis failed ({err}), falling back to algorithmic generation.")

        if not used_neural or not hit_objects:
            print(f"ℹ️ [Pattern Generator] Using algorithmic pattern synthesis (Conditioned on SR {sr:.2f}*)...")
            hit_objects = generate_patterns(
                bpm=effective_bpm,
                offset_ms=effective_offset,
                duration_ms=audio_analysis.duration_ms,
                star_rating=sr,
                onsets_ms=audio_analysis.onsets_ms,
                timing_points=timing_points,
                seed=seed + idx,
            )

        # Set up difficulty metadata
        meta = BeatmapMetadata(
            title=title,
            artist=artist,
            creator=creator,
            version=diff_name,
            audio_filename=Path(wav_path).name,
            overall_difficulty=min(10.0, max(4.0, 4.0 + sr * 0.6)),
            hp_drain=min(10.0, max(5.0, 5.0 + sr * 0.45)),
        )
        packager = BeatmapPackager(meta)
        osu_str = packager.generate_osu_content(timing_points, hit_objects)

        # Initial rosu-pp Star Rating validation
        validated_sr = calculate_osu_mania_star_rating(
            osu_text=osu_str,
            keys=4,
            overall_difficulty=meta.overall_difficulty,
        )
        delta = validated_sr - sr

        # Pack single difficulty package directly without artificial multi-pass inflation
        diff_packages.append((meta, osu_str))
        osu_contents.append(osu_str)

        status_tag = "✓ EXACT" if abs(delta) <= 0.25 else ("~ CLOSE" if abs(delta) <= 0.60 else "ℹ️ VALIDATED")
        print(f"  🔍 [rosu-pp Difficulty Validation] Target: {sr:.2f}* -> Validated: {validated_sr:.2f}* (Delta: {delta:+.2f}*) [{status_tag}]")

    # Bundle all difficulties into single .osz archive
    final_osz = BeatmapPackager.package_multi_diff_osz(
        diff_items=diff_packages,
        audio_source_path=wav_path,
        output_osz_path=output_osz_path,
    )

    return osu_contents, str(final_osz), audio_analysis
