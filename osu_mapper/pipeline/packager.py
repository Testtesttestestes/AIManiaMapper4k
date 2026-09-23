"""
Beatmap packager and .osu v14 / .osz archive builder for osu!mania 4K.

Generates specification-compliant .osu v14 files including:
- [General] with Mode: 3 (mania)
- [Metadata]
- [Difficulty] with CircleSize: 4
- [TimingPoints] with Red and Green lines
- [HitObjects] with 4K columns (64, 192, 320, 448) and Long Notes
- Compression into standard .osz archive.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .pattern_generator import ManiaHitObject
from .timing import TimingPoint


@dataclass
class BeatmapMetadata:
    title: str = "AI Generated Track"
    artist: str = "AI Mania Mapper"
    creator: str = "AI AutoMapper 4K"
    version: str = "AI 4K Normal"
    audio_filename: str = "audio.wav"
    preview_time: int = -1
    overall_difficulty: float = 7.5
    hp_drain: float = 7.0
    source: str = ""
    tags: str = "ai osu mania 4k generated"


class BeatmapPackager:
    """Serializes osu!mania 4K beatmaps to .osu format and bundles .osz archives."""

    def __init__(self, metadata: BeatmapMetadata):
        self.meta = metadata

    def generate_osu_content(
        self,
        timing_points: List[TimingPoint],
        hit_objects: List[ManiaHitObject],
    ) -> str:
        """Constructs the complete valid .osu v14 text."""
        lines: List[str] = []

        # Header
        lines.append("osu file format v14")
        lines.append("")

        # [General]
        lines.append("[General]")
        lines.append(f"AudioFilename: {self.meta.audio_filename}")
        lines.append("AudioLeadIn: 0")
        lines.append(f"PreviewTime: {self.meta.preview_time}")
        lines.append("Countdown: 0")
        lines.append("SampleSet: Soft")
        lines.append("StackLeniency: 0.7")
        lines.append("Mode: 3")  # osu!mania mode
        lines.append("LetterboxInBreaks: 0")
        lines.append("SpecialStyle: 0")
        lines.append("WidescreenStoryboard: 1")
        lines.append("")

        # [Editor]
        lines.append("[Editor]")
        lines.append("DistanceSpacing: 1")
        lines.append("BeatDivisor: 4")
        lines.append("GridSize: 4")
        lines.append("TimelineZoom: 2")
        lines.append("")

        # [Metadata]
        lines.append("[Metadata]")
        lines.append(f"Title:{self.meta.title}")
        lines.append(f"TitleUnicode:{self.meta.title}")
        lines.append(f"Artist:{self.meta.artist}")
        lines.append(f"ArtistUnicode:{self.meta.artist}")
        lines.append(f"Creator:{self.meta.creator}")
        lines.append(f"Version:{self.meta.version}")
        lines.append(f"Source:{self.meta.source}")
        lines.append(f"Tags:{self.meta.tags}")
        lines.append("BeatmapID:0")
        lines.append("BeatmapSetID:-1")
        lines.append("")

        # [Difficulty]
        lines.append("[Difficulty]")
        lines.append(f"HPDrainRate:{self.meta.hp_drain:.1f}")
        lines.append("CircleSize:4")  # 4K keys
        lines.append(f"OverallDifficulty:{self.meta.overall_difficulty:.1f}")
        lines.append("ApproachRate:5")
        lines.append("SliderMultiplier:1.4")
        lines.append("SliderTickRate:1")
        lines.append("")

        # [Events]
        lines.append("[Events]")
        lines.append("//Background and Video events")
        lines.append("//Break Periods")
        lines.append("//Storyboard Layer 0 (Fail)")
        lines.append("//Storyboard Layer 1 (Pass)")
        lines.append("//Storyboard Layer 2 (Foreground)")
        lines.append("//Storyboard Layer 3 (Background)")
        lines.append("//Storyboard Layer 4 (Overlay)")
        lines.append("")

        # [TimingPoints]
        lines.append("[TimingPoints]")
        for tp in timing_points:
            lines.append(tp.to_osu_line())
        lines.append("")

        # [HitObjects]
        lines.append("[HitObjects]")
        for ho in hit_objects:
            lines.append(ho.to_osu_line())
        lines.append("")

        return "\n".join(lines)

    def package_osz(
        self,
        osu_content: str,
        audio_source_path: str,
        output_osz_path: str,
    ) -> Path:
        """
        Creates an .osz ZIP archive bundling the .osu difficulty and the audio file.
        """
        return self.package_multi_diff_osz(
            diff_items=[(self.meta, osu_content)],
            audio_source_path=audio_source_path,
            output_osz_path=output_osz_path,
        )

    @staticmethod
    def package_multi_diff_osz(
        diff_items: List[Tuple[BeatmapMetadata, str]],
        audio_source_path: str,
        output_osz_path: str,
    ) -> Path:
        """
        Bundles MULTIPLE .osu difficulty files and the shared audio track into a single .osz archive.
        """
        out_path = Path(output_osz_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        audio_src = Path(audio_source_path)
        first_meta = diff_items[0][0] if diff_items else BeatmapMetadata()
        audio_name = first_meta.audio_filename or audio_src.name

        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            # Write all difficulty files
            used_filenames = set()
            for meta, osu_content in diff_items:
                safe_artist = "".join(c for c in meta.artist if c.isalnum() or c in " -_").strip() or "AI Artist"
                safe_title = "".join(c for c in meta.title if c.isalnum() or c in " -_").strip() or "AI Track"
                safe_diff = "".join(c for c in meta.version if c.isalnum() or c in " -_[]*.").strip() or "Normal"
                safe_creator = "".join(c for c in meta.creator if c.isalnum() or c in " -_").strip() or "AIManiaMapper"

                base_name = f"{safe_artist} - {safe_title} ({safe_creator}) [{safe_diff}].osu"
                final_name = base_name
                counter = 2
                while final_name in used_filenames:
                    final_name = f"{safe_artist} - {safe_title} ({safe_creator}) [{safe_diff}_{counter}].osu"
                    counter += 1
                used_filenames.add(final_name)

                z.writestr(final_name, osu_content.encode("utf-8"))

            # Copy audio file once
            if audio_src.exists():
                z.write(audio_src, arcname=audio_name)

        return out_path
