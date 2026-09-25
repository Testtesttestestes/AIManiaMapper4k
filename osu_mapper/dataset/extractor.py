from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from osu_mapper.dataset.difficulty_calculator import calculate_osu_mania_star_rating
from osu_mapper.dataset.feature_extractor import parse_osu_for_training
from osu_mapper.dataset.source_resolver import resolve_beatmap_sources


@dataclass
class BeatmapSummary:
    osz_filename: str
    osu_file_name: str
    title: str
    artist: str
    creator: str
    version: str
    mode: int
    circle_size: float
    overall_difficulty: float
    star_rating: float
    bpm: float
    offset_ms: int
    duration_sec: float
    total_notes: int
    single_count: int
    jump_count: int
    hand_count: int
    quad_count: int
    ln_count: int
    ln_ratio: float
    avg_nps: float
    peak_nps: float
    column_distribution: List[int]
    audio_filename: str


class DatasetExtractor:
    def __init__(self, sample_dir: str | Path, output_dir: str | Path):
        self.sample_dir = Path(sample_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_all(self) -> List[BeatmapSummary]:
        osz_files, song_folders = resolve_beatmap_sources(self.sample_dir)
        summaries: List[BeatmapSummary] = []

        for osz in osz_files:
            try:
                with zipfile.ZipFile(osz, "r") as zf:
                    for name in zf.namelist():
                        if name.lower().endswith(".osu"):
                            try:
                                content = zf.read(name).decode("utf-8-sig")
                            except UnicodeDecodeError:
                                content = zf.read(name).decode("latin-1", errors="ignore")

                            s = self._summarize_osu(content, name, osz.name)
                            if s:
                                summaries.append(s)
            except Exception:
                continue

        return summaries

    def _summarize_osu(self, content: str, osu_name: str, osz_name: str) -> Optional[BeatmapSummary]:
        details = parse_osu_for_training(content, osu_name)
        if not details:
            return None

        hit_objs = details.hit_objects
        total_notes = len(hit_objs)
        if total_notes == 0:
            return None

        # Analyze chord counts and column distribution
        time_chords: Dict[int, int] = {}
        col_counts = [0, 0, 0, 0]
        ln_count = 0

        for col, start_ms, end_ms, is_ln in hit_objs:
            if 0 <= col < 4:
                col_counts[col] += 1
            if is_ln and end_ms > start_ms:
                ln_count += 1
            time_chords[start_ms] = time_chords.get(start_ms, 0) + 1

        single_c, jump_c, hand_c, quad_c = 0, 0, 0, 0
        for count in time_chords.values():
            if count == 1:
                single_c += 1
            elif count == 2:
                jump_c += 1
            elif count == 3:
                hand_c += 1
            elif count >= 4:
                quad_c += 1

        min_t = hit_objs[0][1]
        max_t = max(o[2] for o in hit_objs)
        dur_sec = max(1.0, (max_t - min_t) / 1000.0)
        avg_nps = round(total_notes / dur_sec, 2)

        # Peak NPS across 1-second sliding windows
        peak_nps = avg_nps
        sorted_times = sorted([o[1] for o in hit_objs])
        for t in sorted_times:
            win_notes = sum(1 for st in sorted_times if t <= st < t + 1000)
            if win_notes > peak_nps:
                peak_nps = float(win_notes)

        return BeatmapSummary(
            osz_filename=osz_name,
            osu_file_name=osu_name,
            title=details.version,
            artist="Artist",
            creator="Mapper",
            version=details.version,
            mode=3,
            circle_size=4.0,
            overall_difficulty=details.overall_difficulty,
            star_rating=details.star_rating,
            bpm=details.bpm,
            offset_ms=details.offset_ms,
            duration_sec=round(dur_sec, 2),
            total_notes=total_notes,
            single_count=single_c,
            jump_count=jump_c,
            hand_count=hand_c,
            quad_count=quad_c,
            ln_count=ln_count,
            ln_ratio=round(ln_count / max(1, total_notes), 3),
            avg_nps=avg_nps,
            peak_nps=round(peak_nps, 2),
            column_distribution=col_counts,
            audio_filename=details.audio_filename,
        )

    def save_manifests(self, beatmaps: List[BeatmapSummary]) -> Tuple[Path, Path]:
        json_path = self.output_dir / "manifest.json"
        csv_path = self.output_dir / "manifest.csv"

        data = [asdict(b) for b in beatmaps]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if data:
            fields = list(data[0].keys())
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for row in data:
                    row_copy = dict(row)
                    row_copy["column_distribution"] = str(row_copy["column_distribution"])
                    writer.writerow(row_copy)

        return json_path, csv_path
