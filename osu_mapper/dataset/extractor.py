"""
Extractor module for osu!mania 4K beatmaps.

Scans directories for .osz archives, inspects contained .osu files,
filters for Mode: 3 (mania) and CircleSize: 4 (4K), extracts comprehensive
metadata, timing structures, SV dynamics, note density, LN ratio, and pattern distributions,
and generates structured manifest.json and manifest.csv exports.
"""

from __future__ import annotations

import csv
import json
import math
import os
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TimingPointData:
    time: float
    beat_length: float
    bpm: Optional[float]
    sv_multiplier: Optional[float]
    meter: int
    uninherited: bool  # True for red line (BPM), False for green line (SV)
    volume: int


@dataclass
class HitObjectData:
    column: int  # 0 to 3
    start_time: int
    end_time: int
    is_ln: bool


@dataclass
class BeatmapInfo:
    archive_name: str
    osu_file_name: str
    title: str
    artist: str
    creator: str
    version: str  # difficulty name
    audio_filename: str
    audio_size_bytes: int
    mode: int
    circle_size: float
    overall_difficulty: float
    hp_drain: float
    bpm: float
    min_bpm: float
    max_bpm: float
    first_offset_ms: int
    uninherited_timing_count: int
    inherited_timing_count: int
    min_sv: float
    max_sv: float
    mean_sv: float
    total_hit_objects: int
    total_notes: int
    total_lns: int
    ln_ratio: float
    duration_ms: int
    duration_sec: float
    avg_nps: float
    peak_nps: float
    single_count: int
    jump_count: int
    hand_count: int
    quad_count: int
    column_distribution: List[int] = field(default_factory=lambda: [0, 0, 0, 0])


class DatasetExtractor:
    """Extracts metadata and statistics from osu!mania 4K .osz archives."""

    def __init__(self, sample_dir: str = "sample beatmaps", output_dir: str = "dataset_manifest"):
        self.sample_dir = Path(sample_dir)
        self.output_dir = Path(output_dir)

    @staticmethod
    def parse_osu_text(content: str, archive_name: str, osu_file_name: str, audio_size_map: Dict[str, int]) -> Optional[BeatmapInfo]:
        """Parses raw .osu file text and returns BeatmapInfo if valid mania 4K."""
        lines = content.splitlines()
        
        # Section markers
        current_section = ""
        general: Dict[str, str] = {}
        metadata: Dict[str, str] = {}
        difficulty: Dict[str, str] = {}
        timing_raw: List[str] = []
        hitobjects_raw: List[str] = []

        for line in lines:
            line_str = line.strip()
            if not line_str or line_str.startswith("//"):
                continue
            if line_str.startswith("[") and line_str.endswith("]"):
                current_section = line_str[1:-1].strip()
                continue
            
            if current_section == "General":
                if ":" in line_str:
                    k, v = line_str.split(":", 1)
                    general[k.strip()] = v.strip()
            elif current_section == "Metadata":
                if ":" in line_str:
                    k, v = line_str.split(":", 1)
                    metadata[k.strip()] = v.strip()
            elif current_section == "Difficulty":
                if ":" in line_str:
                    k, v = line_str.split(":", 1)
                    difficulty[k.strip()] = v.strip()
            elif current_section == "TimingPoints":
                timing_raw.append(line_str)
            elif current_section == "HitObjects":
                hitobjects_raw.append(line_str)

        # Filter strictly for Mode: 3 (mania) and CircleSize: 4 (4K)
        mode = int(general.get("Mode", 0))
        cs_str = difficulty.get("CircleSize", "0")
        try:
            circle_size = float(cs_str)
        except ValueError:
            circle_size = 0.0

        if mode != 3 or round(circle_size) != 4:
            return None

        # Parse Timing Points
        uninherited_bpms: List[float] = []
        sv_values: List[float] = []
        first_offset = 0
        has_first_offset = False
        uninherited_count = 0
        inherited_count = 0

        for line in timing_raw:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                beat_length = float(parts[1])
                meter = int(parts[2]) if len(parts) > 2 else 4
                uninherited = True if len(parts) <= 6 else (int(parts[6]) == 1)
                
                if uninherited:
                    uninherited_count += 1
                    if beat_length > 0:
                        bpm = 60000.0 / beat_length
                        uninherited_bpms.append(bpm)
                    if not has_first_offset:
                        first_offset = int(round(t))
                        has_first_offset = True
                else:
                    inherited_count += 1
                    if beat_length < 0:
                        sv = -100.0 / beat_length
                        sv_values.append(sv)
            except Exception:
                continue

        base_bpm = uninherited_bpms[0] if uninherited_bpms else 120.0
        min_bpm = min(uninherited_bpms) if uninherited_bpms else base_bpm
        max_bpm = max(uninherited_bpms) if uninherited_bpms else base_bpm
        min_sv = min(sv_values) if sv_values else 1.0
        max_sv = max(sv_values) if sv_values else 1.0
        mean_sv = (sum(sv_values) / len(sv_values)) if sv_values else 1.0

        # Parse HitObjects
        hit_objects: List[HitObjectData] = []
        column_counts = [0, 0, 0, 0]
        notes_by_time: Dict[int, int] = {}

        for line in hitobjects_raw:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                x = int(parts[0])
                t = int(round(float(parts[2])))
                obj_type = int(parts[3])

                # osu!mania 4K column calculation:
                # 0 = left, 1 = mid-left, 2 = mid-right, 3 = right
                col = min(3, max(0, int(math.floor(x * 4 / 512))))
                column_counts[col] += 1

                is_ln = bool(obj_type & 128)
                end_time = t
                if is_ln and len(parts) >= 6:
                    end_raw = parts[5].split(":")[0]
                    try:
                        end_time = int(round(float(end_raw)))
                    except ValueError:
                        end_time = t
                
                hit_objects.append(HitObjectData(column=col, start_time=t, end_time=end_time, is_ln=is_ln))
                notes_by_time[t] = notes_by_time.get(t, 0) + 1
            except Exception:
                continue

        if not hit_objects:
            return None

        hit_objects.sort(key=lambda o: (o.start_time, o.column))
        total_hit_objects = len(hit_objects)
        total_lns = sum(1 for o in hit_objects if o.is_ln)
        total_notes = total_hit_objects
        ln_ratio = total_lns / total_notes if total_notes > 0 else 0.0

        first_time = hit_objects[0].start_time
        last_time = max(o.end_time for o in hit_objects)
        duration_ms = max(1, last_time - first_time)
        duration_sec = duration_ms / 1000.0

        avg_nps = (total_notes / duration_sec) if duration_sec > 0 else 0.0

        # Calculate peak NPS via 1.0-second sliding windows
        window_size_ms = 1000
        step_ms = 250
        peak_nps = 0.0
        cur_start = first_time
        while cur_start <= last_time:
            w_end = cur_start + window_size_ms
            count = sum(1 for o in hit_objects if cur_start <= o.start_time < w_end)
            if count > peak_nps:
                peak_nps = float(count)
            cur_start += step_ms

        # Pattern chord distributions:
        # single (1 note), jump (2 notes), hand (3 notes), quad (4 notes)
        single_count = sum(1 for c in notes_by_time.values() if c == 1)
        jump_count = sum(1 for c in notes_by_time.values() if c == 2)
        hand_count = sum(1 for c in notes_by_time.values() if c == 3)
        quad_count = sum(1 for c in notes_by_time.values() if c >= 4)

        audio_file = general.get("AudioFilename", "")
        audio_size = audio_size_map.get(audio_file.lower(), 0)

        od_str = difficulty.get("OverallDifficulty", "8")
        hp_str = difficulty.get("HPDrainRate", "8")
        try:
            od = float(od_str)
        except ValueError:
            od = 8.0
        try:
            hp = float(hp_str)
        except ValueError:
            hp = 8.0

        return BeatmapInfo(
            archive_name=archive_name,
            osu_file_name=osu_file_name,
            title=metadata.get("Title", "Unknown Title"),
            artist=metadata.get("Artist", "Unknown Artist"),
            creator=metadata.get("Creator", "Unknown Mapper"),
            version=metadata.get("Version", "Normal"),
            audio_filename=audio_file,
            audio_size_bytes=audio_size,
            mode=mode,
            circle_size=circle_size,
            overall_difficulty=od,
            hp_drain=hp,
            bpm=round(base_bpm, 2),
            min_bpm=round(min_bpm, 2),
            max_bpm=round(max_bpm, 2),
            first_offset_ms=first_offset,
            uninherited_timing_count=uninherited_count,
            inherited_timing_count=inherited_count,
            min_sv=round(min_sv, 3),
            max_sv=round(max_sv, 3),
            mean_sv=round(mean_sv, 3),
            total_hit_objects=total_hit_objects,
            total_notes=total_notes,
            total_lns=total_lns,
            ln_ratio=round(ln_ratio, 4),
            duration_ms=duration_ms,
            duration_sec=round(duration_sec, 2),
            avg_nps=round(avg_nps, 2),
            peak_nps=round(peak_nps, 2),
            single_count=single_count,
            jump_count=jump_count,
            hand_count=hand_count,
            quad_count=quad_count,
            column_distribution=column_counts,
        )

    def extract_from_archive(self, archive_path: Path) -> List[BeatmapInfo]:
        """Extracts all 4K mania beatmaps from a single .osz archive."""
        results: List[BeatmapInfo] = []
        try:
            with zipfile.ZipFile(archive_path, "r") as z:
                # Map lowercase names to file sizes
                audio_size_map: Dict[str, int] = {}
                for info in z.infolist():
                    audio_size_map[info.filename.lower()] = info.file_size

                osu_entries = [n for n in z.namelist() if n.lower().endswith(".osu")]
                for entry in osu_entries:
                    try:
                        raw_data = z.read(entry)
                        # Attempt UTF-8 with fallback
                        try:
                            content = raw_data.decode("utf-8-sig")
                        except UnicodeDecodeError:
                            content = raw_data.decode("latin-1")
                        
                        info = self.parse_osu_text(content, archive_path.name, entry, audio_size_map)
                        if info:
                            results.append(info)
                    except Exception as err:
                        print(f"[Warning] Failed parsing {entry} in {archive_path.name}: {err}")
        except Exception as err:
            print(f"[Warning] Failed reading archive {archive_path.name}: {err}")
        return results

    def extract_from_folder(self, folder_path: Path) -> List[BeatmapInfo]:
        """Extracts all 4K mania beatmaps from an uncompressed beatmap folder."""
        results: List[BeatmapInfo] = []
        try:
            # Map lowercase filenames to file sizes
            audio_size_map: Dict[str, int] = {}
            for f in folder_path.iterdir():
                if f.is_file():
                    audio_size_map[f.name.lower()] = f.stat().st_size

            for osu_path in folder_path.glob("*.osu"):
                try:
                    with open(osu_path, "rb") as f:
                        raw_data = f.read()
                    try:
                        content = raw_data.decode("utf-8-sig")
                    except UnicodeDecodeError:
                        content = raw_data.decode("latin-1")

                    info = self.parse_osu_text(content, folder_path.name, osu_path.name, audio_size_map)
                    if info:
                        results.append(info)
                except Exception as err:
                    print(f"[Warning] Failed parsing {osu_path.name} in {folder_path.name}: {err}")
        except Exception as err:
            print(f"[Warning] Failed reading folder {folder_path.name}: {err}")
        return results

    def extract_all(self) -> List[BeatmapInfo]:
        """Scans sample_dir for .osz files as well as beatmap subfolders with .osu files."""
        if not self.sample_dir.exists():
            print(f"[Extractor] Directory {self.sample_dir} not found.")
            return []

        all_infos: List[BeatmapInfo] = []
        osz_files = sorted(self.sample_dir.glob("*.osz"))
        print(f"[Extractor] Found {len(osz_files)} .osz archives in '{self.sample_dir}'")

        for osz in osz_files:
            extracted = self.extract_from_archive(osz)
            print(f"  - {osz.name}: extracted {len(extracted)} valid mania 4K difficulties")
            all_infos.extend(extracted)

        # Also support uncompressed song folders (e.g. osu! Songs directory)
        subdirs = sorted([d for d in self.sample_dir.iterdir() if d.is_dir() and any(d.glob("*.osu"))])
        if subdirs:
            print(f"[Extractor] Found {len(subdirs)} song directories in '{self.sample_dir}'")
            for sdir in subdirs:
                extracted = self.extract_from_folder(sdir)
                print(f"  - {sdir.name}: extracted {len(extracted)} valid mania 4K difficulties")
                all_infos.extend(extracted)

        return all_infos

    def save_manifests(self, beatmaps: List[BeatmapInfo]) -> Tuple[Path, Path]:
        """Saves manifest.json and manifest.csv into output_dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / "manifest.json"
        csv_path = self.output_dir / "manifest.csv"

        # Save JSON
        data = [asdict(b) for b in beatmaps]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Save CSV
        if beatmaps:
            fields = list(asdict(beatmaps[0]).keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for b in beatmaps:
                    row = asdict(b)
                    # Convert column_distribution to formatted string for CSV
                    row["column_distribution"] = ":".join(map(str, row["column_distribution"]))
                    writer.writerow(row)
        else:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                pass

        print(f"[Extractor] Saved manifest JSON: {json_path}")
        print(f"[Extractor] Saved manifest CSV:  {csv_path}")
        return json_path, csv_path


def extract_dataset_from_directory(sample_dir: str = "sample beatmaps", output_dir: str = "dataset_manifest") -> List[BeatmapInfo]:
    extractor = DatasetExtractor(sample_dir, output_dir)
    beatmaps = extractor.extract_all()
    extractor.save_manifests(beatmaps)
    return beatmaps


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "sample beatmaps"
    dst = sys.argv[2] if len(sys.argv) > 2 else "dataset_manifest"
    maps = extract_dataset_from_directory(src, dst)
    print(f"[Done] Total 4K mania beatmaps processed: {len(maps)}")
