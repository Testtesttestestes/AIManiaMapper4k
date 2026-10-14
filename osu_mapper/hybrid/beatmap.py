"""
Exact osu!mania 4K beatmap representation used by the hybrid pipeline.

Everything that matters for gameplay is kept bit-exact: lane, start time (integer ms),
end time (integer ms) and the tap/long-note flag of every hit object, plus the complete
list of timing points (red and green lines with all 8 fields). Hitsound/sample fields of
hit objects are intentionally not modelled (they do not affect gameplay).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

NUM_LANES = 4


@dataclass(frozen=True, order=True)
class Note:
    """One hit object. Ordering is (start, col, end, ln), which is the canonical sort order."""

    start: int
    col: int
    end: int
    ln: bool = False

    def as_tuple(self) -> Tuple[int, int, int, bool]:
        return (self.start, self.col, self.end, self.ln)


@dataclass
class RawTimingPoint:
    """A timing point with all 8 osu! fields preserved."""

    time: float
    beat_len: float
    meter: int = 4
    sample_set: int = 0
    sample_index: int = 0
    volume: int = 100
    uninherited: bool = True
    effects: int = 0

    @property
    def bpm(self) -> Optional[float]:
        if self.uninherited and self.beat_len > 0:
            return 60000.0 / self.beat_len
        return None

    @property
    def sv(self) -> float:
        if not self.uninherited and self.beat_len < 0:
            return -100.0 / self.beat_len
        return 1.0

    def as_tuple(self) -> Tuple[float, float, int, int, int, int, bool, int]:
        return (
            float(self.time), float(self.beat_len), int(self.meter), int(self.sample_set),
            int(self.sample_index), int(self.volume), bool(self.uninherited), int(self.effects),
        )

    def to_osu_line(self) -> str:
        return ",".join([
            _fmt_number(self.time),
            _fmt_number(self.beat_len),
            str(int(self.meter)),
            str(int(self.sample_set)),
            str(int(self.sample_index)),
            str(int(self.volume)),
            "1" if self.uninherited else "0",
            str(int(self.effects)),
        ])


def _fmt_number(v: float) -> str:
    """Formats a float without losing precision and without a trailing '.0' for integers."""
    fv = float(v)
    if math.isfinite(fv) and fv == int(fv):
        return str(int(fv))
    return repr(fv)


@dataclass
class ManiaMap:
    notes: List[Note]
    timing_points: List[RawTimingPoint] = field(default_factory=list)
    keys: int = NUM_LANES
    overall_difficulty: float = 8.0
    hp_drain: float = 8.0
    metadata: Dict[str, str] = field(default_factory=dict)

    def canonical_notes(self) -> List[Tuple[int, int, int, bool]]:
        return sorted(n.as_tuple() for n in self.notes)

    def red_lines(self) -> List[RawTimingPoint]:
        return sorted((tp for tp in self.timing_points if tp.uninherited and tp.beat_len > 0), key=lambda t: t.time)

    @property
    def last_time_ms(self) -> int:
        if not self.notes:
            return 0
        return max(max(n.start, n.end) for n in self.notes)

    @property
    def first_time_ms(self) -> int:
        if not self.notes:
            return 0
        return min(n.start for n in self.notes)


def _split_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _kv(lines: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def parse_timing_point(line: str) -> Optional[RawTimingPoint]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return None
    try:
        def _int(idx: int, default: int) -> int:
            if len(parts) > idx and parts[idx] != "":
                return int(float(parts[idx]))
            return default

        return RawTimingPoint(
            time=float(parts[0]),
            beat_len=float(parts[1]),
            meter=_int(2, 4),
            sample_set=_int(3, 0),
            sample_index=_int(4, 0),
            volume=_int(5, 100),
            uninherited=(_int(6, 1) == 1),
            effects=_int(7, 0),
        )
    except ValueError:
        return None


def parse_hit_object(line: str, keys: int = NUM_LANES) -> Optional[Note]:
    parts = line.split(",")
    if len(parts) < 4:
        return None
    try:
        x = float(parts[0])
        col = int(math.floor(x * keys / 512.0))
        col = max(0, min(keys - 1, col))
        start = int(round(float(parts[2])))
        obj_type = int(float(parts[3]))
    except ValueError:
        return None
    if obj_type & 128:
        end = start
        if len(parts) >= 6:
            try:
                end = int(round(float(parts[5].split(":")[0])))
            except ValueError:
                end = start
        return Note(start=start, col=col, end=end, ln=True)
    return Note(start=start, col=col, end=start, ln=False)


def parse_osu_mania(text: str, keys: int = NUM_LANES) -> Optional[ManiaMap]:
    """Parses .osu text. Returns None unless the map is osu!mania (Mode 3) with ``keys`` keys."""
    sections = _split_sections(text)
    general = _kv(sections.get("General", []))
    difficulty = _kv(sections.get("Difficulty", []))
    metadata = _kv(sections.get("Metadata", []))
    try:
        mode = int(general.get("Mode", "0"))
        cs = float(difficulty.get("CircleSize", "0"))
    except ValueError:
        return None
    if mode != 3 or int(round(cs)) != keys:
        return None

    timing_points: List[RawTimingPoint] = []
    for line in sections.get("TimingPoints", []):
        tp = parse_timing_point(line)
        if tp is not None:
            timing_points.append(tp)

    notes: List[Note] = []
    for line in sections.get("HitObjects", []):
        n = parse_hit_object(line, keys=keys)
        if n is not None:
            notes.append(n)

    meta = dict(metadata)
    if "AudioFilename" in general:
        meta["AudioFilename"] = general["AudioFilename"]
    if "PreviewTime" in general:
        meta["PreviewTime"] = general["PreviewTime"]

    def _float(d: Dict[str, str], k: str, default: float) -> float:
        try:
            return float(d.get(k, default))
        except ValueError:
            return default

    return ManiaMap(
        notes=notes,
        timing_points=timing_points,
        keys=keys,
        overall_difficulty=_float(difficulty, "OverallDifficulty", 8.0),
        hp_drain=_float(difficulty, "HPDrainRate", 8.0),
        metadata=meta,
    )


def note_to_osu_line(n: Note, keys: int = NUM_LANES) -> str:
    x = int(math.floor((n.col * 512 + 256) / keys))
    if n.ln:
        return f"{x},192,{n.start},128,0,{n.end}:0:0:0:0:"
    return f"{x},192,{n.start},1,0,0:0:0:0:"


def map_to_osu_text(m: ManiaMap) -> str:
    """Serializes a map to a valid osu! v14 file."""
    meta = m.metadata
    title = meta.get("Title", "AI Generated Track")
    artist = meta.get("Artist", "AI Artist")
    lines: List[str] = [
        "osu file format v14",
        "",
        "[General]",
        f"AudioFilename: {meta.get('AudioFilename', 'audio.mp3')}",
        "AudioLeadIn: 0",
        f"PreviewTime: {meta.get('PreviewTime', '-1')}",
        "Countdown: 0",
        "SampleSet: Soft",
        "StackLeniency: 0.7",
        "Mode: 3",
        "LetterboxInBreaks: 0",
        "SpecialStyle: 0",
        "WidescreenStoryboard: 1",
        "",
        "[Editor]",
        "DistanceSpacing: 1",
        "BeatDivisor: 4",
        "GridSize: 4",
        "TimelineZoom: 2",
        "",
        "[Metadata]",
        f"Title:{title}",
        f"TitleUnicode:{meta.get('TitleUnicode', title)}",
        f"Artist:{artist}",
        f"ArtistUnicode:{meta.get('ArtistUnicode', artist)}",
        f"Creator:{meta.get('Creator', 'AIManiaMapper Hybrid')}",
        f"Version:{meta.get('Version', 'AI 4K')}",
        f"Source:{meta.get('Source', '')}",
        f"Tags:{meta.get('Tags', 'ai osu mania 4k generated hybrid')}",
        "BeatmapID:0",
        "BeatmapSetID:-1",
        "",
        "[Difficulty]",
        f"HPDrainRate:{m.hp_drain:g}",
        f"CircleSize:{m.keys}",
        f"OverallDifficulty:{m.overall_difficulty:g}",
        "ApproachRate:5",
        "SliderMultiplier:1.4",
        "SliderTickRate:1",
        "",
        "[Events]",
        "//Background and Video events",
        "//Break Periods",
        "",
        "[TimingPoints]",
    ]
    for tp in sorted(m.timing_points, key=lambda t: (t.time, not t.uninherited)):
        lines.append(tp.to_osu_line())
    lines.append("")
    lines.append("[HitObjects]")
    for n in sorted(m.notes):
        lines.append(note_to_osu_line(n, keys=m.keys))
    lines.append("")
    return "\n".join(lines)
