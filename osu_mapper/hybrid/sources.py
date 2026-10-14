"""
Dataset source iteration shared by prepare_hybrid_dataset.py and hybrid_solvability_audit.py.

Reads .osz archives and unpacked song folders (also those listed in 'sample beatmaps/*.txt'
pointer files, exactly like the old pipeline), parses 4K mania difficulties with the exact parser
and groups them by the audio file they use.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from .beatmap import ManiaMap, parse_osu_mania

AUDIO_EXTS = (".mp3", ".ogg", ".wav", ".flac", ".m4a", ".opus")


@dataclass
class SongGroup:
    """One audio file of a beatmap set and all 4K difficulties that use it."""

    source: str                                   # archive / folder path
    audio_name: str                               # entry name inside the archive / folder
    read_audio: Callable[[], bytes]
    maps: List[Tuple[str, str, ManiaMap]] = field(default_factory=list)  # (osu file name, osu text, parsed map)

    @property
    def song_id(self) -> str:
        return stable_id(self.source, self.audio_name.lower())

    @property
    def audio_ext(self) -> str:
        return Path(self.audio_name).suffix.lower() or ".mp3"


def stable_id(*parts: str) -> str:
    h = hashlib.sha1("␟".join(parts).encode("utf-8", errors="ignore")).hexdigest()
    return h[:16]


def _read_zip_entry(path: str, name: str) -> bytes:
    with zipfile.ZipFile(path, "r") as z:
        return z.read(name)


def _decode_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


def _resolve_audio(wanted: str, names: List[str], sizes: Dict[str, int]) -> Optional[str]:
    w = wanted.strip().lower().replace("\\", "/")
    for n in names:
        nl = n.lower().replace("\\", "/")
        if nl == w or nl.split("/")[-1] == w.split("/")[-1]:
            return n
    audio = [n for n in names if n.lower().endswith(AUDIO_EXTS)]
    if not audio:
        return None
    return max(audio, key=lambda n: sizes.get(n, 0))


def iter_song_groups(input_dir: str | Path, keys: int = 4, log: Callable[[str], None] = print) -> Iterator[SongGroup]:
    from osu_mapper.dataset.source_resolver import resolve_beatmap_sources

    osz_files, folders = resolve_beatmap_sources(input_dir)
    for osz in osz_files:
        try:
            with zipfile.ZipFile(osz, "r") as z:
                infos = {i.filename: i.file_size for i in z.infolist()}
                names = list(infos)
                groups: Dict[str, SongGroup] = {}
                for name in names:
                    if not name.lower().endswith(".osu"):
                        continue
                    text = _decode_text(z.read(name))
                    m = parse_osu_mania(text, keys=keys)
                    if m is None or not m.notes:
                        continue
                    audio = _resolve_audio(m.metadata.get("AudioFilename", ""), names, infos)
                    if audio is None:
                        log(f"  [skip] {osz.name} :: {name}: no audio file in archive")
                        continue
                    if audio not in groups:
                        groups[audio] = SongGroup(
                            source=str(osz),
                            audio_name=audio,
                            read_audio=(lambda p=str(osz), a=audio: _read_zip_entry(p, a)),
                        )
                    groups[audio].maps.append((name, text, m))
        except (zipfile.BadZipFile, OSError) as err:
            log(f"  [error] {osz}: {err}")
            continue
        yield from groups.values()

    for folder in folders:
        try:
            files = [p for p in Path(folder).iterdir() if p.is_file()]
        except OSError as err:
            log(f"  [error] {folder}: {err}")
            continue
        names = [p.name for p in files]
        sizes = {p.name: p.stat().st_size for p in files}
        groups = {}
        for p in files:
            if p.suffix.lower() != ".osu":
                continue
            text = _decode_text(p.read_bytes())
            m = parse_osu_mania(text, keys=keys)
            if m is None or not m.notes:
                continue
            audio = _resolve_audio(m.metadata.get("AudioFilename", ""), names, sizes)
            if audio is None:
                log(f"  [skip] {folder} :: {p.name}: no audio file in folder")
                continue
            if audio not in groups:
                groups[audio] = SongGroup(
                    source=str(folder),
                    audio_name=audio,
                    read_audio=(lambda path=str(Path(folder) / audio): Path(path).read_bytes()),
                )
            groups[audio].maps.append((p.name, text, m))
        yield from groups.values()
