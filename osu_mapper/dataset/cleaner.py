"""
Dataset cleaner: keeps in every beatmap archive only what training reads, removes everything else.

What prepare_hybrid_dataset.py reads from an .osz / song folder (osu_mapper/hybrid/sources.py):
  * the .osu difficulties that are osu!mania 4K with notes,
  * the ONE audio file each of them names in AudioFilename (data_checks.audio_reference_problem: no guessing,
    keysounded maps without a song file are not usable).
Everything else is dead weight: videos, backgrounds and other images, hitsound / keysound samples, storyboards
(.osb), skin elements, difficulties of other modes and key counts, difficulties whose audio cannot be used.

Clones: a difficulty whose notes are identical to an already kept one (any archive) is removed, the same rule as
find_duplicate_maps.py / prepare_hybrid_dataset.py (identical text, identical notes + timing, or identical notes
with other green lines; archives whose name starts with a set ID are kept first). An archive left without any
difficulty is deleted.

Dry run by default (prints what would go and how much space it frees); ``apply=True`` rewrites archives in place
(a temporary file next to it, then an atomic replace) or writes cleaned copies to ``out_dir``.
"""

from __future__ import annotations

import os
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

AUDIO_EXTS = (".mp3", ".ogg", ".wav", ".flac", ".m4a", ".opus")
VIDEO_EXTS = (".mp4", ".avi", ".flv", ".mkv", ".mov", ".wmv", ".webm", ".m4v", ".mpg", ".mpeg")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


def kind_of(name: str, audio_kept: bool = False) -> str:
    n = name.lower()
    if n.endswith(".osu"):
        return "osu"
    if n.endswith(".osb"):
        return "storyboard"
    if n.endswith(VIDEO_EXTS):
        return "video"
    if n.endswith(IMAGE_EXTS):
        return "image"
    if n.endswith(AUDIO_EXTS):
        return "audio" if audio_kept else "hitsound/sample"
    return "other"


@dataclass
class DiffDecision:
    name: str
    keep: bool
    reason: str                 # "kept" / "not 4K mania" / "audio: ..." / "clone of ..." / "too few notes"
    audio: Optional[str] = None


@dataclass
class SourcePlan:
    path: Path
    is_archive: bool
    diffs: List[DiffDecision] = field(default_factory=list)
    keep_files: List[str] = field(default_factory=list)
    drop_files: List[Tuple[str, int, str]] = field(default_factory=list)   # (name, bytes, kind)
    size_before: int = 0
    error: Optional[str] = None

    @property
    def delete_whole(self) -> bool:
        return self.error is None and not any(d.keep for d in self.diffs)

    @property
    def bytes_freed(self) -> int:
        return self.size_before if self.delete_whole else sum(b for _, b, _ in self.drop_files)


@dataclass
class CleanReport:
    plans: List[SourcePlan] = field(default_factory=list)
    lfs_pointers: List[Path] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        s: Dict[str, float] = {"sources": len(self.plans), "errors": sum(p.error is not None for p in self.plans),
                               "deleted_sources": sum(p.delete_whole for p in self.plans),
                               "bytes_before": sum(p.size_before for p in self.plans),
                               "bytes_freed": sum(p.bytes_freed for p in self.plans)}
        for p in self.plans:
            for d in p.diffs:
                key = "diffs_kept" if d.keep else "diffs_removed:" + d.reason.split(":")[0].split(" of ")[0]
                s[key] = s.get(key, 0) + 1
            for _, b, k in p.drop_files:
                s["freed:" + k] = s.get("freed:" + k, 0) + b
        return s


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


def plan_source(path: Path, names: Sequence[str], sizes: Dict[str, int], read: Callable[[str], bytes],
                owner: Dict[str, str], keys: int, min_notes: int, keep_clones: bool, strict_timing: bool,
                disk: Optional[Dict[str, int]] = None, disk_total: Optional[int] = None) -> SourcePlan:
    """``sizes``: real (uncompressed) sizes, used by the audio checks; ``disk``: bytes on disk (compressed in an
    archive), used for the space report."""
    from find_duplicate_maps import osu_keys
    from osu_mapper.hybrid.beatmap import parse_osu_mania
    from osu_mapper.hybrid.data_checks import audio_reference_problem

    disk = disk if disk is not None else sizes
    plan = SourcePlan(path, path.suffix.lower() == ".osz",
                      size_before=int(disk_total) if disk_total is not None else sum(disk.values()))
    keep_audio: set = set()
    for name in names:
        if not name.lower().endswith(".osu"):
            continue
        raw = read(name)
        text = _decode(raw)
        m = parse_osu_mania(text, keys=keys)
        if m is None:
            plan.diffs.append(DiffDecision(name, False, f"not {keys}K mania"))
            continue
        if len(m.notes) < min_notes:
            plan.diffs.append(DiffDecision(name, False, f"too few notes ({len(m.notes)})"))
            continue
        audio, problem = audio_reference_problem(text, m.metadata.get("AudioFilename", ""), list(names), sizes)
        if audio is None:
            plan.diffs.append(DiffDecision(name, False, f"audio: {problem}"))
            continue
        ks = osu_keys(raw, not strict_timing)
        clone_of = next((owner[k] for k in ks if k in owner), None)
        if clone_of is not None and not keep_clones:
            plan.diffs.append(DiffDecision(name, False, f"clone of {clone_of}"))
            continue
        for k in ks:
            owner.setdefault(k, f"{path.name} :: {name}")
        plan.diffs.append(DiffDecision(name, True, "kept", audio))
        keep_audio.add(audio)
    kept_osu = {d.name for d in plan.diffs if d.keep}
    for name in names:
        if name.endswith("/"):
            continue
        if name in kept_osu or name in keep_audio:
            plan.keep_files.append(name)
        else:
            k = kind_of(name)
            if k == "hitsound/sample" and int(sizes.get(name, 0)) >= 1_000_000:
                k = "unused song audio"     # e.g. the song of a 7K / osu!standard set or of a rate edit nobody uses
            plan.drop_files.append((name, int(disk.get(name, 0)), k))
    return plan


def plan_dataset(root: str | Path, keys: int = 4, min_notes: int = 20, keep_clones: bool = False,
                 strict_timing: bool = False, log: Callable[[str], None] = print) -> CleanReport:
    from osu_mapper.dataset.source_resolver import resolve_beatmap_sources
    from osu_mapper.hybrid.sources import is_lfs_pointer

    osz_files, folders = resolve_beatmap_sources(root)
    # archives named "<set id> ..." first: they keep their difficulties, re-uploads become the clones
    osz_files = sorted(osz_files, key=lambda p: (re.match(r"^\d+ ", Path(p).name) is None, Path(p).name))
    rep = CleanReport()
    owner: Dict[str, str] = {}
    for k, osz in enumerate(osz_files, 1):
        osz = Path(osz)
        if is_lfs_pointer(osz):
            rep.lfs_pointers.append(osz)
            continue
        try:
            with zipfile.ZipFile(osz) as z:
                infos = {i.filename: i.file_size for i in z.infolist()}
                packed = {i.filename: i.compress_size for i in z.infolist()}
                plan = plan_source(osz, list(infos), infos, z.read, owner, keys, min_notes, keep_clones, strict_timing,
                                   disk=packed, disk_total=osz.stat().st_size)
        except (zipfile.BadZipFile, OSError) as err:
            plan = SourcePlan(osz, True, error=f"broken archive: {err}", size_before=osz.stat().st_size if osz.exists() else 0)
        rep.plans.append(plan)
        if k % 100 == 0:
            log(f"  checked {k}/{len(osz_files)} archives")
    for folder in folders:
        folder = Path(folder)
        files = [p for p in folder.rglob("*") if p.is_file()]
        names = [str(p.relative_to(folder)).replace("\\", "/") for p in files]
        sizes = {n: p.stat().st_size for n, p in zip(names, files)}
        try:
            plan = plan_source(folder, names, sizes, lambda n, f=folder: (f / n).read_bytes(), owner, keys, min_notes,
                               keep_clones, strict_timing)
        except OSError as err:
            plan = SourcePlan(folder, False, error=str(err))
        rep.plans.append(plan)
    return rep


def apply_plan(rep: CleanReport, out_dir: Optional[str | Path] = None, log: Callable[[str], None] = print) -> None:
    """Executes a plan: in place (default) or into ``out_dir`` (originals untouched)."""
    out = Path(out_dir) if out_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    for p in rep.plans:
        if p.error is not None:
            continue
        if p.is_archive:
            if p.delete_whole:
                if out is None:
                    p.path.unlink()
                    log(f"  [deleted] {p.path.name}: no usable 4K difficulty left")
                continue
            if not p.drop_files and out is None:
                continue
            target = (out / p.path.name) if out is not None else p.path
            tmp = target.with_name(target.name + ".cleaning")
            with zipfile.ZipFile(p.path) as src, zipfile.ZipFile(tmp, "w") as dst:
                for name in p.keep_files:
                    data = src.read(name)
                    method = zipfile.ZIP_DEFLATED if name.lower().endswith(".osu") else zipfile.ZIP_STORED
                    dst.writestr(zipfile.ZipInfo(name, date_time=src.getinfo(name).date_time), data, compress_type=method)
            os.replace(tmp, target)
        else:
            if out is not None:
                dest = out / p.path.name
                if p.delete_whole:
                    continue
                for name in p.keep_files:
                    (dest / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p.path / name, dest / name)
                continue
            if p.delete_whole:
                shutil.rmtree(p.path)
                log(f"  [deleted] {p.path}: no usable 4K difficulty left")
                continue
            for name, _, _ in p.drop_files:
                (p.path / name).unlink(missing_ok=True)


def _mb(b: float) -> str:
    return f"{b / 1e6:,.1f} MB" if b < 1e9 else f"{b / 1e9:,.2f} GB"


def format_report(rep: CleanReport, verbose: bool = False) -> str:
    s = rep.summary()
    kept = int(s.get("diffs_kept", 0))
    lines = ["", "=" * 78, " DATASET CLEANUP (only what training reads stays)", "=" * 78,
             f"Sources checked:         {int(s['sources'])} ({_mb(s['bytes_before'])})",
             f"4K difficulties kept:    {kept}"]
    removed = {k.split(":", 1)[1]: int(v) for k, v in s.items() if k.startswith("diffs_removed:")}
    for reason, n in sorted(removed.items(), key=lambda x: -x[1]):
        lines.append(f"  removed - {reason:<28} {n}")
    lines.append(f"Whole sources deleted:   {int(s['deleted_sources'])} (no usable 4K difficulty left)")
    lines.append(f"Space freed:             {_mb(s['bytes_freed'])} of {_mb(s['bytes_before'])}")
    for k in ("video", "image", "hitsound/sample", "unused song audio", "storyboard", "osu", "other"):
        if s.get("freed:" + k):
            lines.append(f"  {k:<22} {_mb(s['freed:' + k])}")
    if s["errors"]:
        lines.append(f"Broken sources (left untouched): {int(s['errors'])}")
        lines += [f"  {p.path.name}: {p.error}" for p in rep.plans if p.error]
    if rep.lfs_pointers:
        lines.append(f"Not downloaded (Git LFS pointers, skipped): {len(rep.lfs_pointers)} - run: git lfs pull")
    if verbose:
        lines.append("-" * 78)
        for p in rep.plans:
            gone = [d for d in p.diffs if not d.keep]
            if p.delete_whole or gone:
                lines.append(f"{p.path.name}{'  -> DELETE' if p.delete_whole else ''}")
                lines += [f"    - {d.name}: {d.reason}" for d in gone]
    lines.append("=" * 78)
    return "\n".join(lines)
