"""
Beatmap Source Resolver for osu!mania 4K dataset preparation.

Supports:
1. Direct .osz files in `sample beatmaps/` directory.
2. Text pointer files (.txt, .path, .link) in `sample beatmaps/` specifying external
   directories/drives (e.g., `D:\\osu!\\Songs`, `E:\\beatmap_archive`, `/mnt/storage/osu`).
3. Comma/semicolon separated paths in CLI arguments or environment variables (`OSU_BEATMAPS_DIR`, `OSU_SAMPLE_DIR`).
4. Windows Junctions / Symlinks / Shortcuts (.lnk).
5. Uncompressed song directories containing .osu files.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Set, Tuple


def _read_pointer_file(txt_path: Path) -> List[Path]:
    """Reads lines from a text pointer file (.txt, .path, .link) as target directory paths."""
    resolved_paths: List[Path] = []
    try:
        content = txt_path.read_text(encoding="utf-8", errors="ignore")
        for line in content.splitlines():
            line_str = line.strip()
            # Ignore comments and empty lines
            if not line_str or line_str.startswith("#") or line_str.startswith("//"):
                continue
            # Strip surrounding quotes
            if (line_str.startswith('"') and line_str.endswith('"')) or (
                line_str.startswith("'") and line_str.endswith("'")
            ):
                line_str = line_str[1:-1].strip()

            # Expand environment variables (e.g. %USERPROFILE% or $HOME) and user ~
            expanded = os.path.expandvars(os.path.expanduser(line_str))
            p = Path(expanded)
            if p.exists():
                resolved_paths.append(p.resolve())
            else:
                print(f"[Source Resolver Warning] Path in '{txt_path.name}' does not exist: '{line_str}'")
    except Exception as e:
        print(f"[Source Resolver Warning] Failed to read pointer file '{txt_path}': {e}")
    return resolved_paths


def resolve_beatmap_sources(
    input_paths: str | Path | List[str | Path] = "sample beatmaps",
    recursive: bool = True,
) -> Tuple[List[Path], List[Path]]:
    """
    Scans given paths, directories, and text pointer files for beatmaps.

    Returns:
    - (osz_files, folder_beatmaps)
      - osz_files: list of unique .osz archive Paths.
      - folder_beatmaps: list of uncompressed beatmap directory Paths containing .osu files.
    """
    raw_targets: List[Path] = []

    # 1. Check environment variables
    for env_key in ("OSU_SAMPLE_DIR", "OSU_BEATMAPS_DIR", "OSU_SOURCES"):
        env_val = os.environ.get(env_key)
        if env_val:
            for piece in env_val.replace(";", ",").split(","):
                if piece.strip():
                    raw_targets.append(Path(piece.strip()))

    # 2. Add input paths passed as argument
    if isinstance(input_paths, (str, Path)):
        # Support comma-separated strings (e.g., "sample beatmaps,D:\osu\Songs")
        str_val = str(input_paths)
        for part in str_val.replace(";", ",").split(","):
            if part.strip():
                raw_targets.append(Path(part.strip()))
    elif isinstance(input_paths, (list, tuple)):
        for item in input_paths:
            raw_targets.append(Path(item))

    # Fallback to default if empty
    if not raw_targets:
        raw_targets = [Path("sample beatmaps")]

    unique_osz: Set[Path] = set()
    unique_folders: Set[Path] = set()
    visited_dirs: Set[Path] = set()

    def scan_directory(dir_path: Path, current_depth: int = 0, max_depth: int = 3):
        try:
            resolved = dir_path.resolve()
        except Exception:
            resolved = dir_path

        if resolved in visited_dirs or not dir_path.exists() or not dir_path.is_dir():
            return
        visited_dirs.add(resolved)

        # 1. Check if this directory itself contains .osu files (uncompressed beatmap folder)
        has_osu = any(f.name.lower().endswith(".osu") for f in dir_path.iterdir() if f.is_file())
        if has_osu:
            unique_folders.add(resolved)

        # 2. Check for .osz files directly inside
        for item in dir_path.iterdir():
            if item.is_file():
                if item.name.lower().endswith(".osz"):
                    unique_osz.add(item.resolve())
                # Check for pointer text files (.txt, .path, .link, .sources)
                elif item.name.lower().endswith((".txt", ".path", ".link", ".sources")):
                    pointer_dirs = _read_pointer_file(item)
                    for p_dir in pointer_dirs:
                        if p_dir.is_dir():
                            scan_directory(p_dir, current_depth=0, max_depth=max_depth)
                        elif p_dir.is_file() and p_dir.name.lower().endswith(".osz"):
                            unique_osz.add(p_dir.resolve())
            elif item.is_dir():
                # Subfolder scan
                if recursive and current_depth < max_depth:
                    scan_directory(item, current_depth=current_depth + 1, max_depth=max_depth)

    for target in raw_targets:
        expanded_target = Path(os.path.expandvars(os.path.expanduser(str(target))))
        if not expanded_target.exists():
            continue

        if expanded_target.is_file():
            if expanded_target.name.lower().endswith(".osz"):
                unique_osz.add(expanded_target.resolve())
            elif expanded_target.name.lower().endswith((".txt", ".path", ".link", ".sources")):
                # Direct pointer file passed
                for p_dir in _read_pointer_file(expanded_target):
                    if p_dir.is_dir():
                        scan_directory(p_dir)
                    elif p_dir.is_file() and p_dir.name.lower().endswith(".osz"):
                        unique_osz.add(p_dir.resolve())
        elif expanded_target.is_dir():
            scan_directory(expanded_target)

    sorted_osz = sorted(list(unique_osz))
    sorted_folders = sorted(list(unique_folders))

    return sorted_osz, sorted_folders
