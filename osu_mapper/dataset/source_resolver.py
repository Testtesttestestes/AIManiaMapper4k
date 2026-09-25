from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple


def resolve_beatmap_sources(sample_dir: str | Path) -> Tuple[List[Path], List[Path]]:
    """
    Scans a directory for .osz archives and directory pointers.
    Supports pointing to external folders (e.g. D:\\osu\\Songs) via .txt pointer files.
    
    Returns:
        (list_of_osz_paths, list_of_unpacked_song_folder_paths)
    """
    base = Path(sample_dir)
    osz_files: List[Path] = []
    song_folders: List[Path] = []

    if not base.exists():
        return [], []

    # 1. Direct .osz files in sample_dir
    for f in sorted(base.glob("*.osz")):
        if f.is_file():
            osz_files.append(f)

    # 2. Check for sources.txt or pointer .txt files
    for txt in sorted(base.glob("*.txt")):
        if txt.name.lower().startswith("sources"):
            try:
                for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                    clean = line.strip()
                    if clean and not clean.startswith("#"):
                        target_p = Path(clean)
                        if target_p.exists():
                            if target_p.is_dir():
                                # Scan directory for .osz or .osu subfolders
                                sub_osz = list(target_p.glob("*.osz"))
                                if sub_osz:
                                    osz_files.extend(sub_osz)
                                else:
                                    song_folders.append(target_p)
                            elif target_p.is_file() and target_p.suffix.lower() == ".osz":
                                osz_files.append(target_p)
            except Exception:
                pass

    return sorted(list(set(osz_files))), sorted(list(set(song_folders)))
