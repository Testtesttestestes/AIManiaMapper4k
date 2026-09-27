from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple


def resolve_beatmap_sources(sample_dir: str | Path) -> Tuple[List[Path], List[Path]]:
    """
    Scans a directory and text files for .osz archives and directory pointers.
    Supports pointing to external folders (e.g. D:\\osu\\Songs) or lists of files via .txt pointer files.
    
    Returns:
        (list_of_osz_paths, list_of_unpacked_song_folder_paths)
    """
    base = Path(sample_dir)
    osz_files: List[Path] = []
    song_folders: List[Path] = []

    # 1. Direct .osz files in sample_dir (if exists)
    if base.exists() and base.is_dir():
        for f in sorted(base.glob("*.osz")):
            if f.is_file():
                osz_files.append(f)

    # 2. Check for all .txt pointer files in sample_dir and current working directory
    txt_candidates = []
    if base.exists() and base.is_dir():
        txt_candidates.extend(list(base.glob("*.txt")))
    
    # Also check root for common pointer file names
    for root_txt in Path(".").glob("*.txt"):
        if root_txt.name.lower().startswith(("source", "song", "map", "osu", "list", "dataset", "track", "music")):
            txt_candidates.append(root_txt)

    seen_txts = set()
    for txt in txt_candidates:
        if txt.resolve() in seen_txts:
            continue
        seen_txts.add(txt.resolve())
        try:
            for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                clean = line.strip().strip('"').strip("'")
                if clean and not clean.startswith("#"):
                    target_p = Path(clean)
                    if target_p.exists():
                        if target_p.is_dir():
                            # Scan directory for .osz or .osu subfolders
                            sub_osz = list(target_p.glob("*.osz"))
                            if not sub_osz:
                                sub_osz = list(target_p.glob("**/*.osz"))
                            if sub_osz:
                                osz_files.extend(sub_osz)
                            
                            # Check if the folder itself or its subdirectories contain .osu files
                            if list(target_p.glob("*.osu")):
                                song_folders.append(target_p)
                            else:
                                for sub_dir in target_p.iterdir():
                                    if sub_dir.is_dir() and list(sub_dir.glob("*.osu")):
                                        song_folders.append(sub_dir)
                        elif target_p.is_file() and target_p.suffix.lower() == ".osz":
                            osz_files.append(target_p)
                        elif target_p.is_file() and target_p.suffix.lower() == ".osu":
                            song_folders.append(target_p.parent)
        except Exception:
            pass

    return sorted(list(set(osz_files))), sorted(list(set(song_folders)))
