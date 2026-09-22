#!/usr/bin/env python3
"""
CLI wrapper for dataset extraction from sample beatmaps.
Usage:
    python3 extract_dataset.py [sample_dir] [output_dir]
"""

import sys
from osu_mapper.dataset.extractor import extract_dataset_from_directory

def main():
    src_dir = sys.argv[1] if len(sys.argv) > 1 else "sample beatmaps"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "dataset_manifest"
    maps = extract_dataset_from_directory(src_dir, out_dir)
    print(f"Extracted {len(maps)} mania 4K difficulties into '{out_dir}'.")

if __name__ == "__main__":
    main()
