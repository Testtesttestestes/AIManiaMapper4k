#!/usr/bin/env bash
set -e

echo "===================================================="
echo " osu!mania Star Rating Comparator (rosu vs native)"
echo "===================================================="

if [ -z "$1" ]; then
    echo "Usage: ./compare_difficulty.sh <path_to_beatmap.osz>"
    echo "Scanning output_beatmaps/ directory..."
    python3 compare_difficulty.py --dir "output_beatmaps"
else
    python3 compare_difficulty.py --map "$1"
fi
