@echo off
setlocal
echo ====================================================
echo  osu!mania Star Rating Comparator (rosu vs native)
echo ====================================================

if "%~1"=="" (
    echo Usage: compare_difficulty.bat "path\to\beatmap.osz"
    echo Example: compare_difficulty.bat output_beatmaps\my_custom_map.osz
    echo.
    python compare_difficulty.py --map "output_beatmaps"
) else (
    python compare_difficulty.py --map "%~1"
)

pause
