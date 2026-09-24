param (
    [string]$MapPath = ""
)

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host " osu!mania Star Rating Comparator (rosu vs native)" -ForegroundColor Yellow
Write-Host "====================================================" -ForegroundColor Cyan

if ($MapPath -eq "") {
    Write-Host "Usage: .\compare_difficulty.ps1 -MapPath 'path\to\beatmap.osz'" -ForegroundColor Gray
    python compare_difficulty.py --dir "output_beatmaps"
} else {
    python compare_difficulty.py --map "$MapPath"
}
