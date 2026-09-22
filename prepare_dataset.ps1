# Windows PowerShell script to prepare osu!mania 4K dataset
param (
    [string]$InputDir = "sample beatmaps",
    [string]$OutputDir = "dataset_processed",
    [int]$ChunkSize = 256,
    [int]$HopSize = 128,
    [string]$Ffmpeg = ""
)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " 📦 PREPARING OSU!MANIA 4K TRAINING DATASET (POWERSHELL)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Input Directory:  $InputDir"
Write-Host " Output Directory: $OutputDir"
Write-Host " Chunk Size:       $ChunkSize"
Write-Host " Hop Size:         $HopSize"

$argsList = @(
    "prepare_dataset.py",
    "--input_dir", $InputDir,
    "--output_dir", $OutputDir,
    "--chunk_size", $ChunkSize,
    "--hop_size", $HopSize
)

if ($Ffmpeg -ne "") {
    $argsList += @("--ffmpeg", $Ffmpeg)
}

python @argsList

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Dataset preparation finished successfully!" -ForegroundColor Green
} else {
    Write-Host "`n❌ Dataset preparation failed. See instructions above." -ForegroundColor Red
}
