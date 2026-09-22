<#
.SYNOPSIS
    osu!mania 4K AI Model Checkpoint Tester (PowerShell).
.EXAMPLE
    .\test_checkpoint.ps1 -Model 1 -Audio "my_track.mp3"
    .\test_checkpoint.ps1 -Model 1 -Dataset "dataset_processed\train_data.pt"
#>

param(
    [int]$Model = 1,
    [string]$Audio = "",
    [string]$Dataset = "",
    [string]$Checkpoint = "",
    [string]$Device = "auto"
)

# Detect Python executable
$pythonExe = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExe = "py"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonExe = "python3"
} else {
    Write-Error "Neither 'python', 'py', nor 'python3' was found in your PATH."
    exit 1
}

$cmdArgs = @("test_checkpoint.py", "--model", $Model, "--device", $Device)

if ($Checkpoint) {
    $cmdArgs += @("--checkpoint", $Checkpoint)
}

if ($Audio) {
    $cmdArgs += @("--audio", $Audio)
} elseif ($Dataset) {
    $cmdArgs += @("--dataset", $Dataset)
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " 🔍 TESTING OSU!MANIA 4K AI CHECKPOINT (Model $Model)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

& $pythonExe $cmdArgs
