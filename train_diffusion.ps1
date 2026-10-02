# ==============================================================================
# Script: train_diffusion.ps1
# Purpose: Launches ManiaDiffusion 1D Latent Beatmap Diffusion training on Windows PowerShell.
# Optimized for multi-GPU setups (e.g. 3x RTX 5060) with batch size 64 per card.
# ==============================================================================

param (
    [int]$Epochs = 100,
    [int]$BatchSize = 64,
    [float]$LR = 0.0001,
    [string]$Resume = ""
)

$PythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" }
             elseif (Get-Command py -ErrorAction SilentlyContinue) { "py" }
             else { Write-Error "Python not found."; exit 1 }

$ResumeFlag = @()
if ($Resume -eq "resume" -or $env:RESUME -eq "1") {
    $ResumeFlag = @("--resume")
}

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "🌊 Starting ManiaDiffusion Training (1D Beatmap Latent Diffusion)" -ForegroundColor Green
Write-Host "   Epochs:       $Epochs"
Write-Host "   Batch/GPU:    $BatchSize"
Write-Host "   LR:           $LR"
Write-Host "==================================================================" -ForegroundColor Cyan

& $PythonCmd train_diffusion.py --epochs $Epochs --batch_size $BatchSize --lr $LR @ResumeFlag
