# Windows PowerShell script to train osu!mania 4K AI models
param (
    [string]$Model = "3",
    [string]$Device = "auto",
    [int]$Epochs = 15,
    [int]$BatchSize = 16,
    [float]$Lr = 0.0002
)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " 🤖 OSU!MANIA 4K AI MODEL TRAINING (POWERSHELL)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Model:      $Model (1=Sync, 2=Dynamics, 3=Transformer, 4=Evaluator, all)"
Write-Host " Device:     $Device"
Write-Host " Epochs:     $Epochs"
Write-Host " Batch Size: $BatchSize"
Write-Host " LR:         $Lr"

python train.py --model $Model --device $Device --epochs $Epochs --batch_size $BatchSize --lr $Lr
