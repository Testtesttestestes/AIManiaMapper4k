# Windows PowerShell script to benchmark CPU vs GPU training speed
param (
    [string]$Model = "3",
    [int]$BatchSize = 16,
    [int]$Steps = 15
)

python benchmark_train.py --model $Model --batch_size $BatchSize --steps $Steps
