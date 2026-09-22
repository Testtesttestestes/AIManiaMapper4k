@echo off
setlocal
chcp 65001 > nul

set MODEL=3
if not "%~1"=="" set MODEL=%~1

set BATCH_SIZE=16
if not "%~2"=="" set BATCH_SIZE=%~2

set STEPS=15
if not "%~3"=="" set STEPS=%~3

python benchmark_train.py --model "%MODEL%" --batch_size "%BATCH_SIZE%" --steps "%STEPS%"
