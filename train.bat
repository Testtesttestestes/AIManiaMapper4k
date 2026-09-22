@echo off
setlocal
chcp 65001 > nul

set MODEL=3
if not "%~1"=="" set MODEL=%~1

set DEVICE=auto
if not "%~2"=="" set DEVICE=%~2

set EPOCHS=15
if not "%~3"=="" set EPOCHS=%~3

set BATCH_SIZE=16
if not "%~4"=="" set BATCH_SIZE=%~4

set LR=0.0002
if not "%~5"=="" set LR=%~5

set TIMING_SOURCE=map
if not "%~6"=="" set TIMING_SOURCE=%~6

set RESUME_FLAG=
if /i "%~7"=="resume" set RESUME_FLAG=--resume
if /i "%~7"=="1" set RESUME_FLAG=--resume
if /i "%~7"=="true" set RESUME_FLAG=--resume

echo ======================================================================
echo  🤖 OSU!MANIA 4K AI MODEL TRAINING (WINDOWS)
echo ======================================================================
echo  Model:       %MODEL% (1=Sync, 2=Dynamics, 3=Transformer, 4=Evaluator, all)
echo  Device:      %DEVICE%
echo  Data Source: %TIMING_SOURCE%
echo  Epochs:      %EPOCHS%
echo  Batch Size:  %BATCH_SIZE%
echo  LR:          %LR%
echo  Resume:      %RESUME_FLAG%
echo ======================================================================

python train.py --model "%MODEL%" --device "%DEVICE%" --epochs "%EPOCHS%" --batch_size "%BATCH_SIZE%" --lr "%LR%" --timing_source "%TIMING_SOURCE%" %RESUME_FLAG%
