@echo off
REM ==============================================================================
REM Script: train_diffusion.bat
REM Purpose: Launches ManiaDiffusion 1D Latent Beatmap Diffusion training on Windows.
REM Optimized for multi-GPU setups (e.g. 3x RTX 5060) with batch size 64 per card.
REM ==============================================================================

setlocal enabledelayedexpansion

where python >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=python"
) else (
    where py >nul 2>nul
    if !errorlevel! equ 0 (
        set "PY_CMD=py"
    ) else (
        echo [Error] Python not found.
        exit /b 1
    )
)

set "EPOCHS=%~1"
if "%EPOCHS%"=="" set "EPOCHS=100"

set "BATCH_SIZE=%~2"
if "%BATCH_SIZE%"=="" set "BATCH_SIZE=64"

set "LR=%~3"
if "%LR%"=="" set "LR=0.0001"

set "RESUME_ARG=%~4"
set "RESUME_FLAG="
if /i "%RESUME_ARG%"=="resume" set "RESUME_FLAG=--resume"
if "%RESUME%"=="1" set "RESUME_FLAG=--resume"

echo ==================================================================
echo 🌊 Starting ManiaDiffusion Training (1D Beatmap Latent Diffusion)
echo    Epochs:       %EPOCHS%
echo    Batch/GPU:    %BATCH_SIZE%
echo    LR:           %LR%
echo    Resume:       %RESUME_FLAG%
echo ==================================================================

%PY_CMD% train_diffusion.py --epochs %EPOCHS% --batch_size %BATCH_SIZE% --lr %LR% %RESUME_FLAG%
endlocal
