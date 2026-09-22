@echo off
REM ==============================================================================
REM OSU!MANIA 4K AI MODEL CHECKPOINT TESTER (Windows CMD Batch)
REM Usage:
REM   test_checkpoint.bat 1 my_track.mp3
REM   test_checkpoint.bat 1 dataset_processed\train_data.pt
REM ==============================================================================

set MODEL=%1
if "%MODEL%"=="" set MODEL=1

set TARGET=%2
if "%TARGET%"=="" set TARGET=output_beatmaps\synthetic_test.wav

set CHECKPOINT=%3

REM Detect Python executable
where python >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_BIN=python
    goto RUN
)

where py >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_BIN=py
    goto RUN
)

where python3 >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_BIN=python3
    goto RUN
)

echo [Error] Python is not installed or not found in PATH.
exit /b 1

:RUN
if not "%CHECKPOINT%"=="" (
    set CKPT_FLAG=--checkpoint %CHECKPOINT%
) else (
    set CKPT_FLAG=
)

echo %TARGET% | findstr /i "\.pt" >nul
if %errorlevel% equ 0 (
    %PYTHON_BIN% test_checkpoint.py --model %MODEL% --dataset "%TARGET%" %CKPT_FLAG%
) else (
    %PYTHON_BIN% test_checkpoint.py --model %MODEL% --audio "%TARGET%" %CKPT_FLAG%
)
