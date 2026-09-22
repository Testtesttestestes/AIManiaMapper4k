@echo off
setlocal
chcp 65001 > nul

echo ======================================================================
echo  📦 PREPARING OSU!MANIA 4K TRAINING DATASET (WINDOWS)
echo ======================================================================

set INPUT_DIR=sample beatmaps
if not "%~1"=="" set INPUT_DIR=%~1

set OUTPUT_DIR=dataset_processed
if not "%~2"=="" set OUTPUT_DIR=%~2

python prepare_dataset.py --input_dir "%INPUT_DIR%" --output_dir "%OUTPUT_DIR%" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Error] Dataset preparation encountered an issue. See instructions above.
    exit /b %ERRORLEVEL%
)

echo.
echo ✅ Dataset preparation complete!
