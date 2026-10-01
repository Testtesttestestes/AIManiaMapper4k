@echo off
REM Quick runner for Dataset Compatibility, Grid Snapping, & Acoustic Critic Audit
echo =========================================================================
echo  OSU!MANIA 4K DATASET AUDITOR: GRID SNAPPING ^& ACOUSTIC CRITIC
echo =========================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

python test_dataset_critic_and_grid.py %*

pause
