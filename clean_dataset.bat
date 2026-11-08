@echo off
echo ============================================================
echo   Dataset cleanup: only what training needs stays
echo   (4K mania difficulties + their audio; no video, bg, hitsounds, clones)
echo ============================================================
echo Step 1: report (nothing is changed)
python audit_dataset_compatibility.py --clean --skip_audit %*
echo.
set /p ok=Remove all of it now? (y/N):
if /i "%ok%"=="y" python audit_dataset_compatibility.py --clean --apply %*
pause
