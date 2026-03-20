@echo off
echo CAHOOTS Study - Full Analysis Pipeline
echo ========================================

python scripts\01_load_and_clean.py
if errorlevel 1 exit /b 1

python scripts\02_omd_analysis.py
if errorlevel 1 exit /b 1

python scripts\03_sdr_pdr_analysis.py
if errorlevel 1 exit /b 1

python scripts\04_supplementary_figures.py
if errorlevel 1 exit /b 1

echo.
echo All analyses complete. Figures saved to figures\
