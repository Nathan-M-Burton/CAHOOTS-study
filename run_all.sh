#!/bin/bash
set -e

echo "CAHOOTS Study — Full Analysis Pipeline"
echo "======================================="

python scripts/01_load_and_clean.py
python scripts/02_omd_analysis.py
python scripts/03_sdr_pdr_analysis.py
python scripts/04_supplementary_figures.py

echo ""
echo "All analyses complete. Figures saved to figures/"
