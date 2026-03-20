# Quantifying CAHOOTS: Mobile Crisis Response Program Diverts and Prevents Police Response

Analysis code for evaluating the impact of CAHOOTS (Crisis Assistance Helping
Out On The Streets) on police call-for-service demand in Eugene, Oregon. The
study quantifies three diversion metrics using computer-aided dispatch records:

- **SDR (Substitution Diversion Rate):** Direct replacement of police responses
- **PDR (Prevention-adjusted Diversion Rate):** Includes downstream prevention effects
- **OMD (Overlapping-mandate Diversion Rate):** Proportion of mutually answerable calls handled by CAHOOTS

## Authors

- Nathan Burton — Department of Data Science, University of Oregon
- Claire Herbert — Department of Sociology, University of Oregon
- Rori Rohlfs — Department of Data Science & Institute for Ecology and Evolution, University of Oregon

## Repository Structure

```
├── config.py
├── src/
│   ├── data_loader.py
│   ├── call_type_formatting.py
│   ├── omd_functions.py
│   ├── did_functions.py
│   ├── omd_plots.py
│   └── did_plots.py
├── scripts/
│   ├── 01_load_and_clean.py
│   ├── 02_omd_analysis.py
│   ├── 03_sdr_pdr_analysis.py
│   └── 04_supplementary_figures.py
├── data/
│   ├── raw/                          # Place CAD_data_clean.csv here
│   └── processed/
├── figures/
├── environment.yml
├── requirements.txt
├── run_all.sh
└── run_all.bat
```

## Data

The analysis uses computer-aided dispatch (CAD) call-for-service records from
the Eugene Police Department (2016–2021). The dataset contains ~737,000 records
with incident type, dispatch timestamps, and responding unit identifiers.

**To obtain the data:** Download `CAD_data_clean.csv` from [DATA REPOSITORY URL]
and place it in `data/raw/`.

## Quickstart

```
conda env create -f environment.yml
conda activate cahoots-311

# Windows
run_all.bat

# Linux / macOS
bash run_all.sh
```

Or run scripts individually:

```
python scripts/01_load_and_clean.py
python scripts/02_omd_analysis.py
python scripts/03_sdr_pdr_analysis.py
python scripts/04_supplementary_figures.py
```

## Outputs

| Script | Terminal Output | Figures |
|--------|----------------|---------|
| `01_load_and_clean.py` | Dataset summary statistics | — |
| `02_omd_analysis.py` | OMD rates by year, threshold sweep, pooled OM summary | `fig_threshold_sweep.pdf`, `fig_omd_annual.pdf`, `fig_pooled_vs_yearspecific.pdf`, `fig_om_heatmap.pdf`, `fig_treemap.pdf` |
| `03_sdr_pdr_analysis.py` | DiD coefficients, significant call types | `fig_hourly_volume.pdf`, `fig_monthly_trends.pdf` |
| `04_supplementary_figures.py` | Heterogeneity test results | `fig_S_composition.pdf` |

## Figure–Manuscript Index

| Manuscript Figure | Generated File |
|-------------------|---------------|
| Figure 1 (hourly patterns) | `fig_hourly_volume.pdf` |
| Figure 2 (DiD results, monthly trends) | `fig_monthly_trends.pdf` |
| Figure 3 (OMD call types) | `fig_om_heatmap.pdf`, `fig_omd_annual.pdf` |
| Figure S (threshold sensitivity) | `fig_threshold_sweep.pdf` |
| Figure S (pooled vs. year-specific) | `fig_pooled_vs_yearspecific.pdf` |
| Figure S (composition stability) | `fig_S_composition.pdf` |
| Figure S (call type treemap) | `fig_treemap.pdf` |

## License

This project is licensed under the MIT License
