#!/usr/bin/env python
"""
Supplementary analyses: call-type composition stability and
within-treatment heterogeneity tests.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd

import config
from src.data_loader import load_processed, dataset_builder
from src.did_functions import test_within_treatment_heterogeneity
from src.did_plots import plot_call_type_composition


def main():
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    df = load_processed(config.DATA_PROCESSED_PATH)

    dispatch_col = config.DISPATCH_COL
    df["year"] = df[dispatch_col].dt.year
    df["hr"] = df[dispatch_col].dt.hour
    df = df.dropna(subset=["PrimaryUnitCallSign"], how="all")
    df.dropna(subset=["hr"], inplace=True)

    processed_data = dataset_builder(
        df, dispatched=True, arrived=True,
        time=["2016", "2018", dispatch_col],
    )

    # Composition figure (Section S3.6)
    print("Generating composition figure...")
    plot_call_type_composition(
        processed_data,
        save_path=os.path.join(config.FIGURES_DIR, "fig_S_composition.pdf"),
    )

    # Heterogeneity test (Table S8)
    print("Running within-treatment heterogeneity test...")
    aug = processed_data.copy()
    aug["hour"] = pd.to_datetime(aug[dispatch_col]).dt.hour
    aug["early_morning"] = (
        (aug["hour"] >= config.TREATMENT_HOURS[0])
        & (aug["hour"] <= config.TREATMENT_HOURS[1])
    ).astype(int)
    aug["month"] = aug[dispatch_col].dt.month

    results_path = os.path.join(config.BASE_DIR, "data", "processed", "did_results.csv")
    if os.path.exists(results_path):
        results_df_main = pd.read_csv(results_path)
    else:
        print("  Warning: did_results.csv not found. Run 03_sdr_pdr_analysis.py first.")
        results_df_main = None

    het_results = test_within_treatment_heterogeneity(
        data=aug,
        cutoff_year=config.CUTOFF_YEAR,
        cutoff_month=config.CUTOFF_MONTH,
        partial_year=config.PARTIAL_YEAR,
        partial_month=config.PARTIAL_MONTH,
        split_hour=5,
        results_df_main=results_df_main,
        cluster_se=True,
    )

    display_cols = [
        "call_type", "effect_size_early", "effect_size_late",
        "heterogeneity_pvalue", "interpretation",
    ]
    available = [c for c in display_cols if c in het_results.columns]
    print("\n" + het_results[available].to_string(index=False))

    het_path = os.path.join(config.BASE_DIR, "data", "processed", "heterogeneity_results.csv")
    het_results.to_csv(het_path, index=False)
    print(f"\n  Saved: {het_path}")
    print("\nSupplementary analyses complete.")


if __name__ == "__main__":
    main()
