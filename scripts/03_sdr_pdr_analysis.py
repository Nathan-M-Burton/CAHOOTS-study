#!/usr/bin/env python
"""
SDR/PDR Difference-in-Differences analysis.

Uses Unit_Dispatched_Time for filtering (2016-2018) and the
early-morning treatment window (3-7 AM).
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
from src.did_functions import analyze_did_offset_nb_canonical_monthly
from src.did_plots import plot_hourly_volume, visualize_monthly_trends


def main():
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.BASE_DIR, "data", "processed"), exist_ok=True)

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
    print(f"Analysis dataset: {len(processed_data):,} rows (2016-2018, dispatch-time filter)\n")

    # Hourly volume figure
    plot_hourly_volume(
        processed_data, year_range=(2016, 2017),
        save_path=os.path.join(config.FIGURES_DIR, "fig_hourly_volume.pdf"),
    )

    # DiD analysis
    print("Running DiD analysis...")
    aug = processed_data.copy()
    aug["hour"] = pd.to_datetime(aug[dispatch_col]).dt.hour
    aug["early_morning"] = (
        (aug["hour"] >= config.TREATMENT_HOURS[0])
        & (aug["hour"] <= config.TREATMENT_HOURS[1])
    ).astype(int)
    aug["month"] = aug[dispatch_col].dt.month

    results_df, monthly_df, fitted_models = analyze_did_offset_nb_canonical_monthly(
        aug,
        treatment_indicator="early_morning",
        cutoff_year=config.CUTOFF_YEAR,
        cutoff_month=config.CUTOFF_MONTH,
        partial_year=config.PARTIAL_YEAR,
        partial_month=config.PARTIAL_MONTH,
        min_calls=config.MIN_CALLS,
        min_total_calls=config.MIN_TOTAL_CALLS,
        return_monthly_data=True,
        p=config.P_VALUE,
        cluster_se=True,
        return_models=True,
    )

    sig = results_df[results_df["significant"] == True].copy()
    inc = sig[sig["effect_size"] > 0]
    dec = sig[sig["effect_size"] < 0]

    if not inc.empty:
        print("\nSignificant increases:")
        for _, r in inc.iterrows():
            pct = r["effect_size"] * 100
            p_str = "p < 0.001" if r["p_value_corrected"] < 0.001 else f"p = {r['p_value_corrected']:.3f}"
            print(f"  {r['call_type']:<25} {pct:+.1f}%  ({p_str})")

    if not dec.empty:
        print("\nSignificant decreases:")
        for _, r in dec.iterrows():
            pct = r["effect_size"] * 100
            p_str = "p < 0.001" if r["p_value_corrected"] < 0.001 else f"p = {r['p_value_corrected']:.3f}"
            print(f"  {r['call_type']:<25} {pct:+.1f}%  ({p_str})")

    print(f"\n{len(sig)} / {len(results_df)} call types significant")

    results_path = os.path.join(config.BASE_DIR, "data", "processed", "did_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  Saved: {results_path}")

    # Monthly trends
    visualize_monthly_trends(
        monthly_df, results_df,
        cutoff_year=config.CUTOFF_YEAR,
        cutoff_month=config.CUTOFF_MONTH,
        rolling_window=config.ROLLING_WINDOW,
        save_path=os.path.join(config.FIGURES_DIR, "fig_monthly_trends.pdf"),
    )

    monthly_path = os.path.join(config.BASE_DIR, "data", "processed", "did_monthly_data.csv")
    monthly_df.to_csv(monthly_path, index=False)

    print("\nSDR/PDR analysis complete.")


if __name__ == "__main__":
    main()
