#!/usr/bin/env python
"""
Overlapping Mandate Diversion (OMD) analysis.

Produces: threshold sweep, annual diversion rates, pooled vs. year-specific
comparison, heatmap, and treemap.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
from src.data_loader import load_processed, dataset_builder
from src.call_type_formatting import format_call_type_name
from src.omd_functions import (
    prop_table, simple_overlap_data, calculate_diversions,
    run_threshold_sweep, year_specific_overlap_flag,
    diversion_rate_by_year_from_flag,
)
from src.omd_plots import (
    plot_threshold_sweep, plot_diversion_by_year,
    plot_pooled_vs_yearspecific_bars, plot_om_heatmap, create_treemap,
)


def main():
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.BASE_DIR, "data", "processed"), exist_ok=True)

    df = load_processed(config.DATA_PROCESSED_PATH)

    base_all = dataset_builder(
        df, dispatched=config.DISPATCHED, arrived=config.ARRIVED,
        time=[config.START_YEAR, config.END_YEAR, config.CALL_CREATED_COL],
    )
    print(f"Analysis dataset: {len(base_all):,} rows "
          f"({config.START_YEAR}-{config.END_YEAR}, dispatched+arrived)\n")

    # Threshold sweep
    print("Threshold sensitivity sweep:")
    rates, thresholds, n_types = run_threshold_sweep(base_all)
    for (lo, hi), r, n in zip(thresholds, rates, n_types):
        print(f"  {hi:.2f}: ODR = {r:.1f}%  ({n} call types)")

    plot_threshold_sweep(
        rates, thresholds, n_types,
        save_path=os.path.join(config.FIGURES_DIR, "fig_threshold_sweep.pdf"),
    )

    # Annual OMD rates
    print("\nAnnual OMD diversion rates:")
    div_by_year = calculate_diversions(
        simple_overlap_data(df, config.OM_LOWER, config.OM_UPPER), by_year=True
    )
    print(div_by_year.to_string(index=False))

    plot_diversion_by_year(
        div_by_year,
        save_path=os.path.join(config.FIGURES_DIR, "fig_omd_annual.pdf"),
    )

    # Year-specific vs. pooled comparison
    print(f"\nPooled vs. year-specific (threshold = {config.OM_THRESHOLD:.2f}):")
    base = base_all.copy()
    base["year"] = base["year"].astype(int)
    base[config.TYPE_COL] = base[config.TYPE_COL].apply(format_call_type_name)

    years_sorted = sorted(base["year"].unique().tolist())
    n_years = len(years_sorted)

    old_om = simple_overlap_data(base, config.OM_LOWER, config.OM_UPPER)
    pooled_types = set(old_om[config.TYPE_COL].unique())
    base["OM_pooled"] = base[config.TYPE_COL].isin(pooled_types)

    pooled_om_call_counts = (
        base[base["OM_pooled"]].groupby(config.TYPE_COL).size().to_dict()
    )

    base, year_to_types = year_specific_overlap_flag(
        base, config.OM_LOWER, config.OM_UPPER, type_col=config.TYPE_COL
    )

    div_pooled = diversion_rate_by_year_from_flag(base, "OM_pooled").rename(
        columns={"Diversion Rate": "Diversion Rate (pooled)"}
    )
    div_yearspec = diversion_rate_by_year_from_flag(base, "OM_year_specific").rename(
        columns={"Diversion Rate": "Diversion Rate (year-specific)"}
    )

    div_compare = (
        pd.DataFrame({"year": years_sorted})
        .merge(div_pooled, on="year", how="left")
        .merge(div_yearspec, on="year", how="left")
        .sort_values("year")
        .reset_index(drop=True)
    )
    print(div_compare.to_string(index=False))

    plot_pooled_vs_yearspecific_bars(
        div_compare, years_sorted,
        save_path=os.path.join(config.FIGURES_DIR, "fig_pooled_vs_yearspecific.pdf"),
    )

    # Heatmap
    all_types_set = set()
    for types in year_to_types.values():
        all_types_set.update(types)
    all_types_set.update(pooled_types)
    all_types_sorted = sorted(all_types_set)

    call_counts_by_year = (
        base[base["OM_year_specific"]]
        .groupby(["year", config.TYPE_COL]).size().reset_index(name="count")
    )

    rows = []
    for ct in all_types_sorted:
        row = {"call_type": ct}
        for y in years_sorted:
            if ct in year_to_types.get(y, set()):
                val = call_counts_by_year.loc[
                    (call_counts_by_year["year"] == y) & (call_counts_by_year[config.TYPE_COL] == ct), "count"
                ]
                row[y] = int(val.values[0]) if len(val) > 0 else 0
            else:
                row[y] = 0
        row["Pooled"] = pooled_om_call_counts.get(ct, 0)
        rows.append(row)

    presence_df = pd.DataFrame(rows)
    pooled_total = sum(pooled_om_call_counts.values())
    presence_df["_in_pooled"] = presence_df["call_type"].isin(pooled_types).astype(int)
    presence_df["_pooled_prop"] = presence_df["call_type"].map(
        lambda ct: pooled_om_call_counts.get(ct, 0) / pooled_total if pooled_total > 0 else 0
    )
    presence_df["_years_included"] = (presence_df[years_sorted] > 0).sum(axis=1)
    presence_df = presence_df.sort_values(
        ["_in_pooled", "_pooled_prop", "_years_included", "call_type"],
        ascending=[False, False, False, True],
    )

    all_columns = years_sorted + ["Pooled"]
    presence_final = presence_df.set_index("call_type")[all_columns]
    presence_binary = (presence_final > 0).astype(int)

    if config.HEATMAP_MODE == "proportions":
        col_totals = presence_final.sum(axis=0)
        presence_display = presence_final.div(col_totals).multiply(100)
    else:
        presence_display = presence_final

    plot_om_heatmap(
        presence_binary, presence_display, config.HEATMAP_MODE,
        save_path=os.path.join(config.FIGURES_DIR, "fig_om_heatmap.pdf"),
    )

    # Pooled OM summary
    print("\nPooled OM summary:")
    pooled_summary = (
        base[base["OM_pooled"]]
        .groupby(config.TYPE_COL).size().reset_index(name="Total Calls")
    )
    total_pooled = pooled_summary["Total Calls"].sum()
    pooled_summary["Proportion (%)"] = (pooled_summary["Total Calls"] / total_pooled * 100).round(2)
    pooled_summary = pooled_summary.sort_values("Total Calls", ascending=False).reset_index(drop=True)
    pooled_summary.columns = ["Call Type", "Total Calls", "Proportion (%)"]
    print(pooled_summary.to_string(index=False))
    print(f"  {len(pooled_summary)} call types, {total_pooled:,} total calls")

    pooled_summary.to_csv(
        os.path.join(config.BASE_DIR, "data", "processed", "omd_pooled_summary.csv"), index=False
    )

    # Treemap
    overlap_table = prop_table(
        dataset_builder(df, dispatched=True, arrived=True,
                        time=["2017", "2021", config.CALL_CREATED_COL])
    )
    create_treemap(
        overlap_table, top_n=config.TOP_N, group_others=config.GROUP_OTHERS,
        save_path=os.path.join(config.FIGURES_DIR, "fig_treemap.pdf"),
    )

    div_compare.to_csv(
        os.path.join(config.BASE_DIR, "data", "processed", "omd_diversion_rates.csv"), index=False
    )

    print("\nOMD analysis complete.")


if __name__ == "__main__":
    main()
