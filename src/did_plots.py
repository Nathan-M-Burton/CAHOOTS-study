"""DiD-related figures: hourly volume, monthly trends, composition stability."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns

import config


def _save_and_close(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def _plot_single_hourly_volume(ax, data_for_plot, plot_title_str, plot_prefix, palette,
                               service_start_hour, service_end_hour, partial_expansion_hour,
                               show_legend=True, show_xlabel=True):
    """Single-panel hourly volume plot."""
    year_col, hr_col, volume_col = "year", "hr", "Call Volume"
    letter = plot_prefix.strip("() ")

    if "CAHOOTS" in plot_title_str:
        fmt_title = "CAHOOTS Calls"
    elif "EPD" in plot_title_str:
        fmt_title = "EPD Calls"
    else:
        fmt_title = plot_title_str.title()

    if data_for_plot is None or data_for_plot.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(fmt_title, fontsize=11, pad=10)
        ax.text(0.0, 1.02, letter.upper(), transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xlim(-0.5, 23.5)
        ax.set_ylim(bottom=0)
        if show_xlabel:
            ax.set_xlabel("Hour of Day")
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        ax.set_ylabel(volume_col)
        return [], [], [], []

    if year_col not in data_for_plot.columns:
        return _plot_single_hourly_volume(ax, pd.DataFrame(), plot_title_str, plot_prefix, palette,
                                          service_start_hour, service_end_hour, partial_expansion_hour,
                                          show_legend, show_xlabel)

    unique_years = sorted(data_for_plot[year_col].unique())
    num_years = len(unique_years)
    markers_list = ["o", "s", "^", "D", "v", ">", "<", "p", "*", "X"]
    if num_years > len(markers_list):
        markers_list = (markers_list * (num_years // len(markers_list) + 1))[:num_years]
    if not isinstance(palette, list) or len(palette) < num_years:
        palette = sns.color_palette("tab10", n_colors=max(num_years, 2))

    year_style = dict(zip(unique_years, zip(palette, markers_list)))
    all_handles, all_labels = [], []

    for yr in unique_years:
        yd = data_for_plot[data_for_plot[year_col] == yr]
        if yd.empty:
            continue
        color, marker = year_style.get(yr, ("gray", "x"))
        sns.lineplot(data=yd, x=hr_col, y=volume_col, color=color, marker=marker,
                     markersize=6, linewidth=1.8, ax=ax, label=str(yr), legend=False)
        if ax.lines:
            h = ax.lines[-1]
            if not any(lh.get_label() == str(yr) for lh in all_handles):
                all_handles.append(h)
                all_labels.append(str(yr))

    ax.set_title(fmt_title, fontsize=11, pad=10)
    ax.text(0.0, 1.02, letter.upper(), transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")
    ax.set_ylabel("Hourly Call Volume")
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(0, 24, 3))
    if show_xlabel:
        ax.set_xticklabels([str(h) for h in np.arange(0, 24, 3)])
        ax.set_xlabel("Hour of Day")
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    ax.set_xlim(-0.5, 23.5)

    is_valid = (service_start_hour is not None and service_end_hour is not None
                and 0 <= service_start_hour <= 23 and 0 <= service_end_hour <= 23
                and service_start_hour < service_end_hour)
    is_partial_valid = (is_valid and partial_expansion_hour is not None
                        and service_start_hour < partial_expansion_hour < service_end_hour)

    if is_partial_valid:
        ax.axvspan(service_start_hour, partial_expansion_hour, color="green", alpha=0.25, zorder=0)
        ax.axvspan(partial_expansion_hour, service_end_hour, color="goldenrod", alpha=0.3, zorder=0)
    elif is_valid:
        ax.axvspan(service_start_hour, service_end_hour, color="green", alpha=0.2, zorder=0)

    if partial_expansion_hour is not None and 0 <= partial_expansion_hour <= 23:
        ax.axvline(x=partial_expansion_hour, color="brown", linestyle="--", linewidth=1, alpha=0.9)

    if show_legend and all_handles:
        seen = set()
        uh, ul = [], []
        for h, l in zip(all_handles, all_labels):
            if l not in seen:
                uh.append(h)
                ul.append(l)
                seen.add(l)
        ax.legend(uh, ul, loc="lower right")

    return all_handles, all_labels, [], []


def plot_hourly_volume(processed_data, year_range=(2016, 2017),
                       incident_types=None, save_path=None):
    """Multi-panel hourly call volume comparison figure."""
    from src.did_functions import ensure_complete_data

    if incident_types is None:
        incident_types = ["CHECK WELFARE", "ASSIST PUBLIC", "TRANSPORT"]

    CAHOOTS_COL = "cahoots_handled"
    INCIDENT_COL = "InitialIncidentTypeDescription"
    YEAR_COL, HR_COL, VOLUME_COL = "year", "hr", "Call Volume"

    years_list = list(range(year_range[0], year_range[1] + 1))
    palette = sns.color_palette("tab10", n_colors=max(len(years_list), 2))
    prefixes = ["(c)", "(d)", "(e)", "(f)", "(g)", "(e)"]

    service_start, service_end, partial_exp = 3, 10, 7

    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=False, sharey=False)
    fig.subplots_adjust(top=0.88, bottom=0.1, left=0.07, right=0.97, hspace=0.2, wspace=0.2)

    plots = [
        {"id": "cahoots", "filter": processed_data[CAHOOTS_COL] == 1, "prefix": 0},
        {"id": "epd", "filter": processed_data[CAHOOTS_COL] == 0, "prefix": 1},
        {"id": "total", "filter": None, "prefix": 2},
    ]
    for idx, it in enumerate(incident_types):
        plots.append({"id": it.lower().replace(" ", "_"), "filter": processed_data[INCIDENT_COL] == it,
                       "prefix": 3 + idx, "type": it})

    titles = {
        "cahoots": "CAHOOTS Calls", "epd": "EPD Calls", "total": "Total Volume",
    }
    for it in incident_types:
        titles[it.lower().replace(" ", "_")] = f"{it} Total Volume"

    ax_map = {
        "cahoots": axes[0, 0], "epd": axes[0, 1], "total": axes[0, 2],
    }
    for idx, it in enumerate(incident_types):
        ax_map[it.lower().replace(" ", "_")] = axes[1, idx]

    xlabel_map = {it.lower().replace(" ", "_"): True for it in incident_types}
    ylabel_clear = {"epd", "total"} | {it.lower().replace(" ", "_") for it in incident_types[1:]}

    grouping_cols = [YEAR_COL, HR_COL]

    for pc in plots:
        pid = pc["id"]
        ax_cur = ax_map[pid]
        prefix = prefixes[pc["prefix"]]
        title = titles[pid]
        specific = pc.get("type")

        try:
            filt = processed_data[pc["filter"]].copy() if pc["filter"] is not None else processed_data.copy()
            grp = [YEAR_COL, INCIDENT_COL] if specific else [YEAR_COL]
            agg = filt.groupby(grouping_cols).size().reset_index(name=VOLUME_COL) if not filt.empty else pd.DataFrame(columns=grouping_cols + [VOLUME_COL])
            agg_complete = ensure_complete_data(agg, grp, year_range, specific_type=specific)

            _plot_single_hourly_volume(
                ax_cur, agg_complete, title, prefix, list(palette),
                service_start, service_end, partial_exp,
                show_legend=True, show_xlabel=xlabel_map.get(pid, False),
            )
            if pid in ylabel_clear:
                ax_cur.set_ylabel("")
        except Exception as e:
            print(f"Error plotting {prefix} {title}: {e}")

    fig.suptitle(f"Hourly Call Volume Comparison ({year_range[0]} vs {year_range[1]})", fontsize=16, y=1.02)
    _save_and_close(fig, save_path)


def visualize_monthly_trends(monthly_data_df, results_df,
                             cutoff_year=2017, cutoff_month=1,
                             significant_only=True, max_types=20,
                             figsize=(15, 20), ncols=3, subplot_height=4,
                             rolling_window=4, save_path=None):
    """Monthly call-rate trend plots for significant call types."""
    if significant_only and "significant" in results_df.columns:
        sig_types = results_df.loc[results_df["significant"], "call_type"].tolist()
        if sig_types:
            filtered = monthly_data_df[monthly_data_df["call_type"].isin(sig_types)]
        else:
            filtered = monthly_data_df
    else:
        filtered = monthly_data_df

    valid_types = list(set(filtered["call_type"].unique()) & set(results_df["call_type"].unique()))
    filtered = filtered[filtered["call_type"].isin(valid_types)]

    if not valid_types:
        print("No valid call types to visualize.")
        return None

    if "effect_size" in results_df.columns:
        rel = results_df[results_df["call_type"].isin(valid_types)].copy()
        rel["abs_effect_size"] = rel["effect_size"].abs().fillna(-1)
        if "significant" in rel.columns:
            rel["is_sig_decrease"] = (rel["significant"]) & (rel["effect_size"] < 0)
            top_types = rel.sort_values(["is_sig_decrease", "abs_effect_size"], ascending=[False, False])["call_type"].head(max_types).tolist()
        else:
            top_types = rel.sort_values("abs_effect_size", ascending=False)["call_type"].head(max_types).tolist()
    else:
        top_types = sorted(valid_types)[:max_types]

    if not top_types:
        print("No valid call types to visualize.")
        return None

    try:
        intervention_date = pd.to_datetime(f"{cutoff_year}-{cutoff_month:02d}-01")
    except ValueError:
        return None

    rolling_window = max(1, int(rolling_window))
    exp_offset = pd.DateOffset(months=rolling_window - 1)
    exp_start = intervention_date - exp_offset
    exp_end = intervention_date + exp_offset

    n_types = len(top_types)
    ncols = max(1, int(ncols))
    if n_types == 1:
        ncols = 1
    nrows = int(np.ceil(n_types / ncols))
    fig_w = figsize[0]
    fig_h = subplot_height * nrows + 2.5

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=False)
    if n_types == 1:
        axes = np.array([axes]).flatten()
    else:
        axes = axes.flatten()

    colors = sns.color_palette("Set1", 2)
    group_colors = {"Control": colors[0], "Treatment": colors[1]}
    group_labels = ["Control", "Treatment"]

    for i, call_type in enumerate(top_types):
        ax = axes[i]
        if call_type not in results_df["call_type"].values:
            ax.text(0.5, 0.5, f"Metadata missing for {call_type}", ha="center", va="center")
            continue

        call_data = filtered[filtered["call_type"] == call_type].copy()
        row = results_df[results_df["call_type"] == call_type]
        effect_size = row["effect_size"].iloc[0] if not row.empty else np.nan
        p_value = row["p_value_corrected"].iloc[0] if not row.empty and "p_value_corrected" in row else np.nan

        if call_data.empty:
            ax.text(0.5, 0.5, f"No data for {call_type}", ha="center", va="center")
            pct = f"{effect_size * 100:+.1f}%" if pd.notna(effect_size) else "N/A"
            ax.set_title(f"({chr(97 + i)}) {call_type}\n(DiD effect: {pct}, p=N/A)", fontsize=11)
            continue

        try:
            call_data["date"] = pd.to_datetime(call_data["date"])
        except Exception:
            continue

        min_y, max_y = np.inf, -np.inf
        for group in group_labels:
            g = call_data[call_data["group"] == group].sort_values("date").set_index("date")
            if g.empty or g["call_rate"].isnull().all():
                continue
            g_rate = g["call_rate"].dropna()
            if g_rate.empty:
                continue
            smoothed = g_rate.rolling(window=rolling_window, center=True, min_periods=1).mean() if len(g_rate) >= rolling_window else g_rate
            color = group_colors[group]
            ax.plot(smoothed.index, smoothed.values, color=color, linewidth=2)
            ax.scatter(g_rate.index, g_rate.values, color=color, alpha=0.30, s=20)
            min_y = min(min_y, min(smoothed.min(), g_rate.min()))
            max_y = max(max_y, max(smoothed.max(), g_rate.max()))

        if np.isfinite(min_y) and np.isfinite(max_y) and max_y > min_y:
            pad = 0.05 * (max_y - min_y)
            ax.set_ylim(min_y - pad, max_y + pad)

        ylim = ax.get_ylim()
        ax.axvspan(exp_start, exp_end, color="yellow", alpha=0.20, zorder=1)
        ax.axvline(exp_start, color="black", linestyle="--", alpha=0.6, zorder=2)
        ax.axvline(exp_end, color="black", linestyle="--", alpha=0.6, zorder=2)
        ax.text(intervention_date, ylim[0] + 0.02 * (ylim[1] - ylim[0]),
                "Expansion", ha="center", va="bottom", fontsize=8, fontweight="bold",
                zorder=3, bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=1))

        pct = f"{effect_size * 100:+.1f}%" if pd.notna(effect_size) else "N/A"
        if pd.notna(p_value):
            p_str = "p < 0.001" if p_value < 0.001 else f"p={p_value:.3f}"
        else:
            p_str = "p=N/A"
        ax.set_title(f"({chr(97 + i)}) {call_type}\n(DiD effect: {pct}, {p_str})", fontsize=11)
        ax.set_ylabel("Hourly Call rate")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    for j in range(n_types, nrows * ncols):
        fig.delaxes(axes[j])

    fig.suptitle("Monthly Call Rates Trends by Type", fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.90, hspace=0.45, bottom=0.20, wspace=0.25)

    legend_handles = [Line2D([0], [0], color=group_colors["Control"], lw=2),
                      Line2D([0], [0], color=group_colors["Treatment"], lw=2)]
    fig.legend(handles=legend_handles, labels=group_labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.08), ncol=2, title="Group", fontsize=10, title_fontsize=11)

    fig.text(0.5, 0.02,
             f"Yellow shaded area = 'Expansion' ({cutoff_year}-{cutoff_month:02d} "
             f"\u00b1 {rolling_window - 1} months).\n"
             f"Call rates shown as a {rolling_window}-month rolling average "
             f"(faint points = raw monthly values).",
             ha="center", va="bottom", fontsize=10)

    _save_and_close(fig, save_path)
    return fig


def plot_call_type_composition(processed_data, save_path=None):
    """4-panel call type composition stability figure (Section S3.6)."""
    key_types = config.KEY_TYPES
    key_type_labels = config.KEY_TYPE_LABELS
    type_label_map = dict(zip(key_types, key_type_labels))
    colors_pal = sns.color_palette("colorblind", n_colors=len(key_types))
    type_color_map = dict(zip(key_types, colors_pal))
    markers = ["o", "s", "^", "D", "v"]
    type_marker_map = dict(zip(key_types, markers))

    baseline = processed_data[processed_data["year"].isin([2016, 2017])].copy()

    hourly_totals = baseline.groupby("hr").size().reset_index(name="total_volume")
    type_hourly = (
        baseline[baseline["InitialIncidentTypeDescription"].isin(key_types)]
        .groupby(["hr", "InitialIncidentTypeDescription"]).size().reset_index(name="volume")
    )
    type_hourly = pd.merge(type_hourly, hourly_totals, on="hr")
    type_hourly["proportion"] = type_hourly["volume"] / type_hourly["total_volume"]

    def assign_window(hr):
        return "Treatment (3-7am)" if 3 <= hr < 7 else "Other Hours"

    baseline["simple_window"] = baseline["hr"].apply(assign_window)
    simple_totals = baseline.groupby("simple_window").size().reset_index(name="window_total")
    simple_hours = {"Treatment (3-7am)": 4, "Other Hours": 20}
    simple_totals["hours_in_window"] = simple_totals["simple_window"].map(simple_hours)

    simple_type_vol = (
        baseline[baseline["InitialIncidentTypeDescription"].isin(key_types)]
        .groupby(["simple_window", "InitialIncidentTypeDescription"]).size().reset_index(name="volume")
    )
    simple_type_vol = pd.merge(simple_type_vol, simple_totals, on="simple_window")
    simple_type_vol["proportion"] = simple_type_vol["volume"] / simple_type_vol["window_total"]
    simple_type_vol["hourly_rate"] = simple_type_vol["volume"] / simple_type_vol["hours_in_window"]

    window_colors = {"Treatment (3-7am)": "#1f77b4", "Other Hours": "#7f7f7f"}
    window_order = ["Treatment (3-7am)", "Other Hours"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.25, top=0.92, bottom=0.08, left=0.08, right=0.95)

    n_types = len(key_types)
    bar_width = 0.35
    x_base = np.arange(n_types)

    # Panel A
    ax_a = axes[0, 0]
    for ct in key_types:
        sub = type_hourly[type_hourly["InitialIncidentTypeDescription"] == ct]
        ax_a.plot(sub["hr"], sub["volume"], color=type_color_map[ct], linewidth=2,
                  marker=type_marker_map[ct], markersize=4, label=type_label_map[ct])
    ax_a.axvspan(3, 7, color="#1f77b4", alpha=0.15, zorder=0)
    ax_a.axvline(3, color="#1f77b4", linestyle="--", linewidth=1, alpha=0.7)
    ax_a.axvline(7, color="#1f77b4", linestyle="--", linewidth=1, alpha=0.7)
    ax_a.set_xlabel("Hour of Day"); ax_a.set_ylabel("Call Volume")
    ax_a.set_xticks(range(24)); ax_a.set_xlim(-0.5, 23.5); ax_a.set_ylim(bottom=0)
    ax_a.legend(loc="center right", fontsize=8, framealpha=0.9)
    ax_a.text(-0.12, 1.05, "A", transform=ax_a.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_a.spines["top"].set_visible(False); ax_a.spines["right"].set_visible(False); ax_a.grid(False)

    # Panel B
    ax_b = axes[0, 1]
    for i_w, window in enumerate(window_order):
        sub = simple_type_vol[simple_type_vol["simple_window"] == window]
        sub_sorted = sub.set_index("InitialIncidentTypeDescription").reindex(key_types).reset_index()
        ax_b.bar(x_base + (i_w - 0.5) * bar_width, sub_sorted["hourly_rate"],
                 width=bar_width, color=window_colors[window], edgecolor="black", linewidth=0.5, label=window)
    ax_b.set_xticks(x_base); ax_b.set_xticklabels(key_type_labels, fontsize=9)
    ax_b.set_ylabel("Mean Hourly Call Volume"); ax_b.set_ylim(bottom=0)
    ax_b.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_b.text(-0.12, 1.05, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False); ax_b.grid(False)

    # Panel C
    ax_c = axes[1, 0]
    for ct in key_types:
        sub = type_hourly[type_hourly["InitialIncidentTypeDescription"] == ct]
        ax_c.plot(sub["hr"], sub["proportion"] * 100, color=type_color_map[ct], linewidth=2,
                  marker=type_marker_map[ct], markersize=4, label=type_label_map[ct])
    ax_c.axvspan(3, 7, color="#1f77b4", alpha=0.15, zorder=0)
    ax_c.axvline(3, color="#1f77b4", linestyle="--", linewidth=1, alpha=0.7)
    ax_c.axvline(7, color="#1f77b4", linestyle="--", linewidth=1, alpha=0.7)
    ax_c.set_xlabel("Hour of Day"); ax_c.set_ylabel("Proportion of Hourly Calls (%)")
    ax_c.set_xticks(range(24)); ax_c.set_xlim(-0.5, 23.5); ax_c.set_ylim(bottom=0)
    ax_c.legend(loc="center right", fontsize=8, framealpha=0.9)
    ax_c.text(-0.12, 1.05, "C", transform=ax_c.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False); ax_c.grid(False)

    # Panel D
    ax_d = axes[1, 1]
    for i_w, window in enumerate(window_order):
        sub = simple_type_vol[simple_type_vol["simple_window"] == window]
        sub_sorted = sub.set_index("InitialIncidentTypeDescription").reindex(key_types).reset_index()
        ax_d.bar(x_base + (i_w - 0.5) * bar_width, sub_sorted["proportion"] * 100,
                 width=bar_width, color=window_colors[window], edgecolor="black", linewidth=0.5, label=window)
    ax_d.set_xticks(x_base); ax_d.set_xticklabels(key_type_labels, fontsize=9)
    ax_d.set_ylabel("Proportion of Calls (%)"); ax_d.set_ylim(bottom=0)
    ax_d.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_d.text(-0.12, 1.05, "D", transform=ax_d.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_d.spines["top"].set_visible(False); ax_d.spines["right"].set_visible(False); ax_d.grid(False)

    fig.suptitle("Call Volume and Composition Across Time of Day", fontsize=14, fontweight="bold", y=0.97)
    _save_and_close(fig, save_path)
