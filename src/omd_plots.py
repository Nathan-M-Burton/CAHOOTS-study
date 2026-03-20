"""OMD-related figures: threshold sweep, annual rates, heatmap, treemap."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import squarify

import config


def _save_and_close(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_threshold_sweep(diversion_rates, thresholds, num_call_types, save_path=None):
    """Diversion rate vs. response-parity threshold."""
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"{ut:.2f}" for _, ut in thresholds]
    ax.plot(labels, diversion_rates, marker="o")

    for i, txt in enumerate(num_call_types):
        ax.text(labels[i], diversion_rates[i] + 2, str(txt), ha="center", va="bottom")

    ax.set_title("Overlapping Mandate Diversion Rate (ODR) Threshold Response", fontweight="bold")
    ax.set_xlabel("Response Parity Threshold", fontweight="bold")
    ax.set_ylabel("ODR (%)", fontweight="bold")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45)
    fig.tight_layout()
    _save_and_close(fig, save_path)


def plot_diversion_by_year(df, save_path=None):
    """Annual bar chart of OMD diversion rate."""
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 8})

    fig, ax = plt.subplots(figsize=(6, 4), dpi=config.FIGURE_DPI)
    years = df["year"].to_numpy()
    rates = df["Diversion Rate"].to_numpy()

    ax.bar(years, rates, color="#4C72B0", width=0.8)
    for x, c in zip(years, rates):
        ax.text(x, c + 0.5, f"{c:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Year")
    ax.set_ylabel("ODR (%)")
    ax.set_title("Annual Overlapping Mandate (OM) Diversion Rate", pad=10)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.spines[["top", "right"]].set_visible(False)
    _save_and_close(fig, save_path)


def plot_pooled_vs_yearspecific_bars(div_compare, years_sorted, save_path=None):
    """Double bar chart comparing pooled and year-specific OM diversion rates."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(years_sorted))
    w = 0.4

    ax.bar(x - w / 2, div_compare["Diversion Rate (pooled)"], width=w,
           color="#4878CF", label="Pooled-year overlapping mandate")
    ax.bar(x + w / 2, div_compare["Diversion Rate (year-specific)"], width=w,
           color="#EECA3B", label="Year-specific overlapping mandate")

    ax.set_xlabel("Year")
    ax.set_ylabel("ODR (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(years_sorted)
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    _save_and_close(fig, save_path)


def plot_om_heatmap(presence_binary, presence_display, heatmap_mode, save_path=None):
    """Heatmap of OM call types by year with cell annotations."""
    annot_fmt = ".2f" if heatmap_mode == "proportions" else "d"

    fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(presence_binary))))
    sns.heatmap(
        presence_binary, cmap="Greys", cbar=False,
        linewidths=0.5, linecolor="lightgray",
        annot=presence_display, fmt=annot_fmt,
        annot_kws={"size": 8, "color": "white"}, ax=ax,
    )

    binary_flat = presence_binary.values.flatten()
    for i, text in enumerate(ax.texts):
        if binary_flat[i] == 0:
            text.set_text("")

    ax.set_xlabel("Year")
    ax.set_ylabel("Call Type")
    fig.tight_layout()
    _save_and_close(fig, save_path)


def create_treemap(data, top_n=14, group_others=True, save_path=None):
    """Treemap of call types with CAHOOTS proportion hatching."""
    color_list = list(plt.get_cmap("tab20").colors)
    data = data.copy()
    data["total"] = data["cahoots_count"] + data["police_count"]
    sorted_data = data.sort_values("total", ascending=False)

    all_cats = sorted_data["InitialIncidentTypeDescription"].unique()
    color_map = {label: color_list[idx % len(color_list)] for idx, label in enumerate(all_cats)}
    color_map["Other"] = "#EFBF04"

    if group_others:
        top = sorted_data.iloc[:top_n].copy()
        other_sum = sorted_data.iloc[top_n:]["total"].sum()
        other_prop = sorted_data.iloc[top_n:]["prop_cahoots"].mean()
        if other_sum > 0:
            other = pd.DataFrame({
                "InitialIncidentTypeDescription": ["Other"],
                "total": [other_sum],
                "prop_cahoots": [other_prop],
            })
            top = pd.concat([top, other], ignore_index=True)
    else:
        top = sorted_data.iloc[:top_n]

    fig, ax = plt.subplots(figsize=(28, 14))
    sizes = top["total"].values
    colors = [color_map[l] for l in top["InitialIncidentTypeDescription"]]
    total_sum = top["total"].sum()

    labels = []
    for inc, val in zip(top["InitialIncidentTypeDescription"], sizes):
        titled = inc.title()
        words = titled.split()
        if len(words) == 2:
            fmt = f"{words[0]}\n{words[1]}"
        elif len(words) == 3:
            fmt = f"{words[0]}\n{words[1]} {words[2]}"
        else:
            fmt = titled
        labels.append(f"{fmt}\n({val / total_sum * 100:.1f}%)")

    squarify.plot(sizes=sizes, label=[""] * len(sizes), color=colors,
                  alpha=0.8, pad=False, ax=ax, ec="black")
    plt.axis("off")

    fontsize = 20
    for text, rect, label_name in zip(labels, ax.patches, top["InitialIncidentTypeDescription"]):
        text_obj = ax.text(0, 0, text, fontsize=fontsize, ha="center", va="center")
        bb = text_obj.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transData.inverted())
        text_obj.remove()

        padding = 1.5
        tw, th = bb.width + padding, bb.height + padding
        rx, ry, rw, rh = rect.get_x(), rect.get_y(), rect.get_width(), rect.get_height()

        if tw < rw and th < rh:
            bg = plt.Rectangle(
                (rx + rw / 2 - tw / 2, ry + rh / 2 - th / 2),
                tw, th, facecolor="#FDF5E6", edgecolor="black", alpha=1, zorder=2,
            )
            ax.add_patch(bg)
            ax.text(rx + rw / 2, ry + rh / 2, text, ha="center", va="center",
                    fontsize=fontsize, zorder=3)

        prop = top.loc[top["InitialIncidentTypeDescription"] == label_name, "prop_cahoots"].values[0]
        hatch = plt.Rectangle(
            (rx, ry), rw * prop, rh,
            hatch="xxxx", facecolor="none", edgecolor="black", lw=0, zorder=1,
        )
        ax.add_patch(hatch)

    _save_and_close(fig, save_path)
