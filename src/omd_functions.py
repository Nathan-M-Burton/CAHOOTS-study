"""Overlapping Mandate Diversion (OMD) analytical functions."""
import pandas as pd
import numpy as np


def common_types(dataset, col="InitialIncidentTypeDescription"):
    """Return incident types handled by both CAHOOTS and EPD."""
    cahoots_types = set(
        dataset[dataset["Cahoots_related"] == 1][col].unique()
    )
    police_types = set(
        dataset[dataset["Cahoots_related"] == 0][col].unique()
    )
    return list(cahoots_types.intersection(police_types))


def prop_table(dataset):
    """Compute CAHOOTS proportion per incident type."""
    cahoots = dataset[dataset["Cahoots_related"] == 1]
    police = dataset[dataset["Cahoots_related"] == 0]

    cahoots_counts = cahoots["InitialIncidentTypeDescription"].value_counts()
    police_counts = police["InitialIncidentTypeDescription"].value_counts()

    merged = cahoots_counts.to_frame("cahoots_count").join(
        police_counts.to_frame("police_count"), how="outer"
    ).fillna(0)

    merged["prop_cahoots"] = merged["cahoots_count"] / (
        merged["cahoots_count"] + merged["police_count"]
    )

    out = merged.reset_index().rename(
        columns={"index": "InitialIncidentTypeDescription"}
    )
    return out.sort_values("prop_cahoots", ascending=False)


def simple_overlap_data(data, lower_threshold, upper_threshold):
    """Return rows whose call type falls within the response-parity thresholds."""
    if lower_threshold > upper_threshold:
        raise ValueError("Lower threshold cannot exceed upper threshold")

    comparison = prop_table(data)
    in_range = comparison[
        (comparison["prop_cahoots"] >= lower_threshold)
        & (comparison["prop_cahoots"] < upper_threshold)
    ]
    types_list = in_range["InitialIncidentTypeDescription"].tolist()
    return data[data["InitialIncidentTypeDescription"].isin(types_list)]


def calculate_diversions(data, by_year=True):
    """Compute the OMD diversion rate."""
    cahoots = data[data["Cahoots_related"] == 1]
    police = data[data["Cahoots_related"] == 0]

    if by_year:
        cah_yr = cahoots.groupby("year").size()
        epd_yr = police.groupby("year").size()
        total = cah_yr + epd_yr
        rates = (cah_yr / total * 100).reset_index().rename(columns={0: "Diversion Rate"})
        return rates
    else:
        return len(cahoots) / (len(cahoots) + len(police)) * 100


def run_threshold_sweep(data, initial_lower=0.0, initial_upper=1.0, step=0.02):
    """Sweep response-parity thresholds and return diversion rates."""
    diversion_rates, thresholds, num_call_types = [], [], []
    lower, upper = initial_lower, initial_upper

    while lower < upper:
        div_data = simple_overlap_data(data, lower, upper)
        rate = calculate_diversions(div_data, by_year=False)
        diversion_rates.append(rate)
        thresholds.append((lower, upper))

        n_types = div_data["InitialIncidentTypeDescription"].nunique() if not div_data.empty else 0
        num_call_types.append(n_types)

        lower += step
        upper -= step

    return diversion_rates, thresholds, num_call_types


# Year-specific OM helpers

def prop_table_within_group(df, type_col="InitialIncidentTypeDescription"):
    """Proportion table computed within a single group (e.g. one year)."""
    cah = df[df["Cahoots_related"] == 1]
    pol = df[df["Cahoots_related"] == 0]

    cah_counts = cah[type_col].value_counts()
    pol_counts = pol[type_col].value_counts()

    merged = cah_counts.to_frame("cahoots_count").join(
        pol_counts.to_frame("police_count"), how="outer"
    ).fillna(0)

    merged["total"] = merged["cahoots_count"] + merged["police_count"]
    merged = merged[merged["total"] > 0].copy()
    merged["prop_cahoots"] = merged["cahoots_count"] / merged["total"]

    return merged.reset_index().rename(columns={"index": type_col}).sort_values(
        "prop_cahoots", ascending=False
    )


def year_specific_overlap_flag(data, lower, upper, type_col="InitialIncidentTypeDescription"):
    """Flag rows belonging to year-specific overlapping-mandate types."""
    df = data.copy()
    year_to_types = {}

    for y, g in df.groupby("year"):
        t = prop_table_within_group(g, type_col=type_col)
        included = set(
            t.loc[
                (t["prop_cahoots"] >= lower) & (t["prop_cahoots"] < upper), type_col
            ].tolist()
        )
        year_to_types[y] = included

    df["OM_year_specific"] = df.apply(
        lambda r: r[type_col] in year_to_types.get(r["year"], set()), axis=1
    )
    return df, year_to_types


def diversion_rate_by_year_from_flag(df, flag_col):
    """Diversion rate per year using a boolean flag column."""
    sub = df[df[flag_col]].copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=["year", "Diversion Rate"])
    grp = sub.groupby("year")["Cahoots_related"]
    cah = grp.sum()
    tot = grp.count()
    return (cah / tot * 100).reset_index().rename(columns={"Cahoots_related": "Diversion Rate"})
