"""Shared data loading and preprocessing."""
import os
import pandas as pd


def load_raw_data(filepath):
    """Load raw CAD CSV and derive analysis columns."""
    df = pd.read_csv(filepath)

    for col in ["Call_Created_Time", "Unit_Dispatched_Time", "Unit_OnScene_Time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["year"] = df["Call_Created_Time"].dt.year
    df["IsPrimary"] = df["PrimaryUnitCallSign"] == df["RespondingUnitCallSign"]
    df["Cahoots_related"] = (
        (df["PrimaryUnitCallSign"] == "CAHOOT")
        | (df["RespondingUnitCallSign"] == "CAHOOT")
    ).astype(int)
    df["cahoots_handled"] = (df["Cahoots_related"] == 1) & (df["IsPrimary"])

    df = df.dropna(subset=["PrimaryUnitCallSign", "RespondingUnitCallSign"], how="all")
    return df


def filter_df_by_time(df, start_time=None, end_time=None, time_column="Call_Created_Time"):
    """Filter a DataFrame to a time range.

    Parameters
    ----------
    time_column : str
        Must be specified explicitly.  The OMD analysis uses
        ``"Call_Created_Time"`` while the SDR/PDR analysis uses
        ``"Unit_Dispatched_Time"``.
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])

    if start_time:
        if len(start_time) == 4:
            start_time = pd.to_datetime(f"{start_time}-01-01")
    else:
        start_time = df[time_column].min()

    if end_time:
        if len(end_time) == 4:
            end_time = pd.to_datetime(f"{end_time}-12-31 23:59:59")
    else:
        end_time = df[time_column].max()

    return df[(df[time_column] >= start_time) & (df[time_column] <= end_time)]


def dataset_builder(data, dispatched=False, arrived=False, time=None):
    """Apply dispatched/arrived filters and an optional time range."""
    if time:
        data = filter_df_by_time(data, time[0], time[1], time[2])

    if dispatched:
        data = data[data["Unit_Dispatched_Time"].notna()]

    if arrived:
        data = data[data["Unit_OnScene_Time"].notna()]
        data = data[data["Unit_Dispatched_Time"].notna()]

    return data


def save_processed(df, path):
    """Save a DataFrame to parquet (or CSV if pyarrow unavailable)."""
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        csv_path = path.replace(".parquet", ".csv")
        df.to_csv(csv_path, index=False)
        path = csv_path
    return path


def load_processed(path):
    """Load a DataFrame from parquet (or CSV fallback)."""
    if path.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except (ImportError, FileNotFoundError):
            csv_path = path.replace(".parquet", ".csv")
            if os.path.exists(csv_path):
                return _load_csv_processed(csv_path)
            raise
    return _load_csv_processed(path)


def _load_csv_processed(path):
    """Load processed CSV with correct datetime parsing."""
    df = pd.read_csv(path)
    for col in ["Call_Created_Time", "Unit_Dispatched_Time", "Unit_OnScene_Time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
