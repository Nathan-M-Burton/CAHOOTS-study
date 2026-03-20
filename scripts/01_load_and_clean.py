#!/usr/bin/env python
"""Load raw CAD data, preprocess, and save for downstream analysis."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import load_raw_data, save_processed


def main():
    print(f"Loading: {config.DATA_RAW_PATH}")
    df = load_raw_data(config.DATA_RAW_PATH)

    print(f"  Rows:           {len(df):,}")
    print(f"  Date range:     {df['Call_Created_Time'].min()} to {df['Call_Created_Time'].max()}")
    print(f"  Incident types: {df[config.TYPE_COL].nunique()}")
    print(f"  CAHOOTS rows:   {df['Cahoots_related'].sum():,}")
    print(f"  EPD rows:       {(df['Cahoots_related'] == 0).sum():,}")

    os.makedirs(os.path.dirname(config.DATA_PROCESSED_PATH), exist_ok=True)
    saved_path = save_processed(df, config.DATA_PROCESSED_PATH)
    size_mb = os.path.getsize(saved_path) / 1e6
    print(f"  Saved: {saved_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
