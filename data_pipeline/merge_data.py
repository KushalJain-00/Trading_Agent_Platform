"""Merge per-ticker CSVs into per-ticker parquets + one pooled parquet."""
import glob
import os
import sys

import pandas as pd

RAW_DIR = os.environ.get(
    "MERGE_RAW_DIR",
    os.path.join(os.path.dirname(__file__), "..", "Data", "archive"),
)
OUT_DIR = os.environ.get(
    "MERGE_OUT_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "processed"),
)
MIN_ROWS = int(os.environ.get("MERGE_MIN_ROWS", "50000"))
SKIP_TICKERS = {"NIFTY 50", "NIFTY BANK"}


def _ticker_from_path(path):
    return os.path.basename(path).replace("_minute_new.csv", "").replace("_minute.csv", "")


def load_one(path):
    ticker = _ticker_from_path(path)
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    df["ticker"] = ticker
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main():
    raw_dir = os.path.abspath(RAW_DIR)
    out_dir = os.path.abspath(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    csvs = sorted(glob.glob(os.path.join(raw_dir, "*_minute*.csv")))
    if not csvs:
        print(f"ERROR: no CSV files in {raw_dir}")
        sys.exit(1)

    # deduplicate: prefer _minute_new over _minute
    ticker_files = {}
    for csv_path in csvs:
        t = _ticker_from_path(csv_path)
        if t in SKIP_TICKERS:
            continue
        if t in ticker_files and "_new" in os.path.basename(csv_path) and "_new" not in os.path.basename(ticker_files[t]):
            ticker_files[t] = csv_path
        elif t not in ticker_files:
            ticker_files[t] = csv_path

    print(f"{len(ticker_files)} tickers from {len(csvs)} files")

    short_tickers = []
    all_ticker_dfs = []
    for ticker, csv_path in sorted(ticker_files.items()):
        df = load_one(csv_path)
        if len(df) < MIN_ROWS:
            short_tickers.append((ticker, len(df)))
            continue
        df = df.sort_values("timestamp").reset_index(drop=True)
        # save per-ticker parquet so downstream never needs to load everything
        df.to_parquet(os.path.join(out_dir, f"{ticker}.parquet"), index=False)
        all_ticker_dfs.append(df)

    if short_tickers:
        print(f"Flagged (< {MIN_ROWS} rows):")
        for t, n in sorted(short_tickers, key=lambda x: x[1]):
            print(f"  {t}: {n:,}")

    # merged parquet — may OOM on 8GB, skip if too large
    try:
        merged = pd.concat(all_ticker_dfs, ignore_index=True)
        merged.sort_values(["ticker", "timestamp"], inplace=True)
        merged.reset_index(drop=True, inplace=True)
        merged.to_parquet(os.path.join(out_dir, "merged_1min.parquet"), index=False)
        print(f"Merged {len(merged):,} rows, {merged['ticker'].nunique()} tickers")
    except MemoryError:
        print("merged_1min.parquet skipped (OOM) — per-ticker parquets saved instead")
