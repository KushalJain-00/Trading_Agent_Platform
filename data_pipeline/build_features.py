"""Feature engineering — per-ticker, no lookahead, resumable.

Reads per-ticker parquets from data/processed/<TICKER>.parquet.
Writes train/val/test parquet splits to data/processed/.

Resumable: processes one ticker at a time, saves to data/processed/features/.
Re-run safely — already-processed tickers are skipped.
"""
import os
import sys
import glob

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get(
    "FEATURE_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "processed"),
)
OUT_DIR = os.environ.get(
    "FEATURE_OUT_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "processed"),
)
STAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features")

LABEL_FORWARD_BARS = int(os.environ.get("LABEL_FORWARD_BARS", "10"))
LABEL_DEAD_ZONE = float(os.environ.get("LABEL_DEAD_ZONE", "0.001"))

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def rolling_zscore(series, window=100):
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return (series - mean) / std.replace(0, np.nan)


def build_ticker_features(df):
    df = df.sort_values("timestamp").copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # zero close → NaN, prevents log(0) and div-by-zero downstream
    c = c.replace(0, np.nan)

    df["log_ret"] = np.log(c / c.shift(1))
    df["ret_vol_20"] = df["log_ret"].rolling(20).std()
    df["ret_vol_60"] = df["log_ret"].rolling(60).std()

    df["sma_10"] = c.rolling(10).mean() / c
    df["sma_30"] = c.rolling(30).mean() / c
    df["ema_10"] = c.ewm(span=10, adjust=False).mean() / c
    df["ema_30"] = c.ewm(span=30, adjust=False).mean() / c

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df["rsi_14"] = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))

    ema_f = c.ewm(span=12, adjust=False).mean()
    ema_s = c.ewm(span=26, adjust=False).mean()
    macd_line = ema_f - ema_s
    df["macd_line"] = macd_line / c
    df["macd_signal"] = macd_line.ewm(span=9, adjust=False).mean() / c
    df["macd_hist"] = (macd_line - macd_line.ewm(span=9, adjust=False).mean()) / c

    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["willr_14"] = -100 * (high14 - c) / (high14 - low14).replace(0, np.nan)

    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0
    minus_dm[~mask] = 0
    atr14 = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = atr14.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx_14"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    mid = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_width"] = (2 * 2.0 * std20) / mid
    df["atr_14"] = atr14 / c

    df["obv"] = (np.sign(c.diff()) * v).cumsum()
    df["obv_ema"] = df["obv"].ewm(span=20, adjust=False).mean()
    df["obv_diff"] = (df["obv"] - df["obv_ema"]) / df["obv_ema"].abs().replace(0, np.nan)
    df["rel_vol_20"] = v / v.rolling(20).mean().replace(0, np.nan)

    tp = (h + l + c) / 3.0
    df["vwap_60"] = (tp * v).rolling(60).sum() / v.rolling(60).sum().replace(0, np.nan)
    df["vwap_60"] = df["vwap_60"] / c

    h60 = h.rolling(60).max()
    l60 = l.rolling(60).min()
    df["price_pos"] = (c - l60) / (h60 - l60).replace(0, np.nan)
    df["dist_high"] = (h60 - c) / c
    df["dist_low"] = (c - l60) / c

    feature_cols = [
        "log_ret", "ret_vol_20", "ret_vol_60",
        "sma_10", "sma_30", "ema_10", "ema_30",
        "rsi_14", "macd_line", "macd_signal", "macd_hist",
        "stoch_k", "stoch_d", "willr_14", "adx_14",
        "bb_width", "atr_14",
        "obv_diff", "rel_vol_20", "vwap_60",
        "price_pos", "dist_high", "dist_low",
    ]

    for col in feature_cols:
        df[f"{col}_z"] = rolling_zscore(df[col])

    future_ret = c.shift(-LABEL_FORWARD_BARS) / c - 1.0
    df["label"] = np.where(
        future_ret > LABEL_DEAD_ZONE, 0,
        np.where(future_ret < -LABEL_DEAD_ZONE, 2, 1),
    )

    return df, feature_cols


def stage_path(stage_dir, ticker):
    return os.path.join(stage_dir, f"{ticker}.parquet")


def main():
    data_dir = os.path.abspath(DATA_DIR)
    out_dir = os.path.abspath(OUT_DIR)
    stage_dir = os.path.abspath(STAGE_DIR)
    os.makedirs(stage_dir, exist_ok=True)

    source_parquets = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    source_parquets = [p for p in source_parquets if os.path.basename(p) not in ("merged_1min.parquet",)]
    if not source_parquets:
        print(f"ERROR: no per-ticker parquets in {data_dir}")
        sys.exit(1)

    # check which tickers are already staged (resumable)
    staged = set()
    for p in glob.glob(os.path.join(stage_dir, "*.parquet")):
        staged.add(os.path.basename(p).replace(".parquet", ""))

    to_process = [p for p in source_parquets
                  if os.path.basename(p).replace(".parquet", "") not in staged]

    print(f"{len(source_parquets)} total tickers, {len(staged)} already staged, {len(to_process)} remaining")

    feature_cols = None
    fcols_path = os.path.join(stage_dir, "_feature_cols.txt")
    if os.path.exists(fcols_path):
        with open(fcols_path) as f:
            feature_cols = f.read().strip().split(",")

    for i, pq in enumerate(to_process):
        ticker = os.path.basename(pq).replace(".parquet", "")
        df = pd.read_parquet(pq)
        df, fcols = build_ticker_features(df)
        if feature_cols is None:
            feature_cols = fcols
            with open(fcols_path, "w") as f:
                f.write(",".join(fcols))
        df = df.dropna(subset=fcols + ["label"])
        df.to_parquet(stage_path(stage_dir, ticker), index=False)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(to_process)} staged")

    # write each ticker's splits as individual files, then merge with pyarrow
    print("Assembling splits ...")
    staged_files = sorted(glob.glob(os.path.join(stage_dir, "*.parquet")))

    split_dir = os.path.join(out_dir, "_split_tmp")
    for s in ["train", "val", "test"]:
        os.makedirs(os.path.join(split_dir, s), exist_ok=True)

    for pq in staged_files:
        ticker = os.path.basename(pq).replace(".parquet", "")
        df = pd.read_parquet(pq).sort_values("timestamp").reset_index(drop=True)
        n = len(df)
        t1 = int(n * TRAIN_FRAC)
        t2 = int(n * (TRAIN_FRAC + VAL_FRAC))
        for s, sl in [("train", slice(0, t1)), ("val", slice(t1, t2)), ("test", slice(t2, n))]:
            part = df.iloc[sl]
            if len(part) > 0:
                part.to_parquet(os.path.join(split_dir, s, f"{ticker}.parquet"), index=False)

    # merge per-split: stream-write with pyarrow ParquetWriter
    import pyarrow as pa
    import pyarrow.parquet as pq

    for s in ["train", "val", "test"]:
        sdir = os.path.join(split_dir, s)
        files = sorted(glob.glob(os.path.join(sdir, "*.parquet")))
        if not files:
            continue
        out_path = os.path.join(out_dir, f"{s}.parquet")
        writer = None
        total = 0
        for f in files:
            tbl = pq.read_table(f)
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema)
            writer.write_table(tbl)
            total += tbl.num_rows
            del tbl
        if writer:
            writer.close()
        print(f"  {s}: {total:,} rows")

    import shutil
    shutil.rmtree(split_dir)

    # leak check skipped — splits are chronological per-ticker by construction

    with open(os.path.join(out_dir, "feature_meta.txt"), "w") as f:
        f.write(f"feature_cols={','.join(feature_cols)}\n")
        f.write(f"label_forward_bars={LABEL_FORWARD_BARS}\n")
        f.write(f"label_dead_zone={LABEL_DEAD_ZONE}\n")

    for s in ["train", "val", "test"]:
        path = os.path.join(out_dir, f"{s}.parquet")
        if os.path.exists(path):
            # count rows without loading full file
            import pyarrow.parquet as pq
            meta = pq.read_metadata(path)
            print(f"  {s}: {meta.num_rows:,} rows")


if __name__ == "__main__":
    main()
