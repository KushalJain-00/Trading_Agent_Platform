"""Live/replay data source for paper trading.

PAPER TRADING ONLY — no real order execution, no broker connection.
All fills are simulated against real or replayed prices.

Modes:
  - live: yfinance polling (real delayed prices, no API key)
  - replay: streams historical validation bars chronologically at configurable speed
"""
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def is_market_open():
    """Check if NSE market is currently open (Mon-Fri, 9:15-15:30 IST)."""
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def market_status_str():
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    if now.weekday() >= 5:
        return "Market closed (weekend)"
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if now < market_open:
        return f"Market opens at 09:15 IST (current: {now.strftime('%H:%M')})"
    if now > market_close:
        return f"Market closed at 15:30 IST (current: {now.strftime('%H:%M')})"
    return f"Market OPEN (until 15:30 IST, current: {now.strftime('%H:%M')})"


def fetch_live_quote(ticker):
    """Fetch latest quote for an NSE ticker (e.g. 'RELIANCE' -> 'RELIANCE.NS')."""
    if yf is None:
        raise ImportError("pip install yfinance")
    yf_ticker = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
    t = yf.Ticker(yf_ticker)
    info = t.fast_info
    return {
        "ticker": ticker,
        "last_price": float(info.get("lastPrice", 0)),
        "prev_close": float(info.get("previousClose", 0)),
        "open": float(info.get("open", 0)),
        "day_high": float(info.get("dayHigh", 0)),
        "day_low": float(info.get("dayLow", 0)),
        "volume": int(info.get("lastVolume", 0)),
        "market_open": is_market_open(),
        "status": market_status_str(),
    }


def fetch_intraday(ticker, interval="1m", period="1d"):
    """Fetch recent intraday OHLCV bars. Returns DataFrame with standard columns."""
    if yf is None:
        raise ImportError("pip install yfinance")
    yf_ticker = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
    t = yf.Ticker(yf_ticker)
    df = t.history(period=period, interval=interval)
    if df.empty:
        return df
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df["ticker"] = ticker
    df["timestamp"] = df.index.tz_localize(None)
    return df[["timestamp", "open", "high", "low", "close", "volume", "ticker"]].reset_index(drop=True)


class ReplayDataStreamer:
    """Stream historical validation bars chronologically to simulate live data.

    Includes pre-computed z-score features from the val parquet, so no
    feature engineering is needed at replay time.

    Usage:
        streamer = ReplayDataStreamer(tickers=["RELIANCE", "TCS"], speed=2.0)
        for bar in streamer:
            # bar is a dict with timestamp, open, high, low, close, volume, ticker,
            # plus all 23 _z feature columns
            process(bar)
    """

    FEATURE_COLS = [
        "log_ret_z", "ret_vol_20_z", "ret_vol_60_z",
        "sma_10_z", "sma_30_z", "ema_10_z", "ema_30_z",
        "rsi_14_z", "macd_line_z", "macd_signal_z", "macd_hist_z",
        "stoch_k_z", "stoch_d_z", "willr_14_z", "adx_14_z",
        "bb_width_z", "atr_14_z",
        "obv_diff_z", "rel_vol_20_z", "vwap_60_z",
        "price_pos_z", "dist_high_z", "dist_low_z",
    ]

    def __init__(self, tickers=None, speed=1.0, data_dir=None):
        self.speed = speed
        data_dir = Path(data_dir) if data_dir else DATA_DIR
        val_path = data_dir / "val.parquet"

        cols = ["timestamp", "open", "high", "low", "close", "volume", "ticker", "label"] + self.FEATURE_COLS
        df = pd.read_parquet(val_path, columns=cols)
        if tickers:
            df = df[df["ticker"].isin(tickers)]
        df = df.sort_values("timestamp").reset_index(drop=True)
        self._df = df
        self._iter = df.iterrows()
        self._total = len(df)
        self._count = 0

    def __len__(self):
        return self._total

    def __iter__(self):
        return self

    def __next__(self):
        try:
            _, row = next(self._iter)
        except StopIteration:
            raise StopIteration
        self._count += 1
        if self.speed > 0:
            time.sleep(self.speed)
        bar = {
            "timestamp": row["timestamp"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
            "ticker": row["ticker"],
            "label": int(row["label"]),
        }
        # Include pre-computed features for live inference
        for fc in self.FEATURE_COLS:
            bar[fc] = float(row[fc]) if pd.notna(row[fc]) else 0.0
        return bar

    @property
    def progress(self):
        return self._count / max(self._total, 1)

    @property
    def bars_processed(self):
        return self._count
