# OHLCV and Market Data

## What It Is

**OHLCV** stands for **Open, High, Low, Close, Volume** — the five pieces of
information recorded for each time period (called a "bar" or "candle"):

- **Open**: The first traded price in the period
- **High**: The highest price traded during the period
- **Low**: The lowest price traded during the period
- **Close**: The last traded price in the period
- **Volume**: How many shares were traded during the period

A **timeframe** determines the period length. Common timeframes:
- **1 minute**: Each bar represents 1 minute of trading
- **5 minutes**: Each bar = 5 minutes
- **1 hour**: Each bar = 1 hour
- **1 day**: Each bar = 1 trading day

## Why It Matters for This Project

This project uses **minute-level Nifty data** — each bar is 1 minute of
trading for stocks in the Nifty index on the National Stock Exchange (NSE) of
India. The data is stored in `data/processed/val.parquet` with columns:
`ticker`, `timestamp`, `open`, `high`, `low`, `close`, `volume`, plus 23
computed feature columns.

Key facts about this dataset:
- **113 tickers** (individual stocks like RELIANCE, TCS, INFY, etc.)
- **~15.8M rows** in the validation set
- **Time range**: October 2022 to February 2026
- **Market hours**: 9:15 AM to 3:30 PM IST (6.25 hours = ~375 minutes per day)
- After September 2024, fewer than 21 tickers have data — the dataset thins out

The `close` price is the most important column for this project — it's used to
compute returns, build features, and determine trade entry/exit prices. The
`open`, `high`, `low` columns are used in feature engineering (e.g., ATR,
Bollinger Bands, VWAP) but the models primarily work with pre-computed features
from these raw values.

## How It's Implemented Here

The raw OHLCV data lives in `data/processed/val.parquet`. Feature engineering
happens upstream (not in this repo's code) and produces 23 `_z`-suffixed
feature columns appended to the parquet. The features include:

```
log_ret_z, ret_vol_20_z, ret_vol_60_z,           # Return features
sma_10_z, sma_30_z, ema_10_z, ema_30_z,           # Moving averages
rsi_14_z, macd_line_z, macd_signal_z, macd_hist_z, # Momentum
stoch_k_z, stoch_d_z, willr_14_z, adx_14_z,       # Oscillators
bb_width_z, atr_14_z,                               # Volatility
obv_diff_z, rel_vol_20_z, vwap_60_z,               # Volume
price_pos_z, dist_high_z, dist_low_z               # Price position
```

These are defined in `FEATURE_COLS` in `backtest/live_data.py`.

The NSE ticker format uses bare symbols (e.g., "RELIANCE"), not the Yahoo
Finance format ("RELIANCE.NS"). The `fetch_live_quote()` function in
`backtest/live_data.py` appends ".NS" when querying yfinance.

## Tunable Parameters

| Parameter | What It Does | How It Affects the Project |
|-----------|--------------|--------------------------|
| Timeframe | Bar granularity | 1-minute bars give maximum detail but maximum noise. 5-minute bars smooth noise but reduce data. We use 1-minute. |
| Date range | How far back data goes | More history = more training data, but older market regimes may not apply |
| Ticker count | How many stocks to include | More stocks = more diversification, more data, but some may have thin liquidity |

## Common Pitfalls

1. **Using adjusted close vs close.** Adjusted close accounts for stock splits
   and dividends. This project uses raw close prices from the parquet, which
   should already be adjusted upstream.

2. **Ignoring market hours.** NSE trades 9:15-15:30 IST, Mon-Fri. Data outside
   these hours doesn't exist. The `is_market_open()` function in
   `backtest/live_data.py` checks this.

3. **Treating all tickers equally.** Some tickers have much more data than
   others (see the month-by-month distribution — 2022-10 has only 7 tickers,
   while 2024-Q2 has 103). This affects walk-forward period design.
