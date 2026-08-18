"""Agent 2: Regime Detection.

Vectorized rules-based market regime classifier. Computes regime features
across the full timeline using rolling window operations (not per-bar loops).

Regime labels:
  - calm-trending: low vol, directional
  - calm-choppy: low vol, no direction
  - volatile-trending: high vol, directional
  - volatile-choppy: high vol, no direction
  - drawdown: significant drawdown from peak

Does NOT forecast regime — only classifies current state reactively.
Writes to: regime table.
Reads from: prices (passed as DataFrame).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.schema import get_conn, log_agent, init_db

# ── Thresholds (defaults, overridable via constructor) ────────────────
VOL_LOOKBACK = 20
VOL_PERCENTILE_WINDOW = 252
VOL_HIGH_PCT = 70
VOL_LOW_PCT = 30
TREND_FAST_MA = 10
TREND_SLOW_MA = 30
TREND_STRENGTH_THRESHOLD = 0.0005  # 0.05% — calibrated for 1-minute data
DRAWDOWN_THRESHOLD = -0.10


def compute_market_index(prices_df):
    """Compute equal-weight index of all tickers."""
    px = prices_df[["ticker", "timestamp", "close"]].copy()
    px["timestamp"] = pd.to_datetime(px["timestamp"])
    pivot = px.pivot_table(index="timestamp", columns="ticker", values="close")
    pivot = pivot.sort_index()
    returns = pivot.pct_change(fill_method=None)
    index_returns = returns.mean(axis=1)
    index_price = (1 + index_returns).cumprod()
    index_price.iloc[0] = 1.0
    return index_price, index_returns


def classify_regime_vectorized(index_price, index_returns,
                                trend_threshold=TREND_STRENGTH_THRESHOLD):
    """Vectorized regime classification across the full timeline."""
    n = len(index_price)

    # ── Rolling realized vol (annualized) ──────────────────────────
    rolling_vol = index_returns.rolling(VOL_LOOKBACK).std() * np.sqrt(252 * 375)

    # ── Vol percentile over rolling window ─────────────────────────
    vol_pctile = rolling_vol.rolling(VOL_PERCENTILE_WINDOW).rank(pct=True) * 100

    is_volatile = vol_pctile >= VOL_HIGH_PCT
    is_calm = vol_pctile <= VOL_LOW_PCT

    # ── Trend detection (MA crossover) ─────────────────────────────
    fast_ma = index_price.rolling(TREND_FAST_MA).mean()
    slow_ma = index_price.rolling(TREND_SLOW_MA).mean()
    ma_spread = (fast_ma - slow_ma) / slow_ma
    is_trending = ma_spread.abs() >= trend_threshold

    # ── Drawdown from peak ─────────────────────────────────────────
    peak = index_price.cummax()
    drawdown = (index_price - peak) / peak
    is_drawdown = drawdown <= DRAWDOWN_THRESHOLD

    # ── Classify per bar ───────────────────────────────────────────
    regime_labels = []
    regime_confidences = []
    features_list = []

    for i in range(n):
        vol_p = vol_pctile.iloc[i] if not np.isnan(vol_pctile.iloc[i]) else 50
        spread = ma_spread.iloc[i] if not np.isnan(ma_spread.iloc[i]) else 0
        dd = drawdown.iloc[i] if not np.isnan(drawdown.iloc[i]) else 0
        rv = rolling_vol.iloc[i] if not np.isnan(rolling_vol.iloc[i]) else 0

        feat = {
            "vol_pctile": round(float(vol_p), 1),
            "recent_vol": round(float(rv), 4),
            "ma_spread": round(float(spread), 6),
            "trend_dir": "up" if spread > 0 else "down",
            "drawdown": round(float(dd), 4),
        }
        features_list.append(feat)

        if is_drawdown.iloc[i]:
            regime = "drawdown"
            confidence = min(1.0, abs(dd) / abs(DRAWDOWN_THRESHOLD))
        elif is_volatile.iloc[i] and is_trending.iloc[i]:
            regime = "volatile-trending"
            confidence = 0.5 + 0.5 * max(0, (vol_p - VOL_HIGH_PCT) / max(100 - VOL_HIGH_PCT, 1))
        elif is_volatile.iloc[i]:
            regime = "volatile-choppy"
            confidence = 0.5 + 0.5 * max(0, (vol_p - VOL_HIGH_PCT) / max(100 - VOL_HIGH_PCT, 1))
        elif is_calm.iloc[i] and is_trending.iloc[i]:
            regime = "calm-trending"
            confidence = 0.5 + 0.5 * max(0, (VOL_LOW_PCT - vol_p) / max(VOL_LOW_PCT, 1))
        else:
            regime = "calm-choppy"
            confidence = 0.5 + 0.5 * max(0, (VOL_LOW_PCT - vol_p) / max(VOL_LOW_PCT, 1))

        confidence = max(0.1, min(1.0, confidence))
        regime_labels.append(regime)
        regime_confidences.append(confidence)

    return regime_labels, regime_confidences, features_list


class RegimeAgent:
    """Regime detection agent — vectorized classification."""

    def __init__(self, db_path=None, trend_threshold=TREND_STRENGTH_THRESHOLD):
        self.db_path = db_path
        self.trend_threshold = trend_threshold

    def run_backtest(self, prices_df):
        """Classify regime for every unique timestamp. Writes to DB."""
        import json

        index_price, index_returns = compute_market_index(prices_df)
        timestamps = index_price.index.tolist()

        regime_labels, regime_confidences, features_list = classify_regime_vectorized(
            index_price, index_returns, trend_threshold=self.trend_threshold
        )

        records = []
        for i, ts in enumerate(timestamps):
            records.append({
                "bar_idx": i,
                "timestamp": str(ts),
                "regime_label": regime_labels[i],
                "regime_confidence": regime_confidences[i],
                "features_json": json.dumps(features_list[i]),
            })

        df = pd.DataFrame(records)

        with get_conn(self.db_path) as conn:
            conn.executemany(
                "INSERT INTO regime (bar_idx, timestamp, regime_label, regime_confidence, features_json) "
                "VALUES (:bar_idx, :timestamp, :regime_label, :regime_confidence, :features_json)",
                df.to_dict("records")
            )

        regime_counts = df["regime_label"].value_counts()
        log_agent("agent2_regime", f"Regime classification complete",
                  details={"regime_counts": regime_counts.to_dict(), "n_bars": len(df)},
                  db_path=self.db_path)

        print(f"Agent 2: {len(df)} bars classified")
        for r, c in regime_counts.items():
            print(f"  {r}: {c} ({c/len(df)*100:.1f}%)")

        return df


def run_agent2(db_path=None):
    """Convenience function."""
    db_path = db_path or init_db()
    prices = pd.read_parquet(
        str(PROJECT_ROOT / "data" / "processed" / "val.parquet"),
        columns=["ticker", "timestamp", "open", "high", "low", "close", "volume"],
    )
    agent = RegimeAgent(db_path)
    return agent.run_backtest(prices)


if __name__ == "__main__":
    run_agent2()
