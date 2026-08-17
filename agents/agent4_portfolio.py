"""Agent 4: Portfolio Optimization.

Consumes Agent 1's signals + Agent 2's regime state.
Decides position sizing with regime-conditional exposure scaling.

Key logic:
  - Base allocation: equal-weight across signaled positions
  - Regime scaling: reduce overall exposure in bad regimes
  - Volatility targeting: scale positions by inverse vol
  - Risk limits: max positions, max per-ticker weight

This is calculation-based (not ML) — explicit, debuggable, no overfitting.
Writes to: portfolio table.
Reads from: signals, regime tables.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.schema import get_conn, log_agent, init_db

# ── Portfolio config ─────────────────────────────────────────────────
MAX_POSITIONS = 20
MAX_POSITION_PCT = 0.05
BASE_POSITION_PCT = 0.02

REGIME_EXPOSURE = {
    "calm-trending": 1.0,
    "calm-choppy": 0.7,
    "volatile-trending": 0.6,
    "volatile-choppy": 0.3,
    "drawdown": 0.2,
}

MIN_CONFIDENCE = 0.45


class PortfolioAgent:
    """Portfolio optimization agent — batch-processed for speed."""

    def __init__(self, db_path=None, config=None):
        self.db_path = db_path
        self.max_positions = config.get("max_positions", MAX_POSITIONS) if config else MAX_POSITIONS
        self.max_position_pct = config.get("max_position_pct", MAX_POSITION_PCT) if config else MAX_POSITION_PCT
        self.base_position_pct = config.get("base_position_pct", BASE_POSITION_PCT) if config else BASE_POSITION_PCT

    def run_backtest(self):
        """Process all signals, write target allocations to DB. Batch-loaded."""
        with get_conn(self.db_path) as conn:
            signals_df = pd.read_sql("SELECT bar_idx, ticker, timestamp, signal, confidence FROM signals", conn)
            regime_df = pd.read_sql("SELECT timestamp, regime_label FROM regime", conn)

        if len(signals_df) == 0 or len(regime_df) == 0:
            print("Agent 4: No signals or regime data")
            return pd.DataFrame()

        # Aggregate multiple signals per (timestamp, ticker) — take max confidence
        signals_df = signals_df.groupby(["timestamp", "ticker"], as_index=False).agg({
            "signal": "first",
            "confidence": "max",
            "bar_idx": "first",
        })

        # Build regime lookup via merge_asof
        regime_df["timestamp"] = pd.to_datetime(regime_df["timestamp"])
        signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])
        signals_df = signals_df.sort_values("timestamp")
        regime_df = regime_df.sort_values("timestamp")

        merged = pd.merge_asof(
            signals_df, regime_df[["timestamp", "regime_label"]],
            on="timestamp", direction="backward"
        )
        merged["regime_label"] = merged["regime_label"].fillna("calm-trending")

        # Filter to buys above confidence
        buys = merged[(merged["signal"] == "Buy") & (merged["confidence"] >= MIN_CONFIDENCE)].copy()
        buys["regime_scale"] = buys["regime_label"].map(REGIME_EXPOSURE).fillna(0.5)

        # Per-timestamp: rank by confidence, limit to max_positions
        records = []
        n_reductions = 0
        for ts, group in buys.groupby("timestamp"):
            group = group.sort_values("confidence", ascending=False).head(self.max_positions)
            regime = group["regime_label"].iloc[0]
            regime_scale = group["regime_scale"].iloc[0]

            for _, row in group.iterrows():
                size_pct = self.base_position_pct * regime_scale
                conf_scale = 0.5 + 0.5 * row["confidence"]
                size_pct *= conf_scale
                size_pct = min(size_pct, self.max_position_pct)

                records.append({
                    "bar_idx": int(row["bar_idx"]),
                    "ticker": row["ticker"],
                    "timestamp": str(row["timestamp"]),
                    "target_weight": size_pct,
                    "signal": "Buy",
                    "size_pct": size_pct,
                    "reason": f"regime={regime}({regime_scale:.1f}),conf={row['confidence']:.3f}",
                })

            if regime_scale < 1.0:
                n_reductions += 1

        if records:
            with get_conn(self.db_path) as conn:
                conn.executemany(
                    "INSERT INTO portfolio (bar_idx, ticker, timestamp, target_weight, signal, size_pct, reason) "
                    "VALUES (:bar_idx, :ticker, :timestamp, :target_weight, :signal, :size_pct, :reason)",
                    records
                )

        log_agent("agent4_portfolio", f"Portfolio optimization complete",
                  details={"n_allocations": len(records), "n_reductions": n_reductions},
                  db_path=self.db_path)

        print(f"Agent 4: {len(records)} allocations written, {n_reductions} regime-reduced timestamps")
        return pd.DataFrame(records) if records else pd.DataFrame()


def run_agent4(db_path=None):
    """Convenience function."""
    db_path = db_path or init_db()
    agent = PortfolioAgent(db_path)
    return agent.run_backtest()


if __name__ == "__main__":
    run_agent4()
