"""Agent 5: Execution & Risk Management.

Consumes Agent 1's full signal stream + Agent 4's sizing decisions.
Merges them and feeds the existing validated simulator.

Writes to: executions table (for logging).
Reads from: portfolio, signals tables.
Produces: equity curve + trade log via existing simulator.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.schema import get_conn, log_agent, init_db
from backtest.simulator import run_historical_backtest

DEFAULT_CONFIG = {
    "capital": 100_000_000,
    "position_size_pct": 0.02,
    "cost_bps": 5,
    "spread_bps": 3,
    "latency_bars": 1,
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.10,
    "confidence_threshold": 0.85,
    "min_holding_bars": 75,
    "max_positions": 20,
    "max_position_pct": 0.05,
}


class ExecutionAgent:
    """Merges Agent 1 signals + Agent 4 sizing, feeds existing simulator."""

    def __init__(self, db_path=None, config=None):
        self.db_path = db_path
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def build_signals_df(self):
        """Build signals: Agent 1 full stream, with filtered-out Buys demoted to Hold."""
        with get_conn(self.db_path) as conn:
            signals = pd.read_sql(
                "SELECT bar_idx, ticker, timestamp, signal, confidence, model FROM signals", conn
            )
            portfolio = pd.read_sql(
                "SELECT ticker, timestamp, size_pct FROM portfolio", conn
            )

        if len(signals) == 0:
            return pd.DataFrame()

        signals["timestamp"] = pd.to_datetime(signals["timestamp"])
        portfolio["timestamp"] = pd.to_datetime(portfolio["timestamp"])

        # Mark which (ticker, timestamp) pairs are in the portfolio (approved Buys)
        portfolio_set = set(zip(portfolio["ticker"], portfolio["timestamp"]))
        # Build size_pct lookup from portfolio
        size_pct_map = dict(zip(
            zip(portfolio["ticker"], portfolio["timestamp"]),
            portfolio["size_pct"]
        ))

        # For Buy signals: keep only if in portfolio (regime-approved)
        # Hold and Sell signals pass through unchanged (needed for exits)
        merged = signals.copy()
        buy_mask = merged["signal"] == "Buy"
        approved = merged.apply(lambda r: (r["ticker"], r["timestamp"]) in portfolio_set, axis=1)
        merged.loc[buy_mask & ~approved, "signal"] = "Hold"

        # Merge size_pct from portfolio for approved buys
        merged["size_pct"] = merged.apply(
            lambda r: size_pct_map.get((r["ticker"], r["timestamp"]), 0.02), axis=1
        )

        merged["predicted_signal"] = merged["signal"]
        merged["predicted_confidence"] = merged["confidence"]

        # Deduplicate: keep last signal per (ticker, timestamp)
        merged = merged.sort_values(["ticker", "timestamp"]).drop_duplicates(
            subset=["ticker", "timestamp"], keep="last"
        )

        n_buys = int((merged["predicted_signal"] == "Buy").sum())
        n_sells = int((merged["predicted_signal"] == "Sell").sum())
        n_holds = int((merged["predicted_signal"] == "Hold").sum())
        log_agent("agent5_execution", f"Built signals DataFrame",
                  details={"n_signals": len(merged), "n_buys": n_buys,
                           "n_sells": n_sells, "n_holds": n_holds},
                  db_path=self.db_path)

        return merged[["ticker", "timestamp", "predicted_signal", "predicted_confidence", "model", "size_pct"]]

    def run_backtest(self, prices_df, output_dir=None):
        """Run the existing simulator with merged multi-agent signals."""
        signals_df = self.build_signals_df()
        if len(signals_df) == 0:
            return None, pd.DataFrame()

        prices_df = prices_df.copy()
        prices_df["timestamp"] = pd.to_datetime(prices_df["timestamp"])
        signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])

        sim = run_historical_backtest(
            signals_df, prices_df,
            capital=self.config["capital"],
            position_size_pct=self.config["position_size_pct"],
            cost_bps=self.config["cost_bps"],
            spread_bps=self.config["spread_bps"],
            latency_bars=self.config["latency_bars"],
            output_dir=output_dir,
            confidence_threshold=self.config["confidence_threshold"],
            min_holding_bars=self.config["min_holding_bars"],
            max_positions=self.config["max_positions"],
            max_position_pct=self.config["max_position_pct"],
            stop_loss_pct=self.config["stop_loss_pct"],
            take_profit_pct=self.config["take_profit_pct"],
        )

        eq_df = sim.get_equity_curve_df()
        trade_df = sim.get_trade_log_df()

        log_agent("agent5_execution", f"Backtest complete",
                  details={"n_trades": len(trade_df),
                           "final_equity": float(eq_df["equity"].iloc[-1]) if len(eq_df) else 0},
                  db_path=self.db_path)

        return eq_df, trade_df


def run_agent5(db_path=None, config=None, prices_df=None):
    """Convenience function."""
    db_path = db_path or init_db()
    if prices_df is None:
        prices_df = pd.read_parquet(
            str(PROJECT_ROOT / "data" / "processed" / "val.parquet"),
            columns=["ticker", "timestamp", "open", "high", "low", "close", "volume"],
        )
    agent = ExecutionAgent(db_path, config)
    return agent.run_backtest(prices_df)


if __name__ == "__main__":
    run_agent5()
