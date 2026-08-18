"""Period-by-period comparison: single-model vs multi-agent (3 variants).

Columns:
  A: Single model (no regime, no exposure scaling)
  B: Multi-agent, original settings (threshold=0.002, choppy=70%)
  C: Multi-agent, threshold-only change (threshold=0.0005, choppy=70%)
  D: Multi-agent, both changes (threshold=0.0005, choppy=90%)
"""
import sys
import time
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.simulator import run_historical_backtest
from backtest.analytics import compute_metrics
from agents.agent2_regime import RegimeAgent
from agents.agent4_portfolio import PortfolioAgent
from agents.agent5_execution import ExecutionAgent
from agents.schema import init_db

# ── Period definitions ───────────────────────────────────────────────
PERIODS = {
    "P1: Jan-Mar'23": ("2023-01-01", "2023-04-01"),
    "P2: Apr-Sep'23": ("2023-04-01", "2023-10-01"),
    "P3: Oct'23-Mar'24": ("2023-10-01", "2024-04-01"),
    "P4: Apr-Sep'24": ("2024-04-01", "2024-10-01"),
}

# ── Config ───────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.85
MIN_HOLDING_BARS = 75
CAPITAL = 100_000_000
POSITION_SIZE_PCT = 0.02
COST_BPS = 5
SPREAD_BPS = 3
LATENCY_BARS = 1
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.10

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SIGNALS_DIR = PROJECT_ROOT / "backtest" / "signals"

# ── The two configs being tested ─────────────────────────────────────
ORIGINAL_EXPOSURE = {
    "calm-trending": 1.0, "calm-choppy": 0.7,
    "volatile-trending": 0.6, "volatile-choppy": 0.3, "drawdown": 0.2,
}
NEW_EXPOSURE = {
    "calm-trending": 1.0, "calm-choppy": 0.9,
    "volatile-trending": 0.6, "volatile-choppy": 0.3, "drawdown": 0.2,
}
ORIGINAL_THRESHOLD = 0.002
NEW_THRESHOLD = 0.0005


def run_single_model(signals_p, prices_p):
    """Single-model backtest (column A)."""
    sim = run_historical_backtest(
        signals_p, prices_p,
        capital=CAPITAL, position_size_pct=POSITION_SIZE_PCT,
        cost_bps=COST_BPS, spread_bps=SPREAD_BPS,
        latency_bars=LATENCY_BARS, output_dir=None,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        min_holding_bars=MIN_HOLDING_BARS,
        stop_loss_pct=STOP_LOSS_PCT, take_profit_pct=TAKE_PROFIT_PCT,
    )
    return compute_metrics(sim.get_equity_curve_df(), sim.get_trade_log_df())


def run_multi_agent_variant(prices_p, signals_df, db_path, trend_threshold, regime_exposure):
    """Multi-agent with specific threshold + exposure mapping."""
    # Reset all tables
    with sqlite3.connect(str(db_path)) as conn:
        for t in ["signals", "regime", "portfolio", "executions", "agent_log"]:
            conn.execute(f"DELETE FROM {t}")

    # Write signals AFTER reset
    write_signals_to_db(db_path, signals_df)

    # Agent 2: regime with given threshold
    agent2 = RegimeAgent(db_path, trend_threshold=trend_threshold)
    agent2.run_backtest(prices_p)

    # Agent 4: portfolio with given exposure mapping
    agent4 = PortfolioAgent(db_path, regime_exposure=regime_exposure)
    agent4.run_backtest()

    # Agent 5: execution
    agent5 = ExecutionAgent(db_path, {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "min_holding_bars": MIN_HOLDING_BARS,
        "capital": CAPITAL, "position_size_pct": POSITION_SIZE_PCT,
        "cost_bps": COST_BPS, "spread_bps": SPREAD_BPS,
        "latency_bars": LATENCY_BARS,
        "stop_loss_pct": STOP_LOSS_PCT, "take_profit_pct": TAKE_PROFIT_PCT,
    })
    eq, trades = agent5.run_backtest(prices_p)
    if eq is not None and len(eq) > 1:
        return compute_metrics(eq, trades)
    return None


def write_signals_to_db(db_path, signals_df):
    """Write signals DataFrame to DB (Agent 1 substitute)."""
    records = []
    for i, (_, row) in enumerate(signals_df.iterrows()):
        records.append({
            "bar_idx": i, "ticker": row["ticker"],
            "timestamp": str(row["timestamp"]),
            "signal": row["predicted_signal"],
            "confidence": float(row["predicted_confidence"]),
            "prob_buy": 0.0, "prob_hold": 0.0, "prob_sell": 0.0, "model": "lstm",
        })
    with sqlite3.connect(str(db_path)) as conn:
        batch = 50000
        for s in range(0, len(records), batch):
            conn.executemany(
                "INSERT INTO signals (bar_idx, ticker, timestamp, signal, confidence, "
                "prob_buy, prob_hold, prob_sell, model) VALUES "
                "(:bar_idx, :ticker, :timestamp, :signal, :confidence, "
                ":prob_buy, :prob_hold, :prob_sell, :model)",
                records[s:s+batch]
            )


def main():
    print("=" * 130)
    print("  4-COLUMN PERIOD-BY-PERIOD COMPARISON")
    print("  A: Single model | B: Original multi-agent | C: Threshold-only | D: Both changes")
    print("=" * 130)

    # Pre-load data
    print("\nLoading data...")
    signals_full = pd.read_parquet(str(SIGNALS_DIR / "lstm_val_signals.parquet"))
    signals_full["timestamp"] = pd.to_datetime(signals_full["timestamp"])
    prices_full = pd.read_parquet(
        str(DATA_DIR / "val.parquet"),
        columns=["ticker", "timestamp", "open", "high", "low", "close", "volume"],
    )
    prices_full["timestamp"] = pd.to_datetime(prices_full["timestamp"])
    print(f"  {len(signals_full):,} signals, {len(prices_full):,} bars loaded")

    tmp_db = Path(tempfile.mktemp(suffix=".db"))
    init_db(tmp_db)

    all_results = {}

    for period_name, (start, end) in PERIODS.items():
        print(f"\n{'='*130}")
        print(f"  {period_name}")
        print(f"{'='*130}")

        mask_p = (prices_full["timestamp"] >= start) & (prices_full["timestamp"] < end)
        prices_p = prices_full[mask_p].copy()
        mask_s = (signals_full["timestamp"] >= start) & (signals_full["timestamp"] < end)
        signals_p = signals_full[mask_s].copy()

        n_tickers = prices_p["ticker"].nunique()
        print(f"  Prices: {len(prices_p):,} bars, {n_tickers} tickers | Signals: {len(signals_p):,}")

        if len(prices_p) < 1000 or len(signals_p) < 100:
            print("  SKIPPED")
            continue

        # ── Col A: Single Model ────────────────────────────────────
        t0 = time.time()
        m_a = run_single_model(signals_p, prices_p)
        print(f"\n  A) SINGLE MODEL ({time.time()-t0:.1f}s)")
        print(f"     Return: {m_a['total_return']:+.2%}  Sharpe: {m_a['sharpe']:.3f}  "
              f"MaxDD: {m_a['max_drawdown']:.2%}  Trades: {m_a['n_trades']:,}")

        # Write signals to DB (done inside each variant call after reset)

        # ── Col B: Original (threshold=0.002, choppy=70%) ──────────
        t0 = time.time()
        m_b = run_multi_agent_variant(prices_p, signals_p, tmp_db, ORIGINAL_THRESHOLD, ORIGINAL_EXPOSURE)
        print(f"\n  B) ORIGINAL multi-agent ({time.time()-t0:.1f}s) [threshold=0.002, choppy=70%]")
        if m_b:
            print(f"     Return: {m_b['total_return']:+.2%}  Sharpe: {m_b['sharpe']:.3f}  "
                  f"MaxDD: {m_b['max_drawdown']:.2%}  Trades: {m_b['n_trades']:,}")
        else:
            m_b = {"total_return": 0, "sharpe": 0, "max_drawdown": 0, "n_trades": 0, "win_rate": 0}
            print("     No results")

        # ── Col C: Threshold-only (threshold=0.0005, choppy=70%) ───
        t0 = time.time()
        m_c = run_multi_agent_variant(prices_p, signals_p, tmp_db, NEW_THRESHOLD, ORIGINAL_EXPOSURE)
        print(f"\n  C) THRESHOLD-ONLY ({time.time()-t0:.1f}s) [threshold=0.0005, choppy=70%]")
        if m_c:
            print(f"     Return: {m_c['total_return']:+.2%}  Sharpe: {m_c['sharpe']:.3f}  "
                  f"MaxDD: {m_c['max_drawdown']:.2%}  Trades: {m_c['n_trades']:,}")
        else:
            m_c = {"total_return": 0, "sharpe": 0, "max_drawdown": 0, "n_trades": 0, "win_rate": 0}
            print("     No results")

        # ── Col D: Both changes (threshold=0.0005, choppy=90%) ─────
        t0 = time.time()
        m_d = run_multi_agent_variant(prices_p, signals_p, tmp_db, NEW_THRESHOLD, NEW_EXPOSURE)
        print(f"\n  D) BOTH CHANGES ({time.time()-t0:.1f}s) [threshold=0.0005, choppy=90%]")
        if m_d:
            print(f"     Return: {m_d['total_return']:+.2%}  Sharpe: {m_d['sharpe']:.3f}  "
                  f"MaxDD: {m_d['max_drawdown']:.2%}  Trades: {m_d['n_trades']:,}")
        else:
            m_d = {"total_return": 0, "sharpe": 0, "max_drawdown": 0, "n_trades": 0, "win_rate": 0}
            print("     No results")

        all_results[period_name] = {"a": m_a, "b": m_b, "c": m_c, "d": m_d}

    # ── Summary Table ───────────────────────────────────────────────
    print("\n\n")
    print("=" * 140)
    print("  SUMMARY: A=Single | B=Original | C=Threshold-only | D=Both")
    print("=" * 140)

    hdr = f"{'Period':<22} {'Metric':<9} {'A:Single':>12} {'B:Original':>12} {'C:Thresh':>12} {'D:Both':>12} {'C-A':>10} {'D-C':>10}"
    print(hdr)
    print("-" * 140)

    for period_name in PERIODS:
        if period_name not in all_results:
            continue
        r = all_results[period_name]

        for metric, label, fmt in [
            ("total_return", "Return", "{:+.2%}"),
            ("sharpe", "Sharpe", "{:.3f}"),
            ("max_drawdown", "MaxDD", "{:.2%}"),
            ("n_trades", "Trades", "{:,}"),
            ("win_rate", "WinRate", "{:.1%}"),
        ]:
            vals = [r[col].get(metric, 0) for col in ["a", "b", "c", "d"]]
            c_minus_a = vals[2] - vals[0]  # threshold-only vs single
            d_minus_c = vals[3] - vals[2]  # both vs threshold-only

            if metric == "n_trades":
                delta1 = f"{c_minus_a:+,.0f}"
                delta2 = f"{d_minus_c:+,.0f}"
            elif "sharpe" in metric:
                delta1 = f"{c_minus_a:+.3f}"
                delta2 = f"{d_minus_c:+.3f}"
            else:
                delta1 = f"{c_minus_a:+.2%}"
                delta2 = f"{d_minus_c:+.2%}"

            print(f"{period_name if metric == 'total_return' else '':<22} "
                  f"{label:<9} "
                  f"{fmt.format(vals[0]):>12} {fmt.format(vals[1]):>12} "
                  f"{fmt.format(vals[2]):>12} {fmt.format(vals[3]):>12} "
                  f"{delta1:>10} {delta2:>10}")
        print("-" * 140)

    # ── Averages ────────────────────────────────────────────────────
    cols = ["a", "b", "c", "d"]
    for metric, label, fmt in [
        ("total_return", "Return", "{:+.2%}"),
        ("sharpe", "Sharpe", "{:.3f}"),
        ("max_drawdown", "MaxDD", "{:.2%}"),
    ]:
        avgs = [np.mean([all_results[p][col].get(metric, 0) for p in all_results]) for col in cols]
        c_minus_a = avgs[2] - avgs[0]
        d_minus_c = avgs[3] - avgs[2]
        if "sharpe" in metric:
            d1, d2 = f"{c_minus_a:+.3f}", f"{d_minus_c:+.3f}"
        else:
            d1, d2 = f"{c_minus_a:+.2%}", f"{d_minus_c:+.2%}"
        print(f"{'AVERAGE':<22} {label:<9} "
              f"{fmt.format(avgs[0]):>12} {fmt.format(avgs[1]):>12} "
              f"{fmt.format(avgs[2]):>12} {fmt.format(avgs[3]):>12} "
              f"{d1:>10} {d2:>10}")

    print("=" * 140)
    print("\n  C-A = effect of threshold change alone (0.002→0.0005)")
    print("  D-C = effect of exposure change alone (choppy 70%→90%)")
    print("  B = original multi-agent (no changes)")
    print("  D = combined effect of both changes")

    tmp_db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
