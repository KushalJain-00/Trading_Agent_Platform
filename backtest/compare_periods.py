"""Period-by-period comparison: single-model vs multi-agent.

Runs both approaches on the same 4 walk-forward periods and shows
a side-by-side comparison table.
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
from agents.schema import init_db, reset_tables

# ── Period definitions (same as walk_forward.py) ─────────────────────
PERIODS = {
    "P1: Jan-Mar'23": ("2023-01-01", "2023-04-01"),
    "P2: Apr-Sep'23": ("2023-04-01", "2023-10-01"),
    "P3: Oct'23-Mar'24": ("2023-10-01", "2024-04-01"),
    "P4: Apr-Sep'24": ("2024-04-01", "2024-10-01"),
}

# ── Config (matching existing walk_forward.py) ───────────────────────
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


def load_period_data(start, end):
    """Load prices and signals for a specific time period."""
    prices = pd.read_parquet(
        str(DATA_DIR / "val.parquet"),
        columns=["ticker", "timestamp", "open", "high", "low", "close", "volume"],
    )
    prices["timestamp"] = pd.to_datetime(prices["timestamp"])
    mask = (prices["timestamp"] >= start) & (prices["timestamp"] < end)
    prices_p = prices[mask].copy()

    signals = pd.read_parquet(str(SIGNALS_DIR / "lstm_val_signals.parquet"))
    signals["timestamp"] = pd.to_datetime(signals["timestamp"])
    mask_s = (signals["timestamp"] >= start) & (signals["timestamp"] < end)
    signals_p = signals[mask_s].copy()

    return prices_p, signals_p


def run_single_model(signals_p, prices_p):
    """Run single-model backtest with SL/TP on a period."""
    sim = run_historical_backtest(
        signals_p, prices_p,
        capital=CAPITAL, position_size_pct=POSITION_SIZE_PCT,
        cost_bps=COST_BPS, spread_bps=SPREAD_BPS,
        latency_bars=LATENCY_BARS, output_dir=None,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        min_holding_bars=MIN_HOLDING_BARS,
        stop_loss_pct=STOP_LOSS_PCT, take_profit_pct=TAKE_PROFIT_PCT,
    )
    eq = sim.get_equity_curve_df()
    trades = sim.get_trade_log_df()
    return compute_metrics(eq, trades)


def run_multi_agent(prices_p, db_path):
    """Run multi-agent pipeline on a period. Assumes signals already in DB."""
    # Agent 2: Regime (re-run on period prices)
    agent2 = RegimeAgent(db_path)
    agent2.run_backtest(prices_p)

    # Agent 4: Portfolio
    agent4 = PortfolioAgent(db_path)
    agent4.run_backtest()

    # Agent 5: Execution
    agent5 = ExecutionAgent(db_path, {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "min_holding_bars": MIN_HOLDING_BARS,
        "capital": CAPITAL,
        "position_size_pct": POSITION_SIZE_PCT,
        "cost_bps": COST_BPS,
        "spread_bps": SPREAD_BPS,
        "latency_bars": LATENCY_BARS,
        "stop_loss_pct": STOP_LOSS_PCT,
        "take_profit_pct": TAKE_PROFIT_PCT,
    })
    eq, trades = agent5.run_backtest(prices_p)
    if eq is not None and len(eq) > 1:
        return compute_metrics(eq, trades)
    return None


def run_agent1_for_period(db_path, start, end):
    """Write Agent 1 signals for a specific period to the temp DB."""
    signals = pd.read_parquet(str(SIGNALS_DIR / "lstm_val_signals.parquet"))
    signals["timestamp"] = pd.to_datetime(signals["timestamp"])
    mask = (signals["timestamp"] >= start) & (signals["timestamp"] < end)
    signals_p = signals[mask]

    # Reset and write
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("DELETE FROM signals")
        conn.execute("DELETE FROM regime")
        conn.execute("DELETE FROM portfolio")
        conn.execute("DELETE FROM executions")
        conn.execute("DELETE FROM agent_log")

    records = []
    for i, (_, row) in enumerate(signals_p.iterrows()):
        records.append({
            "bar_idx": i,
            "ticker": row["ticker"],
            "timestamp": str(row["timestamp"]),
            "signal": row["predicted_signal"],
            "confidence": float(row["predicted_confidence"]),
            "prob_buy": 0.0, "prob_hold": 0.0, "prob_sell": 0.0,
            "model": "lstm",
        })

    with sqlite3.connect(str(db_path)) as conn:
        batch = 50000
        for start_idx in range(0, len(records), batch):
            conn.executemany(
                "INSERT INTO signals (bar_idx, ticker, timestamp, signal, confidence, "
                "prob_buy, prob_hold, prob_sell, model) VALUES "
                "(:bar_idx, :ticker, :timestamp, :signal, :confidence, "
                ":prob_buy, :prob_hold, :prob_sell, :model)",
                records[start_idx:start_idx+batch]
            )


def main():
    print("=" * 100)
    print("  PERIOD-BY-PERIOD COMPARISON: Single Model vs Multi-Agent")
    print(f"  Config: conf>={CONFIDENCE_THRESHOLD}, min_hold={MIN_HOLDING_BARS}, "
          f"SL={STOP_LOSS_PCT:.0%}, TP={TAKE_PROFIT_PCT:.0%}")
    print("=" * 100)

    # Pre-load full signals once
    print("\nLoading cached signals...")
    signals_full = pd.read_parquet(str(SIGNALS_DIR / "lstm_val_signals.parquet"))
    signals_full["timestamp"] = pd.to_datetime(signals_full["timestamp"])
    print(f"  {len(signals_full):,} signals loaded")

    # Pre-load prices once
    print("Loading prices...")
    prices_full = pd.read_parquet(
        str(DATA_DIR / "val.parquet"),
        columns=["ticker", "timestamp", "open", "high", "low", "close", "volume"],
    )
    prices_full["timestamp"] = pd.to_datetime(prices_full["timestamp"])
    print(f"  {len(prices_full):,} bars loaded")

    # Create temp DB for multi-agent
    tmp_db = Path(tempfile.mktemp(suffix=".db"))
    init_db(tmp_db)

    results = {}

    for period_name, (start, end) in PERIODS.items():
        print(f"\n{'='*100}")
        print(f"  {period_name}")
        print(f"{'='*100}")

        # Filter data to period
        mask_p = (prices_full["timestamp"] >= start) & (prices_full["timestamp"] < end)
        prices_p = prices_full[mask_p].copy()

        mask_s = (signals_full["timestamp"] >= start) & (signals_full["timestamp"] < end)
        signals_p = signals_full[mask_s].copy()

        n_tickers = prices_p["ticker"].nunique()
        print(f"  Prices: {len(prices_p):,} bars, {n_tickers} tickers")
        print(f"  Signals: {len(signals_p):,}")

        if len(prices_p) < 1000 or len(signals_p) < 100:
            print("  SKIPPED — insufficient data")
            continue

        # ── Single Model ───────────────────────────────────────────
        t1 = time.time()
        m_single = run_single_model(signals_p, prices_p)
        t_single = time.time() - t1
        print(f"\n  SINGLE MODEL ({t_single:.1f}s):")
        print(f"    Return: {m_single['total_return']:+.2%}  Sharpe: {m_single['sharpe']:.3f}  "
              f"MaxDD: {m_single['max_drawdown']:.2%}  Trades: {m_single['n_trades']:,}  "
              f"WinRate: {m_single['win_rate']:.1%}")

        # ── Multi-Agent ────────────────────────────────────────────
        t2 = time.time()
        # Write period signals to temp DB (Agent 1 substitute)
        run_agent1_for_period(tmp_db, start, end)
        m_multi = run_multi_agent(prices_p, tmp_db)
        t_multi = time.time() - t2

        if m_multi:
            print(f"\n  MULTI-AGENT ({t_multi:.1f}s):")
            print(f"    Return: {m_multi['total_return']:+.2%}  Sharpe: {m_multi['sharpe']:.3f}  "
                  f"MaxDD: {m_multi['max_drawdown']:.2%}  Trades: {m_multi['n_trades']:,}  "
                  f"WinRate: {m_multi['win_rate']:.1%}")

            # Regime breakdown for this period
            with sqlite3.connect(str(tmp_db)) as conn:
                regime_counts = pd.read_sql(
                    "SELECT regime_label, COUNT(*) as cnt FROM regime GROUP BY regime_label",
                    conn
                )
            if len(regime_counts) > 0:
                total = regime_counts["cnt"].sum()
                print(f"\n  Regime breakdown:")
                for _, row in regime_counts.iterrows():
                    print(f"    {row['regime_label']}: {row['cnt']:,} ({row['cnt']/total*100:.1f}%)")
        else:
            print(f"\n  MULTI-AGENT: No results")
            m_multi = {"total_return": 0, "sharpe": 0, "max_drawdown": 0, "n_trades": 0, "win_rate": 0}

        results[period_name] = {"single": m_single, "multi": m_multi}

    # ── Summary Table ───────────────────────────────────────────────
    print("\n\n")
    print("=" * 120)
    print("  COMPARISON SUMMARY")
    print("=" * 120)

    header = f"{'Period':<22} {'Metric':<10} {'Single Model':>14} {'Multi-Agent':>14} {'Delta':>12}"
    print(header)
    print("-" * 120)

    for period_name in PERIODS:
        if period_name not in results:
            continue
        s = results[period_name]["single"]
        m = results[period_name]["multi"]

        for metric, label, fmt in [
            ("total_return", "Return", "{:+.2%}"),
            ("sharpe", "Sharpe", "{:.3f}"),
            ("max_drawdown", "MaxDD", "{:.2%}"),
            ("n_trades", "Trades", "{:,}"),
            ("win_rate", "WinRate", "{:.1%}"),
        ]:
            sv = s.get(metric, 0)
            mv = m.get(metric, 0)
            delta = mv - sv
            if metric == "n_trades":
                delta_str = f"{delta:+,.0f}"
            else:
                delta_str = f"{delta:+.3f}" if "sharpe" in metric else f"{delta:+.2%}"
            print(f"{period_name if metric == 'total_return' else '':<22} "
                  f"{label:<10} {fmt.format(sv):>14} {fmt.format(mv):>14} {delta_str:>12}")
        print("-" * 120)

    # ── Overall averages ────────────────────────────────────────────
    all_single = [results[p]["single"] for p in results]
    all_multi = [results[p]["multi"] for p in results]

    print(f"{'AVERAGE':<22} {'Return':<10} "
          f"{np.mean([s['total_return'] for s in all_single]):>+13.2%} "
          f"{np.mean([m['total_return'] for m in all_multi]):>+13.2%} "
          f"{np.mean([m['total_return'] for m in all_multi]) - np.mean([s['total_return'] for s in all_single]):>+11.2%}")
    print(f"{'':<22} {'Sharpe':<10} "
          f"{np.mean([s['sharpe'] for s in all_single]):>13.3f} "
          f"{np.mean([m['sharpe'] for m in all_multi]):>13.3f} "
          f"{np.mean([m['sharpe'] for m in all_multi]) - np.mean([s['sharpe'] for s in all_single]):>+11.3f}")
    print(f"{'':<22} {'MaxDD':<10} "
          f"{np.mean([s['max_drawdown'] for s in all_single]):>13.2%} "
          f"{np.mean([m['max_drawdown'] for m in all_multi]):>13.2%} "
          f"{np.mean([m['max_drawdown'] for m in all_multi]) - np.mean([s['max_drawdown'] for s in all_single]):>+11.2%}")
    print("=" * 120)

    # Cleanup
    tmp_db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
