"""Multi-agent orchestrator — runs all agents in dependency order.

For backtest: runs Agent 1 (signals) → Agent 2 (regime) → Agent 4 (portfolio) → Agent 5 (execution)
For live/replay: same pipeline, called per-bar.

Usage:
    python -m backtest.run_multiagent
    python -m backtest.run_multiagent --model cnn1d --stop-loss 0.05
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.schema import init_db, reset_tables, get_conn, log_agent
from agents.agent1_signal import SignalAgent
from agents.agent2_regime import RegimeAgent
from agents.agent4_portfolio import PortfolioAgent
from agents.agent5_execution import ExecutionAgent
from backtest.analytics import compute_metrics


def run_full_backtest(model_name="lstm", db_path=None, config=None):
    """Run the complete multi-agent pipeline on validation data."""
    db_path = db_path or init_db()
    reset_tables(db_path)

    print("=" * 80)
    print(f"  MULTI-AGENT BACKTEST — Model: {model_name}")
    print("=" * 80)

    t0 = time.time()

    # ── Agent 1: Signal Generation ────────────────────────────────
    print("\n[1/4] Agent 1: Signal Generation")
    t1 = time.time()
    agent1 = SignalAgent(model_name, db_path=db_path)
    agent1.load()
    signals_df = agent1.run_backtest()
    print(f"  Done ({time.time()-t1:.1f}s)")

    # ── Agent 2: Regime Detection ─────────────────────────────────
    print("\n[2/4] Agent 2: Regime Detection")
    t2 = time.time()
    prices = pd.read_parquet(
        str(PROJECT_ROOT / "data" / "processed" / "val.parquet"),
        columns=["ticker", "timestamp", "open", "high", "low", "close", "volume"],
    )
    agent2 = RegimeAgent(db_path)
    regime_df = agent2.run_backtest(prices)
    print(f"  Done ({time.time()-t2:.1f}s)")

    # ── Agent 4: Portfolio Optimization ───────────────────────────
    print("\n[3/4] Agent 4: Portfolio Optimization")
    t3 = time.time()
    agent4 = PortfolioAgent(db_path, config)
    portfolio_df = agent4.run_backtest()
    print(f"  Done ({time.time()-t3:.1f}s)")

    # ── Agent 5: Execution ────────────────────────────────────────
    print("\n[4/4] Agent 5: Execution")
    t4 = time.time()
    agent5 = ExecutionAgent(db_path, config)
    eq_df, trade_df = agent5.run_backtest(prices)
    print(f"  Done ({time.time()-t4:.1f}s)")

    # ── Results ───────────────────────────────────────────────────
    total_time = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  RESULTS (total: {total_time:.1f}s)")
    print(f"{'='*80}")

    if eq_df is not None and len(eq_df) > 1:
        metrics = compute_metrics(eq_df, trade_df)
        print(f"\n  Return:     {metrics['total_return']:+.2%}")
        print(f"  Sharpe:     {metrics['sharpe']:.3f}")
        print(f"  Max DD:     {metrics['max_drawdown']:.2%}")
        print(f"  Trades:     {metrics['n_trades']:,}")
        print(f"  Win Rate:   {metrics['win_rate']:.1%}")
        print(f"  Final Eq:   {metrics['final_equity']:,.0f}")

        # Reconciliation assertion (Bug #3 check)
        if len(trade_df) > 0:
            closed_pnl = trade_df["net_pnl"].sum()
            expected_final = metrics["initial_equity"] + closed_pnl
            diff = abs(metrics["final_equity"] - expected_final)
            assert diff < 1.0, (
                f"BUG: equity {metrics['final_equity']:,.2f} != expected {expected_final:,.2f}"
            )
            print(f"  ✓ Reconciliation passed (diff: {diff:.2f})")

        # Regime breakdown
        with get_conn(db_path) as conn:
            regime_log = conn.execute(
                "SELECT details_json FROM agent_log WHERE agent='agent4_portfolio' AND event LIKE '%Reduced%'"
            ).fetchall()
        if regime_log:
            print(f"\n  Regime-driven exposure reductions: {len(regime_log)}")

        # Trade exit reasons
        if len(trade_df) > 0 and "exit_reason" in trade_df.columns:
            reasons = trade_df["exit_reason"].value_counts()
            print(f"\n  Exit reasons:")
            for r, c in reasons.items():
                print(f"    {r}: {c}")

        return {"metrics": metrics, "equity_curve": eq_df, "trades": trade_df, "db_path": db_path}
    else:
        print("  No results produced")
        return None


def compare_with_baseline(multiagent_result, model_name="lstm"):
    """Compare multi-agent results against the single-model baseline."""
    if multiagent_result is None:
        return

    # Load existing baseline walk-forward results
    baseline_path = PROJECT_ROOT / "backtest" / "reports" / "walk_forward_results.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        model_baseline = baseline[baseline["model"] == model_name]
        if len(model_baseline) > 0:
            avg_ret = model_baseline["return"].mean()
            avg_sharpe = model_baseline["sharpe"].mean()
            avg_dd = model_baseline["max_drawdown"].mean()

            ma_metrics = multiagent_result["metrics"]
            print(f"\n  COMPARISON vs {model_name.upper()} baseline:")
            print(f"  {'Metric':<16} {'Baseline':>12} {'Multi-Agent':>14} {'Delta':>10}")
            print(f"  {'-'*52}")
            print(f"  {'Return':<16} {avg_ret:>+11.2%} {ma_metrics['total_return']:>+13.2%} "
                  f"{ma_metrics['total_return']-avg_ret:>+9.2%}")
            print(f"  {'Sharpe':<16} {avg_sharpe:>12.3f} {ma_metrics['sharpe']:>14.3f} "
                  f"{ma_metrics['sharpe']-avg_sharpe:>+9.3f}")
            print(f"  {'MaxDD':<16} {avg_dd:>11.2%} {ma_metrics['max_drawdown']:>13.2%} "
                  f"{ma_metrics['max_drawdown']-avg_dd:>+9.2%}")


def main():
    parser = argparse.ArgumentParser(description="Multi-agent backtest orchestrator")
    parser.add_argument("--model", default="lstm", choices=["lstm", "cnn1d", "cnn_lstm"])
    parser.add_argument("--stop-loss", type=float, default=0.05)
    parser.add_argument("--take-profit", type=float, default=0.10)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--max-position-pct", type=float, default=0.05)
    args = parser.parse_args()

    config = {
        "stop_loss_pct": args.stop_loss,
        "take_profit_pct": args.take_profit,
        "max_positions": args.max_positions,
        "max_position_pct": args.max_position_pct,
    }

    result = run_full_backtest(args.model, config=config)
    compare_with_baseline(result, args.model)


if __name__ == "__main__":
    main()
