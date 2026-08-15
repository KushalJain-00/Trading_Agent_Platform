"""Sweep confidence thresholds at fixed min_holding_bars=10."""
import sys, time
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.simulator import run_historical_backtest

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SIGNALS_DIR = PROJECT_ROOT / "backtest" / "signals"
RESULTS_DIR = PROJECT_ROOT / "backtest" / "results"

THRESHOLDS = [0.75, 0.80, 0.85]
MODELS = ["lstm", "cnn1d", "cnn_lstm"]
MIN_HOLD = 10

print("Loading validation prices...")
prices_df = pd.read_parquet(str(DATA_DIR / "val.parquet"),
                             columns=["ticker", "timestamp", "open", "high", "low", "close", "volume"])

rows = []
for threshold in THRESHOLDS:
    print(f"\n{'='*60}  threshold={threshold}  min_hold={MIN_HOLD}  {'='*60}")
    for model in MODELS:
        t0 = time.time()
        signals_df = pd.read_parquet(SIGNALS_DIR / f"{model}_val_signals.parquet")
        sim = run_historical_backtest(
            signals_df, prices_df,
            capital=100_000_000, position_size_pct=0.02,
            cost_bps=5, spread_bps=3, latency_bars=1, output_dir=None,
            confidence_threshold=threshold, min_holding_bars=MIN_HOLD,
        )
        eq = sim.get_equity_curve_df()
        trades = sim.get_trade_log_df()
        fs = sim.get_filter_stats()

        total_ret = (eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1) * 100
        n_trades = len(trades)
        win_rate = (trades["net_pnl"] > 0).mean() * 100 if n_trades else 0
        cost_pct = fs["cost_pct_after"] if fs else 0

        rows.append({
            "threshold": threshold, "model": model,
            "return_pct": total_ret, "trades": n_trades,
            "win_rate": win_rate, "cost_pct": cost_pct,
        })
        print(f"  {model:<10} ret={total_ret:+.2f}%  trades={n_trades:>8,}  win={win_rate:.1f}%  cost%={cost_pct:.1f}%  ({time.time()-t0:.1f}s)")

# Combined table
print("\n\n")
df = pd.DataFrame(rows)
print("=" * 90)
print(f"{'Threshold':>10} {'Model':<12} {'Return%':>9} {'Trades':>10} {'WinRate':>9} {'Cost%P&L':>10}")
print("-" * 90)
for _, r in df.iterrows():
    print(f"{r['threshold']:>10.2f} {r['model']:<12} {r['return_pct']:>+8.2f}% {r['trades']:>10,} "
          f"{r['win_rate']:>8.1f}% {r['cost_pct']:>9.1f}%")
print("=" * 90)

# Also show the original unfiltered row for reference
print(f"\n{'Ref':>10} {'(unfiltered)':<12} {'-6.79 to -12.08%':>17} {'215K-227K':>10} {'32.6%':>9} {'301-304%':>10}")
