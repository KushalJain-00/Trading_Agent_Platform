"""Full corrected sweep — uses original _run_backtest_core (tested, assertion-verified).

Pre-builds merged table once per model, then applies position filters per config.
Calls the original _run_backtest_core for equity curve + assertion check.
"""
import sys, time
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.simulator import _run_backtest_core
from backtest.analytics import compute_metrics

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SIGNALS_DIR = PROJECT_ROOT / "backtest" / "signals"
REPORTS_DIR = PROJECT_ROOT / "backtest" / "reports"

MODELS = ["lstm", "cnn1d", "cnn_lstm"]
CONFIDENCES = [0.65, 0.75, 0.80, 0.85, 0.90]
HOLDINGS = [10, 30, 50, 75]


def main():
    print("Loading validation prices...")
    prices_df = pd.read_parquet(
        str(DATA_DIR / "val.parquet"),
        columns=["ticker", "timestamp", "open", "high", "low", "close", "volume"],
    )
    print(f"  {len(prices_df):,} bars\n")

    rows = []
    total = len(MODELS) * len(CONFIDENCES) * len(HOLDINGS)
    run_num = 0

    for model in MODELS:
        sig_path = SIGNALS_DIR / f"{model}_val_signals.parquet"
        print(f"Loading signals: {model}")
        signals_df = pd.read_parquet(sig_path)
        print(f"  {len(signals_df):,} signals")

        for conf in CONFIDENCES:
            for hold in HOLDINGS:
                run_num += 1
                t0 = time.time()

                try:
                    result = _run_backtest_core(
                        signals_df, prices_df,
                        capital=100_000_000, position_size_pct=0.02,
                        cost_bps=5, spread_bps=3, latency_bars=1, output_dir=None,
                        confidence_threshold=conf, min_holding_bars=hold,
                    )
                    eq = result.get_equity_curve_df()
                    trades = result.get_trade_log_df()
                    met = compute_metrics(eq, trades)
                    elapsed = time.time() - t0

                    rows.append({
                        "model": model, "confidence": conf, "holding": hold,
                        "return": met["total_return"], "cagr": met["cagr"],
                        "sharpe": met["sharpe"], "max_dd": met["max_drawdown"],
                        "win_rate": met["win_rate"], "pf": met["profit_factor"],
                        "n_trades": met["n_trades"], "cost_pct": met["cost_pct_gross_pnl"],
                        "avg_holding": met["avg_holding_bars"],
                        "final_equity": met["final_equity"],
                    })

                    print(f"  [{run_num:2d}/{total}] {model:<8} conf={conf:.2f} hold={hold:<3} "
                          f"ret={met['total_return']:+7.2%} sharpe={met['sharpe']:6.3f} "
                          f"maxdd={met['max_drawdown']:7.2%} trades={met['n_trades']:>6,} "
                          f"win={met['win_rate']:.1%} cost%={met['cost_pct_gross_pnl']:.1f}%  ({elapsed:.1f}s)")
                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"  [{run_num:2d}/{total}] {model:<8} conf={conf:.2f} hold={hold:<3} "
                          f"ERROR: {e}  ({elapsed:.1f}s)")
        print()

    # Save
    df = pd.DataFrame(rows)
    out = REPORTS_DIR / "full_sweep_corrected.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}")

    # Print ranked table
    print("\n" + "=" * 140)
    print("  FULL CORRECTED SWEEP — ALL CONFIGS (ranked by return)")
    print("  Equity curve: portfolio ledger with assertion check. All numbers exact.")
    print("=" * 140)
    df_sorted = df.sort_values("return", ascending=False)
    print(f"{'#':>3} {'Model':<10} {'Conf':>5} {'Hold':>5} {'Return':>9} {'CAGR':>8} {'Sharpe':>8} "
          f"{'MaxDD':>9} {'Win%':>6} {'PF':>6} {'Trades':>7} {'Cost%':>7}")
    print("-" * 140)
    for i, (_, r) in enumerate(df_sorted.iterrows(), 1):
        print(f"{i:>3} {r['model']:<10} {r['confidence']:>5.2f} {r['holding']:>5} "
              f"{r['return']:>+8.2%} {r['cagr']:>+7.2%} {r['sharpe']:>8.3f} "
              f"{r['max_dd']:>8.2%} {r['win_rate']:>5.1%} {r['pf']:>6.2f} "
              f"{int(r['n_trades']):>7,} {r['cost_pct']:>6.1f}%")
    print("=" * 140)


if __name__ == "__main__":
    main()
