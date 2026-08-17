"""Walk-forward validation — fixed config, honest per-period reporting.

Splits the validation period into sub-periods, runs the same model+config
on each, and reports results transparently (including losing periods).
No re-optimization per period. This tests whether the config we selected
on the full set generalizes to unseen time slices.
"""
import argparse
import sys, time
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.simulator import run_historical_backtest
from backtest.analytics import compute_metrics

# ── Config ───────────────────────────────────────────────────────────
# Fixed config — NO per-period re-optimization
CONFIDENCE_THRESHOLD = 0.85
MIN_HOLDING_BARS = 75
CAPITAL = 100_000_000
POSITION_SIZE_PCT = 0.02
COST_BPS = 5
SPREAD_BPS = 3
LATENCY_BARS = 1

MODELS = ["lstm", "cnn1d", "cnn_lstm"]

# ── Period splits ────────────────────────────────────────────────────
# Based on actual data distribution: strong coverage 2023-01 to 2024-09
# After 2024-09, fewer than 21 tickers — too thin for meaningful test.
PERIODS = {
    "P1: 2023-Q1 (Jan-Mar)": ("2023-01-01", "2023-04-01"),
    "P2: 2023-Q2-Q3 (Apr-Sep)": ("2023-04-01", "2023-10-01"),
    "P3: 2023-Q4-2024-Q1 (Oct-Mar)": ("2023-10-01", "2024-04-01"),
    "P4: 2024-Q2-Q3 (Apr-Sep)": ("2024-04-01", "2024-10-01"),
}

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SIGNALS_DIR = PROJECT_ROOT / "backtest" / "signals"
REPORTS_DIR = PROJECT_ROOT / "backtest" / "reports"


def run_walk_forward(stop_loss_pct=0.0, take_profit_pct=0.0):
    print("=" * 100)
    print("  WALK-FORWARD VALIDATION")
    print(f"  Config: confidence={CONFIDENCE_THRESHOLD}, min_hold={MIN_HOLDING_BARS}")
    if stop_loss_pct > 0 or take_profit_pct > 0:
        print(f"  SL/TP: stop_loss={stop_loss_pct:.1%}, take_profit={take_profit_pct:.1%}")
    print(f"  Models: {MODELS}")
    print(f"  Periods: {len(PERIODS)}")
    print("=" * 100)

    # Load full prices once
    print("\nLoading validation prices...")
    prices_full = pd.read_parquet(
        str(DATA_DIR / "val.parquet"),
        columns=["ticker", "timestamp", "open", "high", "low", "close", "volume"],
    )
    prices_full["timestamp"] = pd.to_datetime(prices_full["timestamp"])
    print(f"  Total bars: {len(prices_full):,}")

    all_results = {}  # {model: {period: metrics}}

    for model_name in MODELS:
        print(f"\n{'='*100}")
        print(f"  MODEL: {model_name.upper()}")
        print(f"{'='*100}")

        # Load cached signals
        sig_path = SIGNALS_DIR / f"{model_name}_val_signals.parquet"
        if not sig_path.exists():
            print(f"  Signals not found at {sig_path}, generating...")
            from backtest.generate_signals import generate_historical_signals
            device = __import__("torch").device("cpu")
            signals_full = generate_historical_signals(
                model_name, str(PROJECT_ROOT / "models" / "checkpoints"),
                str(DATA_DIR), device, str(SIGNALS_DIR), stride=15,
            )
        else:
            signals_full = pd.read_parquet(sig_path)
        signals_full["timestamp"] = pd.to_datetime(signals_full["timestamp"])
        print(f"  Total signals: {len(signals_full):,}")

        all_results[model_name] = {}

        for period_name, (start, end) in PERIODS.items():
            print(f"\n  --- {period_name} ---")

            # Filter to this period
            mask_p = (prices_full["timestamp"] >= start) & (prices_full["timestamp"] < end)
            prices_period = prices_full[mask_p].copy()

            mask_s = (signals_full["timestamp"] >= start) & (signals_full["timestamp"] < end)
            signals_period = signals_full[mask_s].copy()

            n_tickers = prices_period["ticker"].nunique()
            print(f"    Prices: {len(prices_period):,} bars, {n_tickers} tickers")
            print(f"    Signals: {len(signals_period):,}")

            if len(prices_period) < 1000 or len(signals_period) < 100:
                print(f"    SKIPPED — insufficient data")
                all_results[model_name][period_name] = None
                continue

            t0 = time.time()
            sim = run_historical_backtest(
                signals_period, prices_period,
                capital=CAPITAL, position_size_pct=POSITION_SIZE_PCT,
                cost_bps=COST_BPS, spread_bps=SPREAD_BPS,
                latency_bars=LATENCY_BARS, output_dir=None,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                min_holding_bars=MIN_HOLDING_BARS,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )
            eq = sim.get_equity_curve_df()
            trades = sim.get_trade_log_df()
            elapsed = time.time() - t0

            metrics = compute_metrics(eq, trades)
            all_results[model_name][period_name] = metrics

            ret = metrics["total_return"]
            sharpe = metrics["sharpe"]
            maxdd = metrics["max_drawdown"]
            n_trades = metrics["n_trades"]
            win_rate = metrics["win_rate"]
            cost_pct = metrics["cost_pct_gross_pnl"]

            print(f"    Return: {ret:+.2%}  Sharpe: {sharpe:.3f}  MaxDD: {maxdd:.2%}  "
                  f"Trades: {n_trades:,}  WinRate: {win_rate:.1%}  Cost%: {cost_pct:.1f}%  ({elapsed:.1f}s)")

    # ── Summary table ────────────────────────────────────────────────
    print("\n\n")
    print("=" * 120)
    print("  WALK-FORWARD SUMMARY")
    print(f"  Config: confidence={CONFIDENCE_THRESHOLD}, min_hold={MIN_HOLDING_BARS}")
    print("=" * 120)

    header = f"{'Model':<12}"
    for pname in PERIODS:
        short = pname.split("(")[0].strip()
        header += f" {short:>16}"
    header += f" {'OVERALL':>16}"
    print(header)
    print("-" * 120)

    for model_name in MODELS:
        row = f"{model_name:<12}"
        period_returns = []
        for pname in PERIODS:
            m = all_results[model_name].get(pname)
            if m is None:
                row += f" {'N/A':>16}"
            else:
                row += f" {m['total_return']:>+15.2%}"
                period_returns.append(m["total_return"])
        # Overall = average of period returns (equal-weight)
        if period_returns:
            avg_ret = np.mean(period_returns)
            row += f" {avg_ret:>+15.2%}"
        else:
            row += f" {'N/A':>16}"
        print(row)

    print("-" * 120)

    # Sharpe row
    for model_name in MODELS:
        row = f"{'  Sharpe':<12}"
        sharpes = []
        for pname in PERIODS:
            m = all_results[model_name].get(pname)
            if m is None:
                row += f" {'N/A':>16}"
            else:
                row += f" {m['sharpe']:>15.3f}"
                sharpes.append(m["sharpe"])
        if sharpes:
            row += f" {np.mean(sharpes):>15.3f}"
        else:
            row += f" {'N/A':>16}"
        print(row)

    # MaxDD row
    for model_name in MODELS:
        row = f"{'  MaxDD':<12}"
        dds = []
        for pname in PERIODS:
            m = all_results[model_name].get(pname)
            if m is None:
                row += f" {'N/A':>16}"
            else:
                row += f" {m['max_drawdown']:>15.2%}"
                dds.append(m["max_drawdown"])
        if dds:
            row += f" {np.mean(dds):>15.2%}"
        else:
            row += f" {'N/A':>16}"
        print(row)

    # Trades row
    for model_name in MODELS:
        row = f"{'  Trades':<12}"
        total_t = 0
        for pname in PERIODS:
            m = all_results[model_name].get(pname)
            if m is None:
                row += f" {'N/A':>16}"
            else:
                row += f" {m['n_trades']:>15,}"
                total_t += m["n_trades"]
        row += f" {total_t:>15,}"
        print(row)

    print("=" * 120)

    # ── Verdict ──────────────────────────────────────────────────────
    print("\n  VERDICT:")
    for model_name in MODELS:
        period_rets = []
        for pname in PERIODS:
            m = all_results[model_name].get(pname)
            if m is not None:
                period_rets.append(m["total_return"])
        if not period_rets:
            print(f"    {model_name}: no data")
            continue
        n_profitable = sum(1 for r in period_rets if r > 0)
        avg_ret = np.mean(period_rets)
        print(f"    {model_name}: {n_profitable}/{len(period_rets)} periods profitable, "
              f"avg return {avg_ret:+.2%}")

    # Save results
    rows = []
    for model_name in MODELS:
        for pname, m in all_results[model_name].items():
            if m is not None:
                rows.append({
                    "model": model_name, "period": pname,
                    "return": m["total_return"], "sharpe": m["sharpe"],
                    "max_drawdown": m["max_drawdown"], "n_trades": m["n_trades"],
                    "win_rate": m["win_rate"], "cost_pct": m["cost_pct_gross_pnl"],
                })
    if rows:
        df = pd.DataFrame(rows)
        suffix = ""
        if stop_loss_pct > 0 or take_profit_pct > 0:
            suffix = f"_sl{stop_loss_pct:.2f}_tp{take_profit_pct:.2f}"
        out = REPORTS_DIR / f"walk_forward_results{suffix}.csv"
        df.to_csv(out, index=False)
        print(f"\n  Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-loss", type=float, default=0.0, help="Stop-loss fraction (e.g. 0.05 for 5%%)")
    parser.add_argument("--take-profit", type=float, default=0.0, help="Take-profit fraction (e.g. 0.10 for 10%%)")
    args = parser.parse_args()
    run_walk_forward(stop_loss_pct=args.stop_loss, take_profit_pct=args.take_profit)
