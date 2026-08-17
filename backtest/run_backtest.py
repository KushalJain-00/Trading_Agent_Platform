"""CLI entry-point: run full historical backtest for all 3 models.

Steps:
  1. Generate signals (or load cached)
  2. Run simulator per model
  3. Compute analytics
  4. Generate static visualizations
  5. Run Monte Carlo

All outputs saved to backtest/signals/, backtest/results/, backtest/reports/.
"""
import argparse
import sys
import time
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run historical backtest for all models")
    parser.add_argument("--mode", choices=["portfolio", "single"], default="portfolio",
                        help="portfolio=multi-stock, single=one ticker")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Single ticker symbol (required for single mode)")
    parser.add_argument("--allocation", choices=["equal", "confidence-weighted", "top-N"],
                        default="equal", help="Position sizing mode")
    parser.add_argument("--max-positions", type=int, default=0,
                        help="Max concurrent positions (0=unlimited)")
    parser.add_argument("--max-position-pct", type=float, default=0.0,
                        help="Max fraction of capital per position (0=unlimited)")
    parser.add_argument("--capital", type=float, default=100_000_000)
    parser.add_argument("--position-size", type=float, default=0.02)
    parser.add_argument("--cost-bps", type=float, default=5)
    parser.add_argument("--spread-bps", type=float, default=3)
    parser.add_argument("--latency", type=int, default=1)
    parser.add_argument("--mc-iterations", type=int, default=2000)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--confidence-threshold", type=float, default=0.65,
                        help="Min confidence to act on Buy signal (0=off)")
    parser.add_argument("--min-holding-bars", type=int, default=10,
                        help="Min bars to hold position before allowing exit (1=off)")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "models" / "checkpoints"))
    args = parser.parse_args()

    if args.mode == "single" and not args.ticker:
        parser.error("--ticker is required for single mode")

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}" + (f" ({args.ticker})" if args.ticker else ""))
    print(f"Allocation: {args.allocation}")

    from backtest.generate_signals import generate_historical_signals, LABEL_MAP
    from backtest.simulator import run_historical_backtest
    from backtest.analytics import comparison_table, print_comparison_table, save_comparison_csv
    from backtest.visualize import generate_all_static_charts
    from backtest.monte_carlo import run_monte_carlo

    models = ["lstm", "cnn1d", "cnn_lstm"]
    signals_dir = PROJECT_ROOT / "backtest" / "signals"
    results_dir = PROJECT_ROOT / "backtest" / "results"
    reports_dir = PROJECT_ROOT / "backtest" / "reports"
    signals_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading validation prices...")
    prices_df = pd.read_parquet(str(data_dir / "val.parquet"),
                                 columns=["ticker", "timestamp", "open", "high", "low", "close", "volume"])
    if args.ticker:
        prices_df = prices_df[prices_df["ticker"] == args.ticker]
        print(f"  Filtered to {args.ticker}: {len(prices_df):,} bars")
    else:
        print(f"  {len(prices_df):,} bars")

    results_dict = {}
    signals_dict = {}
    mc_results = {}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  {model_name.upper()}")
        print(f"{'='*60}")

        t0 = time.time()
        sig_path = signals_dir / f"{model_name}_val_signals.parquet"
        if sig_path.exists():
            print(f"  Loading cached signals → {sig_path}")
            signals_df = pd.read_parquet(sig_path)
        else:
            print(f"  Generating signals...")
            signals_df = generate_historical_signals(
                model_name, str(ckpt_dir), str(data_dir), device,
                str(signals_dir), args.batch_size, args.stride
            )
        if args.ticker:
            signals_df = signals_df[signals_df["ticker"] == args.ticker]
        print(f"  Signals: {len(signals_df):,} ({time.time()-t0:.1f}s)")

        signals_dict[model_name] = signals_df

        t0 = time.time()
        model_results_dir = results_dir / model_name
        sim = run_historical_backtest(
            signals_df, prices_df,
            capital=args.capital, position_size_pct=args.position_size,
            cost_bps=args.cost_bps, spread_bps=args.spread_bps,
            latency_bars=args.latency, output_dir=str(model_results_dir),
            confidence_threshold=args.confidence_threshold,
            min_holding_bars=args.min_holding_bars,
            max_positions=args.max_positions,
            max_position_pct=args.max_position_pct,
            allocation=args.allocation,
        )
        eq_df = sim.get_equity_curve_df()
        trades_df = sim.get_trade_log_df()
        print(f"  Simulator: {len(eq_df):,} bars, {len(trades_df)} trades ({time.time()-t0:.1f}s)")
        results_dict[model_name] = {"equity_curve": eq_df, "trade_log": trades_df, "sim": sim}

        t0 = time.time()
        mc_results[model_name] = run_monte_carlo(
            trades_df, eq_df, model_name,
            n_iterations=args.mc_iterations, initial_capital=args.capital,
        )
        print(f"  Monte Carlo: {time.time()-t0:.1f}s")

    print("\n\n")
    metrics_df = print_comparison_table(results_dict, args.risk_free_rate)
    save_comparison_csv(results_dict, str(reports_dir / "comparison_table.csv"), args.risk_free_rate)

    if args.confidence_threshold > 0 or args.min_holding_bars > 1:
        print("\n" + "=" * 90)
        print("  FILTER IMPACT: Before vs After")
        print(f"  confidence_threshold={args.confidence_threshold}  min_holding_bars={args.min_holding_bars}")
        print("=" * 90)
        print(f"{'Model':<14} {'Trades Before':>14} {'Trades After':>13} {'Reduction':>10} "
              f"{'Cost% Before':>13} {'Cost% After':>12}")
        print("-" * 90)
        for model_name in models:
            fs = results_dict[model_name]["sim"].get_filter_stats()
            if fs:
                print(f"{model_name:<14} {fs['n_trades_before']:>14,} {fs['n_trades_after']:>13,} "
                      f"{fs['trades_reduction_pct']:>9.1f}% "
                      f"{fs['cost_pct_before']:>12.1f}% {fs['cost_pct_after']:>11.1f}%")
        print("=" * 90)

    print("\nGenerating static charts...")
    generate_all_static_charts(results_dict, signals_dict, prices_df)

    print("\n" + "=" * 60)
    print("  BACKTEST COMPLETE")
    print("=" * 60)
    print(f"  Signals: {signals_dir}/")
    print(f"  Results: {results_dir}/")
    print(f"  Reports: {reports_dir}/")
    print(f"\n  Launch dashboard: streamlit run backtest/dashboard.py")
    print(f"  Launch live paper: python backtest/run_live.py --mode replay")


if __name__ == "__main__":
    main()
