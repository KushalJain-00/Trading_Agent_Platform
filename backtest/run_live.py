"""Live paper trading entry point.

Modes:
  --mode live    : real yfinance polling (default)
  --mode replay  : streams historical val bars at configurable speed

Run in tmux so it persists. Dashboard reads latest state from
backtest/live_state/.

PAPER TRADING ONLY — no real order execution.
"""
import argparse
import sys
import time
import json
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Live paper trading loop")
    parser.add_argument("--mode", choices=["live", "replay"], default="live")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--replay-speed", type=float, default=2.0)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--capital", type=float, default=100_000_000)
    parser.add_argument("--position-size", type=float, default=0.02)
    parser.add_argument("--cost-bps", type=float, default=5)
    parser.add_argument("--spread-bps", type=float, default=3)
    parser.add_argument("--latency", type=int, default=1)
    parser.add_argument("--models", nargs="*", default=["lstm", "cnn1d", "cnn_lstm"])
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "models" / "checkpoints"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    state_dir = PROJECT_ROOT / "backtest" / "live_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from backtest.generate_signals import load_model, generate_live_signal, LABEL_MAP
    from backtest.simulator import LiveSimulator
    from backtest.live_data import (
        fetch_intraday, is_market_open, market_status_str, fetch_live_quote,
        ReplayDataStreamer, DATA_DIR,
    )
    from train import load_meta

    meta = load_meta(str(data_dir))
    feature_cols = meta["feature_cols"]
    ns = np.load(data_dir / "train_npy" / "norm_stats.npz")
    norm_mean, norm_std = ns["mean"], ns["std"]

    models = {}
    for name in args.models:
        try:
            model, input_dim, window_size = load_model(name, str(ckpt_dir), device)
            models[name] = model
            print(f"  Loaded {name} (input={input_dim}, window={window_size})")
        except Exception as e:
            print(f"  WARNING: Could not load {name}: {e}")

    if not models:
        print("ERROR: No models loaded")
        sys.exit(1)

    sims = {name: LiveSimulator(
        capital=args.capital, position_size_pct=args.position_size,
        cost_bps=args.cost_bps, spread_bps=args.spread_bps,
        latency_bars=args.latency,
    ) for name in models}

    window_buffers = {name: deque(maxlen=window_size) for name in models}
    ticker_bars = {name: [] for name in models}
    save_counter = {name: 0 for name in models}

    def process_bar(bar, source="replay"):
        ticker = bar["ticker"]

        for name, model in models.items():
            # Get features: pre-computed _z columns from replay, or compute from raw for live
            if all(fc in bar for fc in feature_cols):
                feats = np.array([bar[fc] for fc in feature_cols], dtype=np.float32)
            else:
                # Live mode: need to compute from raw OHLCV (skip until enough history)
                ticker_bars[name].append(bar)
                if len(ticker_bars[name]) < args.window_size:
                    continue
                recent = pd.DataFrame(ticker_bars[name][-args.window_size:])
                from data_pipeline.build_features import build_ticker_features
                try:
                    feat_df, fcols = build_ticker_features(recent)
                    if len(feat_df) == 0:
                        continue
                    last_row = feat_df.iloc[-1]
                    z_cols_build = [f"{c}_z" for c in fcols]  # build_ticker_features returns raw col names
                    feats = last_row[z_cols_build].values.astype(np.float32)
                except Exception:
                    continue

            window_buffers[name].append(feats)
            if len(window_buffers[name]) < window_size:
                continue

            window_array = np.array(list(window_buffers[name])[-window_size:])
            sig = generate_live_signal(
                model, pd.DataFrame(window_array, columns=feature_cols),
                norm_mean, norm_std, device,
            )

            sim = sims[name]
            sim.process_bar(bar, sig["signal"])

            save_counter[name] += 1
            if save_counter[name] % 50 == 0:
                _save_state(state_dir, sims)

        with open(state_dir / "latest_bar.json", "w") as f:
            json.dump({
                "timestamp": str(bar["timestamp"]),
                "ticker": bar["ticker"],
                "close": bar["close"],
                "source": source,
                "market_status": market_status_str() if source == "live" else "replay",
            }, f, indent=2)

    # Main loop
    print(f"\nMode: {args.mode.upper()}")

    if args.mode == "replay":
        print(f"Replay speed: {args.replay_speed}s per bar")
        streamer = ReplayDataStreamer(
            tickers=args.tickers, speed=args.replay_speed, data_dir=data_dir,
        )
        print(f"Total bars: {len(streamer):,}")
        print("Starting...\n")

        try:
            for i, bar in enumerate(streamer):
                process_bar(bar, source="replay")
                if i % 5000 == 0:
                    print(f"  [{i}/{len(streamer)}] {bar['ticker']} @ {bar['close']:.2f} "
                          f"({streamer.progress:.1%})", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted.")

        _save_state(state_dir, sims)
        print("\nReplay complete.")

    else:
        print(f"Poll interval: {args.poll_interval}s")
        tickers = args.tickers or ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
        print(f"Tickers: {tickers}")

        while True:
            status = market_status_str()
            print(f"\n[{time.strftime('%H:%M:%S')}] {status}")

            if not is_market_open():
                for ticker in tickers:
                    try:
                        quote = fetch_live_quote(ticker)
                        print(f"  {ticker}: ₹{quote['last_price']:,.2f} — {quote['status']}")
                    except Exception as e:
                        print(f"  {ticker}: {e}")
                print(f"  Sleeping {args.poll_interval}s...")
                time.sleep(args.poll_interval)
                continue

            for ticker in tickers:
                try:
                    df = fetch_intraday(ticker, interval="1m", period="1d")
                    if len(df) > 0:
                        bar = df.iloc[-1].to_dict()
                        bar["timestamp"] = str(bar["timestamp"])
                        process_bar(bar, source="live")
                except Exception as e:
                    print(f"  {ticker} error: {e}")

            _save_state(state_dir, sims)
            print(f"  Sleeping {args.poll_interval}s...")
            time.sleep(args.poll_interval)


def _save_state(state_dir, sims):
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, sim in sims.items():
        model_dir = state_dir / name
        model_dir.mkdir(exist_ok=True)
        eq_df = sim.get_equity_curve_df()
        if len(eq_df):
            eq_df.to_parquet(model_dir / "equity_curve.parquet", index=False)
        td = sim.get_trade_log_df()
        if len(td):
            td.to_parquet(model_dir / "trade_log.parquet", index=False)
        open_pos = []
        for ticker, pos in sim.open_positions.items():
            open_pos.append({
                "ticker": ticker, "entry_price": pos["entry_price"],
                "size": pos["size"], "entry_time": pos["entry_time"],
                "holding_bars": pos["holding_bars"],
            })
        with open(model_dir / "open_positions.json", "w") as f:
            json.dump(open_pos, f, indent=2)
        summary[name] = {
            "equity": sim.equity, "cash": sim.cash,
            "n_positions": len(sim.open_positions),
            "total_trades": len(sim.trade_log),
            "bars_processed": sim.bars_processed,
        }
    with open(state_dir / "portfolio_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
