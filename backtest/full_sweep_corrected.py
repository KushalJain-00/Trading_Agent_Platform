"""Full corrected sweep — precomputed merge per model, shared across configs.

The merge (1M signals × 15.8M prices) takes ~6s and is identical for all
configs within a model. Precompute once, apply filters per config.
"""
import sys, time
from pathlib import Path
import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.analytics import compute_metrics

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SIGNALS_DIR = PROJECT_ROOT / "backtest" / "signals"
REPORTS_DIR = PROJECT_ROOT / "backtest" / "reports"
CKPT_DIR = PROJECT_ROOT / "models" / "checkpoints"

MODELS = ["lstm", "cnn1d", "cnn_lstm"]
CONFIDENCES = [0.65, 0.75, 0.80, 0.85, 0.90]
HOLDINGS = [10, 30, 50, 75]

CAPITAL = 100_000_000
POS_PCT = 0.02
COST_FRAC = 5 / 10000
HALF_SPREAD = 3 / 2 / 10000


def precompute_merged(model_name):
    """Load signals, merge with prices, return merged DataFrame (ticker-sorted)."""
    sig_path = SIGNALS_DIR / f"{model_name}_val_signals.parquet"
    if not sig_path.exists():
        print(f"  Generating signals for {model_name}...")
        from backtest.generate_signals import generate_historical_signals
        device = torch.device("cpu")
        generate_historical_signals(model_name, str(CKPT_DIR), str(DATA_DIR), device, str(SIGNALS_DIR), stride=15)

    print(f"  Loading signals: {model_name}")
    signals_df = pd.read_parquet(sig_path)
    print(f"    {len(signals_df):,} signals")

    sigs = signals_df[["ticker", "timestamp", "predicted_signal", "predicted_confidence"]].copy()
    sigs["timestamp"] = pd.to_datetime(sigs["timestamp"])

    px = pd.read_parquet(str(DATA_DIR / "val.parquet"), columns=["ticker", "timestamp", "close"])
    px["timestamp"] = pd.to_datetime(px["timestamp"])
    print(f"    {len(px):,} price bars")

    t0 = time.time()
    merged = pd.merge(sigs, px, on=["ticker", "timestamp"], how="inner", sort=False)
    merged = merged.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    print(f"    Merged: {len(merged):,} bars ({time.time()-t0:.1f}s)")
    return merged


def apply_filters_and_backtest(merged, confidence, min_hold):
    """Apply config filters, run equity loop, return metrics."""
    m = merged.copy(deep=False)  # shallow copy — we only modify position

    # Confidence filter
    if confidence > 0.0:
        weak = (m["predicted_signal"] == "Buy") & (m["predicted_confidence"] < confidence)
        m.loc[weak, "predicted_signal"] = "Hold"

    # Base position
    m["raw_pos"] = (m["predicted_signal"] == "Buy").astype(int)
    m["position"] = m.groupby("ticker")["raw_pos"].shift(1).fillna(0).astype(int)

    # Min-holding-bars
    if min_hold > 1:
        pos = m["position"].values.copy()
        tickers = m["ticker"].values
        entry_bar = {}
        in_trade = {}
        for i in range(len(pos)):
            t = tickers[i]
            if pos[i] == 1 and not in_trade.get(t, False):
                in_trade[t] = True
                entry_bar[t] = i
            elif pos[i] == 0 and in_trade.get(t, False):
                if i - entry_bar[t] < min_hold:
                    pos[i] = 1
                else:
                    in_trade[t] = False
        m["position"] = pos

    # Trade log
    prev_pos = m.groupby("ticker")["position"].shift(1).fillna(0)
    entries = m[(m["position"] == 1) & (prev_pos == 0)]
    exits = m[(m["position"] == 0) & (prev_pos == 1)]

    trades = []
    for ticker in m["ticker"].unique():
        te = entries[entries["ticker"] == ticker]
        tx = exits[exits["ticker"] == ticker]
        te_idx = te.index.values
        tx_idx = tx.index.values
        n = min(len(te_idx), len(tx_idx))
        for i in range(n):
            ei, xi = te_idx[i], tx_idx[i]
            ep = m.at[ei, "close"] * (1 + HALF_SPREAD)
            xp = m.at[xi, "close"] * (1 - HALF_SPREAD)
            sz = (CAPITAL * POS_PCT) / ep
            gpnl = (xp - ep) * sz
            tc = abs(xp * sz) * COST_FRAC + sz * xp * HALF_SPREAD
            trades.append({"net_pnl": gpnl - tc, "gross_pnl": gpnl, "costs": tc,
                           "entry_price": ep, "exit_price": xp, "size": sz,
                           "entry_time": str(m.at[ei, "timestamp"]),
                           "exit_time": str(m.at[xi, "timestamp"]),
                           "ticker": ticker, "direction": "long",
                           "holding_bars": int(xi - ei)})
        if len(te_idx) > len(tx_idx):
            ei = te_idx[len(tx_idx)]
            ep = m.at[ei, "close"] * (1 + HALF_SPREAD)
            sz = (CAPITAL * POS_PCT) / ep
            trades.append({"net_pnl": 0, "gross_pnl": 0, "costs": 0,
                           "entry_price": ep, "exit_price": m.at[ei, "close"], "size": sz,
                           "entry_time": str(m.at[ei, "timestamp"]),
                           "exit_time": str(m.index[-1]),
                           "ticker": ticker, "direction": "long", "holding_bars": 0})

    trade_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "ticker", "direction", "entry_price",
                 "exit_price", "size", "gross_pnl", "net_pnl", "costs", "holding_bars"])

    # Equity curve — array-indexed loop
    all_bars = m.sort_values("timestamp").reset_index(drop=True)
    n_bars = len(all_bars)
    bar_closes = all_bars["close"].values.astype(np.float64)
    bar_positions = all_bars["position"].values.astype(np.int64)
    bar_tickers = all_bars["ticker"].values
    bar_timestamps = all_bars["timestamp"].values
    bar_prev = all_bars.groupby("ticker")["position"].shift(1).fillna(0).values.astype(np.int64)

    unique_tickers = np.unique(bar_tickers)
    t2id = {t: i for i, t in enumerate(unique_tickers)}
    n_t = len(unique_tickers)
    bar_tid = np.array([t2id[t] for t in bar_tickers], dtype=np.int32)

    sh_arr = np.zeros(n_t, dtype=np.float64)
    ep_arr = np.zeros(n_t, dtype=np.float64)
    lp_arr = np.zeros(n_t, dtype=np.float64)
    cash = float(CAPITAL)
    mtm = 0.0
    eq_ts = np.empty(n_bars, dtype=bar_timestamps.dtype)
    eq_vals = np.empty(n_bars, dtype=np.float64)

    for idx in range(n_bars):
        tid = bar_tid[idx]
        c = bar_closes[idx]
        p = bar_positions[idx]
        pv = bar_prev[idx]

        if p == 1 and pv == 0:
            ep = c * (1 + HALF_SPREAD)
            sz = (CAPITAL * POS_PCT) / ep
            cash -= ep * sz
            ep_arr[tid] = ep
            sh_arr[tid] = sz
            mtm += sz * c
        elif p == 0 and pv == 1:
            sz = sh_arr[tid]
            xp = c * (1 - HALF_SPREAD)
            tc = abs(xp * sz) * COST_FRAC + sz * xp * HALF_SPREAD
            cash += xp * sz - tc
            mtm -= sz * lp_arr[tid]
            sh_arr[tid] = 0.0
            ep_arr[tid] = 0.0
        elif p == 1 and pv == 1:
            mtm += sh_arr[tid] * (c - lp_arr[tid])

        lp_arr[tid] = c
        eq_ts[idx] = bar_timestamps[idx]
        eq_vals[idx] = cash + mtm

    eq_df = pd.DataFrame({"timestamp": eq_ts, "equity": eq_vals})
    eq_df = eq_df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    # Validation
    closed_pnl = float(trade_df["net_pnl"].sum()) if len(trade_df) else 0.0
    open_unr = sum(sh_arr[i] * (lp_arr[i] - ep_arr[i]) for i in range(n_t) if sh_arr[i] > 0)
    expected = CAPITAL + closed_pnl + open_unr
    actual = float(eq_df["equity"].iloc[-1]) if len(eq_df) else CAPITAL
    assert abs(actual - expected) < 1.0, (
        f"BUG: equity {actual:,.2f} != expected {expected:,.2f} (diff {actual-expected:,.2f})")

    met = compute_metrics(eq_df, trade_df)
    return met, eq_df, trade_df


def main():
    rows = []
    total = len(MODELS) * len(CONFIDENCES) * len(HOLDINGS)
    run_num = 0

    for model in MODELS:
        t_model = time.time()
        merged = precompute_merged(model)
        print(f"  Model {model} prep: {time.time()-t_model:.1f}s")

        for conf in CONFIDENCES:
            for hold in HOLDINGS:
                run_num += 1
                t0 = time.time()
                try:
                    met, eq, trades = apply_filters_and_backtest(merged, conf, hold)
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
        print(f"  Model {model} total: {time.time()-t_model:.1f}s\n")

    df = pd.DataFrame(rows)
    out = REPORTS_DIR / "full_sweep_corrected.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}")

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
