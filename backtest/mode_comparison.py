"""Full comparison: B&H baseline, single-stock, equal-weight, confidence-weighted.

Uses the existing simulator. No reimplemented logic.
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

CAPITAL = 100_000_000
CONFIDENCE = 0.90
HOLDING = 75
POSITION_SIZE = 0.02
MAX_POSITIONS = 20
MAX_POS_PCT = 0.10
MODELS = ["lstm", "cnn1d", "cnn_lstm"]


def bh_baseline(prices_df, capital=CAPITAL):
    """Equal-weight B&H. Average per-ticker daily returns, compounded."""
    prices_df = prices_df.copy()
    prices_df["date"] = pd.to_datetime(prices_df["timestamp"]).dt.date

    daily = prices_df.groupby(["ticker", "date"])["close"].last().reset_index()
    daily = daily.sort_values(["ticker", "date"])
    daily["ret"] = daily.groupby("ticker")["close"].pct_change()
    daily = daily.dropna(subset=["ret"])

    # Equal-weight: average of all tickers' daily returns
    port_ret = daily.groupby("date")["ret"].mean()

    eq = capital * (1 + port_ret).cumprod()
    eq.iloc[0] = capital

    eq_df = pd.DataFrame({"timestamp": pd.to_datetime(eq.index), "equity": eq.values})
    return compute_metrics(eq_df, pd.DataFrame(columns=["net_pnl"]))


def single_stock(model_name, ticker, prices_df, conf=CONFIDENCE, hold=HOLDING):
    """Single-ticker backtest via existing simulator."""
    sigs = pd.read_parquet(SIGNALS_DIR / f"{model_name}_val_signals.parquet")
    sig_t = sigs[sigs["ticker"] == ticker]
    px_t = prices_df[prices_df["ticker"] == ticker]
    if len(sig_t) < 100 or len(px_t) < 100:
        return None
    result = _run_backtest_core(
        sig_t, px_t, CAPITAL, POSITION_SIZE, 5, 3, 1, None, conf, hold,
    )
    return compute_metrics(result.get_equity_curve_df(), result.get_trade_log_df())


def portfolio_run(model_name, prices_df, allocation, conf=CONFIDENCE, hold=HOLDING,
                  max_pos=MAX_POSITIONS, max_pct=MAX_POS_PCT):
    """Portfolio backtest via existing simulator with new allocation params."""
    sigs = pd.read_parquet(SIGNALS_DIR / f"{model_name}_val_signals.parquet")
    result = _run_backtest_core(
        sigs, prices_df, CAPITAL, POSITION_SIZE, 5, 3, 1, None,
        conf, hold,
        max_positions=max_pos, max_position_pct=max_pct, allocation=allocation,
    )
    return compute_metrics(result.get_equity_curve_df(), result.get_trade_log_df())


def main():
    print("Loading prices...")
    prices_full = pd.read_parquet(
        str(DATA_DIR / "val.parquet"),
        columns=["ticker", "timestamp", "open", "high", "low", "close", "volume"],
    )
    print(f"  {len(prices_full):,} bars, {prices_full['ticker'].nunique()} tickers\n")

    results = []

    # ── 1. B&H ────────────────────────────────────────────────────
    print("=" * 70)
    print("  BUY-AND-HOLD BASELINE")
    print("=" * 70)
    bh = bh_baseline(prices_full)
    print(f"  Return: {bh['total_return']*100:+.2f}%  Sharpe: {bh['sharpe']:.3f}  "
          f"MaxDD: {bh['max_drawdown']*100:.2f}%  WinRate: {bh['win_rate']:.1%}")
    results.append({"mode": "B&H", "model": "-", "config": "equal-weight", **bh})

    # ── 2. Equal-weight portfolio (current behavior) ───────────────
    print("\n" + "=" * 70)
    print("  EQUAL-WEIGHT PORTFOLIO (conf=0.90/hold=75)")
    print("=" * 70)
    for m in MODELS:
        r = portfolio_run(m, prices_full, "equal")
        print(f"  {m:<10} Return: {r['total_return']*100:+.2f}%  Sharpe: {r['sharpe']:.3f}  "
              f"MaxDD: {r['max_drawdown']*100:.2f}%  Trades: {r['n_trades']:,}")
        results.append({"mode": "equal-weight", "model": m,
                         "config": f"conf={CONFIDENCE}/hold={HOLDING}", **r})

    # ── 3. Confidence-weighted ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CONFIDENCE-WEIGHTED (conf=0.90/hold=75/max_pos=20)")
    print("=" * 70)
    for m in MODELS:
        r = portfolio_run(m, prices_full, "confidence-weighted")
        print(f"  {m:<10} Return: {r['total_return']*100:+.2f}%  Sharpe: {r['sharpe']:.3f}  "
              f"MaxDD: {r['max_drawdown']*100:.2f}%  Trades: {r['n_trades']:,}")
        results.append({"mode": "conf-weighted", "model": m,
                         "config": f"conf={CONFIDENCE}/hold={HOLDING}/top{MAX_POSITIONS}", **r})

    # ── 4. Top-N allocation ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TOP-N ALLOCATION (conf=0.90/hold=75/max_pos=20)")
    print("=" * 70)
    for m in MODELS:
        r = portfolio_run(m, prices_full, "top-N")
        print(f"  {m:<10} Return: {r['total_return']*100:+.2f}%  Sharpe: {r['sharpe']:.3f}  "
              f"MaxDD: {r['max_drawdown']*100:.2f}%  Trades: {r['n_trades']:,}")
        results.append({"mode": "top-N", "model": m,
                         "config": f"conf={CONFIDENCE}/hold={HOLDING}/top{MAX_POSITIONS}", **r})

    # ── 5. Single-stock model vs B&H ───────────────────────────────
    print("\n" + "=" * 70)
    print("  SINGLE-STOCK: LSTM MODEL vs BUY-AND-HOLD")
    print("=" * 70)
    top = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
           "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT"]
    for t in top:
        m = single_stock("lstm", t, prices_full)
        px = prices_full[prices_full["ticker"] == t].sort_values("timestamp")
        bh_ret = (px["close"].iloc[-1] / px["close"].iloc[0]) - 1 if len(px) > 1 else 0
        status = f"model={m['total_return']*100:+.1f}%  B&H={bh_ret*100:+.1f}%" if m else "insufficient data"
        if m:
            alpha = m['total_return'] - bh_ret
            status += f"  alpha={alpha*100:+.1f}pp"
        print(f"  {t:<12} {status}")
        if m:
            results.append({"mode": "single-model", "model": "lstm",
                             "config": t, **m})
            results.append({"mode": "single-BH", "model": "-",
                             "config": t, "total_return": bh_ret,
                             "sharpe": 0, "max_drawdown": 0, "n_trades": 0,
                             "win_rate": 0, "cagr": 0, "pf": 0,
                             "cost_pct_gross_pnl": 0})

    # ── Save ───────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    out = REPORTS_DIR / "mode_comparison.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print(f"{'Mode':<18} {'Model':<10} {'Config':<28} {'Return':>9} {'Sharpe':>8} {'MaxDD':>9} {'Trades':>8}")
    print("-" * 100)
    for _, r in df.iterrows():
        print(f"{r['mode']:<18} {str(r.get('model','')):<10} {str(r.get('config','')):<28} "
              f"{r['total_return']*100:>+8.2f}% {r['sharpe']:>8.3f} "
              f"{r['max_drawdown']*100:>8.2f}% {int(r.get('n_trades',0)):>8,}")
    print("=" * 100)


if __name__ == "__main__":
    main()
