"""Monte Carlo validation for CNN1D at 0.65/10 config. 5000 iterations."""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TRADE_LOG = PROJECT_ROOT / "backtest" / "results" / "cnn1d" / "trade_log.parquet"
REPORTS_DIR = PROJECT_ROOT / "backtest" / "reports"
INITIAL_CAPITAL = 100_000_000
N_ITER = 5000

trades = pd.read_parquet(TRADE_LOG)
trade_pnls = trades["net_pnl"].values.astype(np.float64)
n_trades = len(trade_pnls)
observed_total_pnl = trade_pnls.sum()
observed_return_pct = (observed_total_pnl / INITIAL_CAPITAL) * 100

print(f"CNN1D trade log: {n_trades:,} trades")
print(f"Observed total net P&L: ₹{observed_total_pnl:,.0f} ({observed_return_pct:+.2f}%)")
print(f"Win rate: {(trade_pnls > 0).mean():.1%}")
print(f"Avg win: ₹{trade_pnls[trade_pnls > 0].mean():,.0f}  |  Avg loss: ₹{abs(trade_pnls[trade_pnls <= 0].mean()):,.0f}")
print(f"Running {N_ITER:,} Monte Carlo iterations...\n")

# ── 1. Trade-sequence resampling ───────────────────────────────────
print("=" * 70)
print("  1. TRADE-SEQUENCE RESAMPLING (5,000 iterations)")
print("=" * 70)

trade_returns = trade_pnls / INITIAL_CAPITAL
t0 = time.time()

final_equities = np.empty(N_ITER)
max_drawdowns = np.empty(N_ITER)
final_return_pcts = np.empty(N_ITER)

rng = np.random.default_rng(42)
for i in range(N_ITER):
    perm = rng.permutation(n_trades)
    resampled = trade_returns[perm]
    equity_path = INITIAL_CAPITAL * np.cumprod(1 + resampled)
    equity_path = np.insert(equity_path, 0, float(INITIAL_CAPITAL))
    final_equities[i] = equity_path[-1]
    final_return_pcts[i] = (equity_path[-1] / INITIAL_CAPITAL - 1) * 100
    cummax = np.maximum.accumulate(equity_path)
    dd = np.where(cummax > 0, (equity_path - cummax) / cummax, 0)
    max_drawdowns[i] = dd.min() * 100

elapsed = time.time() - t0
pcts = [5, 25, 50, 75, 95]
print(f"\n  Completed in {elapsed:.1f}s\n")

print(f"  {'Percentile':>12}  {'Final Return%':>15}  {'Final Equity':>18}")
print(f"  {'-'*12}  {'-'*15}  {'-'*18}")
for p in pcts:
    ret = np.percentile(final_return_pcts, p)
    eq = np.percentile(final_equities, p)
    print(f"  {'P'+str(p):>12}  {ret:>+14.2f}%  ₹{eq:>16,.0f}")

print(f"\n  {'Percentile':>12}  {'Max Drawdown%':>15}")
print(f"  {'-'*12}  {'-'*15}")
for p in pcts:
    dd = np.percentile(max_drawdowns, p)
    print(f"  {'P'+str(p):>12}  {dd:>14.2f}%")

pct_positive = (final_return_pcts > 0).mean() * 100
pct_negative = (final_return_pcts <= 0).mean() * 100
print(f"\n  ► {pct_positive:.1f}% of {N_ITER:,} paths ended with POSITIVE return")
print(f"  ► {pct_negative:.1f}% ended with NEGATIVE return")
print(f"  ► Observed actual: {observed_return_pct:+.2f}%")

# ── 2. Return bootstrapping ───────────────────────────────────────
print(f"\n\n{'=' * 70}")
print("  2. RETURN BOOTSTRAPPING (5,000 iterations)")
print("=" * 70)

# Build per-bar returns from trade log
# Group trades by exit time to get daily returns
trades_sorted = trades.copy()
trades_sorted["exit_date"] = pd.to_datetime(trades_sorted["exit_time"]).dt.date
daily_pnl = trades_sorted.groupby("exit_date")["net_pnl"].sum()
daily_returns = (daily_pnl / INITIAL_CAPITAL).values.astype(np.float64)
n_days = len(daily_returns)
trading_days_per_year = 252

print(f"  Trading days in trade log: {n_days}")

t0 = time.time()
cagrs = np.empty(N_ITER)
sharpes = np.empty(N_ITER)
boot_max_dds = np.empty(N_ITER)

for i in range(N_ITER):
    sample = rng.choice(daily_returns, size=n_days, replace=True)
    equity = INITIAL_CAPITAL * np.cumprod(1 + sample)
    equity = np.insert(equity, 0, float(INITIAL_CAPITAL))

    years = max(n_days / trading_days_per_year, 0.01)
    total_ret = equity[-1] / INITIAL_CAPITAL - 1
    cagrs[i] = ((1 + total_ret) ** (1 / years) - 1) * 100

    std = sample.std()
    sharpes[i] = (sample.mean() / std * np.sqrt(trading_days_per_year)) if std > 0 else 0

    cummax = np.maximum.accumulate(equity)
    dd = np.where(cummax > 0, (equity - cummax) / cummax, 0)
    boot_max_dds[i] = dd.min() * 100

elapsed = time.time() - t0
print(f"  Completed in {elapsed:.1f}s\n")

print(f"  {'Percentile':>12}  {'CAGR%':>10}  {'Sharpe':>10}  {'Max DD%':>10}")
print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
for p in pcts:
    c = np.percentile(cagrs, p)
    s = np.percentile(sharpes, p)
    d = np.percentile(boot_max_dds, p)
    print(f"  {'P'+str(p):>12}  {c:>+9.2f}%  {s:>10.3f}  {d:>9.2f}%")

pct_sharpe_positive = (sharpes > 0).mean() * 100
print(f"\n  ► {pct_sharpe_positive:.1f}% of paths had Sharpe Ratio > 0")

# ── 3. Distribution plot ──────────────────────────────────────────
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["Final Return Distribution (Trade Resampling)",
                                    "Sharpe Ratio Distribution (Return Bootstrapping)"])

fig.add_trace(go.Histogram(
    x=final_return_pcts, nbinsx=100, name="Return%",
    marker_color="#58a6ff", opacity=0.7,
), row=1, col=1)
fig.add_vline(x=observed_return_pct, line_dash="dash", line_color="#d29922", line_width=3,
              annotation_text=f"Observed: {observed_return_pct:+.2f}%",
              annotation_font_color="#d29922", row=1, col=1)
fig.add_vline(x=0, line_dash="dot", line_color="#8b949e", row=1, col=1)
for p, color in [(50, "#3fb950"), (5, "#f85149"), (95, "#3fb950")]:
    fig.add_vline(x=np.percentile(final_return_pcts, p), line_dash="dash",
                  line_color=color, annotation_text=f"P{p}", row=1, col=1)

fig.add_trace(go.Histogram(
    x=sharpes, nbinsx=100, name="Sharpe",
    marker_color="#3fb950", opacity=0.7,
), row=1, col=2)
fig.add_vline(x=0, line_dash="dot", line_color="#8b949e", row=1, col=2)
for p, color in [(50, "#d29922"), (5, "#f85149"), (95, "#3fb950")]:
    fig.add_vline(x=np.percentile(sharpes, p), line_dash="dash",
                  line_color=color, annotation_text=f"P{p}", row=1, col=2)

fig.update_layout(
    title="CNN1D Monte Carlo Validation (0.65 confidence / 10-bar hold) — 5,000 iterations",
    template="plotly_dark", height=450,
    margin=dict(l=60, r=30, t=60, b=40),
    showlegend=False,
)
fig.update_xaxes(title_text="Final Return (%)", row=1, col=1)
fig.update_xaxes(title_text="Sharpe Ratio", row=1, col=2)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
fig.write_html(str(REPORTS_DIR / "mc_cnn1d_065_10_validation.html"))
print(f"\n  Plot saved → {REPORTS_DIR / 'mc_cnn1d_065_10_validation.html'}")

# ── 4. Verdict ────────────────────────────────────────────────────
print(f"\n\n{'=' * 70}")
print("  VERDICT")
print("=" * 70)
print(f"""
  Config: CNN1D, confidence≥0.65, min_holding=10 bars
  Observed result: {observed_return_pct:+.2f}% total return, {n_trades:,} trades

  Trade-sequence resampling ({N_ITER:,} iterations):
    • {pct_positive:.1f}% of simulated trade orderings produced a POSITIVE return
    • {pct_negative:.1f}% produced a NEGATIVE return
    • Median outcome: {np.percentile(final_return_pcts, 50):+.2f}%
    • Worst 5%: {np.percentile(final_return_pcts, 5):+.2f}%  |  Best 5%: {np.percentile(final_return_pcts, 95):+.2f}%

  Return bootstrapping ({N_ITER:,} iterations):
    • {pct_sharpe_positive:.1f}% of paths had Sharpe Ratio > 0
    • Median Sharpe: {np.percentile(sharpes, 50):.3f}
    • Median CAGR: {np.percentile(cagrs, 50):+.2f}%

  Bottom line:
""")
if pct_positive > 60:
    print(f"    ✅ ROBUST — {pct_positive:.0f}% of orderings are profitable.")
    print(f"       The +{observed_return_pct:.2f}% edge is NOT just lucky trade ordering.")
    print(f"       This config is suitable for live paper trading testing.")
elif pct_positive > 45:
    print(f"    ⚠️  MARGINAL — {pct_positive:.0f}% of orderings are profitable.")
    print(f"       The edge is weak and depends somewhat on trade ordering.")
    print(f"       Paper trading is OK for further validation, but monitor closely.")
else:
    print(f"    ❌ NOT ROBUST — only {pct_positive:.0f}% of orderings are profitable.")
    print(f"       The observed +{observed_return_pct:.2f}% was likely lucky trade ordering.")
    print(f"       Do NOT trust this edge in live trading without further investigation.")
print()
