"""Monte Carlo simulation — trade resampling + return bootstrapping.

Per model, using actual historical trade log (not re-running inference):
  - Trade-sequence resampling: randomly reshuffle realized trade P&Ls
  - Return bootstrapping: resample bar returns for CI on CAGR, Sharpe, MaxDD

Vectorized with numpy — fast and light.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


REPORTS_DIR = Path(__file__).parent / "reports"


def trade_resampling(trade_log_df, equity_curve_df, n_iterations=2000,
                     initial_capital=100_000_000):
    """Resample trade P&L sequence to generate distribution of equity outcomes."""
    if trade_log_df is None or len(trade_log_df) == 0:
        return None

    trade_pnls = trade_log_df["net_pnl"].values.astype(np.float64)
    n_trades = len(trade_pnls)

    # Use trade returns (pnl/capital) to avoid overflow
    trade_returns = trade_pnls / initial_capital

    final_equities = np.empty(n_iterations)
    max_drawdowns = np.empty(n_iterations)

    for i in range(n_iterations):
        perm = np.random.permutation(n_trades)
        resampled = trade_returns[perm]
        equity_path = initial_capital * np.cumprod(1 + resampled)
        equity_path = np.insert(equity_path, 0, initial_capital)
        final_equities[i] = equity_path[-1]
        cummax = np.maximum.accumulate(equity_path)
        dd = np.where(cummax > 0, (equity_path - cummax) / cummax, 0)
        max_drawdowns[i] = dd.min()

    percentiles = [5, 25, 50, 75, 95]
    eq_pcts = np.percentile(final_equities, percentiles)
    dd_pcts = np.percentile(max_drawdowns, percentiles)

    return {
        "final_equities": final_equities,
        "max_drawdowns": max_drawdowns,
        "percentiles": {p: {"equity": eq_pcts[i], "max_dd": dd_pcts[i]}
                        for i, p in enumerate(percentiles)},
        "n_iterations": n_iterations,
        "n_trades": n_trades,
    }


def return_bootstrapping(equity_curve_df, n_iterations=2000,
                          trading_days_per_year=252, bars_per_day=375):
    """Bootstrap bar returns for confidence intervals on CAGR, Sharpe, MaxDD."""
    if equity_curve_df is None or len(equity_curve_df) < 10:
        return None

    eq = equity_curve_df.copy().sort_values("timestamp")
    bar_returns = eq["equity"].pct_change().dropna().values.astype(np.float64)
    # Clip extreme returns to avoid overflow in cumprod
    bar_returns = np.clip(bar_returns, -0.5, 0.5)
    n_bars = len(bar_returns)

    cagrs = np.empty(n_iterations)
    sharpes = np.empty(n_iterations)
    max_dds = np.empty(n_iterations)

    for i in range(n_iterations):
        sample = np.random.choice(bar_returns, size=n_bars, replace=True)
        equity = np.cumprod(1 + sample)
        equity = np.insert(equity, 0, 1.0)

        years = max(n_bars / (trading_days_per_year * bars_per_day), 0.01)
        cagrs[i] = (equity[-1] ** (1 / years)) - 1

        std = sample.std()
        sharpes[i] = (sample.mean() / std * np.sqrt(trading_days_per_year * bars_per_day)) if std > 0 else 0

        cummax = np.maximum.accumulate(equity)
        dd = np.where(cummax > 0, (equity - cummax) / cummax, 0)
        max_dds[i] = dd.min()

    percentiles = [5, 25, 50, 75, 95]
    return {
        "cagr_ci": {p: float(np.percentile(cagrs, p)) for p in percentiles},
        "sharpe_ci": {p: float(np.percentile(sharpes, p)) for p in percentiles},
        "maxdd_ci": {p: float(np.percentile(max_dds, p)) for p in percentiles},
        "cagr_samples": cagrs,
        "sharpe_samples": sharpes,
        "maxdd_samples": max_dds,
        "n_iterations": n_iterations,
    }


def plot_trade_resampling(result, model_name, output_path=None):
    """Plot trade resampling distribution."""
    if result is None:
        return None
    output_path = output_path or REPORTS_DIR / f"monte_carlo_trade_{model_name}.html"

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Final Equity Distribution", "Max Drawdown Distribution"])

    fig.add_trace(go.Histogram(
        x=result["final_equities"], nbinsx=80, name="Equity",
        marker_color="#2196F3", opacity=0.7,
    ), row=1, col=1)

    pcts = result["percentiles"]
    for p, color in [(50, "#FFA726"), (5, "#EF5350"), (95, "#26A69A")]:
        fig.add_vline(x=pcts[p]["equity"], line_dash="dash", line_color=color,
                      annotation_text=f"P{p}: {pcts[p]['equity']:,.0f}", row=1, col=1)

    fig.add_trace(go.Histogram(
        x=result["max_drawdowns"] * 100, nbinsx=80, name="Max DD",
        marker_color="#EF5350", opacity=0.7,
    ), row=1, col=2)

    for p, color in [(50, "#FFA726"), (5, "#EF5350"), (95, "#26A69A")]:
        fig.add_vline(x=pcts[p]["max_dd"] * 100, line_dash="dash", line_color=color,
                      annotation_text=f"P{p}: {pcts[p]['max_dd']:.2f}%", row=1, col=2)

    fig.update_layout(
        title=f"Trade Resampling — {model_name.upper()} ({result['n_iterations']} iterations, {result['n_trades']} trades)",
        template="plotly_dark", height=400,
        margin=dict(l=60, r=30, t=60, b=40),
    )
    fig.write_html(str(output_path))
    return fig


def plot_return_bootstrapping(result, model_name, output_path=None):
    """Plot return bootstrapping CI distributions."""
    if result is None:
        return None
    output_path = output_path or REPORTS_DIR / f"monte_carlo_bootstrap_{model_name}.html"

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["CAGR Distribution", "Sharpe Distribution", "Max Drawdown Distribution"])

    fig.add_trace(go.Histogram(
        x=result["cagr_samples"] * 100, nbinsx=80, name="CAGR",
        marker_color="#2196F3", opacity=0.7,
    ), row=1, col=1)
    for p, color in [(50, "#FFA726"), (5, "#EF5350"), (95, "#26A69A")]:
        fig.add_vline(x=result["cagr_ci"][p] * 100, line_dash="dash", line_color=color,
                      annotation_text=f"P{p}: {result['cagr_ci'][p]*100:.2f}%", row=1, col=1)

    fig.add_trace(go.Histogram(
        x=result["sharpe_samples"], nbinsx=80, name="Sharpe",
        marker_color="#4CAF50", opacity=0.7,
    ), row=1, col=2)
    for p, color in [(50, "#FFA726"), (5, "#EF5350"), (95, "#26A69A")]:
        fig.add_vline(x=result["sharpe_ci"][p], line_dash="dash", line_color=color,
                      annotation_text=f"P{p}: {result['sharpe_ci'][p]:.3f}", row=1, col=2)

    fig.add_trace(go.Histogram(
        x=result["maxdd_samples"] * 100, nbinsx=80, name="Max DD",
        marker_color="#EF5350", opacity=0.7,
    ), row=1, col=3)
    for p, color in [(50, "#FFA726"), (5, "#EF5350"), (95, "#26A69A")]:
        fig.add_vline(x=result["maxdd_ci"][p] * 100, line_dash="dash", line_color=color,
                      annotation_text=f"P{p}: {result['maxdd_ci'][p]*100:.2f}%", row=1, col=3)

    fig.update_layout(
        title=f"Return Bootstrapping — {model_name.upper()} ({result['n_iterations']} iterations)",
        template="plotly_dark", height=400,
        margin=dict(l=60, r=30, t=60, b=40),
    )
    fig.write_html(str(output_path))
    return fig


def run_monte_carlo(trade_log_df, equity_curve_df, model_name,
                     n_iterations=2000, initial_capital=100_000_000):
    """Run full Monte Carlo for one model. Returns results dict."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    trade_result = trade_resampling(trade_log_df, equity_curve_df, n_iterations, initial_capital)
    bootstrap_result = return_bootstrapping(equity_curve_df, n_iterations)

    if trade_result:
        plot_trade_resampling(trade_result, model_name)
    if bootstrap_result:
        plot_return_bootstrapping(bootstrap_result, model_name)

    # Save percentile table
    if trade_result:
        pcts = trade_result["percentiles"]
        rows = []
        for p in sorted(pcts.keys()):
            rows.append({"percentile": p, "final_equity": pcts[p]["equity"],
                         "max_drawdown": pcts[p]["max_dd"]})
        pd.DataFrame(rows).to_csv(REPORTS_DIR / f"monte_carlo_trade_{model_name}.csv", index=False)

    if bootstrap_result:
        rows = []
        for p in sorted(bootstrap_result["cagr_ci"].keys()):
            rows.append({
                "percentile": p,
                "cagr": bootstrap_result["cagr_ci"][p],
                "sharpe": bootstrap_result["sharpe_ci"][p],
                "max_dd": bootstrap_result["maxdd_ci"][p],
            })
        pd.DataFrame(rows).to_csv(REPORTS_DIR / f"monte_carlo_bootstrap_{model_name}.csv", index=False)

    return {"trade_resampling": trade_result, "bootstrap": bootstrap_result}
