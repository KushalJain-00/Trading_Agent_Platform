"""Portfolio & Delta analytics.

Per-model metrics: total return, CAGR, Sharpe, Max Drawdown, win rate,
avg win/loss, profit factor, Delta (net directional exposure), trade count,
total costs, cost as % of gross P&L.

Works for both historical backtest results and live paper trading state.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json


def compute_metrics(equity_curve_df, trade_log_df, risk_free_rate=0.0,
                     trading_days_per_year=252, bars_per_day=375):
    """Compute all performance metrics from equity curve and trade log.

    Args:
        equity_curve_df: DataFrame with [timestamp, equity] columns
        trade_log_df: DataFrame with [entry_time, exit_time, ticker, direction,
                       entry_price, exit_price, size, gross_pnl, net_pnl, costs,
                       holding_bars] columns
        risk_free_rate: annual risk-free rate (default 0%)
        trading_days_per_year: NSE ~252
        bars_per_day: NSE ~375 minutes per day

    Returns:
        dict of all metrics
    """
    if equity_curve_df is None or len(equity_curve_df) < 2:
        return _empty_metrics()

    eq = equity_curve_df.copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"])
    eq = eq.sort_values("timestamp").reset_index(drop=True)

    initial_equity = eq["equity"].iloc[0]
    final_equity = eq["equity"].iloc[-1]
    total_return = (final_equity / initial_equity) - 1.0

    # Compute actual time span from timestamps for accurate annualization
    eq["timestamp"] = pd.to_datetime(eq["timestamp"])
    time_span_days = (eq["timestamp"].iloc[-1] - eq["timestamp"].iloc[0]).total_seconds() / 86400
    years = max(time_span_days / trading_days_per_year, 0.01)
    cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1.0

    # Sharpe: resample to daily equity for clean annualization
    eq_daily = eq.set_index("timestamp").resample("1D")["equity"].last().dropna()
    if len(eq_daily) < 2:
        sharpe = 0.0
    else:
        daily_ret = eq_daily.pct_change().dropna()
        if daily_ret.std() == 0:
            sharpe = 0.0
        else:
            rf_daily = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
            sharpe = (daily_ret.mean() - rf_daily) / daily_ret.std() * np.sqrt(trading_days_per_year)

    # Max drawdown
    cummax = eq["equity"].cummax()
    drawdown = (eq["equity"] - cummax) / cummax
    max_dd = drawdown.min()

    # Drawdown duration (in trading days)
    current_dd_duration = 0
    max_dd_duration_days = 0
    for i in range(len(drawdown)):
        if drawdown.iloc[i] < 0:
            current_dd_duration += 1
        else:
            max_dd_duration_days = max(max_dd_duration_days, current_dd_duration)
            current_dd_duration = 0
    max_dd_duration_days = max(max_dd_duration_days, current_dd_duration)

    # Trade metrics
    trades = trade_log_df if trade_log_df is not None and len(trade_log_df) > 0 else pd.DataFrame()
    n_trades = len(trades)

    if n_trades > 0:
        wins = trades[trades["net_pnl"] > 0]
        losses = trades[trades["net_pnl"] <= 0]
        win_rate = len(wins) / n_trades
        avg_win = wins["net_pnl"].mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses["net_pnl"].mean()) if len(losses) > 0 else 0.0
        profit_factor = (wins["net_pnl"].sum() / losses["net_pnl"].abs().sum()
                        if len(losses) > 0 and losses["net_pnl"].abs().sum() > 0 else float("inf"))
        avg_holding = trades["holding_bars"].mean()
        total_costs = trades["costs"].sum()
        total_gross_pnl = trades["gross_pnl"].sum()
        cost_pct = (total_costs / abs(total_gross_pnl) * 100) if total_gross_pnl != 0 else 0.0
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0.0
        avg_holding = 0.0
        total_costs = total_gross_pnl = cost_pct = 0.0

    # Delta: net directional exposure over time
    # For long-only: Delta = % of equity invested at any time
    if "n_positions" in eq.columns and len(eq) > 0:
        avg_exposure = eq["n_positions"].mean()
    else:
        avg_exposure = 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_dd_duration_days": max_dd_duration_days,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "delta_exposure": avg_exposure,
        "n_trades": n_trades,
        "total_costs": total_costs,
        "total_gross_pnl": total_gross_pnl,
        "cost_pct_gross_pnl": cost_pct,
        "avg_holding_bars": avg_holding,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "bars_processed": len(eq),
    }


def _empty_metrics():
    return {k: 0.0 for k in [
        "total_return", "cagr", "sharpe",         "max_drawdown", "max_dd_duration_days",
        "win_rate", "avg_win", "avg_loss", "profit_factor", "delta_exposure",
        "n_trades", "total_costs", "total_gross_pnl", "cost_pct_gross_pnl",
        "avg_holding_bars", "initial_equity", "final_equity", "bars_processed",
    ]}


def comparison_table(results_dict, risk_free_rate=0.0):
    """Build comparison table across models.

    Args:
        results_dict: {model_name: {"equity_curve": df, "trade_log": df}}

    Returns:
        DataFrame with models as rows, metrics as columns
    """
    rows = []
    for name, data in results_dict.items():
        metrics = compute_metrics(data["equity_curve"], data["trade_log"], risk_free_rate)
        metrics["model"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("model")
    return df


def save_comparison_csv(results_dict, output_path, risk_free_rate=0.0):
    """Save comparison table to CSV."""
    df = comparison_table(results_dict, risk_free_rate)
    df.to_csv(output_path)
    print(f"Comparison table saved → {output_path}")
    return df


def print_comparison_table(results_dict, risk_free_rate=0.0):
    """Print formatted comparison table to terminal."""
    df = comparison_table(results_dict, risk_free_rate)

    fmt = {
        "total_return": "{:.2%}",
        "cagr": "{:.2%}",
        "sharpe": "{:.3f}",
        "max_drawdown": "{:.2%}",
        "win_rate": "{:.1%}",
        "profit_factor": "{:.2f}",
        "cost_pct_gross_pnl": "{:.2f}%",
        "n_trades": "{:d}",
        "avg_win": "{:,.0f}",
        "avg_loss": "{:,.0f}",
        "total_costs": "{:,.0f}",
    }

    print("\n" + "=" * 90)
    print(f"{'Model':<14} {'Return':>9} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>9} "
          f"{'WinRate':>8} {'PF':>7} {'Trades':>8} {'Cost%':>8}")
    print("-" * 90)
    for name, row in df.iterrows():
        print(f"{name:<14} {row['total_return']:9.2%} {row['cagr']:8.2%} "
              f"{row['sharpe']:8.3f} {row['max_drawdown']:9.2%} "
              f"{row['win_rate']:8.1%} {row['profit_factor']:7.2f} "
              f"{int(row['n_trades']):8d} {row['cost_pct_gross_pnl']:7.2f}%")
    print("=" * 90)
    return df
