"""Static visualization — equity curves, drawdowns, price charts, metrics.

Saves to backtest/reports/. Uses Plotly for all charts.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json


REPORTS_DIR = Path(__file__).parent / "reports"


def _ensure_reports_dir():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_equity_curves(results_dict, output_path=None):
    """Overlaid equity curves for all models. Downsampled for HTML size."""
    _ensure_reports_dir()
    output_path = output_path or REPORTS_DIR / "equity_curves.html"

    fig = go.Figure()
    colors = {"lstm": "#2196F3", "cnn1d": "#FF9800", "cnn_lstm": "#4CAF50", "baseline": "#9E9E9E"}

    for name, data in results_dict.items():
        eq = data.get("equity_curve", pd.DataFrame())
        if len(eq) == 0:
            continue
        eq = eq.copy()
        eq["timestamp"] = pd.to_datetime(eq["timestamp"])
        # Downsample: keep every 50th point for HTML size
        if len(eq) > 20000:
            eq = eq.iloc[::len(eq)//10000].copy()
        color = colors.get(name, "#666")
        fig.add_trace(go.Scattergl(
            x=eq["timestamp"], y=eq["equity"],
            name=name.upper(), line=dict(color=color, width=2),
            hovertemplate=f"<b>{name.upper()}</b><br>%{{x}}<br>Equity: %{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(
        title="Equity Curves — All Models",
        xaxis_title="Date", yaxis_title="Portfolio Equity (₹)",
        template="plotly_dark", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500, margin=dict(l=60, r=30, t=50, b=40),
    )
    fig.write_html(str(output_path))
    return fig


def plot_drawdowns(results_dict, output_path=None):
    """Overlaid drawdown charts for all models. Downsampled for HTML size."""
    _ensure_reports_dir()
    output_path = output_path or REPORTS_DIR / "drawdowns.html"

    fig = go.Figure()
    colors = {"lstm": "#2196F3", "cnn1d": "#FF9800", "cnn_lstm": "#4CAF50"}

    for name, data in results_dict.items():
        eq = data.get("equity_curve", pd.DataFrame())
        if len(eq) == 0:
            continue
        eq = eq.copy()
        eq["timestamp"] = pd.to_datetime(eq["timestamp"])
        if len(eq) > 20000:
            eq = eq.iloc[::len(eq)//10000].copy()
        cummax = eq["equity"].cummax()
        dd = (eq["equity"] - cummax) / cummax * 100
        color = colors.get(name, "#666")
        fig.add_trace(go.Scatter(
            x=eq["timestamp"], y=dd,
            name=name.upper(), line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in color else None,
            hovertemplate=f"<b>{name.upper()}</b><br>%{{x}}<br>Drawdown: %{{y:.2f}}%<extra></extra>",
        ))

    fig.update_layout(
        title="Drawdowns — All Models",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        template="plotly_dark", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400, margin=dict(l=60, r=30, t=50, b=40),
    )
    fig.write_html(str(output_path))
    return fig


def plot_price_with_signals(signals_df, prices_df, ticker=None, output_path=None):
    """Price chart with Buy/Sell markers for a representative ticker."""
    _ensure_reports_dir()
    output_path = output_path or REPORTS_DIR / "price_signals.html"

    if ticker is None:
        ticker = signals_df["ticker"].mode().iloc[0] if len(signals_df) else None
    if ticker is None:
        return None

    px = prices_df[prices_df["ticker"] == ticker].copy()
    sx = signals_df[signals_df["ticker"] == ticker].copy()
    px["timestamp"] = pd.to_datetime(px["timestamp"])
    sx["timestamp"] = pd.to_datetime(sx["timestamp"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=px["timestamp"], open=px["open"], high=px["high"],
        low=px["low"], close=px["close"], name="Price",
        increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
    ), row=1, col=1)

    buys = sx[sx["predicted_signal"] == "Buy"]
    sells = sx[sx["predicted_signal"] == "Sell"]

    if len(buys):
        fig.add_trace(go.Scatter(
            x=buys["timestamp"], y=buys.get("close", [None]*len(buys)),
            mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=10, color="#26A69A"),
        ), row=1, col=1)

    if len(sells):
        fig.add_trace(go.Scatter(
            x=sells["timestamp"], y=sells.get("close", [None]*len(sells)),
            mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", size=10, color="#EF5350"),
        ), row=1, col=1)

    if "predicted_confidence" in sx.columns:
        fig.add_trace(go.Scatter(
            x=sx["timestamp"], y=sx["predicted_confidence"],
            name="Confidence", line=dict(color="#FFA726", width=1),
        ), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — Price & Signals",
        template="plotly_dark", height=600,
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=30, t=50, b=40),
    )
    fig.write_html(str(output_path))
    return fig


def plot_metrics_bar(metrics_df, output_path=None):
    """Summary metrics bar chart comparing models."""
    _ensure_reports_dir()
    output_path = output_path or REPORTS_DIR / "metrics_comparison.html"

    display_metrics = ["total_return", "cagr", "sharpe", "max_drawdown", "win_rate", "profit_factor"]
    labels = ["Total Return", "CAGR", "Sharpe", "Max Drawdown", "Win Rate", "Profit Factor"]
    colors = {"lstm": "#2196F3", "cnn1d": "#FF9800", "cnn_lstm": "#4CAF50"}

    fig = go.Figure()
    for name in metrics_df.index:
        vals = [abs(metrics_df.loc[name, m]) if m in ["total_return", "cagr", "max_drawdown", "win_rate"]
                else metrics_df.loc[name, m] for m in display_metrics]
        fig.add_trace(go.Bar(
            name=name.upper(), x=labels, y=vals,
            marker_color=colors.get(name, "#666"),
        ))

    fig.update_layout(
        title="Model Comparison",
        barmode="group", template="plotly_dark",
        height=450, margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.write_html(str(output_path))
    return fig


def generate_all_static_charts(results_dict, signals_dict=None, prices_df=None):
    """Generate all static reports. Returns dict of output paths."""
    _ensure_reports_dir()
    paths = {}

    fig = plot_equity_curves(results_dict)
    paths["equity"] = str(REPORTS_DIR / "equity_curves.html")

    fig = plot_drawdowns(results_dict)
    paths["drawdown"] = str(REPORTS_DIR / "drawdowns.html")

    if signals_dict is not None and len(signals_dict) > 0 and prices_df is not None and len(prices_df) > 0:
        first_model = list(signals_dict.keys())[0]
        fig = plot_price_with_signals(signals_dict[first_model], prices_df)
        paths["price_signals"] = str(REPORTS_DIR / "price_signals.html")

    from .analytics import comparison_table
    metrics_df = comparison_table(results_dict)
    fig = plot_metrics_bar(metrics_df)
    paths["metrics"] = str(REPORTS_DIR / "metrics_comparison.html")
    metrics_df.to_csv(REPORTS_DIR / "comparison_table.csv")

    print(f"Static charts saved → {REPORTS_DIR}/")
    return paths
