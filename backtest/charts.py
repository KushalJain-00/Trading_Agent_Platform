"""Charting primitives — Plotly candlestick, indicators, signals, projections.

All functions return go.Figure objects. No Streamlit calls here —
dashboard.py composes these into layouts.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Color palette ─────────────────────────────────────────────────────
COLORS = {
    "bg": "#0d1117",
    "card": "#161b22",
    "border": "#30363d",
    "text": "#e6edf3",
    "text_dim": "#8b949e",
    "blue": "#58a6ff",
    "green": "#3fb950",
    "red": "#f85149",
    "orange": "#d29922",
    "purple": "#bc8cff",
    "cyan": "#39d2c0",
    "gray": "#484f58",
}

MODEL_COLORS = {"lstm": "#58a6ff", "cnn1d": "#d29922", "cnn_lstm": "#3fb950"}

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
    hovermode="x unified",
    margin=dict(l=55, r=20, t=35, b=40),
    xaxis=dict(gridcolor=COLORS["border"], showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=COLORS["border"], showgrid=True, zeroline=False),
    legend=dict(orientation="h", y=1.08, font=dict(size=11)),
)


def _apply_layout(fig, **overrides):
    fig.update_layout(**LAYOUT_DEFAULTS, **overrides)
    return fig


# ── Candlestick ───────────────────────────────────────────────────────
def candlestick(df, title=None, height=500):
    """OHLC candlestick chart. df must have: timestamp, open, high, low, close."""
    fig = go.Figure(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color=COLORS["green"], decreasing_line_color=COLORS["red"],
        increasing_fillcolor=COLORS["green"], decreasing_fillcolor=COLORS["red"],
    ))
    return _apply_layout(fig, title=title, height=height,
                         xaxis_rangeslider_visible=False)


def candlestick_with_volume(df, title=None, height=600):
    """Candlestick + volume bars in subplots."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.78, 0.22])
    fig.add_trace(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color=COLORS["green"], decreasing_line_color=COLORS["red"],
    ), row=1, col=1)

    colors = [COLORS["green"] if c >= o else COLORS["red"]
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["timestamp"], y=df["volume"], name="Volume",
        marker_color=colors, opacity=0.6,
    ), row=2, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return _apply_layout(fig, title=title, height=height,
                         xaxis_rangeslider_visible=False)


# ── Indicators ────────────────────────────────────────────────────────
def sma(series, period):
    return series.rolling(period, min_periods=1).mean()


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def bollinger_bands(close, period=20, std=2):
    mid = sma(close, period)
    rolling_std = close.rolling(period, min_periods=1).std()
    return mid, mid + std * rolling_std, mid - std * rolling_std


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


def macd(close, fast=12, slow=26, signal_period=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def vwap(df):
    """Volume-weighted average price (cumulative intraday approximation)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical * df["volume"]).cumsum()
    return cum_tp_vol / (cum_vol + 1e-10)


def _has_subplots(fig):
    return hasattr(fig, '_grid_ref') and fig._grid_ref is not None


def add_indicators(fig, df, indicators, row=1, col=1):
    """Add indicator overlays to an existing figure.

    indicators: list of str, e.g. ['sma_10', 'sma_50', 'bb', 'vwap']
    """
    close = df["close"]
    use_grid = _has_subplots(fig)

    for ind in indicators:
        if ind.startswith("sma_"):
            period = int(ind.split("_")[1])
            trace = go.Scatter(x=df["timestamp"], y=sma(close, period),
                               name=f"SMA {period}", line=dict(width=1, color=COLORS["orange"]),
                               opacity=0.7)
            fig.add_trace(trace, row=row, col=col) if use_grid else fig.add_trace(trace)

        elif ind.startswith("ema_"):
            period = int(ind.split("_")[1])
            trace = go.Scatter(x=df["timestamp"], y=ema(close, period),
                               name=f"EMA {period}", line=dict(width=1, color=COLORS["purple"]),
                               opacity=0.7)
            fig.add_trace(trace, row=row, col=col) if use_grid else fig.add_trace(trace)

        elif ind == "bb":
            mid, upper, lower = bollinger_bands(close)
            for y, name, dash in [(upper, "BB Upper", "dot"), (lower, "BB Lower", "dot"), (mid, "BB Mid", "dash")]:
                fill = "tonexty" if name == "BB Lower" else None
                fillcolor = "rgba(57,210,192,0.08)" if name == "BB Lower" else None
                trace = go.Scatter(x=df["timestamp"], y=y, name=name,
                                   line=dict(width=1, color=COLORS["cyan"], dash=dash),
                                   fill=fill, fillcolor=fillcolor, showlegend=(name != "BB Lower"))
                fig.add_trace(trace, row=row, col=col) if use_grid else fig.add_trace(trace)

        elif ind == "vwap":
            trace = go.Scatter(x=df["timestamp"], y=vwap(df),
                               name="VWAP", line=dict(width=1.5, color=COLORS["text_dim"], dash="dash"))
            fig.add_trace(trace, row=row, col=col) if use_grid else fig.add_trace(trace)

    return fig


# ── Signal markers ────────────────────────────────────────────────────
def add_signals(fig, signals_df, row=1, col=1):
    """Add Buy/Sell triangle markers from a signals DataFrame."""
    buys = signals_df[signals_df["predicted_signal"] == "Buy"]
    sells = signals_df[signals_df["predicted_signal"] == "Sell"]
    use_grid = _has_subplots(fig)

    if len(buys):
        trace = go.Scatter(
            x=buys["timestamp"], mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=10, color=COLORS["green"],
                        line=dict(width=1, color="#fff")),
            hovertemplate="BUY %{x}<br>Conf: %{customdata:.2f}<extra></extra>",
            customdata=buys.get("predicted_confidence", pd.Series(dtype=float)))
        fig.add_trace(trace, row=row, col=col) if use_grid else fig.add_trace(trace)

    if len(sells):
        trace = go.Scatter(
            x=sells["timestamp"], mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", size=10, color=COLORS["red"],
                        line=dict(width=1, color="#fff")),
            hovertemplate="SELL %{x}<br>Conf: %{customdata:.2f}<extra></extra>",
            customdata=sells.get("predicted_confidence", pd.Series(dtype=float)))
        fig.add_trace(trace, row=row, col=col) if use_grid else fig.add_trace(trace)

    return fig


# ── RSI / MACD sub-charts ────────────────────────────────────────────
def rsi_chart(df, period=14, height=200):
    """Standalone RSI oscillator."""
    rsi_val = rsi(df["close"], period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=rsi_val, name=f"RSI {period}",
        line=dict(color=COLORS["blue"], width=1.5),
    ))
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["red"], opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], opacity=0.5)
    fig.add_hline(y=50, line_dash="dot", line_color=COLORS["gray"], opacity=0.3)
    fig.update_yaxes(range=[0, 100])
    return _apply_layout(fig, height=height, yaxis_title="RSI",
                         showlegend=False)


def macd_chart(df, fast=12, slow=26, signal_period=9, height=200):
    """Standalone MACD oscillator."""
    macd_line, signal_line, hist = macd(df["close"], fast, slow, signal_period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=macd_line, name="MACD",
        line=dict(color=COLORS["blue"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=signal_line, name="Signal",
        line=dict(color=COLORS["orange"], width=1),
    ))
    colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in hist]
    fig.add_trace(go.Bar(
        x=df["timestamp"], y=hist, name="Histogram",
        marker_color=colors, opacity=0.6,
    ))
    fig.add_hline(y=0, line_color=COLORS["gray"], opacity=0.3)
    return _apply_layout(fig, height=height, yaxis_title="MACD")


# ── Equity / drawdown ────────────────────────────────────────────────
def equity_curve(equity_dfs, title="Equity Curve", height=450):
    """Plot one or more equity curves. equity_dfs: dict of {name: DataFrame}."""
    fig = go.Figure()
    for name, eq in equity_dfs.items():
        if len(eq) == 0:
            continue
        ts = pd.to_datetime(eq["timestamp"]) if "timestamp" in eq.columns else eq.index
        color = MODEL_COLORS.get(name.lower(), COLORS["blue"])
        fig.add_trace(go.Scatter(
            x=ts, y=eq["equity"], name=name.upper(),
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{name.upper()}</b><br>%{{x}}<br>₹%{{y:,.0f}}<extra></extra>",
        ))
    return _apply_layout(fig, title=title, height=height,
                         yaxis_title="Equity (₹)")


def drawdown_chart(equity_dfs, title="Drawdown", height=250):
    """Plot drawdown curves."""
    fig = go.Figure()
    for name, eq in equity_dfs.items():
        if len(eq) == 0:
            continue
        ts = pd.to_datetime(eq["timestamp"]) if "timestamp" in eq.columns else eq.index
        cummax = eq["equity"].cummax()
        dd = (eq["equity"] - cummax) / cummax * 100
        color = MODEL_COLORS.get(name.lower(), COLORS["blue"])
        fig.add_trace(go.Scatter(
            x=ts, y=dd, name=name.upper(),
            line=dict(color=color, width=1.5), fill="tozeroy",
        ))
    return _apply_layout(fig, title=title, height=height,
                         yaxis_title="Drawdown (%)")


# ── Prediction projection ────────────────────────────────────────────
def prediction_projection(df, signal_history, n_bars=10, confidence=0.5,
                          title="Model Projection", height=300):
    """Show recent price + forward projection cone.

    The projection is a straight-line extrapolation of the last signal's
    direction, with a confidence band that widens over time.
    Clearly labeled as a model projection, not a guarantee.
    """
    fig = go.Figure()

    # Recent price
    recent = df.tail(30)
    fig.add_trace(go.Scatter(
        x=recent["timestamp"], y=recent["close"],
        name="Price", line=dict(color=COLORS["text"], width=1.5),
    ))

    # Projection from last bar
    last_price = df["close"].iloc[-1]
    last_ts = df["timestamp"].iloc[-1]

    # Determine direction from signal
    if signal_history and len(signal_history) > 0:
        last_signal = signal_history[-1]
        direction = 1 if last_signal == "Buy" else (-1 if last_signal == "Sell" else 0)
    else:
        direction = 0

    if direction != 0:
        # Generate forward timestamps (approximate 1-min bars)
        forward_ts = pd.date_range(last_ts, periods=n_bars + 1, freq="1min")[1:]
        # Simple linear extrapolation with random walk noise
        drift = direction * 0.0002  # small drift per bar
        noise = np.random.normal(0, 0.0005, n_bars)
        returns = drift + noise
        prices = last_price * np.cumprod(1 + returns)

        # Confidence band widens with sqrt(t)
        base_vol = df["close"].pct_change().std() if len(df) > 1 else 0.001
        band_width = confidence * base_vol * last_price * np.sqrt(np.arange(1, n_bars + 1))

        fig.add_trace(go.Scatter(
            x=forward_ts, y=prices, name="Projected",
            line=dict(color=COLORS["blue"], width=1.5, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=list(forward_ts) + list(forward_ts[::-1]),
            y=list(prices + band_width) + list((prices - band_width)[::-1]),
            fill="toself", fillcolor="rgba(88,166,255,0.1)",
            line=dict(width=0), showlegend=False, name="Confidence Band",
        ))

    fig.add_vline(x=last_ts, line_dash="dot", line_color=COLORS["gray"],
                  annotation_text="Now")

    return _apply_layout(fig, title=f"{title} — labeled as model projection only",
                         height=height, yaxis_title="Price (₹)")


# ── Multi-model comparison ───────────────────────────────────────────
def comparison_bar(comp_df, metrics=("total_return", "sharpe", "max_drawdown", "win_rate"),
                   height=400):
    """Side-by-side bar chart for model comparison."""
    labels = {"total_return": "Return (%)", "sharpe": "Sharpe",
              "max_drawdown": "Max DD (%)", "win_rate": "Win Rate (%)"}
    n = len(metrics)
    fig = make_subplots(rows=1, cols=n,
                        subplot_titles=[labels.get(m, m) for m in metrics])

    for i, metric in enumerate(metrics):
        if metric not in comp_df.columns and metric not in comp_df.index:
            continue
        vals = comp_df[metric].values if metric in comp_df.columns else []
        names = comp_df.index.tolist() if metric in comp_df.columns else []
        for j, name in enumerate(names):
            v = vals[j]
            if isinstance(v, (int, float)):
                if metric in ("total_return", "max_drawdown", "win_rate"):
                    v = v * 100
                color = MODEL_COLORS.get(name, COLORS["gray"])
                fig.add_trace(go.Bar(
                    name=name.upper(), x=[name.upper()], y=[v],
                    marker_color=color, showlegend=(i == 0),
                ), row=1, col=i + 1)

    return _apply_layout(fig, height=height, barmode="group")
