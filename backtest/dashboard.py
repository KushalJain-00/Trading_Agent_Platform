"""Trading Arena — Professional Trading Terminal.

Run: streamlit run backtest/dashboard.py

PAPER TRADING ONLY — no real money at risk.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import json
import time

from backtest.charts import (
    candlestick, candlestick_with_volume, add_indicators, add_signals,
    rsi_chart, macd_chart, equity_curve, drawdown_chart,
    prediction_projection, comparison_bar, MODEL_COLORS, COLORS,
)
SIGNALS_DIR = PROJECT_ROOT / "backtest" / "signals"
RESULTS_DIR = PROJECT_ROOT / "backtest" / "results"
REPORTS_DIR = PROJECT_ROOT / "backtest" / "reports"
LIVE_STATE_DIR = PROJECT_ROOT / "backtest" / "live_state"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

st.set_page_config(
    page_title="Trading Arena",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0d1117; --bg2: #161b22; --card: #1c2128;
    --border: #30363d; --text: #e6edf3; --dim: #8b949e;
    --blue: #58a6ff; --green: #3fb950; --red: #f85149;
    --orange: #d29922; --purple: #bc8cff;
}

.stApp { background: var(--bg); color: var(--text); }
[data-testid="stSidebar"] { background: var(--bg2); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .stMarkdown { color: var(--text); }

h1, h2, h3 { font-family: 'Inter', sans-serif !important; color: var(--text) !important; letter-spacing: -0.02em; }
h1 { font-weight: 700 !important; font-size: 1.6rem !important; }
h2 { font-weight: 600 !important; font-size: 1.2rem !important; }
h3 { font-weight: 500 !important; font-size: 1rem !important; }

[data-testid="stMetric"] {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 16px;
}
[data-testid="stMetric"] label { color: var(--dim) !important; font-size: 0.78rem !important; font-weight: 500 !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text) !important; font-weight: 600 !important; font-size: 1.1rem !important; }
[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

div[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif; font-weight: 500;
    color: var(--dim); padding: 8px 16px; font-size: 0.85rem;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--blue) !important; border-bottom: 2px solid var(--blue) !important;
}

.stDataFrame { border: 1px solid var(--border); border-radius: 6px; }

.paper-banner {
    background: linear-gradient(90deg, #d29922 0%, #e3b341 100%);
    color: #0d1117; text-align: center; padding: 6px 16px;
    font-weight: 700; font-size: 0.75rem; letter-spacing: 0.1em;
    text-transform: uppercase; border-radius: 4px; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# Paper trading banner — always visible
st.markdown('<div class="paper-banner">⚡ PAPER TRADING — NO REAL MONEY</div>', unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_signals(model_name):
    path = SIGNALS_DIR / f"{model_name}_val_signals.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_equity(model_name):
    path = RESULTS_DIR / model_name / "equity_curve.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_trades(model_name):
    path = RESULTS_DIR / model_name / "trade_log.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_comparison():
    path = REPORTS_DIR / "comparison_table.csv"
    return pd.read_csv(path, index_col=0) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_prices():
    return pd.read_parquet(str(DATA_DIR / "val.parquet"),
                           columns=["ticker", "timestamp", "open", "high", "low", "close", "volume"])


@st.cache_data(ttl=300, show_spinner=False)
def load_ticker_prices(ticker):
    prices = load_prices()
    return prices[prices["ticker"] == ticker].copy().reset_index(drop=True)


def load_live_state():
    path = LIVE_STATE_DIR / "portfolio_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_latest_bar():
    path = LIVE_STATE_DIR / "latest_bar.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Trading Arena")
    st.caption("Professional Trading Terminal")

    mode = st.radio("Mode", ["Terminal", "Historical Backtest"], index=0, label_visibility="collapsed")

    st.markdown("---")
    model_opts = ["lstm", "cnn1d", "cnn_lstm"]
    selected_models = st.multiselect("Models", model_opts, default=model_opts)

    st.markdown("---")
    st.markdown("### Backtest Config")
    capital = st.number_input("Capital (₹)", value=100_000_000, step=10_000_000)
    position_size = st.slider("Position Size %", 0.5, 10.0, 2.0, 0.5) / 100
    cost_bps = st.slider("Cost (bps)", 0, 20, 5)
    spread_bps = st.slider("Spread (bps)", 0, 15, 3)
    mc_iters = st.slider("MC Iterations", 500, 5000, 2000, 500)

    if mode == "Terminal":
        st.markdown("---")
        st.markdown("### Live Settings")
        live_tickers = st.text_input("Watchlist", "RELIANCE TCS INFY HDFCBANK ICICIBANK")
        refresh_sec = st.slider("Refresh (sec)", 5, 60, 15)


# ── Available tickers ─────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def get_available_tickers():
    prices = load_prices()
    return sorted(prices["ticker"].unique()) if len(prices) else []


all_tickers = get_available_tickers()

# ── Tabs ──────────────────────────────────────────────────────────────
tabs = st.tabs(["⚡ Terminal", "📈 Charts", "🎯 Strategy", "💰 Portfolio",
                "📋 Trades", "🎲 Monte Carlo", "🤖 Agents"])


# ══════════════════════════════════════════════════════════════════════
# TAB: Terminal — live market overview
# ══════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## Market Terminal")

    latest = load_latest_bar()
    live_state = load_live_state()

    if latest:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Bar", f"{latest['ticker']} @ ₹{latest['close']:,.2f}")
        c2.metric("Source", latest.get("source", "—").upper())
        c3.metric("Market", latest.get("market_status", "—"))
        c4.metric("Time", latest.get("timestamp", "—")[:19])
    else:
        st.info("No live data yet. Start the live loop: `python backtest/run_live.py --mode replay`")

    if live_state:
        st.markdown("### Portfolio Summary")
        cols = st.columns(len(live_state))
        for i, (model, data) in enumerate(live_state.items()):
            with cols[i]:
                st.markdown(f"**{model.upper()}**")
                st.metric("Equity", f"₹{data['equity']:,.0f}")
                st.metric("Positions", data["n_positions"])
                st.metric("Trades", data["total_trades"])

    # Watchlist — latest signals per ticker
    st.markdown("### Watchlist Signals")
    if selected_models:
        model = selected_models[0]
        sigs = load_signals(model)
        if len(sigs):
            latest_sigs = sigs.sort_values("timestamp").groupby("ticker").tail(1)
            latest_sigs = latest_sigs[["ticker", "timestamp", "predicted_signal", "predicted_confidence"]].copy()
            latest_sigs.columns = ["Ticker", "Last Signal Time", "Signal", "Confidence"]
            latest_sigs["Confidence"] = latest_sigs["Confidence"].apply(lambda x: f"{x:.2f}")
            st.dataframe(latest_sigs, use_container_width=True, hide_index=True)

    # News for selected ticker
    st.markdown("### News Intelligence")
    news_ticker = st.selectbox("Ticker", all_tickers, key="news_ticker")
    try:
        from news import get_news_for_ticker, get_sentiment_summary
        news_df = get_news_for_ticker(news_ticker, limit=10)
        if len(news_df):
            st.dataframe(news_df[["headline", "source", "published_at", "sentiment_score"]].rename(
                columns={"headline": "Headline", "source": "Source", "published_at": "Published",
                         "sentiment_score": "Sentiment"}),
                use_container_width=True, hide_index=True, height=300)
            # Sentiment trend
            sent_df = get_sentiment_summary(news_ticker, days=30)
            if len(sent_df) > 1:
                import plotly.graph_objects as go
                fig_sent = go.Figure()
                fig_sent.add_trace(go.Scatter(
                    x=sent_df["date"], y=sent_df["avg_sentiment"],
                    mode="lines+markers", name="Sentiment",
                    line=dict(color=COLORS["blue"], width=2),
                ))
                fig_sent.add_hline(y=0, line_dash="dash", line_color=COLORS["gray"])
                fig_sent.update_layout(template="plotly_dark", title=f"{news_ticker} Sentiment Trend",
                                       yaxis_title="Sentiment Score", height=220,
                                       margin=dict(l=50, r=20, t=40, b=30))
                st.plotly_chart(fig_sent, use_container_width=True)
        else:
            st.info(f"No news yet for {news_ticker}. News pipeline fetches every 30 min.")
    except ImportError:
        st.info("News module not installed. Install tavily-python to enable.")


# ══════════════════════════════════════════════════════════════════════
# TAB: Charts — candlestick + indicators
# ══════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## Charts")

    chart_ticker = st.selectbox("Ticker", all_tickers, key="chart_ticker")
    indicator_opts = st.multiselect(
        "Indicators",
        ["sma_10", "sma_20", "sma_50", "ema_12", "ema_26", "bb", "vwap"],
        default=["sma_20"],
    )
    show_volume = st.checkbox("Volume", value=True)
    show_rsi = st.checkbox("RSI", value=False)
    show_macd = st.checkbox("MACD", value=False)

    df = load_ticker_prices(chart_ticker)
    if len(df) == 0:
        st.info("No price data for this ticker.")
    else:
        if show_volume:
            fig = candlestick_with_volume(df, title=f"{chart_ticker}", height=550)
        else:
            fig = candlestick(df, title=f"{chart_ticker}", height=450)

        if indicator_opts:
            add_indicators(fig, df, indicator_opts)

        st.plotly_chart(fig, use_container_width=True)

        if show_rsi:
            st.plotly_chart(rsi_chart(df), use_container_width=True)
        if show_macd:
            st.plotly_chart(macd_chart(df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB: Strategy — signals + regime overlay
# ══════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## Strategy Viewer")

    if not selected_models:
        st.info("Select at least one model.")
    else:
        strat_model = st.selectbox("Model", selected_models, key="strat_model")
        strat_ticker = st.selectbox("Ticker", all_tickers, key="strat_ticker")

        prices = load_ticker_prices(strat_ticker)
        signals = load_signals(strat_model)

        if len(prices) == 0 or len(signals) == 0:
            st.info("No data available.")
        else:
            ticker_signals = signals[signals["ticker"] == strat_ticker].copy()

            fig = candlestick(prices, title=f"{strat_ticker} — {strat_model.upper()} Signals", height=500)
            if len(ticker_signals):
                add_signals(fig, ticker_signals)

            # Add regime background shading if available
            st.plotly_chart(fig, use_container_width=True)

            # Signal distribution
            if len(ticker_signals):
                c1, c2, c3 = st.columns(3)
                counts = ticker_signals["predicted_signal"].value_counts()
                c1.metric("Buy Signals", counts.get("Buy", 0))
                c2.metric("Hold Signals", counts.get("Hold", 0))
                c3.metric("Sell Signals", counts.get("Sell", 0))

            # Prediction projection
            st.markdown("### Model Projection")
            st.caption("Forward extrapolation based on last signal direction — NOT a guarantee.")
            signal_history = ticker_signals["predicted_signal"].tolist() if len(ticker_signals) else []
            fig_proj = prediction_projection(prices, signal_history, n_bars=10,
                                              confidence=0.6, title=f"{strat_ticker} Projection")
            st.plotly_chart(fig_proj, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB: Portfolio — equity curves + drawdown
# ══════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## Portfolio Performance")

    if not selected_models:
        st.info("Select at least one model.")
    else:
        equity_dfs = {}
        for m in selected_models:
            eq = load_equity(m)
            if len(eq):
                equity_dfs[m] = eq

        if equity_dfs:
            st.plotly_chart(equity_curve(equity_dfs), use_container_width=True)
            st.plotly_chart(drawdown_chart(equity_dfs), use_container_width=True)

            # Key metrics
            st.markdown("### Key Metrics")
            rows = []
            for m, eq in equity_dfs.items():
                trades = load_trades(m)
                init = eq["equity"].iloc[0]
                final = eq["equity"].iloc[-1]
                ret = (final / init - 1) * 100
                days = len(eq) / (252 * 375)
                cagr = ((final / init) ** (1 / max(days, 0.01)) - 1) * 100
                bar_ret = eq["equity"].pct_change().dropna()
                sharpe = (bar_ret.mean() / bar_ret.std() * np.sqrt(252 * 375)) if bar_ret.std() > 0 else 0
                dd = ((eq["equity"] - eq["equity"].cummax()) / eq["equity"].cummax()).min() * 100
                wr = (trades["net_pnl"] > 0).mean() * 100 if len(trades) else 0
                rows.append({"Model": m.upper(), "Return": f"{ret:+.1f}%", "CAGR": f"{cagr:+.1f}%",
                             "Sharpe": f"{sharpe:.3f}", "Max DD": f"{dd:.1f}%", "Win Rate": f"{wr:.0f}%",
                             "Trades": len(trades)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No equity curve data. Run backtest first.")


# ══════════════════════════════════════════════════════════════════════
# TAB: Trades — log + audit
# ══════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## Trade Log")

    if not selected_models:
        st.info("Select at least one model.")
    else:
        trade_model = st.selectbox("Model", selected_models, key="trade_model")
        trades = load_trades(trade_model)

        if len(trades) == 0:
            st.info("No trades. Run backtest first.")
        else:
            # Summary
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Trades", len(trades))
            c2.metric("Win Rate", f"{(trades['net_pnl'] > 0).mean() * 100:.1f}%")
            c3.metric("Total P&L", f"₹{trades['net_pnl'].sum():,.0f}")
            c4.metric("Total Costs", f"₹{trades['costs'].sum():,.0f}")

            # Filter
            ticker_list = ["All"] + sorted(trades["ticker"].unique().tolist())
            filter_ticker = st.selectbox("Filter by Ticker", ticker_list, key="trade_filter")
            filtered = trades if filter_ticker == "All" else trades[trades["ticker"] == filter_ticker]

            # Exit reason breakdown
            if "exit_reason" in filtered.columns:
                st.markdown("### Exit Reasons")
                reason_counts = filtered["exit_reason"].value_counts()
                st.bar_chart(reason_counts)

            # Trade table
            display_cols = ["entry_time", "exit_time", "ticker", "direction",
                            "entry_price", "exit_price", "size", "gross_pnl",
                            "net_pnl", "costs", "holding_bars", "exit_reason"]
            display_cols = [c for c in display_cols if c in filtered.columns]
            st.dataframe(
                filtered[display_cols].style.format({
                    "entry_price": "₹{:,.2f}", "exit_price": "₹{:,.2f}",
                    "gross_pnl": "₹{:,.0f}", "net_pnl": "₹{:,.0f}",
                    "costs": "₹{:,.0f}", "holding_bars": "{:d}",
                }).applymap(
                    lambda v: "color: #3fb950" if isinstance(v, (int, float)) and v > 0
                    else ("color: #f85149" if isinstance(v, (int, float)) and v < 0 else ""),
                    subset=["net_pnl", "gross_pnl"]
                ),
                use_container_width=True, height=500,
            )


# ══════════════════════════════════════════════════════════════════════
# TAB: Monte Carlo
# ══════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## Monte Carlo Simulation")

    if not selected_models:
        st.info("Select at least one model.")
    else:
        mc_model = st.selectbox("Model", selected_models, key="mc_model")
        trades = load_trades(mc_model)
        eq = load_equity(mc_model)

        if len(trades) == 0 or len(eq) == 0:
            st.info("No data. Run backtest first.")
        elif st.button("Run Simulation", key="mc_run"):
            with st.spinner(f"Running {mc_iters} iterations..."):
                from backtest.monte_carlo import trade_resampling, return_bootstrapping
                trade_result = trade_resampling(trades, eq, mc_iters, capital)
                boot_result = return_bootstrapping(eq, mc_iters)

            if trade_result:
                st.markdown("### Trade Resampling")
                c1, c2 = st.columns(2)
                with c1:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=trade_result["final_equities"], nbinsx=80,
                                               marker_color=COLORS["blue"], opacity=0.7))
                    pcts = trade_result["percentiles"]
                    for p, color in [(50, COLORS["orange"]), (5, COLORS["red"]), (95, COLORS["green"])]:
                        fig.add_vline(x=pcts[p]["equity"], line_dash="dash",
                                      line_color=color, annotation_text=f"P{p}")
                    fig.update_layout(template="plotly_dark", title="Final Equity Distribution",
                                      height=350, margin=dict(l=50, r=20, t=40, b=30))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=trade_result["max_drawdowns"] * 100, nbinsx=80,
                                               marker_color=COLORS["red"], opacity=0.7))
                    for p, color in [(50, COLORS["orange"]), (5, COLORS["red"]), (95, COLORS["green"])]:
                        fig.add_vline(x=pcts[p]["max_dd"] * 100, line_dash="dash",
                                      line_color=color, annotation_text=f"P{p}")
                    fig.update_layout(template="plotly_dark", title="Max Drawdown Distribution",
                                      height=350, margin=dict(l=50, r=20, t=40, b=30))
                    st.plotly_chart(fig, use_container_width=True)

            if boot_result:
                st.markdown("### Return Bootstrapping")
                c1, c2, c3 = st.columns(3)
                for col, (key, title, color) in zip(
                    [c1, c2, c3],
                    [("cagr_ci", "CAGR", COLORS["blue"]),
                     ("sharpe_ci", "Sharpe", COLORS["green"]),
                     ("maxdd_ci", "Max DD", COLORS["red"])],
                ):
                    with col:
                        fig = go.Figure()
                        samples_key = f"{key.replace('_ci', '')}_samples"
                        if samples_key in boot_result:
                            mult = 100 if "cagr" in key or "maxdd" in key else 1
                            fig.add_trace(go.Histogram(
                                x=boot_result[samples_key] * mult, nbinsx=60,
                                marker_color=color, opacity=0.7,
                            ))
                            for p, c in [(50, COLORS["orange"]), (5, COLORS["red"]), (95, COLORS["green"])]:
                                val = boot_result[key][p] * mult
                                fig.add_vline(x=val, line_dash="dash", line_color=c,
                                              annotation_text=f"P{p}: {val:.2f}")
                        fig.update_layout(template="plotly_dark", title=title,
                                          height=280, margin=dict(l=50, r=20, t=40, b=30))
                        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB: Agents — control + decision log + latency
# ══════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("## Agent Control & Monitoring")

    # Live state
    live_state = load_live_state()
    if live_state:
        st.markdown("### Live Agent Status")
        cols = st.columns(len(live_state))
        for i, (model, data) in enumerate(live_state.items()):
            with cols[i]:
                st.markdown(f"**{model.upper()}**")
                st.metric("Equity", f"₹{data['equity']:,.0f}")
                st.metric("Bars Processed", data["bars_processed"])
                st.metric("Open Positions", data["n_positions"])

    # ── Latency Report ────────────────────────────────────────────────
    st.markdown("### Latency Report")
    latency_path = LIVE_STATE_DIR / "latency.jsonl"
    if latency_path.exists():
        lat_records = []
        with open(latency_path) as f:
            for line in f:
                try:
                    lat_records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if lat_records:
            lat_df = pd.DataFrame(lat_records)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Median Inference", f"{lat_df['infer_ms'].median():.0f}ms")
            c2.metric("P95 Inference", f"{lat_df['infer_ms'].quantile(0.95):.0f}ms")
            c3.metric("Median Pipeline", f"{lat_df['pipeline_ms'].median():.0f}ms")
            c4.metric("Total Bars", f"{len(lat_df):,}")

            import plotly.graph_objects as go
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Histogram(x=lat_df["infer_ms"], name="Inference",
                                           marker_color=COLORS["blue"], opacity=0.7, nbinsx=40))
            fig_lat.add_trace(go.Histogram(x=lat_df["pipeline_ms"], name="Full Pipeline",
                                           marker_color=COLORS["orange"], opacity=0.5, nbinsx=40))
            fig_lat.update_layout(template="plotly_dark", title="Latency Distribution (ms)",
                                  barmode="overlay", height=280,
                                  margin=dict(l=50, r=20, t=40, b=30))
            st.plotly_chart(fig_lat, use_container_width=True)
        else:
            st.info("No latency data yet. Start the live loop to collect metrics.")
    else:
        st.info("No latency log found. Start the live loop: `python backtest/run_live.py`")

    # ── Manual Paper Trading ──────────────────────────────────────────
    st.markdown("### Manual Paper Trade")
    st.caption("Place your own paper trades — clearly separated from bot-generated trades.")
    manual_col1, manual_col2, manual_col3, manual_col4 = st.columns(4)
    with manual_col1:
        manual_ticker = st.selectbox("Ticker", all_tickers, key="manual_ticker")
    with manual_col2:
        manual_action = st.radio("Action", ["Buy", "Sell"], horizontal=True, key="manual_action")
    with manual_col3:
        manual_price = st.number_input("Price (₹)", value=0.0, step=0.05, format="%.2f", key="manual_price")
    with manual_col4:
        manual_size = st.number_input("Size (₹)", value=100000.0, step=10000.0, key="manual_size")
    manual_reason = st.text_input("Reason (optional)", key="manual_reason")
    if st.button("Place Paper Trade", key="place_manual"):
        if manual_price > 0 and manual_size > 0:
            import sqlite3 as _sqlite3
            _conn = _sqlite3.connect(str(PROJECT_ROOT / "backtest" / "multiagent_state.db"))
            _conn.execute(
                "INSERT INTO manual_trades (ticker, timestamp, action, price, size, reason) VALUES (?, ?, ?, ?, ?, ?)",
                (manual_ticker, pd.Timestamp.now().isoformat(), manual_action, manual_price, manual_size, manual_reason or None))
            _conn.commit()
            _conn.close()
            st.success(f"Paper trade placed: {manual_action} {manual_ticker} @ ₹{manual_price:,.2f} (₹{manual_size:,.0f})")
            st.rerun()
        else:
            st.warning("Enter a valid price and size.")

    # Manual trade history
    import sqlite3 as _sqlite3
    _db = PROJECT_ROOT / "backtest" / "multiagent_state.db"
    if _db.exists():
        _conn = _sqlite3.connect(str(_db))
        try:
            manual_df = pd.read_sql_query(
                "SELECT ticker, timestamp, action, price, size, reason FROM manual_trades ORDER BY id DESC LIMIT 20", _conn)
            if len(manual_df):
                st.markdown("#### Manual Trade History")
                st.dataframe(manual_df, use_container_width=True, hide_index=True)
        except Exception:
            pass
        _conn.close()

    # ── Decision Log from SQLite ──────────────────────────────────────
    st.markdown("### Decision Log")
    db_path = PROJECT_ROOT / "backtest" / "multiagent_state.db"
    if db_path.exists():
        import sqlite3
        conn = sqlite3.connect(str(db_path))

        # Latest signals
        st.markdown("#### Latest Signals")
        try:
            sigs_df = pd.read_sql_query(
                "SELECT bar_idx, ticker, timestamp, signal, confidence, model "
                "FROM signals ORDER BY id DESC LIMIT 20", conn)
            if len(sigs_df):
                st.dataframe(sigs_df, use_container_width=True, hide_index=True)
            else:
                st.info("No signals logged yet.")
        except Exception:
            st.info("Signals table not available.")

        # Agent log
        st.markdown("#### Agent Activity Log")
        try:
            logs_df = pd.read_sql_query(
                "SELECT bar_idx, agent, event, created_at FROM agent_log ORDER BY id DESC LIMIT 30", conn)
            if len(logs_df):
                st.dataframe(logs_df, use_container_width=True, hide_index=True)
            else:
                st.info("No agent activity logged.")
        except Exception:
            st.info("Agent log not available.")

        # Executions
        st.markdown("#### Executions")
        try:
            exec_df = pd.read_sql_query(
                "SELECT bar_idx, ticker, timestamp, action, price, size, pnl, exit_reason "
                "FROM executions ORDER BY id DESC LIMIT 20", conn)
            if len(exec_df):
                st.dataframe(exec_df, use_container_width=True, hide_index=True)
            else:
                st.info("No executions logged.")
        except Exception:
            st.info("Executions table not available.")

        conn.close()
    else:
        st.info("No agent database found. Start the live loop to populate.")


# ── Auto-refresh for terminal mode ────────────────────────────────────
if mode == "Terminal":
    time.sleep(refresh_sec)
    st.rerun()
