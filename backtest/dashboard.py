"""Professional trading dashboard — Streamlit + Plotly.

Run: streamlit run backtest/dashboard.py

Tabs: Overview, Equity Curves, Trade Explorer, Monte Carlo, Model Comparison, Live Paper Trading
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGNALS_DIR = PROJECT_ROOT / "backtest" / "signals"
RESULTS_DIR = PROJECT_ROOT / "backtest" / "results"
REPORTS_DIR = PROJECT_ROOT / "backtest" / "reports"
LIVE_STATE_DIR = PROJECT_ROOT / "backtest" / "live_state"

st.set_page_config(
    page_title="Trading Arena — Backtest Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark professional theme ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-card: #1c2128;
    --border: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --accent-blue: #58a6ff;
    --accent-green: #3fb950;
    --accent-red: #f85149;
    --accent-orange: #d29922;
}

.stApp { background: var(--bg-primary); color: var(--text-primary); }
[data-testid="stSidebar"] { background: var(--bg-secondary); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .stMarkdown { color: var(--text-primary); }

h1, h2, h3, h4 { font-family: 'Inter', sans-serif !important; color: var(--text-primary) !important; }
h1 { font-weight: 700 !important; }
h2 { font-weight: 600 !important; }

[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
}
[data-testid="stMetric"] label { color: var(--text-secondary) !important; font-size: 0.85rem !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-weight: 600 !important; }

div[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    color: var(--text-secondary);
    padding: 10px 20px;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-blue) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
}

.stDataFrame { border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }

div[data-baseweb="tab"] { font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_signals(model_name):
    path = SIGNALS_DIR / f"{model_name}_val_signals.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_equity_curve(model_name):
    path = RESULTS_DIR / model_name / "equity_curve.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_trade_log(model_name):
    path = RESULTS_DIR / model_name / "trade_log.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_comparison():
    path = REPORTS_DIR / "comparison_table.csv"
    return pd.read_csv(path, index_col=0) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_prices():
    val_path = PROJECT_ROOT / "data" / "processed" / "val.parquet"
    return pd.read_parquet(str(val_path), columns=["ticker", "timestamp", "open", "high", "low", "close", "volume"])


def load_live_state():
    summary_path = LIVE_STATE_DIR / "portfolio_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def load_latest_bar():
    path = LIVE_STATE_DIR / "latest_bar.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    mode = st.radio("Mode", ["Historical Backtest", "Live Paper Trading"], index=0)
    model_options = ["lstm", "cnn1d", "cnn_lstm"]
    selected_models = st.multiselect("Models", model_options, default=model_options)

    st.markdown("---")
    st.markdown("### Trading Mode")
    trading_mode = st.radio("Trading Mode", ["Portfolio (Multi-Stock)", "Single Stock"],
                            index=0, key="trading_mode")
    ticker_filter = None
    if trading_mode == "Single Stock":
        ticker_filter = st.text_input("Ticker Symbol", "RELIANCE")
        st.caption("e.g., RELIANCE, TCS, INFY, HDFCBANK")

    allocation = st.selectbox("Allocation", ["equal", "confidence-weighted", "top-N"], index=0)
    max_positions = st.number_input("Max Positions (0=unlimited)", value=0, min_value=0, step=1)
    max_pos_pct = st.slider("Max Position %", 0.0, 50.0, 10.0, 1.0) / 100 if max_positions > 0 else 0.0

    st.markdown("---")
    st.markdown("### Position & Costs")
    capital = st.number_input("Starting Capital (₹)", value=100_000_000, step=10_000_000)
    position_size = st.slider("Position Size %", 0.5, 10.0, 2.0, 0.5) / 100
    cost_bps = st.slider("Transaction Cost (bps)", 0, 20, 5)
    spread_bps = st.slider("Bid-Ask Spread (bps)", 0, 15, 3)
    latency = st.slider("Execution Latency (bars)", 0, 5, 1)

    st.markdown("---")
    mc_iterations = st.slider("Monte Carlo Iterations", 500, 5000, 2000, 500)

    if mode == "Live Paper Trading":
        st.markdown("---")
        st.markdown("### Live Settings")
        live_tickers = st.text_input("Tickers (space-separated)",
                                      ticker_filter if ticker_filter else "RELIANCE TCS INFY HDFCBANK ICICIBANK")
        replay_speed = st.slider("Replay Speed (sec/bar)", 0.1, 5.0, 2.0, 0.1)
        refresh_rate = st.slider("Refresh (sec)", 5, 60, 10)


# ── Plotly defaults ──────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_COLORS = {"lstm": "#58a6ff", "cnn1d": "#d29922", "cnn_lstm": "#3fb950"}


# ── Tabs ─────────────────────────────────────────────────────────────
tab_overview, tab_equity, tab_trades, tab_mc, tab_compare, tab_modes, tab_live = st.tabs([
    "📊 Overview", "📈 Equity Curves", "🔍 Trade Explorer",
    "🎲 Monte Carlo", "⚖️ Model Comparison", "🔀 Mode Comparison", "🔴 Live Paper Trading",
])

# ── Overview ─────────────────────────────────────────────────────────
with tab_overview:
    st.markdown("## Portfolio Overview")

    if not selected_models:
        st.info("Select at least one model in the sidebar.")
    else:
        cols = st.columns(len(selected_models))
        for i, model in enumerate(selected_models):
            eq = load_equity_curve(model)
            trades = load_trade_log(model)
            if len(eq) == 0:
                cols[i].warning(f"No data for {model.upper()}")
                continue

            initial = eq["equity"].iloc[0]
            final = eq["equity"].iloc[-1]
            total_ret = (final / initial - 1) * 100

            n_bars = len(eq)
            days = n_bars / (252 * 375)
            cagr = ((final / initial) ** (1 / max(days, 0.01)) - 1) * 100

            bar_ret = eq["equity"].pct_change().dropna()
            sharpe = (bar_ret.mean() / bar_ret.std() * np.sqrt(252 * 375)) if bar_ret.std() > 0 else 0

            cummax = eq["equity"].cummax()
            max_dd = ((eq["equity"] - cummax) / cummax).min() * 100

            n_trades = len(trades)
            win_rate = (trades["net_pnl"] > 0).mean() * 100 if n_trades > 0 else 0

            with cols[i]:
                st.markdown(f"### {model.upper()}")
                c1, c2 = st.columns(2)
                c1.metric("Total Return", f"{total_ret:+.2f}%")
                c2.metric("CAGR", f"{cagr:+.2f}%")
                c1.metric("Sharpe", f"{sharpe:.3f}")
                c2.metric("Max Drawdown", f"{max_dd:.2f}%")
                c1.metric("Trades", f"{n_trades:,}")
                c2.metric("Win Rate", f"{win_rate:.1f}%")

# ── Equity Curves ────────────────────────────────────────────────────
with tab_equity:
    st.markdown("## Equity Curves")

    if selected_models:
        fig = go.Figure()
        for model in selected_models:
            eq = load_equity_curve(model)
            if len(eq) == 0:
                continue
            eq["timestamp"] = pd.to_datetime(eq["timestamp"])
            fig.add_trace(go.Scatter(
                x=eq["timestamp"], y=eq["equity"],
                name=model.upper(), line=dict(color=PLOTLY_COLORS.get(model, "#666"), width=2),
                hovertemplate=f"<b>{model.upper()}</b><br>%{{x}}<br>₹%{{y:,.0f}}<extra></extra>",
            ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, hovermode="x unified",
            xaxis_title="Date", yaxis_title="Portfolio Equity (₹)",
            legend=dict(orientation="h", y=1.12), height=550,
            margin=dict(l=60, r=30, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown subchart
        fig2 = go.Figure()
        for model in selected_models:
            eq = load_equity_curve(model)
            if len(eq) == 0:
                continue
            eq["timestamp"] = pd.to_datetime(eq["timestamp"])
            cummax = eq["equity"].cummax()
            dd = (eq["equity"] - cummax) / cummax * 100
            fig2.add_trace(go.Scatter(
                x=eq["timestamp"], y=dd, name=model.upper(),
                line=dict(color=PLOTLY_COLORS.get(model, "#666"), width=1.5),
                fill="tozeroy",
            ))
        fig2.update_layout(
            title="Drawdowns", template=PLOTLY_TEMPLATE, hovermode="x unified",
            yaxis_title="Drawdown (%)", height=300,
            margin=dict(l=60, r=30, t=40, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Trade Explorer ───────────────────────────────────────────────────
with tab_trades:
    st.markdown("## Trade Explorer")

    if selected_models:
        model = st.selectbox("Model", selected_models, key="trade_model")
        prices = load_prices()
        signals = load_signals(model)
        trades = load_trade_log(model)

        if len(signals) == 0:
            st.info("No signals found. Run backtest first.")
        else:
            ticker = st.selectbox("Ticker", sorted(signals["ticker"].unique()), key="trade_ticker")

            px = prices[prices["ticker"] == ticker].copy()
            sx = signals[signals["ticker"] == ticker].copy()
            px["timestamp"] = pd.to_datetime(px["timestamp"])
            sx["timestamp"] = pd.to_datetime(sx["timestamp"])

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.03, row_heights=[0.75, 0.25])

            fig.add_trace(go.Candlestick(
                x=px["timestamp"], open=px["open"], high=px["high"],
                low=px["low"], close=px["close"], name="Price",
                increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            ), row=1, col=1)

            buys = sx[sx["predicted_signal"] == "Buy"]
            sells = sx[sx["predicted_signal"] == "Sell"]

            if len(buys):
                fig.add_trace(go.Scatter(
                    x=buys["timestamp"], mode="markers", name="Buy",
                    marker=dict(symbol="triangle-up", size=9, color="#3fb950"),
                    hovertemplate="Buy<br>%{x}<extra></extra>",
                ), row=1, col=1)

            if len(sells):
                fig.add_trace(go.Scatter(
                    x=sells["timestamp"], mode="markers", name="Sell",
                    marker=dict(symbol="triangle-down", size=9, color="#f85149"),
                    hovertemplate="Sell<br>%{x}<extra></extra>",
                ), row=1, col=1)

            if "predicted_confidence" in sx.columns:
                fig.add_trace(go.Scatter(
                    x=sx["timestamp"], y=sx["predicted_confidence"],
                    name="Confidence", line=dict(color="#d29922", width=1),
                ), row=2, col=1)

            fig.update_layout(
                template=PLOTLY_TEMPLATE, height=600,
                xaxis_rangeslider_visible=False,
                margin=dict(l=60, r=30, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Trade log table
            if len(trades) > 0:
                st.markdown("### Trade Log")
                ticker_trades = trades[trades["ticker"] == ticker] if ticker else trades
                st.dataframe(
                    ticker_trades.style.format({
                        "entry_price": "₹{:,.2f}", "exit_price": "₹{:,.2f}",
                        "gross_pnl": "₹{:,.0f}", "net_pnl": "₹{:,.0f}",
                        "costs": "₹{:,.0f}", "holding_bars": "{:d}",
                    }).applymap(
                        lambda v: "color: #3fb950" if isinstance(v, (int, float)) and v > 0
                        else ("color: #f85149" if isinstance(v, (int, float)) and v < 0 else ""),
                        subset=["net_pnl", "gross_pnl"]
                    ),
                    use_container_width=True, height=400,
                )

# ── Monte Carlo ──────────────────────────────────────────────────────
with tab_mc:
    st.markdown("## Monte Carlo Simulation")

    if selected_models:
        model = st.selectbox("Model", selected_models, key="mc_model")
        trades = load_trade_log(model)
        eq = load_equity_curve(model)

        if len(trades) == 0 or len(eq) == 0:
            st.info("No trade data. Run backtest first.")
        else:
            if st.button("Run Monte Carlo", key="mc_run"):
                with st.spinner(f"Running {mc_iterations} iterations..."):
                    from backtest.monte_carlo import trade_resampling, return_bootstrapping
                    trade_result = trade_resampling(trades, eq, mc_iterations, capital)
                    boot_result = return_bootstrapping(eq, mc_iterations)

                if trade_result:
                    st.markdown("### Trade Resampling")
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=trade_result["final_equities"], nbinsx=80,
                            marker_color="#58a6ff", opacity=0.7,
                        ))
                        pcts = trade_result["percentiles"]
                        for p, color in [(50, "#d29922"), (5, "#f85149"), (95, "#3fb950")]:
                            fig.add_vline(x=pcts[p]["equity"], line_dash="dash",
                                          line_color=color, annotation_text=f"P{p}")
                        fig.update_layout(template=PLOTLY_TEMPLATE, title="Final Equity",
                                          height=350, margin=dict(l=50, r=20, t=40, b=30))
                        st.plotly_chart(fig, use_container_width=True)

                    with c2:
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=trade_result["max_drawdowns"] * 100, nbinsx=80,
                            marker_color="#f85149", opacity=0.7,
                        ))
                        for p, color in [(50, "#d29922"), (5, "#f85149"), (95, "#3fb950")]:
                            fig.add_vline(x=pcts[p]["max_dd"] * 100, line_dash="dash",
                                          line_color=color, annotation_text=f"P{p}")
                        fig.update_layout(template=PLOTLY_TEMPLATE, title="Max Drawdown",
                                          height=350, margin=dict(l=50, r=20, t=40, b=30))
                        st.plotly_chart(fig, use_container_width=True)

                    # Percentile table
                    pct_df = pd.DataFrame([
                        {"Percentile": f"P{p}", "Final Equity": f"₹{pcts[p]['equity']:,.0f}",
                         "Max Drawdown": f"{pcts[p]['max_dd']*100:.2f}%"}
                        for p in sorted(pcts.keys())
                    ])
                    st.dataframe(pct_df, use_container_width=True, hide_index=True)

                if boot_result:
                    st.markdown("### Return Bootstrapping")
                    c1, c2, c3 = st.columns(3)
                    for col, (key, title, color) in zip(
                        [c1, c2, c3],
                        [("cagr_ci", "CAGR", "#58a6ff"), ("sharpe_ci", "Sharpe", "#3fb950"), ("maxdd_ci", "Max DD", "#f85149")],
                    ):
                        with col:
                            fig = go.Figure()
                            samples_key = f"{key.replace('_ci', '')}_samples"
                            if samples_key in boot_result:
                                fig.add_trace(go.Histogram(
                                    x=boot_result[samples_key] * (100 if "cagr" in key or "maxdd" in key else 1),
                                    nbinsx=60, marker_color=color, opacity=0.7,
                                ))
                                for p, c in [(50, "#d29922"), (5, "#f85149"), (95, "#3fb950")]:
                                    val = boot_result[key][p] * (100 if "cagr" in key or "maxdd" in key else 1)
                                    fig.add_vline(x=val, line_dash="dash", line_color=c,
                                                  annotation_text=f"P{p}: {val:.2f}")
                            fig.update_layout(template=PLOTLY_TEMPLATE, title=title,
                                              height=300, margin=dict(l=50, r=20, t=40, b=30))
                            st.plotly_chart(fig, use_container_width=True)

# ── Model Comparison ─────────────────────────────────────────────────
with tab_compare:
    st.markdown("## Model Comparison")

    comp = load_comparison()
    if len(comp) == 0:
        st.info("No comparison data. Run backtest first.")
    else:
        # Format display
        display = comp.copy()
        for col in ["total_return", "cagr", "max_drawdown", "win_rate", "cost_pct_gross_pnl"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"{x*100:.2f}%" if col != "cost_pct_gross_pnl" else f"{x:.2f}%")
        if "sharpe" in display.columns:
            display["sharpe"] = display["sharpe"].apply(lambda x: f"{x:.3f}")
        if "profit_factor" in display.columns:
            display["profit_factor"] = display["profit_factor"].apply(lambda x: f"{x:.2f}")
        for col in ["total_costs", "total_gross_pnl", "avg_win", "avg_loss", "initial_equity", "final_equity"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"₹{x:,.0f}")

        rename = {
            "total_return": "Total Return", "cagr": "CAGR", "sharpe": "Sharpe",
            "max_drawdown": "Max Drawdown", "win_rate": "Win Rate",
            "profit_factor": "Profit Factor", "n_trades": "Trades",
            "total_costs": "Total Costs", "cost_pct_gross_pnl": "Cost % P&L",
            "avg_win": "Avg Win", "avg_loss": "Avg Loss",
            "total_gross_pnl": "Gross P&L", "delta_exposure": "Avg Exposure",
            "avg_holding_bars": "Avg Hold", "max_dd_duration_days": "Max DD Duration",
        }
        display = display.rename(columns={k: v for k, v in rename.items() if k in display.columns})
        st.dataframe(display, use_container_width=True)

        # Bar chart comparison
        metrics_to_plot = ["total_return", "sharpe", "max_drawdown", "win_rate"]
        fig = make_subplots(rows=1, cols=4,
                            subplot_titles=["Total Return", "Sharpe", "Max Drawdown", "Win Rate"])
        for i, metric in enumerate(metrics_to_plot):
            if metric in comp.index or metric in comp.columns:
                vals = comp[metric].values if metric in comp.columns else []
                names = comp.index.tolist()
                for j, name in enumerate(names):
                    color = PLOTLY_COLORS.get(name, "#666")
                    fig.add_trace(go.Bar(
                        name=name.upper(), x=[name.upper()], y=[abs(vals[j]) if isinstance(vals[j], (int, float)) else 0],
                        marker_color=color, showlegend=(i == 0),
                    ), row=1, col=i+1)

        fig.update_layout(template=PLOTLY_TEMPLATE, height=400, barmode="group",
                          margin=dict(l=50, r=20, t=50, b=30))
        st.plotly_chart(fig, use_container_width=True)

# ── Mode Comparison ───────────────────────────────────────────────────
with tab_modes:
    st.markdown("## 🔀 Mode Comparison: B&H vs Model vs Allocation")

    mode_csv = REPORTS_DIR / "mode_comparison.csv"
    if not mode_csv.exists():
        st.info("Run `python backtest/mode_comparison.py` to generate mode comparison data.")
    else:
        mdf = pd.read_csv(mode_csv)
        # Rename for display
        rename_map = {
            "B&H": "Buy & Hold",
            "equal-weight": "Equal Weight",
            "conf-weighted": "Conf-Weighted",
            "top-N": "Top-N",
            "single-model": "Single (Model)",
            "single-BH": "Single (B&H)",
        }
        mdf["mode_label"] = mdf["mode"].map(rename_map).fillna(mdf["mode"])

        # Key metrics table
        st.markdown("### All Modes")
        display_cols = ["mode_label", "model", "config", "total_return", "sharpe",
                         "max_drawdown", "n_trades", "win_rate"]
        display = mdf[display_cols].copy()
        display["total_return"] = display["total_return"].apply(lambda x: f"{x*100:+.2f}%")
        display["sharpe"] = display["sharpe"].apply(lambda x: f"{x:.3f}")
        display["max_drawdown"] = display["max_drawdown"].apply(lambda x: f"{x*100:.2f}%")
        display["win_rate"] = display["win_rate"].apply(lambda x: f"{x:.1%}")
        display = display.rename(columns={
            "mode_label": "Mode", "model": "Model", "config": "Config",
            "total_return": "Return", "sharpe": "Sharpe",
            "max_drawdown": "MaxDD", "n_trades": "Trades", "win_rate": "WinRate",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)

        # Portfolio modes comparison chart
        st.markdown("### Portfolio Modes — Return & Sharpe")
        portfolio_modes = mdf[mdf["mode"].isin(["B&H", "equal-weight", "conf-weighted", "top-N"])]
        if len(portfolio_modes) > 0:
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["Total Return (%)", "Sharpe Ratio"])
            for _, row in portfolio_modes.iterrows():
                label = f"{row['mode_label']} ({row['model']})" if row["mode"] != "B&H" else "Buy & Hold"
                fig.add_trace(go.Bar(
                    name=label, x=[label], y=[row["total_return"] * 100],
                    showlegend=False,
                ), row=1, col=1)
                fig.add_trace(go.Bar(
                    name=label, x=[label], y=[row["sharpe"]],
                    showlegend=False,
                ), row=1, col=2)
            fig.update_layout(template=PLOTLY_TEMPLATE, height=400, barmode="group",
                              margin=dict(l=50, r=20, t=50, b=30))
            st.plotly_chart(fig, use_container_width=True)

        # Single-stock comparison
        st.markdown("### Single-Stock: Model vs Buy-and-Hold")
        single_data = mdf[mdf["mode"].isin(["single-model", "single-BH"])]
        if len(single_data) > 0:
            pivot = single_data.pivot_table(index="config", columns="mode",
                                             values="total_return", aggfunc="first")
            if "single-model" in pivot.columns and "single-BH" in pivot.columns:
                pivot["alpha"] = pivot["single-model"] - pivot["single-BH"]
                pivot = pivot.sort_values("alpha", ascending=False)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Model", x=pivot.index.tolist(),
                    y=(pivot["single-model"] * 100).tolist(),
                    marker_color="#58a6ff",
                ))
                fig.add_trace(go.Bar(
                    name="Buy & Hold", x=pivot.index.tolist(),
                    y=(pivot["single-BH"] * 100).tolist(),
                    marker_color="#666",
                ))
                fig.update_layout(
                    template=PLOTLY_TEMPLATE, barmode="group",
                    title="Single-Stock: Model Return vs Buy-and-Hold",
                    yaxis_title="Return (%)", height=400,
                    margin=dict(l=50, r=20, t=50, b=30),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Alpha table
                alpha_df = pd.DataFrame({
                    "Ticker": pivot.index,
                    "Model Return": (pivot["single-model"] * 100).apply(lambda x: f"{x:+.2f}%"),
                    "B&H Return": (pivot["single-BH"] * 100).apply(lambda x: f"{x:+.2f}%"),
                    "Alpha (pp)": (pivot["alpha"] * 100).apply(lambda x: f"{x:+.1f}pp"),
                })
                st.dataframe(alpha_df, use_container_width=True, hide_index=True)

# ── Live Paper Trading ───────────────────────────────────────────────
with tab_live:
    st.markdown("## 🔴 Live Paper Trading")
    st.warning("⚠️ PAPER TRADING ONLY — No real money at risk. All fills are simulated.")

    live_state = load_live_state()
    latest_bar = load_latest_bar()

    if live_state is None:
        st.info("No live state found. Start the live loop: `python backtest/run_live.py --mode replay`")
    else:
        # Status bar
        if latest_bar:
            source = latest_bar.get("source", "unknown")
            status = latest_bar.get("market_status", "unknown")
            c1, c2, c3 = st.columns(3)
            c1.metric("Last Bar", f"{latest_bar['ticker']} @ ₹{latest_bar['close']:,.2f}")
            c2.metric("Source", source.upper())
            c3.metric("Market", status)

        st.markdown("---")

        # Per-model live metrics
        cols = st.columns(len(live_state))
        for i, (model, data) in enumerate(live_state.items()):
            with cols[i]:
                st.markdown(f"### {model.upper()}")
                st.metric("Equity", f"₹{data['equity']:,.0f}")
                st.metric("Cash", f"₹{data['cash']:,.0f}")
                st.metric("Open Positions", data["n_positions"])
                st.metric("Trades", data["total_trades"])

        # Live equity curves
        st.markdown("### Live Equity Curve")
        fig = go.Figure()
        for model in live_state:
            eq_path = LIVE_STATE_DIR / model / "equity_curve.parquet"
            if eq_path.exists():
                eq = pd.read_parquet(eq_path)
                eq["timestamp"] = pd.to_datetime(eq["timestamp"])
                fig.add_trace(go.Scatter(
                    x=eq["timestamp"], y=eq["equity"],
                    name=model.upper(), line=dict(color=PLOTLY_COLORS.get(model, "#666"), width=2),
                ))
        fig.update_layout(template=PLOTLY_TEMPLATE, hovermode="x unified",
                          yaxis_title="Equity (₹)", height=400,
                          margin=dict(l=60, r=30, t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

        # Open positions
        st.markdown("### Open Positions")
        for model in live_state:
            pos_path = LIVE_STATE_DIR / model / "open_positions.json"
            if pos_path.exists():
                with open(pos_path) as f:
                    positions = json.load(f)
                if positions:
                    st.markdown(f"**{model.upper()}**")
                    st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)

        # Auto-refresh
        if mode == "Live Paper Trading":
            time.sleep(refresh_rate)
            st.rerun()
