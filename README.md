# Regime-Aware Trading Intelligence Platform
*(working title — rename freely)*

[![Status](https://img.shields.io/badge/status-in--development-yellow)]()
[![Timeline](https://img.shields.io/badge/timeline-2.5%20weeks-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

A trading research platform built for the Trading Arena hackathon. Instead of demoing six independent models, everything is fused into one explainable decision engine — a **Regime-Aware Strategy Router** — that decides, at any point in time, which signals to trust and how much risk to take.

This document is the shared knowledge base for the group. Everyone building any component should read this before writing code — it defines what data looks like at every handoff point, so five people's work actually fits together instead of needing to be glued at the end.

> **Disclaimer:** This is a research/education tool. It surfaces analytics and recommendations, not personalized investment advice, and does not auto-execute trades without explicit user confirmation. Not registered with SEBI or any regulator. Built for demonstration purposes only.

---

## Table of contents

- [Problem statement](#problem-statement)
- [The core idea](#the-core-idea)
- [Architecture](#architecture)
- [The data contract](#the-data-contract)
- [Components](#components)
- [What's original here](#whats-original-here)
- [Tech stack](#tech-stack)
- [Repo structure](#repo-structure)
- [Team](#team)
- [License](#license)

---

## Problem statement

Selected from the Trading Arena problem set: Stock Price Forecasting (ARIMA), Volatility Prediction (GARCH), Portfolio Optimization (MPT/efficient frontier), News Sentiment Analysis (NLP), Pairs Trading (cointegration), and Trading Journal — plus a Backtesting Engine and an original fusion layer that none of the individual problems require on their own.

The individual problems are well-trodden; the gap we're targeting is that none of them talk to each other in a typical submission. Our thesis: a system that knows *when* to trust which signal is more valuable than five isolated models that are each right some of the time and wrong the rest, with no way to tell which is which.

## The core idea

```
 Market data          News headlines
      │                     │
      ▼                     ▼
┌───────────┐  ┌──────────────┐  ┌───────────┐  ┌──────────────┐
│Forecasting│  │ Vol & regime │  │ Sentiment │  │ Pairs signal │
│  (ARIMA)  │  │ (GARCH+HMM)  │  │ (FinBERT) │  │(cointegration)│
└─────┬─────┘  └──────┬───────┘  └─────┬─────┘  └──────┬───────┘
      └───────────────┴────────┬────────┴───────────────┘
                                ▼
                 ┌───────────────────────────┐
                 │ Regime-Aware Strategy      │  ◄── our original piece
                 │ Router                     │
                 └──────────────┬─────────────┘
                                 ▼
              ┌──────────────────┴──────────────────┐
              ▼                                       ▼
   ┌────────────────────┐                  ┌─────────────────────┐
   │ Portfolio optimizer │                  │  Trading journal +   │
   │ (Black-Litterman)   │                  │  AI trade coach      │
   └──────────┬──────────┘                  └──────────┬───────────┘
              └───────────────────┬────────────────────┘
                                   ▼
                        ┌────────────────────┐
                        │ Backtest engine /   │
                        │ Live execution gate │
                        └────────────────────┘
```

## Architecture

Two rules govern everything, because this is a multi-person build and data moves between people, not just between functions:

1. **Single source of truth for raw data.** Every model reads from the same raw OHLCV/news pull. If two people pull data independently, or at different times, the router and the backtest will silently disagree on what actually happened on a given day — this is the single most common way a team project like this breaks at integration time.
2. **Everyone's output speaks the same language.** Whoever builds forecasting, regime detection, volatility, sentiment, or pairs never hands their teammates a custom format. Everyone writes to the same schema below. Whoever builds the router never needs to know or care whose model produced a given signal.

## The data contract

This is the part every team member needs to know cold — it's what lets your component connect to everyone else's without a meeting. Any model (forecasting, regime, volatility, sentiment, pairs) outputs a record in this shape:

| Field | Type | Description |
|---|---|---|
| `ticker` | string | Instrument the signal applies to (or pair, e.g. `"RELIANCE-TCS"`) |
| `timestamp` | date | As-of date for the signal, always using data available up to `t-1` — never leak future information |
| `signal_type` | enum | `forecast` \| `regime` \| `volatility` \| `sentiment` \| `pairs` |
| `value` | float | Signal's core output (e.g. forecast return, regime label, vol estimate, sentiment score, spread z-score) |
| `confidence` | float [0,1] | Model's own confidence — regime probability, forecast interval width, sentiment softmax score, etc. |
| `metadata` | dict | Signal-specific extras (e.g. regime name, forecast horizon, headline count) |

The router consumes a batch of these per ticker per day and hands back:

| Field | Type | Description |
|---|---|---|
| `target_weight` | float | Recommended portfolio weight for the ticker |
| `dominant_signal` | string | Which signal drove the decision, for journal/explainability |
| `regime_context` | dict | Active regime, its probability, and the weight table used |
| `reasoning` | string | Human-readable explanation, logged to the journal |

Whoever owns the backtest engine and whoever owns the dashboard both consume this same router output — so if the router's shape changes, that's a conversation the whole group has, not a silent breaking change.

## Components

| # | Component | Method | Consumes | Produces (schema `signal_type`) |
|---|---|---|---|---|
| 1 | Stock price forecasting | ARIMA/SARIMA, pooled across Nifty 50 | Raw OHLCV | `forecast` |
| 2 | Regime detection | HMM (3 states: trending / range-bound / crisis) | Returns, realized vol, India VIX | `regime` |
| 3 | Volatility prediction | GARCH(1,1)/EGARCH | Daily returns | `volatility` |
| 4 | Backtesting engine | Vectorized, pluggable strategy interface | Any strategy implementing `strategy(data, positions) → target_weights` | Performance report |
| 5 | Portfolio optimizer | Black-Litterman | Router's fused signal as views, regime confidence × inverse GARCH vol as view confidence | Portfolio weights |
| 6 | Market sentiment indicator | FinBERT on daily headlines | News headlines per ticker | `sentiment` |
| 7 | Algo trading | Semi-automated, broker API, manual confirm gate | Router recommendation | Execution log |
| 8 | Unified indicator system | The schema above, exposed as an internal API | All model outputs | Standardized feed for dashboard/journal/router |
| 9 | Quality-of-life features | *(to be finalized by the group)* | — | — |

## What's original here

- **The router itself** — regime-conditioned signal weighting, confidence scaling by regime probability, volatility-targeting overlay, and a hard exposure cap in crisis regimes. Every decision is traceable to a specific regime call and signal weight — not a black box.
- **Sentiment feeding Black-Litterman as views** — sentiment directly moves portfolio weights through a principled optimizer rather than sitting as a separate, disconnected dashboard tile.
- **AI trade coach** — the journal logs the regime/sentiment/confidence context behind each trade and generates plain-English critiques, not just a P&L log.
- **Explainability as regulatory strategy, not just UX** — white-box design keeps the system clear of India's stricter registration requirements that apply to opaque ("black-box") algo strategies, which matters if this goes beyond the hackathon.

## Tech stack

| Layer | Choice |
|---|---|
| Data | `yfinance` (Nifty 50 + `^NSEI` + `^INDIAVIX`), news API for headlines |
| Forecasting | `statsmodels` / `pmdarima` |
| Volatility | `arch` (GARCH/EGARCH) |
| Regime | `hmmlearn` |
| Sentiment | `transformers` + FinBERT |
| Pairs | `statsmodels.tsa.stattools.coint` (Engle-Granger) |
| Portfolio optimization | `PyPortfolioOpt` (Black-Litterman) |
| Backtest | custom vectorized engine |
| Backend | FastAPI |
| Frontend | Streamlit |
| Storage | SQLite |
| Broker integration | Kite Connect / Upstox sandbox |

## Repo structure

```
├── data/
│   ├── raw/                  # single source of truth, untouched pulls
│   └── processed/            # per-model feature sets, all derived from raw/
├── models/
│   ├── forecasting.py
│   ├── regime.py
│   ├── volatility.py
│   ├── sentiment.py
│   └── pairs.py
├── router/
│   └── strategy_router.py
├── portfolio/
│   └── optimizer.py
├── backtest/
│   └── engine.py
├── journal/
│   └── journal.py
├── dashboard/
│   └── app.py
├── PLAN.md
└── README.md
```

## Team

See the tracker and handoff map in [`PLAN.md`](./PLAN.md) for component ownership, current status, and who delivers what to whom.

## License

MIT — for hackathon/educational use. Not licensed or intended for live capital deployment as-is.
# Trading_Agent_Platform
