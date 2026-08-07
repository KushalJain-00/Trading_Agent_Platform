# Build Plan & Project Tracker

**Total timeline:** 2.5 weeks · **Demo checkpoint:** end of Day 9 · **Final presentation:** end of Week 2.5

This document is the team's single source of truth for scope, ownership, status, and — since multiple people's work has to fit together — exactly what gets handed off to whom, in what shape. Update the tracker tables directly as tasks move; don't let status live only in chat.

---

## Table of contents

- [Status legend](#status-legend)
- [Timeline overview](#timeline-overview)
- [Handoff map](#handoff-map)
- [Tier 0 — Core system tracker](#tier-0--core-system-days-19-non-negotiable)
- [Tier 1 — Fundability layer tracker](#tier-1--fundability-layer-days-1014)
- [Tier 2 — Stretch tracker](#tier-2--stretch-only-if-ahead-of-schedule-by-day-14)
- [Buffer & rehearsal](#days-1517--buffer)
- [Team assignment](#team-assignment)
- [Risk register](#risk-register)
- [Milestones](#milestones)
- [Pitch reminders](#pitch-reminders)

---

## Status legend

| Symbol | Meaning |
|---|---|
| 🔴 | Not started |
| 🟡 | In progress |
| 🟢 | Done |
| ⚪ | Blocked — see notes |

Priority: **P0** = blocks the demo · **P1** = important, not blocking · **P2** = nice to have

---

## Timeline overview

| Phase | Days | Focus | Checkpoint |
|---|---|---|---|
| Phase 1 | 1–3 | Raw data + standalone signal models | Each model outputs a valid schema record |
| Phase 2 | 4–6 | Router + portfolio optimizer | One call: tickers in → weights + reasoning out |
| Phase 3 | 7–8 | Backtest engine + journal | Reproducible performance report, any strategy |
| Phase 4 | 9 | Dashboard | **Demo-ready checkpoint** |
| Phase 5 | 10–14 | Fundability layer | Live broker sync, AI coach, robustness proof |
| Phase 6 | 15–17 | Buffer + rehearsal | Nothing new — fix, polish, practice |

---

## Handoff map

The most important thing for a team build: knowing exactly what you owe the next person, and in what shape. Every row below is a real handoff — treat a broken or late one as blocking whoever's downstream.

| From (owner) | Delivers | Format | To (owner) |
|---|---|---|---|
| Data owner | Raw OHLCV + India VIX, one shared file/table | Signal schema's underlying raw dataset — same rows, same dates, for everyone | Forecasting, volatility, regime, pairs owners |
| Data owner | Raw news headlines | Ticker-tagged headline set | Sentiment owner |
| Forecasting owner | Forecast signal | Schema record, `signal_type: forecast` | Router owner |
| Volatility owner | Vol forecast | Schema record, `signal_type: volatility` | Regime owner, router owner, portfolio optimizer owner |
| Regime owner | Regime label + probability | Schema record, `signal_type: regime` | Router owner |
| Sentiment owner | Sentiment score | Schema record, `signal_type: sentiment` | Router owner |
| Pairs owner | Spread z-score / trade signal | Schema record, `signal_type: pairs` | Router owner |
| Router owner | Target weights + reasoning | Router output shape (see README) | Portfolio optimizer, backtest engine, journal, dashboard owners |
| Portfolio optimizer owner | Final portfolio weights | Weight vector per ticker | Backtest engine, algo trading/broker owner |
| Backtest engine owner | Performance report | Standard report shape, reusable for any strategy | Dashboard owner, journal owner |
| Journal owner | Logged trade + context | Trade record w/ regime/sentiment/confidence attached | Dashboard owner, AI coach |

Anyone changing the shape of what they hand off flags it to the group before merging — a silent schema change upstream breaks everyone downstream without an obvious error.

---

## Tier 0 — Core system (Days 1–9, non-negotiable)

### Phase 1 (Days 1–3): Raw data + individual signals

| ID | Task | Owner | Priority | Status | Depends on | Notes |
|---|---|---|---|---|---|---|
| 1.1 | Shared raw OHLCV + India VIX dataset | | P0 | 🔴 | — | Single source of truth — every model reads this, nobody pulls their own copy |
| 1.2 | Shared raw news headline dataset | | P0 | 🔴 | — | Feeds sentiment only, kept separate from price data |
| 1.3 | Forecasting model, outputs to schema | | P0 | 🔴 | 1.1 | Pooled across tickers, not per-stock |
| 1.4 | Volatility model, outputs to schema | | P0 | 🔴 | 1.1 | |
| 1.5 | Regime detection model, outputs to schema | | P0 | 🔴 | 1.1, 1.4 | 3 states: trending / range-bound / crisis |
| 1.6 | Sentiment model, outputs to schema | | P0 | 🔴 | 1.2 | |
| 1.7 | Pairs signal, outputs to schema | | P1 | 🔴 | 1.1 | Cointegration across the ticker universe |

**Phase 1 definition of done:** each model runs independently and emits a valid signal-schema record — no dashboard, no fusion yet.

### Phase 2 (Days 4–6): Router + portfolio optimizer

| ID | Task | Owner | Priority | Status | Depends on | Notes |
|---|---|---|---|---|---|---|
| 2.1 | Regime → signal weight table | | P0 | 🔴 | 1.5 | Starting weights already drafted — see README |
| 2.2 | Confidence scaling by regime probability | | P0 | 🔴 | 2.1 | Not just the hard label |
| 2.3 | Volatility-targeting position sizing | | P0 | 🔴 | 1.4 | |
| 2.4 | Hard exposure cap in crisis regime | | P0 | 🔴 | 2.1 | Rule, not learned — the circuit breaker |
| 2.5 | Router integration — one callable combining 2.1–2.4 | | P0 | 🔴 | 2.1–2.4, all Phase 1 signals | Input: tickers. Output: weights + reasoning string |
| 2.6 | Black-Litterman optimizer, router output as views | | P0 | 🔴 | 2.5 | |
| 2.7 | View confidence = regime prob × inverse vol | | P0 | 🔴 | 2.6 | |

**Phase 2 definition of done:** one function call takes tickers in, returns target portfolio weights plus a human-readable reason.

### Phase 3 (Days 7–8): Backtest engine + journal

| ID | Task | Owner | Priority | Status | Depends on | Notes |
|---|---|---|---|---|---|---|
| 3.1 | Backtest engine core — pluggable strategy interface | | P0 | 🔴 | — | Can start in parallel with Phase 1/2 |
| 3.2 | No-look-ahead enforcement (structural) | | P0 | 🔴 | 3.1 | |
| 3.3 | Transaction cost / slippage modeling | | P0 | 🔴 | 3.1 | |
| 3.4 | Portfolio accounting | | P0 | 🔴 | 3.1 | Pairs sleeve tracked separately (market-neutral) |
| 3.5 | Performance report generator | | P0 | 🔴 | 3.4 | Sharpe, Sortino, drawdown, win rate, turnover |
| 3.6 | Regime-conditioned performance breakdown | | P0 | 🔴 | 3.5, 1.5 | Key differentiator chart |
| 3.7 | Backtest run: buy-and-hold baseline | | P0 | 🔴 | 3.1–3.5 | |
| 3.8 | Backtest run: each standalone signal | | P1 | 🔴 | 3.1–3.5 | Comparison charts |
| 3.9 | Backtest run: fused router | | P0 | 🔴 | 2.5, 3.1–3.5 | The headline result |
| 3.10 | Trading journal — log context per trade | | P0 | 🔴 | 3.9 | Regime, sentiment, confidence attached to each record |
| 3.11 | Journal metrics | | P1 | 🔴 | 3.10 | |

**Phase 3 definition of done:** one report format, reusable across any strategy passed into the engine.

### Phase 4 (Day 9): Dashboard

| ID | Task | Owner | Priority | Status | Depends on | Notes |
|---|---|---|---|---|---|---|
| 4.1 | Dashboard scaffold + navigation | | P0 | 🔴 | — | |
| 4.2 | Per-model pages | | P0 | 🔴 | Phase 1 | Forecast, vol/regime, sentiment, pairs |
| 4.3 | Unified "router decision" page | | P0 | 🔴 | Phase 2 | The page judges remember |
| 4.4 | Backtest results page | | P0 | 🔴 | Phase 3 | Equity curve vs. baseline, front and center |
| 4.5 | Journal page | | P1 | 🔴 | 3.10 | |

**🎯 Phase 4 is the demo checkpoint — if nothing else gets built, this is presentable.**

---

## Tier 1 — Fundability layer (Days 10–14)

| ID | Task | Owner | Priority | Status | Notes |
|---|---|---|---|---|---|
| 5.1 | Broker read-only sync (real portfolio) | | P1 | 🔴 | |
| 5.2 | Run router live against real portfolio during demo | | P1 | 🔴 | Depends on 5.1 |
| 5.3 | Manual confirm-before-execute gate | | P1 | 🔴 | No auto-trading |
| 5.4 | AI trade coach — summary critique per backtest run | | P1 | 🔴 | Tied to regime context in journal |
| 5.5 | Two-period robustness backtest (calm vs. volatile) | | P1 | 🔴 | Strongest pitch chart |
| 5.6 | VaR panel | | P2 | 🔴 | |
| 5.7 | RSI/MACD as router inputs + baseline comparison | | P2 | 🔴 | |

## Tier 2 — Stretch (only if ahead of schedule by Day 14)

| ID | Task | Owner | Priority | Status | Notes |
|---|---|---|---|---|---|
| 6.1 | Dynamic hedge ratio for pairs | | P2 | 🔴 | Upgrade from static |
| 6.2 | Walk-forward retraining | | P2 | 🔴 | Shows adaptation, not a fixed fit |
| 6.3 | Public deployment | | P2 | 🔴 | Judges can click through independently |
| 6.4 | Recorded 90-second demo video | | P1 | 🔴 | Live-demo insurance |
| 6.5 | Quality-of-life features (finalize as a group) | | P2 | 🔴 | Watchlists / alerts / exports / saved configs |

## Days 15–17 — Buffer

No new features. Fix what broke, re-run the robustness backtest if anything changed, rehearse the pitch, tighten the story.

---

## Team assignment

| Component | Owner | Status | Contact |
|---|---|---|---|
| Raw data (shared) | | 🔴 | |
| Forecasting | | 🔴 | |
| Regime detection | | 🔴 | |
| Volatility | | 🔴 | |
| Sentiment | | 🔴 | |
| Pairs signal | | 🔴 | |
| Router | | 🔴 | |
| Portfolio optimizer | | 🔴 | |
| Backtest engine | | 🔴 | |
| Journal + AI coach | | 🔴 | |
| Dashboard | | 🔴 | |
| Broker integration | | 🔴 | |
| Pitch deck | | 🔴 | |

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| News data historical depth too short for backtesting sentiment | High | Medium | Confirm depth on Day 1 before building around it |
| Underlying price data source breaks or changes silently | Medium | High | Cache raw pulls once fetched; don't re-pull mid-sprint |
| Regimes don't align with intuition / too noisy | Medium | High | Validate visually against known events before trusting them in the router |
| Router tuned on the same data it's backtested on (overfitting) | Medium | High | Hold out a final period untouched until the robustness check (task 5.5) |
| Broker sandbox access delayed | Medium | Medium | Request access on Day 1, don't wait until Tier 1 |
| Schema mismatches between team members' models | High | Medium | Enforce the shared schema strictly — reject any output that doesn't conform |
| Running out of time before Tier 0 is demo-ready | Medium | Critical | Tier 0 scope is fixed and P0-prioritized; cut Tier 1/2, never Tier 0 |
| Live demo failure during presentation | Low | High | Recorded backup video (task 6.4) |

---

## Milestones

- [ ] **Day 3:** All standalone signals producing valid schema output
- [ ] **Day 6:** Router + portfolio optimizer callable end-to-end
- [ ] **Day 8:** Backtest report generated for baseline, standalone signals, and fused router
- [ ] **Day 9:** 🎯 Demo-ready dashboard — full Tier 0 checkpoint
- [ ] **Day 14:** Fundability layer complete
- [ ] **Day 17:** Final rehearsal complete, pitch locked

---

## Pitch reminders

- Lead with the router, not the individual models
- Show the backtest curve early
- Highlight the two-independent-risk-brakes story (regime de-risking + vol targeting agreeing with each other)
- Explainability isn't just a feature — it's the regulatory strategy (white-box vs. black-box algo classification)
- Close on the robustness check: the router behaving sensibly in a known bad period is more convincing than any single accuracy number
