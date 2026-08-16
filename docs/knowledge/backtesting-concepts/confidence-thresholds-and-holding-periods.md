# Confidence Thresholds and Holding Periods

## What They Are

**Confidence threshold** filters weak predictions. If the model predicts "Buy"
with only 55% confidence, that's barely better than random — acting on it adds
noise. Setting a confidence threshold of 0.85 means: only act on Buy signals
where the model is ≥85% confident.

**Minimum holding period** prevents rapid open/close cycling. Without it, the
model might say "Buy" at bar 100, "Sell" at bar 101, "Buy" at bar 102 —
generating 3 trades in 3 bars, each incurring full transaction costs. The
holding period forces the strategy to stay in a position for at least N bars.

## Why It Matters for This Project

These two filters are the difference between -59% and +85% return. They
transform the raw model output into a profitable strategy by addressing two
fundamental problems:

### Problem 1: Too many trades (cost problem)

Without filtering, the models generate ~75,000 trades across the validation
period. Each trade costs ~₹129,000 in transaction costs + spread. Total costs:
₹9.7 billion — **251% of gross P&L**. The strategy makes money on gross trades
but loses it all (and more) to costs.

### Problem 2: Weak signals (noise problem)

The model's Buy predictions include many low-confidence ones (55-65%). These
are essentially random — the model isn't sure. Acting on them adds noise
without adding signal.

## The Before/After Numbers

### No filtering (confidence=0, holding=10)

```
Trades: 75,001
Win rate: 45.3%
Cost % of gross P&L: 251%
Return: -58.5%
Sharpe: -0.95
```

### With filtering (confidence=0.85, holding=75)

```
Trades: 13,269 (82% reduction)
Win rate: 52.8%
Cost % of gross P&L: 16%
Return: +85.1%
Sharpe: 1.45
```

The filtering:
1. Reduced trades by 82% (75K → 13K)
2. Improved win rate by 7.5 percentage points (45.3% → 52.8%)
3. Reduced cost ratio from 251% to 16%
4. Turned a -59% losing strategy into an +85% winner

## How It's Implemented Here

**Confidence filter** — in `simulator.py`, `_run_backtest_core()`:
```python
if confidence_threshold > 0.0:
    weak_buy = (merged["predicted_signal"] == "Buy") & \
               (merged["predicted_confidence"] < confidence_threshold)
    merged.loc[weak_buy, "predicted_signal"] = "Hold"
```
Weak Buy signals (below threshold) become Hold — they're simply ignored.

**Holding period filter** — in `simulator.py`, `_run_backtest_core()`:
```python
if min_holding_bars > 1:
    pos = merged["position"].values.copy()
    for i in range(len(pos)):
        t = tickers[i]
        if pos[i] == 1 and not in_trade.get(t, False):
            in_trade[t] = True
            entry_bar[t] = i
        elif pos[i] == 0 and in_trade.get(t, False):
            if i - entry_bar[t] < min_holding_bars:
                pos[i] = 1  # suppress exit, stay in trade
            else:
                in_trade[t] = False
```
If the model says "Sell" before the minimum holding period expires, the exit
is suppressed — the position stays open.

## Tunable Parameters

| Parameter | Default | Range to Explore | Tradeoff |
|-----------|---------|-----------------|----------|
| Confidence threshold | 0.85 | 0.65-0.95 | Higher = fewer but higher-quality trades. Lower = more trades, more costs |
| Min holding bars | 75 | 10-100 | Higher = fewer trades, longer holds, less reactive. Lower = more responsive, more costs |

### What happens at each confidence level (hold=75):

| Confidence | Trades | Win Rate | Cost% | Return |
|------------|--------|----------|-------|--------|
| 0.65 | 13,347-13,414 | 52.9-53.2% | 16.6-16.8% | +83-84% |
| 0.75 | 13,301-13,375 | 53.0-53.1% | 16.2-16.6% | +83-85% |
| 0.80 | 13,268-13,345 | 53.2-53.4% | 16.1-16.7% | +83-86% |
| 0.85 | 13,220-13,293 | 52.8-53.2% | 16.2-16.4% | +84-85% |
| 0.90 | 13,163-13,218 | 53.1-53.2% | 15.8-16.4% | +83-87% |

The differences between confidence levels are small at hold=75 — the holding
period does most of the heavy lifting.

### What happens at each holding period (conf=0.85):

| Holding | Trades | Win Rate | Cost% | Return |
|---------|--------|----------|-------|--------|
| 10 | 70,140-72,312 | 45.5% | 215-219% | -49% to -51% |
| 30 | 30,114-30,506 | 49.8-50.0% | 44-45% | +47% to +48% |
| 50 | 19,209-19,375 | 51.1-51.4% | 25-26% | +69% to +71% |
| 75 | 13,220-13,293 | 52.8-53.2% | 16-16.4% | +84% to +85% |

Holding period is the dominant factor. Going from hold=10 to hold=75:
- Trades drop 82%
- Win rate improves 7.5 points
- Cost ratio drops from 219% to 16%
- Return swings from -51% to +85%

## Common Pitfalls

1. **Not filtering at all.** The default (confidence=0, holding=1) produces
   -59% return. Always use filters.

2. **Filtering Sell signals.** Our implementation only filters Buy signals.
   Sell signals are less frequent and higher quality — filtering them would
   reduce the edge.

3. **Tuning filters on the same data you test on.** If you sweep 60 configs
   and pick the best, you've overfitted to the validation set. The walk-forward
   test is the honest check.

4. **Confusing holding period with rebalancing.** The holding period doesn't
   force rebalancing at 75 bars — it just prevents premature exits. Positions
   can stay open longer.
