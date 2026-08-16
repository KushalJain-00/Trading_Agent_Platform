# Portfolio Delta

## What It Is

**Delta** in this project's context is the **net directional exposure** of the
portfolio — how much of the portfolio's equity is invested in long positions at
any given time.

This is **NOT** options Greeks delta (which measures an option's price
sensitivity to the underlying). In this project, delta is a simple metric:

```
delta = (total value of open long positions) / total equity
```

A delta of 0.5 means 50% of equity is invested. A delta of 0 means the
portfolio is fully in cash. A delta of 1.0 means 100% invested.

Since this project only takes long positions (buy to enter, sell to exit),
delta is always between 0 and 1 (or slightly above 1 with leverage).

## Why It Matters for This Project

Delta tells you how "active" the portfolio is. Key observations from this
project:

- With no filtering (confidence=0, holding=1), delta is very high — the
  portfolio is almost always invested in many positions simultaneously
- With filtering (confidence=0.85, holding=75), delta is lower — fewer
  positions, held longer

Higher delta means:
- More capital at work (higher potential returns)
- More exposure to market-wide moves (higher drawdown risk)
- Higher transaction costs (more entries and exits)

## How It's Implemented Here

The delta computation is in `analytics.py` (line ~97):

```python
if "n_positions" in eq.columns and len(eq) > 0:
    avg_exposure = eq["n_positions"].mean()
else:
    avg_exposure = 0.0
```

Currently, the equity curve DataFrame doesn't include an `n_positions` column,
so delta reports as 0.0 in the metrics. The metric is defined but not actively
populated.

To compute delta properly, you would need to track the number of open positions
at each bar in the equity curve — something the portfolio ledger in
`simulator.py` already has the data for (the `positions` dictionary tracks
open positions).

## Tunable Parameters

Delta is an output metric, not an input parameter. But it's influenced by:

| Parameter | Effect on Delta |
|-----------|----------------|
| Confidence threshold ↑ | Lower delta (fewer trades) |
| Holding period ↑ | Moderate delta (fewer entries, longer holds) |
| Position size ↑ | Higher dollar exposure per position |
| Number of tickers ↑ | Higher delta (more positions possible) |

## Common Pitfalls

1. **Confusing with options delta.** This project's delta is directional
   exposure, not options sensitivity. The `analytics.py` docstring explicitly
   notes this.

2. **Ignoring delta when comparing strategies.** A strategy with 90% return
   but 95% delta is riskier than one with 85% return but 60% delta. Always
   consider risk-adjusted returns (Sharpe ratio) alongside raw returns.

3. **Assuming constant delta.** Delta changes bar by bar as positions open and
   close. The average delta over the backtest period is a summary, not a
   constant.
