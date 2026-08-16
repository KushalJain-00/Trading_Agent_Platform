# Walk-Forward Testing

## What It Is

**Walk-forward testing** splits the validation period into multiple
sub-periods, runs the same strategy on each, and reports results per-period.
It's a stronger test than a single backtest because it reveals whether the
strategy works consistently across different market conditions (regimes).

The key rule: **the config must be fixed before running walk-forward.**
No per-period re-optimization — that would be overfitting.

## Why It Matters for This Project

A single backtest on the full validation period (2022-10 to 2026-02) could be
inflated by a few lucky periods. Walk-forward asks: "Does this strategy work
in Q1 2023 AND Q2-Q3 2023 AND Q4'23-Q1'24 AND Q2-Q3'24?"

**This project's discovery:** The walk-forward test revealed that the strategy
**loses money in Q1 2023** (-8% to -9%) across all three models, while
profitable in all other periods (+18% to +45%). This means:

1. The edge is real — it works in 3 out of 4 independent periods
2. But it's not universal — there's at least one regime where it fails
3. The losing period is consistent across models, suggesting a market regime
   the models didn't learn

This is honest, valuable information that a single-backtest metric would hide.

## How It's Implemented Here

In `backtest/walk_forward.py`:

### Period Design

The validation data has uneven coverage across time:
- 2022-Q4: Only 7-20 tickers (dataset ramp-up)
- 2023-2024-Q3: 85-103 tickers (full coverage)
- 2024-Q4 onward: Drops to <21 tickers (data thins out)

The walk-forward periods are designed around the full-coverage window:

```python
PERIODS = {
    "P1: 2023-Q1 (Jan-Mar)": ("2023-01-01", "2023-04-01"),
    "P2: 2023-Q2-Q3 (Apr-Sep)": ("2023-04-01", "2023-10-01"),
    "P3: 2023-Q4-2024-Q1 (Oct-Mar)": ("2023-10-01", "2024-04-01"),
    "P4: 2024-Q2-Q3 (Apr-Sep)": ("2024-04-01", "2024-10-01"),
}
```

Periods after 2024-09 are excluded because <21 tickers is too thin for a
meaningful backtest.

### Fixed Config

```python
CONFIDENCE_THRESHOLD = 0.85
MIN_HOLDING_BARS = 75
```

These were selected from the full sweep BEFORE running walk-forward. The
walk-forward tests whether this config generalizes.

### Results

| Period | LSTM | CNN1D | CNN-LSTM |
|--------|------|-------|----------|
| P1: Q1'23 | **-8.31%** | **-9.05%** | **-8.74%** |
| P2: Q2-Q3'23 | +33.22% | +33.68% | +34.81% |
| P3: Q4'23-Q1'24 | +43.91% | +44.80% | +45.05% |
| P4: Q2-Q3'24 | +20.63% | +18.43% | +19.26% |
| **Average** | **+22.36%** | **+21.96%** | **+22.59%** |

**Verdict:** 3/4 periods profitable, average return +22%.

### How to Interpret a Losing Period

A strategy that loses in one period isn't necessarily broken. Possible
explanations:

1. **Regime dependence:** The market had conditions the models didn't see in
   training (e.g., specific volatility regime, sector rotation)
2. **Statistical noise:** With ~13K-14K trades per period, the result is
   statistically significant, so this isn't just noise
3. **Model limitation:** The models may not have learned certain patterns
   present in Q1 2023

The fact that ALL THREE models lose in P1 (not just one) suggests the issue is
in the data or market regime, not the model architecture.

## Tunable Parameters

| Parameter | What It Does | How to Change It |
|-----------|--------------|-----------------|
| Period boundaries | How you split the validation data | Choose periods with enough data (>=50 tickers) |
| Config selection | Which config to test | Must be fixed BEFORE running walk-forward |
| Excluded periods | Periods with too little data | Exclude periods with <20 tickers |

## Common Pitfalls

1. **Re-optimizing per period.** If you pick a different config for each period,
   you've overfitted to each period's data. The config must be fixed.

2. **Using periods with too little data.** P1 has ~2M bars and 89 tickers.
   A period with 5 tickers and 100K bars would give unreliable results.

3. **Ignoring the losing period.** Reporting only the average (+22%) without
   mentioning the -9% period hides important risk information.

4. **Confusing walk-forward with cross-validation.** Walk-forward is temporal —
   periods are sequential in time. K-fold cross-validation shuffles data
   randomly, which is wrong for time series (it leaks future information into
   past folds).
