# Performance Metrics Explained

## What Each Metric Means

### Total Return

**What it is:** The percentage change in portfolio value from start to finish.

**Formula:** `(final_equity / initial_equity) - 1`

**Example:** Starting with ₹100M, ending with ₹185M → +85% return

**What's "good":** Depends on timeframe. +85% over ~2.5 years is excellent
(~30% annualized). But returns must be evaluated alongside risk (drawdown).

### CAGR (Compound Annual Growth Return)

**What it is:** The annualized return rate — what you'd get if the return
grew at a steady rate every year.

**Formula:** `(1 + total_return) ^ (1 / years) - 1`

**Example:** +85% over 2.5 years → CAGR ≈ +13.4%

**What's "good":** 10-15% CAGR is strong for a systematic strategy. 20%+ is
exceptional but may indicate overfitting.

### Sharpe Ratio

**What it is:** A measure of risk-adjusted return — how much return you get
per unit of volatility (risk). It penalizes strategies that are volatile even
if profitable.

**Formula:** `(mean_daily_return - risk_free_rate) / std_daily_return × √252`

The `√252` annualizes the result (252 trading days per year).

**Example:** Sharpe of 1.45 means you get 1.45 units of return per unit of
volatility. This is a good Sharpe for a systematic strategy.

**What's "good":**
- < 0.5: Poor — return doesn't compensate for risk
- 0.5-1.0: Okay — reasonable risk-adjusted return
- 1.0-2.0: Good — strong risk-adjusted return
- \> 2.0: Excellent — but may indicate overfitting

Our models achieve Sharpe of 1.4-1.5 at the best config, which is strong.

### Maximum Drawdown (MaxDD)

**What it is:** The largest peak-to-trough decline in equity. It measures the
worst-case loss if you'd bought at the peak and held through the bottom.

**Formula:** `min((equity - cummax(equity)) / cummax(equity))`

**Example:** MaxDD of -17.9% means at the worst point, the portfolio lost
17.9% from its peak. A ₹100M portfolio would have dropped to ₹82.1M.

**What's "good":**
- < -10%: Excellent — very controlled risk
- -10% to -20%: Good — typical for diversified strategies
- -20% to -30%: Moderate — significant but recoverable
- \> -30%: Concerning — may indicate concentrated risk

Our MaxDD of -17.9% is reasonable for a multi-stock long-only strategy.

### Win Rate

**What it is:** The percentage of trades that are profitable.

**Formula:** `count(trades with net_pnl > 0) / total_trades`

**Example:** 53% win rate means 53 out of 100 trades made money.

**What's "good":** Depends on average win/loss size. A 45% win rate can be
profitable if wins are much larger than losses. Our 53% win rate combined with
profit factor > 1.3 indicates a genuine edge.

### Profit Factor

**What it is:** The ratio of gross profits to gross losses. It measures how
much you make for every unit you lose.

**Formula:** `sum(winning trades) / sum(abs(losing trades))`

**Example:** Profit factor of 1.35 means you make ₹1.35 for every ₹1 you lose.

**What's "good":**
- < 1.0: Losing strategy (losses exceed profits)
- 1.0-1.5: Marginal — small edge
- 1.5-2.0: Good — solid edge
- \> 2.0: Excellent — but verify it's not overfitting

Our models achieve 1.33-1.36 at the best config.

## How They're Computed Here

In `backtest/analytics.py`, `compute_metrics()`:

```python
total_return = (final_equity / initial_equity) - 1.0
cagr = (1 + total_return) ** (1 / years) - 1

# Sharpe: resample to daily, compute daily returns
eq_daily = eq.set_index("timestamp").resample("1D")["equity"].last()
daily_ret = eq_daily.pct_change()
sharpe = (daily_ret.mean() - rf_daily) / daily_ret.std() * np.sqrt(252)

# Max drawdown
cummax = eq["equity"].cummax()
drawdown = (eq["equity"] - cummax) / cummax
max_dd = drawdown.min()

# Win rate and profit factor
wins = trades[trades["net_pnl"] > 0]
losses = trades[trades["net_pnl"] <= 0]
win_rate = len(wins) / len(trades)
profit_factor = wins["net_pnl"].sum() / losses["net_pnl"].abs().sum()
```

## This Project's Actual Numbers

From the full corrected sweep at confidence=0.85, holding=75:

| Metric | LSTM | CNN1D | CNN-LSTM |
|--------|------|-------|----------|
| Return | +85.15% | +85.05% | +83.87% |
| CAGR | +13.46% | +13.45% | +13.30% |
| Sharpe | 1.447 | 1.453 | 1.416 |
| MaxDD | -17.88% | -17.99% | -18.40% |
| Win Rate | 53.0% | 52.8% | 53.2% |
| Profit Factor | 1.35 | 1.34 | 1.34 |

All three models perform within 2% of each other on every metric.

## Common Pitfalls

1. **Looking at return alone.** A +85% return with -60% MaxDD is very different
   from +85% with -18% MaxDD. Always consider Sharpe and MaxDD together.

2. **Ignoring trade count.** 85% return from 13K trades is more reliable than
   85% from 50 trades. Our trade count (13K+) provides statistical significance.

3. **Using daily returns for minute-level data.** Our Sharpe uses daily
   resampled returns (not per-bar) for cleaner annualization. Per-bar Sharpe
   would be inflated by the 375 bars/day multiplier.
