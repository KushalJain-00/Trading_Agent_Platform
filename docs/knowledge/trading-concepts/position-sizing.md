# Position Sizing

## What It Is

**Position sizing** determines how much money you put into each trade. It's
one of the most important risk management decisions — too large and a single
bad trade can wipe out your account; too small and your returns are negligible.

The most common approach is **fixed-fractional sizing**: each trade uses a fixed
percentage of your capital.

```
position_value = capital × position_size_pct
shares = position_value / entry_price
```

For example, with ₹100M capital and 2% sizing:
- Each trade uses ₹2,000,000 (2% of ₹100M)
- If entry price is ₹2,500, you buy 800 shares
- If entry price is ₹50,000, you buy 40 shares

## Why It Matters for This Project

Position sizing directly affects:
1. **Risk per trade**: Larger positions mean larger losses when wrong
2. **Total exposure**: With 113 tickers all trading simultaneously, 2% per
   position means up to 226% total exposure (113 × 2%)
3. **Transaction costs**: Larger positions = larger trade values = higher costs
4. **Portfolio equity**: The equity curve's volatility scales with position size

This project uses **₹100M (₹10 Crore) starting capital** with **2% position
sizing** — ₹2M per trade. This is a paper trading simulation, not real money.
The position size is fixed as a fraction of INITIAL capital, not current equity.

## How It's Implemented Here

In `simulator.py`, the position size calculation appears in both the equity
curve loop and the trade log:

```python
entry_price = close * (1 + half_spread)  # include spread cost
size = (capital * position_size_pct) / entry_price
```

Key details:
- `capital` is always the INITIAL capital (₹100M), not current equity
- `position_size_pct` defaults to 0.02 (2%)
- Entry price includes the half-spread (buy at ask price)
- The position size in shares is constant for the duration of the trade

This means the position size doesn't change as equity grows or shrinks. A
simpler and more common approach is to size based on current equity, but our
fixed approach makes the backtest deterministic and easier to verify.

## Tunable Parameters

| Parameter | Default | What It Does | Effect of Increasing | Effect of Decreasing |
|-----------|---------|--------------|---------------------|---------------------|
| `position_size_pct` | 0.02 (2%) | Fraction of capital per trade | Higher returns AND higher risk | Lower returns, lower risk |
| `capital` | 100,000,000 | Starting capital in ₹ | Larger absolute P&L | Smaller absolute P&L |

**Typical ranges:**
- Conservative: 0.5-1% per trade
- Moderate: 2-3% per trade
- Aggressive: 5-10% per trade

With 113 tickers potentially in position simultaneously, 2% per trade means
up to 226% total exposure. This works in a simulation (you can have negative
cash) but would require margin in real trading.

## Common Pitfalls

1. **Sizing based on current equity.** If you lose 50% and then size at 2%,
   you're trading with half the capital. Our approach sizes from initial
   capital, which is simpler but doesn't adapt to drawdowns.

2. **Ignoring correlated positions.** With 113 Nifty stocks, positions are
   highly correlated. A market crash hits all positions simultaneously. The
   max drawdown of -17.9% reflects this — all positions lose together.

3. **Using too large a position.** With 5% sizing and 113 tickers, total
   exposure would be 565%. A 10% market drop would cause a 56.5% portfolio
   loss. The 2% default keeps max exposure at ~226%.
