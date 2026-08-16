# Monte Carlo Simulation

## What It Is

Monte Carlo (MC) simulation tests whether a strategy's results are robust by
randomly varying the inputs and seeing if the strategy still profits. There are
two main approaches:

**Trade-sequence resampling:** Takes the actual trade P&Ls and randomly reshuffles
their order. This tests whether the strategy's profitability depends on the
specific sequence of wins and losses, or whether it would profit regardless of
order.

**Return bootstrapping:** Takes bar-level returns from the equity curve and
resamples them with replacement. This generates confidence intervals for
metrics like CAGR, Sharpe, and Max Drawdown.

## Why It Matters for This Project

MC simulation answers: "Is this strategy's profit real, or did we get lucky?"

**What MC protects against:**
- Random luck in the specific sequence of trades
- Outlier trades that disproportionately affect results
- Overfitting to specific market conditions in the test period

**What MC does NOT protect against (critical limitation):**
- **Selection bias from sweeping many configs and picking the winner**

This project swept 60 configurations (3 models × 5 confidence thresholds × 4
holding periods) and selected the best one (confidence=0.85, holding=75). The
MC simulation then tests that specific config's trade sequence — but it doesn't
account for the fact that we cherry-picked the best config from 60 candidates.

This is like rolling a die 60 times, picking the highest roll, and then running
MC to see if that roll is "typical" — of course it is, because you picked the
highest one. The MC tests the roll's distribution, not the selection process.

## How It's Implemented Here

### Trade Resampling

In `backtest/monte_carlo.py`, `trade_resampling()` (line ~19):

```python
def trade_resampling(trade_log_df, equity_curve_df, n_iterations=2000,
                     initial_capital=100_000_000):
    trade_pnls = trade_log_df["net_pnl"].values
    trade_returns = trade_pnls / initial_capital

    for i in range(n_iterations):
        perm = np.random.permutation(n_trades)
        resampled = trade_returns[perm]
        equity_path = initial_capital * np.cumprod(1 + resampled)
        final_equities[i] = equity_path[-1]
        max_drawdowns[i] = compute_maxdd(equity_path)
```

- Output: Distribution of final equity and max drawdown across 2000 permutations
- Percentiles reported: P5, P25, P50 (median), P75, P95
- If P5 (worst 5% case) is still profitable, the edge is robust to trade order

### Return Bootstrapping

In `backtest/monte_carlo.py`, `return_bootstrapping()` (line ~58):

```python
def return_bootstrapping(equity_curve_df, n_iterations=2000):
    bar_returns = eq["equity"].pct_change().dropna().values
    bar_returns = np.clip(bar_returns, -0.5, 0.5)  # prevent overflow

    for i in range(n_iterations):
        sample = np.random.choice(bar_returns, size=n_bars, replace=True)
        equity = np.cumprod(1 + sample)
        cagrs[i] = compute_cagr(equity)
        sharpes[i] = compute_sharpe(sample)
```

- Output: Distribution of CAGR, Sharpe, and MaxDD across 2000 bootstrap samples
- The `np.clip` prevents extreme return values from causing overflow in
  `np.cumprod`

### Running MC

```python
from backtest.monte_carlo import run_monte_carlo
run_monte_carlo(trade_log_df, equity_curve_df, model_name,
                n_iterations=2000, initial_capital=100_000_000)
```

Output: HTML charts (Plotly) and CSV percentile tables in `backtest/reports/`.

## Tunable Parameters

| Parameter | Default | What It Does | Effect of Increasing | Effect of Decreasing |
|-----------|---------|--------------|---------------------|---------------------|
| `n_iterations` | 2000 | Number of MC simulations | More stable percentiles, slower | Less stable, faster |
| Clip range | [-0.5, 0.5] | Caps extreme bar returns | More conservative bootstrap | May include outliers |

## Common Pitfalls

1. **Trusting MC results blindly.** MC tests order-independence, not whether
   the config was selected by cherry-picking. Our walk-forward test is the
   honest check for that.

2. **Using too few iterations.** 100 iterations give noisy percentiles. 2000 is
   a reasonable default.

3. **Not clipping returns.** Without `np.clip(bar_returns, -0.5, 0.5)`, a
   single extreme return can make `np.cumprod` overflow to inf.

4. **Confusing MC with walk-forward.** MC tests robustness of a single config.
   Walk-forward tests generalization across time periods. Both are needed.
