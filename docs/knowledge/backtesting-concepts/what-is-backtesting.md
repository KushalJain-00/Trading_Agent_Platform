# What is Backtesting

## What It Is

**Backtesting** is running a trading strategy on historical data to see how it
would have performed. You take past prices, apply your strategy's rules
retroactively, and measure the simulated profit or loss.

The core idea: "If I had bought when the model said Buy and sold when it said
Sell, how much money would I have made (or lost)?"

## Why It Matters for This Project

Backtesting is the primary validation tool for this project. Before risking
real money (even paper trading money), we need to know if the strategy
profitable on data the model hasn't seen.

**Critical rule: Only backtest on validation data, never on training data.**

- Training data: `data/processed/train_npy/` — used to train the model
- Validation data: `data/processed/val.parquet` — used for backtesting
  (15.8M rows, 113 tickers, Oct 2022 to Feb 2026)

The validation set is a **temporal holdout** — it comes AFTER all training data
in time. This mimics real trading where you test on the future, not the past.

**Lookahead bias** is the most common backtesting mistake. It happens when
the strategy uses information that wouldn't be available at the time of the
trade. For example:
- Using today's close price to decide a trade at today's open (impossible)
- Including a future indicator value in a feature (data leakage)
- Executing at the exact signal bar's price instead of the next bar

This project avoids lookahead bias by:
1. Using `latency_bars=1` — signals execute at the next bar, not the current one
2. Normalization stats computed on training data only
3. Features use only past data (rolling windows look backward, not forward)

## How It's Implemented Here

The backtesting pipeline has these stages:

1. **Signal generation** (`backtest/generate_signals.py`):
   - Load trained model, run inference on validation data
   - Output: DataFrame with (ticker, timestamp, predicted_signal, confidence)

2. **Simulation** (`backtest/simulator.py`):
   - Merge signals with price data
   - Apply filters (confidence threshold, holding period, latency)
   - Track portfolio state: cash, open positions, mark-to-market equity
   - Output: equity curve + trade log

3. **Analytics** (`backtest/analytics.py`):
   - Compute metrics: return, CAGR, Sharpe, MaxDD, win rate, profit factor
   - Compare across models

4. **Validation**:
   - Walk-forward testing (`backtest/walk_forward.py`)
   - Monte Carlo simulation (`backtest/monte_carlo.py`)
   - Full parameter sweep (`backtest/full_sweep_corrected.py`)

## Common Pitfalls

1. **Backtesting on training data.** The model memorized this data — of course
   it looks good. Our pipeline strictly separates train and val.

2. **Lookahead bias.** The most insidious bug. Our `latency_bars=1` ensures
   signals execute at the next bar, not the current one.

3. **Overfitting to the validation set.** If you tweak parameters until the
   validation backtest looks great, you've overfitted to validation. This is
   exactly what we discovered — see [Monte Carlo Simulation](monte-carlo-simulation.md).

4. **Ignoring transaction costs.** A strategy that profits before costs may
   lose after costs. Our simulator always includes 5 bps cost + 3 bps spread.

5. **Survivorship bias.** Only testing on stocks that exist today ignores
   delisted stocks. Our dataset includes 113 Nifty tickers — a reasonable
   universe but not exhaustive.
