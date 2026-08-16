# Trading Arena — Knowledge Base

A complete reference for every concept used in this project. Written for someone
who knows basic Python and general ML but may not know trading-specific or
advanced time-series concepts.

## Recommended Reading Order (start here)

If you're new to this project, read these in order:

1. [Windowing and Sequences](ml-fundamentals/windowing-and-sequences.md) — how raw data becomes model input
2. [Normalization and Scaling](ml-fundamentals/normalization-and-scaling.md) — why raw features break neural nets
3. [Model Architectures Compared](ml-fundamentals/model-architectures-compared.md) — what each model actually does
4. [What is Backtesting](backtesting-concepts/what-is-backtesting.md) — the core validation idea
5. [Equity Curves and Portfolio Simulation](backtesting-concepts/equity-curves-and-portfolio-simulation.md) — how we measure performance (and the bug we fixed)
6. [Confidence Thresholds and Holding Periods](backtesting-concepts/confidence-thresholds-and-holding-periods.md) — why raw signals lost money and filters saved them
7. [Walk-Forward Testing](backtesting-concepts/walk-forward-testing.md) — stronger validation than a single window
8. [Monte Carlo Simulation](backtesting-concepts/monte-carlo-simulation.md) — what MC does and doesn't tell you

## ML Fundamentals

- [Windowing and Sequences](ml-fundamentals/windowing-and-sequences.md) — sliding windows, stride, the OOM crash and how stride fixed it
- [Normalization and Scaling](ml-fundamentals/normalization-and-scaling.md) — z-score standardization, the NaN loss bug
- [Class Imbalance and Weighting](ml-fundamentals/class-imbalance-and-weighting.md) — why Buy/Hold/Sell aren't equal, inverse-frequency weights
- [Gradient Clipping and Stability](ml-fundamentals/gradient-clipping-and-stability.md) — exploding gradients in LSTMs, clip_grad_norm_
- [Overfitting and Generalization](ml-fundamentals/overfitting-and-generalization.md) — train/val split, why we never backtest on training data
- [Model Architectures Compared](ml-fundamentals/model-architectures-compared.md) — LSTM, 1D CNN, CNN-LSTM, Transformer explained

## Trading Concepts

- [OHLCV and Market Data](trading-concepts/ohlcv-and-market-data.md) — what bars are, how Nifty minute data is structured
- [Buy/Hold/Sell Signals](trading-concepts/buy-hold-sell-signals.md) — how classification becomes a trade, confidence scores
- [Position Sizing](trading-concepts/position-sizing.md) — fixed-fractional sizing, the 2% default
- [Transaction Costs and Slippage](trading-concepts/transaction-costs-and-slippage.md) — basis points, the cost-eating-P&L problem
- [Execution Latency](trading-concepts/execution-latency.md) — next-bar fills, lookahead bias
- [Portfolio Delta](trading-concepts/portfolio-delta.md) — net directional exposure

## Backtesting Concepts

- [What is Backtesting](backtesting-concepts/what-is-backtesting.md) — the core idea
- [Equity Curves and Portfolio Simulation](backtesting-concepts/equity-curves-and-portfolio-simulation.md) — the per-ticker compounding bug and its fix
- [Performance Metrics Explained](backtesting-concepts/performance-metrics-explained.md) — CAGR, Sharpe, MaxDD, win rate, profit factor
- [Monte Carlo Simulation](backtesting-concepts/monte-carlo-simulation.md) — trade resampling vs return bootstrapping, selection bias
- [Walk-Forward Testing](backtesting-concepts/walk-forward-testing.md) — multi-period validation, regime dependence
- [Confidence Thresholds and Holding Periods](backtesting-concepts/confidence-thresholds-and-holding-periods.md) — how filters turned -59% into +87%

## Infrastructure

- [Memory-Mapped Datasets](infrastructure/memory-mapped-datasets.md) — mmap, LazyTickerWindows, the OOM crash
- [Training on Constrained Hardware](infrastructure/training-on-constrained-hardware.md) — CPU-only 8GB laptop lessons
- [Reproducibility and Caching](infrastructure/reproducibility-and-caching.md) — what's cached vs recomputed, determinism
