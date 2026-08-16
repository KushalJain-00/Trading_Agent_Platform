# Overfitting and Generalization

## What It Is

**Overfitting** means the model memorizes the training data instead of learning
patterns that generalize to new data. An overfitted model scores 99% on
training data but 51% on validation data — it learned noise, not signal.

**Generalization** is the opposite: the model performs similarly on training and
validation data because it learned real patterns.

Key tools to prevent overfitting:
- **Train/validation split**: hold out data the model never sees during training
- **Early stopping**: stop training when validation loss stops improving
- **Dropout**: randomly zero out neurons during training to prevent co-adaptation
- **Regularization**: penalize large weights

## Why It Matters for This Project

In financial time series, overfitting is especially dangerous because:
1. Markets have low signal-to-noise ratio — most variation is random
2. Past patterns may not repeat (non-stationarity)
3. If you accidentally peek at validation data, your backtest looks great but
   live trading loses money

This project's entire validation approach is designed to prevent data leakage:
- Training data: `data/processed/train_npy/` (numpy cache)
- Validation data: `data/processed/val.parquet` — never used during training
- Backtesting: only on validation data, never on training data
- Walk-forward testing: validates on sub-periods the config wasn't selected on

## How It's Implemented Here

**Train/validation split**: The dataset is pre-split into `train.parquet` and
`val.parquet`. The validation set spans 2022-10 to 2026-02 (~15.8M rows across
113 Nifty tickers). This is a **temporal split** — all validation data comes
after all training data, mimicking real trading where you test on the future.

**Early stopping**: In `train_one()` in `train.py` (line ~265):
```python
best_val_loss = float('inf')
patience_counter = 0
for epoch in range(epochs):
    # ... train ...
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # save best model
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break  # stop training
```
With `PATIENCE=4`, training stops if validation loss doesn't improve for 4
consecutive epochs.

**Dropout**: Both LSTM (`dropout=0.2` in 2-layer config) and the CNN-LSTM
hybrid use dropout. The 1D CNN backbone doesn't use dropout (convolutional
layers are less prone to co-adaptation).

## Tunable Parameters

| Parameter | Default | What It Does | Effect of Increasing | Effect of Decreasing |
|-----------|---------|--------------|---------------------|---------------------|
| `PATIENCE` | 4 | Epochs to wait before stopping | Trains longer, may overfit more | Stops sooner, may underfit |
| `EPOCHS` | 15 | Maximum training epochs | Longer training possible | May stop before learning |
| Dropout (LSTM) | 0.2 | Fraction of neurons zeroed during training | More regularization, may underfit | Less regularization, may overfit |

## Common Pitfalls

1. **Evaluating on training data.** The cardinal sin. Our backtest engine
   (`backtest/simulator.py`) only processes `val.parquet` signals. The training
   numpy cache is never loaded during backtesting.

2. **Tuning hyperparameters on the test set.** If you select the best
   confidence threshold (0.85) and holding period (75) based on validation
   performance, then report that same performance as "unbiased", you've
   overfitted to validation. This is exactly what we discovered — see
   [Monte Carlo Simulation](../backtesting-concepts/monte-carlo-simulation.md)
   for the selection bias problem.

3. **Using future data in features.** Our features use rolling windows
   (SMA, EMA, RSI, etc.) that only look backward. The `_z` suffix features
   in `FEATURE_COLS` in `backtest/live_data.py` are all computed from past
   data only.

4. **Temporal leakage in normalization.** If normalization stats include
   validation data, the model sees the future through the stats. Our
   `compute_norm_stats()` in `train.py` reads from `train_npy/` only.
