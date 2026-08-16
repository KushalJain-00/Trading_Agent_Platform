# Buy/Hold/Sell Signals

## What It Is

A **trading signal** is a recommendation to buy, hold, or sell an asset at a
given time. In this project, the neural network outputs a probability
distribution over three classes at each bar:

- **Buy (class 0)**: The model predicts the price will go up
- **Hold (class 1)**: The model predicts the price will stay flat or the
  direction is ambiguous
- **Sell (class 2)**: The model predicts the price will go down

The model outputs raw probabilities via softmax (e.g., `[0.6, 0.3, 0.1]`).
The **predicted class** is the one with the highest probability (0.6 → Buy).
The **confidence** is that highest probability (0.6).

## Why It Matters for This Project

The model's classification output is the bridge between ML and trading. A
prediction of "Buy" with 85% confidence means the model is quite sure the
price will rise. A prediction of "Buy" with 55% confidence is barely better
than a coin flip.

The **confidence threshold** is the key parameter that turns raw predictions
into actionable signals. If we only act on Buy signals with confidence ≥ 0.85,
we filter out weak, noisy predictions. This is critical because:

- Raw model signals (all Buy/Hold/Sell predictions) produce **-59% return**
  (at hold=10, costs eat everything)
- With confidence threshold 0.85 and holding period 75, the same model
  produces **+85% return**

See [Confidence Thresholds and Holding Periods](../backtesting-concepts/confidence-thresholds-and-holding-periods.md)
for the full before/after numbers.

## How It's Implemented Here

**Label mapping** — defined in `backtest/generate_signals.py` (line ~25):
```python
LABEL_MAP = {0: "Buy", 1: "Hold", 2: "Sell"}
```

**Signal generation** — `generate_historical_signals()` in
`backtest/generate_signals.py`:
1. Loads the trained model checkpoint
2. Creates a `LazyTickerWindows` dataset with stride=15
3. Runs inference: `probs = model(batch).softmax(dim=1)`
4. Extracts predicted class and confidence:
   ```python
   pred_class = preds.argmax(axis=1)
   confidence = preds.max(axis=1)
   ```
5. Maps class 0→"Buy", 1→"Hold", 2→"Sell"
6. Saves to `{model}_val_signals.parquet` with columns:
   `ticker`, `timestamp`, `predicted_signal`, `predicted_confidence`, `model`

**Signal filtering in the backtest** — in `simulator.py`:
```python
# Confidence filter: weak Buy signals become Hold
if confidence_threshold > 0.0:
    weak_buy = (merged["predicted_signal"] == "Buy") & \
               (merged["predicted_confidence"] < confidence_threshold)
    merged.loc[weak_buy, "predicted_signal"] = "Hold"
```

This only filters Buy signals — Sell signals are always acted on (the model's
Sell predictions are less frequent and higher quality).

## Tunable Parameters

| Parameter | Default | What It Does | Effect of Increasing | Effect of Decreasing |
|-----------|---------|--------------|---------------------|---------------------|
| Confidence threshold | 0.85 | Minimum confidence to act on Buy signals | Fewer trades, higher quality, may miss opportunities | More trades, more noise, higher costs |
| Which signals to filter | Buy only | Could filter Sell too | — | — |

## Common Pitfalls

1. **Treating confidence as probability.** A confidence of 0.85 doesn't mean
   there's an 85% chance the price goes up. It means the model is 85% sure
   the next bar is a Buy (class 0). The relationship between model confidence
   and actual win rate is learned, not guaranteed.

2. **Not filtering at all.** Without confidence filtering, you trade every
   signal — including the model's uncertain, noisy predictions. Our initial
   backtest showed -59% return with no filtering.

3. **Filtering too aggressively.** If you set confidence=0.95, you may filter
   out all but a handful of trades. The edge exists in the aggregate — too few
   trades means the edge doesn't manifest.
