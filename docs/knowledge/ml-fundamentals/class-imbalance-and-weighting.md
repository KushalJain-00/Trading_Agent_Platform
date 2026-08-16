# Class Imbalance and Weighting

## What It Is

In classification problems, **class imbalance** means some classes have many
more examples than others. If 70% of training labels are "Hold", 20% are "Buy",
and 10% are "Sell", the model will naturally bias toward predicting "Hold"
because that's the easiest way to minimize overall error.

**Class weighting** compensates by making the loss function treat each class
equally. A common approach is **inverse-frequency weighting**: classes with
fewer examples get higher weights, so the model "cares more" about getting
them right.

```
weight(class) = total_samples / (num_classes × count_of_class)
```

If "Sell" appears 10% of the time and "Hold" appears 70%, "Sell" gets a weight
7x higher than "Hold" in the loss function.

## Why It Matters for This Project

This project classifies minute-level market data into three classes:
- **0 = Buy** (model expects price to go up)
- **1 = Hold** (model expects price to stay flat or ambiguous)
- **2 = Sell** (model expects price to go down)

In real market data, "Hold" dominates — most minute bars don't show strong
directional moves. The class distribution is heavily skewed toward Hold. Without
weighting, the model would learn to predict "Hold" most of the time and achieve
high accuracy while being useless for trading.

## How It's Implemented Here

**Computing weights** — `compute_class_weights()` in `train.py` (line ~253):
```python
def compute_class_weights(data_path, num_classes=3):
    counts = np.zeros(num_classes, dtype=np.float64)
    # Iterate through label parquet in chunks of 65536
    counts = np.maximum(counts, 1)  # avoid division by zero
    weights = 1.0 / counts
    return torch.from_numpy(weights / weights.sum() * num_classes).float()
```
- Reads raw labels from the training parquet in streaming chunks
- Counts occurrences of each class (0, 1, 2)
- Computes inverse frequency: `1.0 / count`
- Normalizes so `sum(weights) == num_classes` (mean weight = 1.0)
- Returns a tensor of shape `(3,)` — one weight per class

**Using weights in training** — in `train_one()` in `train.py`:
```python
class_weights = compute_class_weights(data_path)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```
The weights are passed to `CrossEntropyLoss`, which multiplies the per-sample
loss by the weight of the true class. A "Sell" example with weight 2.0 produces
2x more gradient than a "Hold" example with weight 0.5.

## Tunable Parameters

| Parameter | What It Does | How to Change It |
|-----------|--------------|-----------------|
| Weighting scheme | Inverse frequency is the default | Can try sqrt-inverse, or manual weights like [1.0, 0.3, 2.0] |
| Normalization | Weights sum to num_classes by default | Changing normalization affects the absolute scale of gradients |

In practice, the default inverse-frequency weighting works well for this
project's class distribution. You might adjust if:
- You care more about one class (e.g., want to catch every Sell signal)
- The class distribution changes significantly between train and validation

## Common Pitfalls

1. **Computing weights on the wrong split.** Weights must come from training
   data only. If computed on validation, the model sees future class
   distribution information. Our `compute_class_weights()` reads from
   `data_dir` (training path).

2. **Over-weighting rare classes.** If "Sell" is 2% of data, inverse-frequency
   gives it weight 50x. This can cause the model to over-predict "Sell" and
   produce noisy, unprofitable signals. Our normalization (`weights / sum * 3`)
   keeps the maximum weight moderate.

3. **Ignoring imbalance at inference.** Class weighting only affects training.
   At inference, the model outputs raw probabilities. If the model is biased
   toward Hold despite weighting, you may need post-processing (like the
   confidence threshold we use in the backtest).
