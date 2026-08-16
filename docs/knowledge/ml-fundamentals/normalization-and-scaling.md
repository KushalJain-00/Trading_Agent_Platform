# Normalization and Scaling

## What It Is

**Normalization** (or **standardization**) rescales input features so they have
similar ranges. Without it, features like stock price (₹1,000–₹50,000) and
relative volume (0.1–5.0) would dominate the model's learning in wildly
different ways.

The most common method is **z-score standardization**: subtract the mean and
divide by the standard deviation of each feature, computed on the training set.
After this, every feature has mean ≈ 0 and standard deviation ≈ 1.

```
normalized_value = (raw_value - training_mean) / training_std
```

## Why It Matters for This Project

This project uses 23 features (log returns, RSI, MACD, Bollinger Band width,
OBV, VWAP, etc.) — all computed from minute-level Nifty stock data. These
features have completely different scales:
- `log_ret_z` might range from -0.05 to +0.05
- `rel_vol_20_z` might range from 0.1 to 5.0
- `rsi_14_z` ranges from 0 to 100

If fed raw, the model would see `rsi_14_z` values 1000x larger than
`log_ret_z` values. The gradient updates would be dominated by the large-scale
features, and the model would effectively ignore the small-scale ones.

**The NaN loss bug:** This project experienced a real training failure where
the loss became NaN (Not a Number) during training. The root cause was that
some features contained NaN or Inf values from division-by-zero in rolling
z-score computation (e.g., computing standard deviation over a window with zero
variance). These NaN values propagated through the model, causing the loss
function to produce NaN.

The fix was two-pronged:
1. `compute_norm_stats()` in `train.py` (line ~60) skips non-finite values when
   computing mean/std, using `np.isfinite(block)` as a mask
2. `LazyTickerWindows.__getitem__()` in `train.py` (line ~240) calls
   `np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)` to replace
   any remaining NaN/Inf with zeros before feeding to the model
3. The training loop in `train_one()` (line ~291) also checks
   `if torch.isnan(loss) or torch.isinf(loss)` and skips the batch with a
   warning — a safety net

## How It's Implemented Here

**Computing normalization stats** — `compute_norm_stats()` in `train.py`:
- Streams through the training `.npy` cache in chunks of 65,536 rows
- Tracks count, sum, and sum-of-squares in float64 for numerical stability
- Skips non-finite values (NaN, Inf)
- Saves result to `data/processed/train_npy/norm_stats.npz` containing `mean`
  and `std` arrays, each of shape `(23,)` — one value per feature

**Applying normalization** — in `LazyTickerWindows.__getitem__()`:
```python
x = (x - self.norm_mean) / (self.norm_std + 1e-8)
np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
```
The `1e-8` in the denominator prevents division by zero for features with zero
variance.

**Normalization stats are computed once** from the training set only — never
from validation or test data. This prevents "information leakage" where the
model accidentally sees the future through the normalization parameters.

## Tunable Parameters

| Parameter | Default | What It Does | Effect of Changing |
|-----------|---------|--------------|-------------------|
| Epsilon in division | `1e-8` | Prevents division by zero | Too small: division by zero possible. Too large: artificially compresses low-variance features |
| Chunk size for stat computation | 65,536 | Memory usage during stat computation | Larger: faster but uses more RAM. Smaller: slower but safer on low-RAM machines |

In practice, these are not parameters you'd tune — they're engineering choices.

## Common Pitfalls

1. **Computing stats on the full dataset.** If you compute mean/std on
   train+val combined, your normalization includes future information. We
   compute it on `train_npy/` only — see `compute_norm_stats()` which reads
   from the training cache directory.

2. **Forgetting to apply the same normalization at inference time.** The live
   signal generator (`generate_live_signal()` in
   `backtest/generate_signals.py`) loads the same `norm_stats.npz` and applies
   the same formula. If you use different stats, predictions will be garbage.

3. **Not handling NaN/Inf after normalization.** Even with correct stats, edge
   cases (zero-variance features, extreme outliers) can produce NaN or Inf.
   Our `np.nan_to_num()` call zeros these out — without it, the NaN loss bug
   would recur.

4. **Normalizing features that are already normalized.** Our feature columns
   have `_z` suffixes (e.g., `rsi_14_z`, `log_ret_z`) indicating they were
   already z-scored during feature engineering. We still normalize again using
   the training-set stats to ensure consistent scaling across the full pipeline.
