# Windowing and Sequences

## What It Is

Neural networks process fixed-size inputs. A time series (like stock price data
over time) is a sequence of arbitrary length. **Windowing** means chopping that
sequence into fixed-size chunks — each chunk is a "window" that the model sees
at once.

A **sliding window** moves forward by a fixed number of bars each time, creating
overlapping windows. The **window size** determines how many past bars the model
sees. The **stride** determines how far the window advances between consecutive
samples.

```
Data:    [bar1, bar2, bar3, bar4, bar5, bar6, bar7, bar8, ...]

Window=4, Stride=1:
  Window 1: [bar1, bar2, bar3, bar4]  → label at bar4
  Window 2: [bar2, bar3, bar4, bar5]  → label at bar5
  Window 3: [bar3, bar4, bar5, bar6]  → label at bar6
  ...

Window=4, Stride=4:
  Window 1: [bar1, bar2, bar3, bar4]  → label at bar4
  Window 2: [bar5, bar6, bar7, bar8]  → label at bar8
  ...
```

The **label** for each window is the bar immediately after the last bar in the
window — the model predicts what happens next given these past observations.

## Why It Matters for This Project

This project predicts Buy/Hold/Sell signals from minute-level Nifty stock data.
Each model (LSTM, CNN1D, CNN-LSTM) takes a window of 60 consecutive bars
containing 23 features each, and predicts the next bar's direction.

The window size (60 bars) was chosen to capture ~60 minutes (1 hour) of
intraday context. The stride (15 bars) means we sample every 15th window —
producing ~1.05M signals from the validation set instead of ~15.8M (one per bar).

This project hit a real problem: with stride=1 and window_size=60, the naive
approach produces ~73 million windows from the full dataset. On 8GB RAM, loading
all of these into memory causes an **OOM (Out of Memory) crash**. The fix was
two-fold: stride=15 (reducing windows by 15x) and memory-mapped numpy arrays
(see [Memory-Mapped Datasets](../infrastructure/memory-mapped-datasets.md)).

## How It's Implemented Here

The windowing logic lives in two places:

**Training** — `LazyTickerWindows` in `train.py` (line ~199):
- Opens memory-mapped `feats.npy` and `labels.npy` files
- Uses binary search (`bisect.bisect_right`) to map a flat window index to a
  specific ticker and bar offset
- `__getitem__(idx)` slices the memmap at the right position and returns a
  `(window_tensor, label)` pair

**Signal generation** — `generate_historical_signals()` in
`backtest/generate_signals.py` (line ~46):
- Creates the same `LazyTickerWindows` with `stride=15`
- Runs model inference in batches
- Maps each prediction back to the "decision bar" — the last bar in the window
  — using `bisect` and the same offset math

The window size (60), stride (15), and batch size (256 for training, 512 for
inference) are all configurable via environment variables or CLI arguments.

## Tunable Parameters

| Parameter | Default | What It Does | Effect of Increasing | Effect of Decreasing |
|-----------|---------|--------------|---------------------|---------------------|
| `WINDOW_SIZE` | 60 | Number of bars the model sees at once | More context, more memory, slower training | Less context, faster, may miss patterns |
| `STRIDE` | 15 | How far the window advances per sample | Fewer windows, faster training, less data reuse | More windows, slower, more redundancy |
| `BATCH_SIZE` | 256 | Windows processed per gradient step | Faster training, more GPU/RAM needed | Slower, smoother gradients, less RAM |

**Window size tradeoffs:** Larger windows (e.g., 120 or 240) give the model
more history to work with but increase memory proportionally. On 8GB RAM, 60
bars is practical; 240 bars would require batch size reduction or more RAM.

**Stride tradeoffs:** Stride=1 gives maximum data but creates massive redundancy
(neighboring windows share 59/60 bars). Stride=window_size (non-overlapping)
gives maximum independence but may miss signals that occur mid-window. Our
stride=15 means 75% overlap — a balance between data density and redundancy.

## Common Pitfalls

1. **Window leaks into the label.** The label must come from AFTER the window
   ends, not from within it. Our implementation correctly uses the "decision
   bar" (last bar of window) as the signal timestamp, with the label from
   training data being the next bar's direction.

2. **Stride too small → false confidence.** If you evaluate on stride=1, your
   test metrics look great but are inflated because consecutive predictions are
   nearly identical. We use stride=15 for both training and evaluation.

3. **Window size chosen without data.** Picking 60 because "it's about an hour"
   is a starting point, not a rigorous choice. We did not tune window size —
   it stayed at 60 throughout the project.
