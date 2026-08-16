# Reproducibility and Caching

## What It Is

**Reproducibility** means getting the same results when you re-run the same
code with the same data. In ML, this is harder than it sounds — random
initialization, data shuffling, and floating-point non-determinism can all
cause different results across runs.

**Caching** means storing intermediate results (signals, trade logs, numpy
arrays) to avoid recomputing them. Caching speeds up iteration but introduces
the risk of stale data — using cached results from an old code version.

## Why It Matters for This Project

This project's comparison results must be reproducible. If LSTM shows +85%
return, re-running the sweep should show the same number. Without this, you
can't trust that one model is actually better than another.

The project verified reproducibility by checking that:
1. Re-running the same config produces identical equity curves
2. The assertion check (`abs(actual - expected) < 1.0`) passes consistently
3. No stale-cache bugs exist (cached signals from old code versions)

## What's Cached vs Recomputed

### Cached (one-time computation, reused across runs)

| What | Location | Recomputed when |
|------|----------|----------------|
| Numpy arrays (feats.npy, labels.npy) | `data/processed/train_npy/` | Never (data doesn't change) |
| Normalization stats | `data/processed/train_npy/norm_stats.npz` | Never |
| Model checkpoints | `models/checkpoints/{model}.pt` | Never (models are fixed) |
| Feature metadata | `data/processed/feature_meta.txt` | Never |
| Generated signals | `backtest/signals/{model}_val_signals.parquet` | Never (signals are deterministic given model+data+stride) |

### Recomputed every run

| What | Why |
|------|-----|
| Equity curve | Depends on config (confidence, holding) |
| Trade log | Depends on config |
| Performance metrics | Derived from equity curve + trade log |
| Monte Carlo results | Stochastic (random resampling) |
| Walk-forward results | Depends on config |

### The key invariant: signals are immutable

Generated signals (`{model}_val_signals.parquet`) are deterministic given:
- The same model checkpoint
- The same validation data
- The same stride (15)
- The same normalization stats

Changing the confidence threshold or holding period does NOT change the
signals — it only changes how the backtest filters them. This is why the sweep
can share signals across configs.

## How It's Implemented Here

**Signal caching** — in `run_backtest.py`:
```python
sig_path = signals_dir / f"{model_name}_val_signals.parquet"
if sig_path.exists():
    signals_df = pd.read_parquet(sig_path)  # load cached
else:
    signals_df = generate_historical_signals(...)  # generate and save
```

**Numpy cache** — in `ensure_numpy_cache()`:
```python
expected_bytes = n_rows * n_features * 4  # float32
if os.path.exists(feats_path) and os.path.getsize(feats_path) == expected_bytes:
    return cache_dir  # already cached
```

**No caching between sweep runs** — `run_historical_backtest()` always
recomputes the equity curve and trade log. This ensures each config's result
is freshly computed, not cached from a previous config.

**Determinism verification** — the assertion check in `_run_backtest_core()`:
```python
assert abs(actual_final - expected_final) < 1.0
```
This verifies that the equity curve matches the trade log's mathematical
expectation. If any caching bug causes inconsistent results, this assertion
catches it.

## Tunable Parameters

Reproducibility isn't tunable — it's a property of the code. But you can
control:

| Setting | Effect |
|---------|--------|
| PyTorch deterministic mode | `torch.backends.cudnn.deterministic = True` (we're CPU-only, so not critical) |
| Random seed | Not explicitly set in this project — results are deterministic because models are pre-trained and signals are cached |
| Numpy random seed | Used in Monte Carlo (different each run by design) |

## Common Pitfalls

1. **Using cached signals after changing the model.** If you retrain a model
   but don't regenerate signals, the backtest uses stale predictions. The
   signal cache key is just the model name — if you overwrite the checkpoint,
   you must delete the cached signals.

2. **Caching equity curves across configs.** If you run confidence=0.85 and
   then confidence=0.90, but the equity curve is cached from the first run,
   you'd see the same results. Our sweep always recomputes (output_dir=None).

3. **Stale parquet files.** If you modify the backtest code but don't clear
   old result files, the dashboard may show results from the old code. Always
   clear `backtest/results/` before a fresh run.

4. **Forgetting that Monte Carlo is stochastic.** MC uses `np.random` without
   a fixed seed, so results vary between runs. This is intentional — the
   percentiles should converge with enough iterations (2000 is sufficient).
