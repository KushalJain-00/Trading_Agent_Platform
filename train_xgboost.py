"""Train XGBoost on featurized windows from the same memory-mapped data.

Featurization:
  Each 60-step window of 23 features is summarized into 138 tabular features:
    For each of the 23 raw features, compute 6 statistics across the window:
      - mean, std, min, max, last value, linear regression slope
    Total: 23 × 6 = 138 features.

  The slope captures trend direction/magnitude within the window, which the
  other stats miss. This is a fundamentally different representation from
  feeding raw sequences to LSTM/CNN — XGBoost sees compressed summaries,
  not temporal order.

Usage:
  python train_xgboost.py                          # defaults
  python train_xgboost.py --window-size 60 --stride 15
  python train_xgboost.py --max-depth 8 --n-estimators 500 --learning-rate 0.05
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier

# Reuse existing infrastructure from train.py — no reimplementation
sys.path.insert(0, os.path.dirname(__file__))
from train import (
    load_meta, check_memory, ensure_numpy_cache,
    compute_norm_stats, LazyTickerWindows, compute_class_weights,
)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data", "processed"))
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(os.path.dirname(__file__), "models", "checkpoints"))


# ── Featurization ────────────────────────────────────────────────────
def featurize_window(window):
    """Convert (window_size, n_features) raw sequence → (n_features * 6,) tabular vector.

    Stats per feature: mean, std, min, max, last, slope.
    NaN/inf in input are replaced with 0 before computing stats.
    Slope uses least-squares on a normalized time axis [0, 1].
    """
    w = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
    n_steps, n_feat = w.shape

    means = w.mean(axis=0)
    stds = w.std(axis=0)
    mins = w.min(axis=0)
    maxs = w.max(axis=0)
    lasts = w[-1]

    # Slope: least-squares on [0,1] time axis — vectorized across features
    t = np.linspace(0, 1, n_steps)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    if t_var > 0:
        w_mean = w.mean(axis=0, keepdims=True)
        slopes = ((w - t_mean) * (t[:, None] - t_mean)).sum(axis=0) / t_var
    else:
        slopes = np.zeros(n_feat)

    return np.concatenate([means, stds, mins, maxs, lasts, slopes])


def materialize_features(dataset, desc="Featurizing"):
    """Iterate dataset, featurize every window, return (X, y) arrays.

    Stores results in memory-mapped numpy for low peak RAM — same pattern
    as the existing pipeline, just different output shape.
    """
    n = len(dataset)
    if n == 0:
        raise ValueError("Empty dataset")

    # Peek at first window to get feature dim
    sample_x, _ = dataset[0]
    n_raw_features = sample_x.shape[1]
    n_tabular = n_raw_features * 6  # 23 × 6 = 138

    # Write to temp mmap files, then return as arrays
    tmp_dir = os.path.join(os.path.dirname(dataset.feats.filename), "xgboost_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    x_path = os.path.join(tmp_dir, "X.npy")
    y_path = os.path.join(tmp_dir, "y.npy")

    X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(n, n_tabular))
    y_mm = np.memmap(y_path, dtype=np.int64, mode="w+", shape=(n,))

    t0 = time.time()
    for i in range(n):
        x_torch, y = dataset[i]
        X_mm[i] = featurize_window(x_torch.numpy())
        y_mm[i] = y

        if (i + 1) % 50000 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            sys.stdout.write(
                f"\r  {desc}: {i+1:,}/{n:,}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s"
            )
            sys.stdout.flush()

    print()
    X_mm.flush()
    y_mm.flush()

    # Copy to regular arrays (mmap files stay around for debugging but aren't needed)
    X = np.array(X_mm, dtype=np.float32)
    y = np.array(y_mm, dtype=np.int64)
    del X_mm, y_mm

    # Clean up tmp files
    for f in [x_path, y_path]:
        try:
            os.unlink(f)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return X, y


def compute_xgb_class_weights(labels, num_classes=3):
    """Class weights matching train.py's compute_class_weights() logic.

    train.py: weights = (1 / counts) normalized to sum to num_classes.
    XGBoost expects a dict {class_label: weight}.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return {i: float(w) for i, w in enumerate(weights)}


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost on featurized windows")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--ckpt-dir", default=CKPT_DIR)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--max-depth", type=int, default=6,
                        help="Max tree depth (6 is a good default; deeper risks overfitting on noisy features)")
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="Max boosting rounds (early stopping will likely cut short)")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="Step size shrinkage (0.1 with 300 rounds = reasonable; lower lr + more rounds can help")
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"Device: CPU (XGBoost)")
    check_memory()

    meta = load_meta(args.data_dir)
    feature_cols = meta["feature_cols"]
    n_features = len(feature_cols)
    print(f"Features: {n_features}, Window: {args.window_size}, Stride: {args.stride}")

    # ── Data loading (reuses existing memory-mapped pipeline) ────────
    print("Preparing numpy caches ...")
    train_cache = ensure_numpy_cache(args.data_dir, "train", feature_cols)
    val_cache = ensure_numpy_cache(args.data_dir, "val", feature_cols)

    print("Computing normalization stats ...")
    norm_mean, norm_std = compute_norm_stats(train_cache, n_features)

    print("Building datasets (memory-mapped) ...")
    train_ds = LazyTickerWindows(train_cache, args.window_size, args.stride,
                                 norm_mean=norm_mean, norm_std=norm_std)
    val_ds = LazyTickerWindows(val_cache, args.window_size, args.stride,
                               norm_mean=norm_mean, norm_std=norm_std)
    print(f"  Train windows: {len(train_ds):,}  Val windows: {len(val_ds):,}")

    # ── Materialize tabular features ────────────────────────────────
    print("\nFeaturizing training windows ...")
    X_train, y_train = materialize_features(train_ds, desc="Train")
    print(f"  X_train: {X_train.shape}  y_train: {np.bincount(y_train)}")

    print("Featurizing validation windows ...")
    X_val, y_val = materialize_features(val_ds, desc="Val")
    print(f"  X_val: {X_val.shape}  y_val: {np.bincount(y_val)}")

    # ── Class weights ───────────────────────────────────────────────
    class_weights = compute_xgb_class_weights(y_train)
    print(f"Class weights: {class_weights}")

    # ── XGBoost training ────────────────────────────────────────────
    print(f"\nTraining XGBoost (max_depth={args.max_depth}, n_estimators={args.n_estimators}, "
          f"lr={args.learning_rate}, subsample={args.subsample})")

    clf = XGBClassifier(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=args.early_stopping_rounds,
        tree_method="hist",  # ponytail: hist is fastest for CPU, exact is slower but no memory overhead
        nthread=os.cpu_count(),
        verbosity=1,
        random_state=42,
    )

    t0 = time.time()
    clf.fit(
        X_train, y_train,
        sample_weight=np.array([class_weights[y] for y in y_train], dtype=np.float32),
        eval_set=[(X_val, y_val)],
        verbose=True,
    )
    train_time = time.time() - t0
    print(f"\nTraining complete in {train_time:.1f}s  Best iteration: {clf.best_iteration}")

    # ── Evaluation ──────────────────────────────────────────────────
    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)

    tr_acc = accuracy_score(y_train, train_preds)
    vl_acc = accuracy_score(y_val, val_preds)
    tr_f1 = f1_score(y_train, train_preds, average=None, labels=[0, 1, 2])
    vl_f1 = f1_score(y_val, val_preds, average=None, labels=[0, 1, 2])

    print("\n" + "=" * 75)
    print(f"{'Phase':<8} {'Acc':>7} {'F1-macro':>9} {'F1-Buy':>8} {'F1-Hold':>9} {'F1-Sell':>9}")
    print("-" * 75)
    for phase, acc, f1s in [("Train", tr_acc, tr_f1), ("Val", vl_acc, vl_f1)]:
        macro = f1s.mean()
        print(f"{phase:<8} {acc:7.4f} {macro:9.4f} {f1s[0]:8.4f} {f1s[1]:9.4f} {f1s[2]:9.4f}")
    print("=" * 75)

    # ── Save ────────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.ckpt_dir, "xgboost.json")
    clf.save_model(ckpt_path)
    print(f"\nModel saved → {ckpt_path}")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("SMOKE TEST PASSED — model trained and saved successfully.")
    print(f"{'='*75}")
    print(f"\nTo run full training with custom params:")
    print(f"  python train_xgboost.py --window-size {args.window_size} --stride {args.stride} "
          f"--max-depth {args.max_depth} --n-estimators {args.n_estimators} "
          f"--learning-rate {args.learning_rate}")
    print(f"\nModel checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
