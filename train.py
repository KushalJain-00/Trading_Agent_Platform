"""Train all four model architectures. Memory-safe: memory-mapped numpy arrays."""
import argparse
import bisect
import gc
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
import pyarrow.parquet as pq

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data", "processed"))
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(os.path.dirname(__file__), "models", "checkpoints"))
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "60"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
EPOCHS = int(os.environ.get("EPOCHS", "15"))
LR = float(os.environ.get("LR", "1e-3"))
PATIENCE = int(os.environ.get("PATIENCE", "4"))
STRIDE = int(os.environ.get("STRIDE", "15"))

MODELS = {
    "lstm": "models.lstm_model",
    "cnn1d": "models.cnn_1d_model",
    "cnn_lstm": "models.cnn_lstm_hybrid",
    "transformer": "models.transformer_model",
}


def load_meta(data_dir):
    meta = {}
    with open(os.path.join(data_dir, "feature_meta.txt")) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                meta[k] = v.split(",") if k == "feature_cols" else v
    return meta


def check_memory(min_gb=1.5):
    """Print available RAM; warn if below threshold."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    avail_gb = int(line.split()[1]) / (1024 ** 2)
                    print(f"Available RAM: {avail_gb:.1f} GB")
                    if avail_gb < min_gb:
                        print(f"WARNING: Below {min_gb:.1f} GB threshold — training may OOM.")
                    return avail_gb
    except FileNotFoundError:
        pass
    print("Could not determine available RAM (non-Linux?).")
    return None


def compute_norm_stats(cache_dir, n_features):
    """Compute per-feature mean/std from training .npy cache (streaming, memory-safe).

    Skips NaN values from rolling_zscore division-by-zero. Saves stats to
    norm_stats.npz in cache_dir so subsequent runs skip recomputation.
    """
    stats_path = os.path.join(cache_dir, "norm_stats.npz")
    if os.path.exists(stats_path):
        data = np.load(stats_path)
        return data["mean"], data["std"]

    feats_path = os.path.join(cache_dir, "feats.npy")
    n_rows = os.path.getsize(feats_path) // (n_features * 4)
    feats = np.memmap(feats_path, dtype=np.float32, mode="r", shape=(n_rows, n_features))

    chunk = 65536
    count = np.zeros(n_features, dtype=np.float64)
    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_x2 = np.zeros(n_features, dtype=np.float64)

    for start in range(0, n_rows, chunk):
        end = min(start + chunk, n_rows)
        block = np.array(feats[start:end], dtype=np.float64)
        valid = np.isfinite(block)
        count += valid.sum(axis=0)
        block_clean = np.where(valid, block, 0.0)
        sum_x += block_clean.sum(axis=0)
        sum_x2 += (block_clean ** 2).sum(axis=0)

    mean = sum_x / np.maximum(count, 1)
    var = sum_x2 / np.maximum(count, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 0))

    np.savez_compressed(stats_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"  Saved norm stats → {stats_path}")
    return mean.astype(np.float32), std.astype(np.float32)


def diagnose_features(train_ds, n_features, n_samples=1000):
    """Sample n_windows from train_ds and print per-feature stats. Count NaN/inf."""
    rng = np.random.default_rng(42)
    indices = rng.choice(len(train_ds), size=min(n_samples, len(train_ds)), replace=False)
    stack = np.stack([train_ds[i][0].numpy() for i in indices], axis=0)  # (n_samples, window, features)
    flat = stack.reshape(-1, n_features)

    nan_count = np.isnan(flat).sum()
    inf_count = np.isinf(flat).sum()
    print(f"\n  Feature diagnostic ({n_samples} windows, {flat.shape[0]:,} values):")
    print(f"  NaN: {nan_count:,}  Inf: {inf_count:,}")
    for fi in range(n_features):
        col = flat[:, fi]
        valid = col[np.isfinite(col)]
        if len(valid) > 0:
            print(f"    feat[{fi:2d}] min={valid.min():.4f} max={valid.max():.4f} "
                  f"mean={valid.mean():.4f} std={valid.std():.4f}")
        else:
            print(f"    feat[{fi:2d}] ALL NON-FINITE")


def ensure_numpy_cache(data_dir, split, feature_cols):
    """Convert parquet split to memory-mapped numpy arrays (one-time cost).

    Writes .npy files that are memory-mapped during training — zero RAM
    overhead for the full dataset, only accessed pages are paged in by the OS.
    Resumes from partial conversions (skips files that already have correct size).
    """
    cache_dir = os.path.join(data_dir, f"{split}_npy")
    feats_path = os.path.join(cache_dir, "feats.npy")
    labels_path = os.path.join(cache_dir, "labels.npy")
    ranges_path = os.path.join(cache_dir, "ticker_ranges.npz")

    os.makedirs(cache_dir, exist_ok=True)
    parquet_path = os.path.join(data_dir, f"{split}.parquet")
    pf = pq.ParquetFile(parquet_path)
    n_rows = pf.metadata.num_rows
    n_feats = len(feature_cols)
    expected_feats_bytes = n_rows * n_feats * 4
    expected_labels_bytes = n_rows * 8

    feats_ready = os.path.exists(feats_path) and os.path.getsize(feats_path) >= expected_feats_bytes
    labels_ready = os.path.exists(labels_path) and os.path.getsize(labels_path) >= expected_labels_bytes
    ranges_ready = os.path.exists(ranges_path)

    if feats_ready and labels_ready and ranges_ready:
        print(f"  Using cached numpy arrays for {split}/")
        return cache_dir

    print(f"  Converting {split}.parquet to numpy (one-time) ...")

    # memmap writes directly to disk — never holds the full array in RAM
    feats_mm = np.memmap(feats_path, dtype=np.float32, mode="w+", shape=(n_rows, n_feats)) if not feats_ready else None
    labels_mm = np.memmap(labels_path, dtype=np.int64, mode="w+", shape=(n_rows,)) if not labels_ready else None

    ticker_ranges = []
    current_ticker = None
    ticker_start = 0
    write_pos = 0

    for batch in pf.iter_batches(columns=feature_cols + ["label", "ticker"], batch_size=65536):
        tickers = batch.column("ticker").to_pylist()
        batch_size = len(tickers)

        if labels_mm is not None:
            labels_mm[write_pos : write_pos + batch_size] = batch.column("label").to_numpy(zero_copy_only=False).astype(np.int64)

        if feats_mm is not None:
            for fi, fc in enumerate(feature_cols):
                feats_mm[write_pos : write_pos + batch_size, fi] = batch.column(fc).to_numpy(zero_copy_only=False).astype(np.float32)

        # Track ticker boundaries regardless (needed for ranges)
        for i, t in enumerate(tickers):
            row = write_pos + i
            if t != current_ticker:
                if current_ticker is not None:
                    ticker_ranges.append((current_ticker, ticker_start, row))
                current_ticker = t
                ticker_start = row
        write_pos += batch_size

    if current_ticker is not None:
        ticker_ranges.append((current_ticker, ticker_start, write_pos))

    if feats_mm is not None:
        feats_mm.flush()
        del feats_mm
    if labels_mm is not None:
        labels_mm.flush()
        del labels_mm

    if not ranges_ready and ticker_ranges:
        names = [r[0] for r in ticker_ranges]
        starts = np.array([r[1] for r in ticker_ranges], dtype=np.int64)
        ends = np.array([r[2] for r in ticker_ranges], dtype=np.int64)
        np.savez_compressed(ranges_path, names=names, starts=starts, ends=ends)

    print(f"  Wrote {n_rows:,} rows, {n_feats} features → {cache_dir}/")
    return cache_dir


class LazyTickerWindows(Dataset):
    """Memory-mapped dataset: O(1) window access via numpy mmap + binary search.

    The full dataset lives on disk as .npy files. Only the rows touched by
    the current batch are paged into RAM by the OS. Peak RAM for data is
    ~batch_size * window_size * n_features * 4 bytes.
    """

    def __init__(self, cache_dir, window_size, stride=1, n_features=None,
                 norm_mean=None, norm_std=None):
        self.window_size = window_size
        self.stride = stride
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        rng = np.load(os.path.join(cache_dir, "ticker_ranges.npz"), allow_pickle=True)
        self.ticker_names = rng["names"]
        self.ticker_starts = rng["starts"]
        self.ticker_ends = rng["ends"]
        n_rows = int(self.ticker_ends[-1])

        feats_path = os.path.join(cache_dir, "feats.npy")
        if n_features is None:
            n_features = os.path.getsize(feats_path) // (n_rows * 4)

        self.feats = np.memmap(feats_path, dtype=np.float32, mode="r", shape=(n_rows, n_features))
        self.labels = np.memmap(os.path.join(cache_dir, "labels.npy"), dtype=np.int64, mode="r", shape=(n_rows,))

        # Compute strided window offsets for each ticker
        self._offsets = []  # global_offset for each ticker (for bisect)
        global_offset = 0
        for i in range(len(self.ticker_names)):
            self._offsets.append(global_offset)
            ticker_rows = self.ticker_ends[i] - self.ticker_starts[i]
            usable = ticker_rows - window_size
            n_windows = max(0, (usable + stride - 1) // stride) if usable > 0 else 0
            global_offset += n_windows
        self.total = global_offset

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        i = bisect.bisect_right(self._offsets, idx) - 1
        local = idx - self._offsets[i]
        row_start = int(self.ticker_starts[i] + local * self.stride)
        x = self.feats[row_start : row_start + self.window_size].copy()
        if self.norm_mean is not None:
            x = (x - self.norm_mean) / (self.norm_std + 1e-8)
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        y = int(self.labels[row_start])
        return torch.from_numpy(x), y


def compute_class_weights(data_path, num_classes=3):
    """Exact class weights from raw labels — no window iteration needed."""
    counts = np.zeros(num_classes, dtype=np.float64)
    pf = pq.ParquetFile(data_path)
    for batch in pf.iter_batches(columns=["label"], batch_size=65536):
        labels = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
        counts += np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    return torch.from_numpy(weights / weights.sum() * num_classes).float()


def train_one(name, build_fn, input_dim, window_size, train_ds, val_ds,
              device, class_weights, epochs, lr, patience, batch_size,
              num_workers=3):
    model = build_fn(input_dim, window_size).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    n_batches = (len(train_ds) + batch_size - 1) // batch_size
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=False)
    vloader = DataLoader(val_ds, batch_size=batch_size,
                         num_workers=num_workers, pin_memory=False)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, correct, total = 0, 0, 0
        skipped = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            if torch.isnan(loss) or torch.isinf(loss):
                skipped += 1
                print(f"\n  [{name}] WARNING: NaN/inf loss at ep {epoch} batch {skipped}, skipping")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

            batch_idx = total // batch_size
            if batch_idx % 200 == 0 and batch_idx > 0:
                elapsed = time.time() - t0
                batches_done = batch_idx
                eta = elapsed / batches_done * (n_batches - batches_done)
                sys.stdout.write(
                    f"\r  [{name}] ep {epoch:3d} batch {batch_idx:5d}/{n_batches}  "
                    f"loss={total_loss/total:.4f}  "
                    f"elapsed={elapsed:.0f}s  eta={eta:.0f}s"
                )
                sys.stdout.flush()
        if total > 0:
            sys.stdout.write(f"\r  [{name}] ep {epoch:3d} batch {n_batches:5d}/{n_batches}  "
                             f"loss={total_loss/total:.4f}  elapsed={time.time()-t0:.0f}s        \n")
        if skipped:
            print(f"  [{name}] {skipped} batches skipped (NaN/inf loss)")

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in vloader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(yb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += len(yb)
                all_preds.append(logits.argmax(1).cpu())
                all_labels.append(yb.cpu())

        v_loss = val_loss / val_total
        v_acc = val_correct / val_total
        epoch_time = time.time() - t0
        remaining = epochs - epoch
        projected = remaining * epoch_time
        print(f"  [{name}] ep {epoch:3d}  tr_loss={total_loss/total:.4f}  tr_acc={correct/total:.3f}  "
              f"vl_loss={v_loss:.4f}  vl_acc={v_acc:.3f}  "
              f"ep_time={epoch_time:.0f}s  projected_total={projected/3600:.1f}h")

        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{name}] early stop ep {epoch}")
                break

    model.load_state_dict(best_state)
    del best_state
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers):
            preds.append(model(xb.to(device)).argmax(1).cpu())
            labels.append(yb)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    per_class = f1_score(labels, preds, average=None, labels=[0, 1, 2])
    return model, {
        "name": name, "val_acc": (preds == labels).mean(),
        "val_f1_macro": f1_score(labels, preds, average="macro"),
        "f1_buy": per_class[0], "f1_hold": per_class[1], "f1_sell": per_class[2],
    }


def main():
    torch.set_num_threads(os.cpu_count())

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--ckpt-dir", default=CKPT_DIR)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Threads: {torch.get_num_threads()}")

    check_memory()

    meta = load_meta(args.data_dir)
    feature_cols = meta["feature_cols"]
    input_dim = len(feature_cols)
    print(f"Features: {input_dim}, Window: {args.window_size}, Stride: {args.stride}, "
          f"Batch: {args.batch_size}, Workers: {args.num_workers}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Patience: {args.patience}")

    # One-time: convert parquet → memory-mapped numpy (writes to disk, not RAM)
    print("Preparing numpy caches ...")
    train_cache = ensure_numpy_cache(args.data_dir, "train", feature_cols)
    val_cache = ensure_numpy_cache(args.data_dir, "val", feature_cols)

    print("Computing normalization stats ...")
    norm_mean, norm_std = compute_norm_stats(train_cache, input_dim)

    print("Building datasets (memory-mapped) ...")
    train_ds = LazyTickerWindows(train_cache, args.window_size, args.stride,
                                 norm_mean=norm_mean, norm_std=norm_std)
    val_ds = LazyTickerWindows(val_cache, args.window_size, args.stride,
                               norm_mean=norm_mean, norm_std=norm_std)
    print(f"  Train windows: {len(train_ds):,}  Val windows: {len(val_ds):,}")

    print("Running feature diagnostic ...")
    diagnose_features(train_ds, input_dim)

    print("Computing class weights ...")
    class_weights = compute_class_weights(
        os.path.join(args.data_dir, "train.parquet")
    )
    print(f"Class weights: {class_weights.numpy()}")

    # ponytail: cheaper models first so partial runs get usable checkpoints
    model_order = ["lstm", "cnn1d", "cnn_lstm", "transformer"]
    model_names = args.models or model_order
    results = []
    for name in model_names:
        if name not in MODELS:
            continue
        try:
            mod = __import__(MODELS[name], fromlist=["build_model"])
            print(f"\n=== {name} ===")
            model, metrics = train_one(
                name, mod.build_model, input_dim, args.window_size,
                train_ds, val_ds, device, class_weights,
                args.epochs, args.lr, args.patience, args.batch_size,
                num_workers=args.num_workers,
            )
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim, "window_size": args.window_size,
            }, os.path.join(args.ckpt_dir, f"{name}.pt"))
            results.append(metrics)
        except Exception as e:
            print(f"\n  [{name}] FAILED: {e}")
        finally:
            if "model" in dir():
                del model
            gc.collect()

    print("\n" + "=" * 75)
    print(f"{'Model':<14} {'Acc':>7} {'F1-macro':>9} {'F1-Buy':>8} {'F1-Hold':>9} {'F1-Sell':>9}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<14} {r['val_acc']:7.4f} {r['val_f1_macro']:9.4f} "
              f"{r['f1_buy']:8.4f} {r['f1_hold']:9.4f} {r['f1_sell']:9.4f}")
    print("=" * 75)
    return results


if __name__ == "__main__":
    main()
