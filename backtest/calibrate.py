"""Temperature scaling — post-hoc calibration for existing models.

Fits a single temperature parameter T on validation logits, then applies
softmax(logits/T) at inference. No retraining required.

Usage:
    python -m backtest.calibrate --model lstm
    python -m backtest.calibrate --model all
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train import load_meta, LazyTickerWindows, ensure_numpy_cache
from models.lstm_model import build_model as build_lstm
from models.cnn_1d_model import build_model as build_cnn1d
from models.cnn_lstm_hybrid import build_model as build_cnn_lstm

LABEL_MAP = {0: "Buy", 1: "Hold", 2: "Sell"}
MODEL_BUILDERS = {"lstm": build_lstm, "cnn1d": build_cnn1d, "cnn_lstm": build_cnn_lstm}
CKPT_DIR = PROJECT_ROOT / "models" / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CAL_DIR = PROJECT_ROOT / "models" / "calibration"


def collect_logits(model, data_loader, device):
    """Collect raw logits and true labels from a dataset."""
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in data_loader:
            logits = model(xb.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(yb)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits, labels):
    """Fit temperature T via NLL minimization on validation set."""
    nll = nn.CrossEntropyLoss()

    def eval_nll(T):
        return nll(logits / T, labels).item()

    result = minimize_scalar(eval_nll, bounds=(0.1, 10.0), method="bounded")
    return result.x


def reliability_curve(probs, labels, n_bins=10):
    """Compute calibration data: (mean_confidence, accuracy) per bin."""
    confs = probs.max(dim=1).values.numpy()
    preds = probs.argmax(dim=1).numpy()
    correct = (preds == labels.numpy()).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_accs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (confs >= bin_edges[i]) & (confs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_accs.append(correct[mask].mean())
            bin_counts.append(int(mask.sum()))

    return np.array(bin_centers), np.array(bin_accs), np.array(bin_counts)


def expected_calibration_error(probs, labels, n_bins=10):
    """ECE: weighted average of |accuracy - confidence| per bin."""
    bin_centers, bin_accs, bin_counts = reliability_curve(probs, labels, n_bins)
    if len(bin_centers) == 0:
        return 0.0
    total = bin_counts.sum()
    return float(np.sum(np.abs(bin_accs - bin_centers) * bin_counts / total))


def conf_histogram(conf_tensor, bins=10):
    """Bucket confidence scores into histogram."""
    c = conf_tensor.numpy()
    edges = np.linspace(0, 1, bins + 1)
    return [int(((c >= edges[i]) & (c < edges[i + 1])).sum()) for i in range(bins)]


def calibrate_model(model_name, device, batch_size=512, stride=15):
    """Full calibration pipeline for one model."""
    print(f"\n{'='*60}")
    print(f"  CALIBRATING: {model_name.upper()}")
    print(f"{'='*60}")

    ckpt = torch.load(CKPT_DIR / f"{model_name}.pt", map_location=device, weights_only=True)
    model = MODEL_BUILDERS[model_name](ckpt["input_dim"], ckpt["window_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = load_meta(str(DATA_DIR))
    feature_cols = meta["feature_cols"]
    val_cache = ensure_numpy_cache(str(DATA_DIR), "val", feature_cols)
    ns = np.load(DATA_DIR / "train_npy" / "norm_stats.npz")
    norm_mean, norm_std = ns["mean"], ns["std"]

    val_ds = LazyTickerWindows(val_cache, ckpt["window_size"], stride=stride,
                                norm_mean=norm_mean, norm_std=norm_std)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    print("  Collecting validation logits...")
    logits, labels = collect_logits(model, loader, device)
    print(f"  {len(labels):,} samples collected")

    # Before calibration
    probs_before = F.softmax(logits, dim=1)
    ece_before = expected_calibration_error(probs_before, labels)
    conf_before = probs_before.max(dim=1).values
    print(f"\n  BEFORE: ECE={ece_before:.4f}  "
          f"Conf: min={conf_before.min():.4f} max={conf_before.max():.4f} "
          f"std={conf_before.std():.4f} mean={conf_before.mean():.4f}")

    # Fit temperature
    T = fit_temperature(logits, labels)
    print(f"  Fitted T = {T:.4f}")

    # After calibration
    probs_after = F.softmax(logits / T, dim=1)
    ece_after = expected_calibration_error(probs_after, labels)
    conf_after = probs_after.max(dim=1).values
    print(f"  AFTER:  ECE={ece_after:.4f}  "
          f"Conf: min={conf_after.min():.4f} max={conf_after.max():.4f} "
          f"std={conf_after.std():.4f} mean={conf_after.mean():.4f}")

    hist_before = conf_histogram(conf_before)
    hist_after = conf_histogram(conf_after)
    print(f"\n  Confidence histogram (10 buckets):")
    print(f"    Before: {hist_before}")
    print(f"    After:  {hist_after}")

    # Save
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CAL_DIR / f"{model_name}_temperature.npz",
             temperature=np.float32(T),
             ece_before=np.float32(ece_before), ece_after=np.float32(ece_after))

    bins_b, accs_b, counts_b = reliability_curve(probs_before, labels)
    bins_a, accs_a, counts_a = reliability_curve(probs_after, labels)
    pd.DataFrame({
        "bin_center": bins_b,
        "accuracy_before": accs_b, "accuracy_after": accs_a,
        "count_before": counts_b, "count_after": counts_a,
    }).to_csv(CAL_DIR / f"{model_name}_reliability.csv", index=False)

    print(f"  Saved → {CAL_DIR / f'{model_name}_temperature.npz'}")

    return {
        "model": model_name, "temperature": T,
        "ece_before": ece_before, "ece_after": ece_after,
        "conf_before_std": float(conf_before.std()),
        "conf_after_std": float(conf_after.std()),
    }


def generate_calibrated_signals(model_name, device, batch_size=512, stride=15):
    """Generate validation signals with calibrated probabilities."""
    import bisect

    print(f"\n  Generating calibrated signals for {model_name}...")

    ckpt = torch.load(CKPT_DIR / f"{model_name}.pt", map_location=device, weights_only=True)
    model = MODEL_BUILDERS[model_name](ckpt["input_dim"], ckpt["window_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    window_size = ckpt["window_size"]

    T = float(np.load(CAL_DIR / f"{model_name}_temperature.npz")["temperature"])

    meta = load_meta(str(DATA_DIR))
    feature_cols = meta["feature_cols"]
    val_cache = ensure_numpy_cache(str(DATA_DIR), "val", feature_cols)
    ns = np.load(DATA_DIR / "train_npy" / "norm_stats.npz")
    norm_mean, norm_std = ns["mean"], ns["std"]

    rng = np.load(os.path.join(val_cache, "ticker_ranges.npz"), allow_pickle=True)
    ticker_names, ticker_starts, ticker_ends = rng["names"], rng["starts"], rng["ends"]

    offsets = []
    global_offset = 0
    for i in range(len(ticker_names)):
        offsets.append(global_offset)
        usable = int(ticker_ends[i] - ticker_starts[i]) - window_size
        n_windows = max(0, (usable + stride - 1) // stride) if usable > 0 else 0
        global_offset += n_windows

    val_df = pd.read_parquet(str(DATA_DIR / "val.parquet"), columns=["ticker", "timestamp"])
    val_ds = LazyTickerWindows(val_cache, window_size, stride=stride,
                                norm_mean=norm_mean, norm_std=norm_std)

    records = []
    with torch.no_grad():
        for start in range(0, len(val_ds), batch_size):
            end = min(start + batch_size, len(val_ds))
            batch_x = torch.stack([val_ds[i][0] for i in range(start, end)])
            probs = F.softmax(model(batch_x.to(device)) / T, dim=1).cpu().numpy()

            for j, idx in enumerate(range(start, end)):
                ti = bisect.bisect_right(offsets, idx) - 1
                local = idx - offsets[ti]
                row = int(ticker_starts[ti] + local * stride + window_size - 1)
                if row >= len(val_df):
                    continue
                records.append({
                    "ticker": str(ticker_names[ti]),
                    "timestamp": val_df.iloc[row]["timestamp"],
                    "predicted_signal": LABEL_MAP[int(probs[j].argmax())],
                    "predicted_confidence": float(probs[j].max()),
                    "prob_buy": float(probs[j, 0]),
                    "prob_hold": float(probs[j, 1]),
                    "prob_sell": float(probs[j, 2]),
                    "model": model_name,
                })

    df = pd.DataFrame(records)
    out_path = PROJECT_ROOT / "backtest" / "signals" / f"{model_name}_val_signals_calibrated.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved {len(df):,} calibrated signals → {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_BUILDERS.keys()) + ["all"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = list(MODEL_BUILDERS.keys()) if args.model == "all" else [args.model]

    results = []
    for name in models:
        results.append(calibrate_model(name, device, args.batch_size, args.stride))
        generate_calibrated_signals(name, device, args.batch_size, args.stride)

    print("\n\n" + "=" * 70)
    print("  CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"{'Model':<12} {'Temp':>7} {'ECE before':>11} {'ECE after':>10} {'Conf std(b)':>12} {'Conf std(a)':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<12} {r['temperature']:7.3f} {r['ece_before']:11.4f} "
              f"{r['ece_after']:10.4f} {r['conf_before_std']:12.4f} {r['conf_after_std']:12.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
