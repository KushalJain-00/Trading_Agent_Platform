"""Signal generation — historical backtest + live inference.

Loads trained checkpoints, runs inference on validation set (historical)
or on rolling live windows (paper trading). Uses same normalization stats
from training — never recomputed.

Label mapping: 0 = Buy, 1 = Hold, 2 = Sell
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train import load_meta, LazyTickerWindows, ensure_numpy_cache, compute_norm_stats
from models.lstm_model import build_model as build_lstm
from models.cnn_1d_model import build_model as build_cnn1d
from models.cnn_lstm_hybrid import build_model as build_cnn_lstm

LABEL_MAP = {0: "Buy", 1: "Hold", 2: "Sell"}
MODEL_BUILDERS = {
    "lstm": build_lstm,
    "cnn1d": build_cnn1d,
    "cnn_lstm": build_cnn_lstm,
}


def load_model(name, checkpoint_dir, device):
    """Load a trained model from checkpoint."""
    ckpt_path = Path(checkpoint_dir) / f"{name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    builder = MODEL_BUILDERS[name]
    model = builder(ckpt["input_dim"], ckpt["window_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["input_dim"], ckpt["window_size"]


def generate_historical_signals(model_name, checkpoint_dir, data_dir, device,
                                 output_dir=None, batch_size=512, stride=15):
    """Run inference on validation set, save signals as parquet.

    Uses stride=15 (matching training) for speed on CPU. Each signal maps to
    the last bar of its window (the "decision bar").

    Returns DataFrame with columns: ticker, timestamp, predicted_signal,
    predicted_confidence, model.
    """
    import bisect

    data_dir = Path(data_dir)
    ckpt_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "signals"
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(str(data_dir))
    feature_cols = meta["feature_cols"]
    n_features = len(feature_cols)

    model, input_dim, window_size = load_model(model_name, ckpt_dir, device)
    assert input_dim == n_features, f"Feature mismatch: model={input_dim}, data={n_features}"

    val_cache = ensure_numpy_cache(str(data_dir), "val", feature_cols)
    norm_stats_path = data_dir / "train_npy" / "norm_stats.npz"
    ns = np.load(norm_stats_path)
    norm_mean, norm_std = ns["mean"], ns["std"]

    val_ds = LazyTickerWindows(val_cache, window_size, stride=stride,
                                norm_mean=norm_mean, norm_std=norm_std)

    rng = np.load(os.path.join(val_cache, "ticker_ranges.npz"), allow_pickle=True)
    ticker_names = rng["names"]
    ticker_starts = rng["starts"]
    ticker_ends = rng["ends"]

    offsets = []
    global_offset = 0
    for i in range(len(ticker_names)):
        offsets.append(global_offset)
        ticker_rows = ticker_ends[i] - ticker_starts[i]
        usable = ticker_rows - window_size
        n_windows = max(0, (usable + stride - 1) // stride) if usable > 0 else 0
        global_offset += n_windows

    # Read ticker+timestamp arrays once (vectorized, no slow iloc)
    val_df = pd.read_parquet(str(data_dir / "val.parquet"), columns=["ticker", "timestamp"])

    all_ticker = []
    all_ts = []
    all_pred = []
    all_conf = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(val_ds), batch_size):
            end = min(start + batch_size, len(val_ds))
            batch_x = torch.stack([val_ds[i][0] for i in range(start, end)])
            preds = model(batch_x.to(device)).softmax(dim=1).cpu().numpy()
            pred_class = preds.argmax(axis=1)
            confidence = preds.max(axis=1)

            for j, idx in enumerate(range(start, end)):
                ti = bisect.bisect_right(offsets, idx) - 1
                local = idx - offsets[ti]
                # Signal maps to the last bar of the window (decision bar)
                decision_row = int(ticker_starts[ti] + local * stride + window_size - 1)
                if decision_row >= len(val_df):
                    continue
                all_ticker.append(str(ticker_names[ti]))
                all_ts.append(val_df.iloc[decision_row]["timestamp"])
                all_pred.append(LABEL_MAP[int(pred_class[j])])
                all_conf.append(float(confidence[j]))

    df = pd.DataFrame({
        "ticker": all_ticker, "timestamp": all_ts,
        "predicted_signal": all_pred, "predicted_confidence": all_conf,
        "model": model_name,
    })
    out_path = output_dir / f"{model_name}_val_signals.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} signals → {out_path}")
    return df


def generate_live_signal(model, window_buffer, norm_mean, norm_std, device):
    """Given a buffer of recent bars (list of dicts or DataFrame rows), produce
    a live Buy/Hold/Sell signal + confidence.

    Args:
        model: loaded PyTorch model
        window_buffer: last `window_size` rows, each with 23 feature columns
        norm_mean, norm_std: normalization stats from training
        device: torch device

    Returns:
        dict with 'signal' (str), 'confidence' (float), 'probs' (dict)
    """
    if isinstance(window_buffer, pd.DataFrame):
        feats = window_buffer.values.astype(np.float32)
    else:
        feats = np.array([[r[c] for c in window_buffer.columns] if hasattr(window_buffer, 'columns')
                          else list(r.values()) for r in window_buffer], dtype=np.float32)

    if feats.ndim != 2 or feats.shape[1] != len(norm_mean):
        return {"signal": "Hold", "confidence": 0.0, "probs": {"Buy": 0.33, "Hold": 0.34, "Sell": 0.33}}

    x = (feats - norm_mean) / (norm_std + 1e-8)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(x).unsqueeze(0).to(device)
        probs = model(tensor).softmax(dim=1).cpu().numpy()[0]

    pred_class = int(probs.argmax())
    return {
        "signal": LABEL_MAP[pred_class],
        "confidence": float(probs[pred_class]),
        "probs": {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)},
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "models" / "checkpoints"))
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "backtest" / "signals"))
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_historical_signals(args.model, args.checkpoint_dir, args.data_dir,
                                 device, args.output_dir, args.batch_size)
