"""Agent 1: Signal Generation.

Wraps trained models + temperature calibration. For each ticker+bar,
outputs calibrated Buy/Hold/Sell signal with probabilities.
Writes to: signals table.
Reads from: nothing (standalone).
"""
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train import load_meta, LazyTickerWindows, ensure_numpy_cache
from models.lstm_model import build_model as build_lstm
from models.cnn_1d_model import build_model as build_cnn1d
from models.cnn_lstm_hybrid import build_model as build_cnn_lstm
from agents.schema import get_conn, log_agent, init_db

LABEL_MAP = {0: "Buy", 1: "Hold", 2: "Sell"}
MODEL_BUILDERS = {"lstm": build_lstm, "cnn1d": build_cnn1d, "cnn_lstm": build_cnn_lstm}
CKPT_DIR = PROJECT_ROOT / "models" / "checkpoints"
CAL_DIR = PROJECT_ROOT / "models" / "calibration"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


class SignalAgent:
    """Signal generation agent — runs inference on one model with calibration."""

    def __init__(self, model_name, device=None, db_path=None):
        self.model_name = model_name
        self.device = device or torch.device("cpu")
        self.db_path = db_path
        self.model = None
        self.window_size = None
        self.T = 1.0
        self.norm_mean = None
        self.norm_std = None
        self.feature_cols = None

    def load(self):
        """Load model checkpoint and calibration temperature."""
        ckpt = torch.load(CKPT_DIR / f"{self.model_name}.pt",
                          map_location=self.device, weights_only=True)
        self.model = MODEL_BUILDERS[self.model_name](ckpt["input_dim"], ckpt["window_size"])
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.window_size = ckpt["window_size"]

        cal_path = CAL_DIR / f"{self.model_name}_temperature.npz"
        if cal_path.exists():
            self.T = float(np.load(cal_path)["temperature"])
        else:
            self.T = 1.0

        meta = load_meta(str(DATA_DIR))
        self.feature_cols = meta["feature_cols"]
        ns = np.load(DATA_DIR / "train_npy" / "norm_stats.npz")
        self.norm_mean, self.norm_std = ns["mean"], ns["std"]
        return self

    def run_backtest(self, val_cache=None, batch_size=512, stride=15):
        """Run on validation set, write all signals to DB. Returns DataFrame."""
        import bisect

        if self.model is None:
            self.load()

        val_cache = val_cache or ensure_numpy_cache(str(DATA_DIR), "val", self.feature_cols)
        val_ds = LazyTickerWindows(val_cache, self.window_size, stride=stride,
                                    norm_mean=self.norm_mean, norm_std=self.norm_std)

        rng = np.load(os.path.join(val_cache, "ticker_ranges.npz"), allow_pickle=True)
        ticker_names, ticker_starts, ticker_ends = rng["names"], rng["starts"], rng["ends"]

        offsets = []
        global_offset = 0
        for i in range(len(ticker_names)):
            offsets.append(global_offset)
            usable = int(ticker_ends[i] - ticker_starts[i]) - self.window_size
            n_windows = max(0, (usable + stride - 1) // stride) if usable > 0 else 0
            global_offset += n_windows

        val_df = pd.read_parquet(str(DATA_DIR / "val.parquet"), columns=["ticker", "timestamp"])

        records = []
        with torch.no_grad():
            for start in range(0, len(val_ds), batch_size):
                end = min(start + batch_size, len(val_ds))
                batch_x = torch.stack([val_ds[i][0] for i in range(start, end)])
                logits = self.model(batch_x.to(self.device)) / self.T
                probs = F.softmax(logits, dim=1).cpu().numpy()

                for j, idx in enumerate(range(start, end)):
                    ti = bisect.bisect_right(offsets, idx) - 1
                    local = idx - offsets[ti]
                    row = int(ticker_starts[ti] + local * stride + self.window_size - 1)
                    if row >= len(val_df):
                        continue
                    records.append({
                        "bar_idx": row,
                        "ticker": str(ticker_names[ti]),
                        "timestamp": str(val_df.iloc[row]["timestamp"]),
                        "signal": LABEL_MAP[int(probs[j].argmax())],
                        "confidence": float(probs[j].max()),
                        "prob_buy": float(probs[j, 0]),
                        "prob_hold": float(probs[j, 1]),
                        "prob_sell": float(probs[j, 2]),
                        "model": self.model_name,
                    })

        df = pd.DataFrame(records)

        # Write to DB
        if len(df) > 0:
            with get_conn(self.db_path) as conn:
                conn.executemany(
                    "INSERT INTO signals (bar_idx, ticker, timestamp, signal, confidence, "
                    "prob_buy, prob_hold, prob_sell, model) VALUES "
                    "(:bar_idx, :ticker, :timestamp, :signal, :confidence, "
                    ":prob_buy, :prob_hold, :prob_sell, :model)",
                    df.to_dict("records")
                )
            log_agent("agent1_signal", f"Wrote {len(df)} signals to DB",
                      details={"model": self.model_name, "n_signals": len(df)},
                      db_path=self.db_path)

        return df


def run_agent1(model_name="lstm", db_path=None):
    """Convenience function to run Agent 1 end-to-end."""
    db_path = db_path or init_db()
    agent = SignalAgent(model_name, db_path=db_path)
    agent.load()
    df = agent.run_backtest()
    print(f"Agent 1 ({model_name}): {len(df):,} signals written")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    run_agent1(args.model, args.db)
