#!/usr/bin/env python3
"""End-to-end: merge -> features -> train -> backtest.

Usage:
    python run_all.py
    python run_all.py --skip-merge --skip-features
    python run_all.py --epochs 5 --window-size 30
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    t0 = time.time()

    if not args.skip_merge:
        print("\n" + "=" * 60 + "\nSTEP 1: Merge\n" + "=" * 60)
        from data_pipeline.merge_data import main as merge_main
        merge_main()

    if not args.skip_features:
        print("\n" + "=" * 60 + "\nSTEP 2: Features\n" + "=" * 60)
        from data_pipeline.build_features import main as features_main
        features_main()

    if not args.skip_train:
        print("\n" + "=" * 60 + "\nSTEP 3: Train\n" + "=" * 60)
        sys.argv = ["train.py", "--window-size", str(args.window_size), "--epochs", str(args.epochs)]
        from train import main as train_main
        train_main()

    print("\n" + "=" * 60 + "\nSTEP 4: Backtest\n" + "=" * 60)
    import torch
    import pandas as pd
    from train import load_meta
    from backtest.engine import run_all_backtests

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(ROOT, "data", "processed")
    ckpt_dir = os.path.join(ROOT, "models", "checkpoints")
    meta = load_meta(data_dir)
    feature_cols = meta["feature_cols"]
    ws = args.window_size

    test_df = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
    print(f"Test: {len(test_df):,} rows, {test_df['ticker'].nunique()} tickers")

    model_map = {
        "lstm": ("models.lstm_model", "lstm.pt"),
        "cnn1d": ("models.cnn_1d_model", "cnn1d.pt"),
        "cnn_lstm": ("models.cnn_lstm_hybrid", "cnn_lstm.pt"),
        "transformer": ("models.transformer_model", "transformer.pt"),
    }
    model_configs = {}
    for name, (mod_path, ckpt_file) in model_map.items():
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        if not os.path.exists(ckpt_path):
            continue
        mod = __import__(mod_path, fromlist=["build_model"])
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = mod.build_model(ckpt["input_dim"], ckpt["window_size"])
        model.load_state_dict(ckpt["model_state_dict"])
        model_configs[name] = (model, ckpt_path)

    if model_configs:
        run_all_backtests(model_configs, test_df, feature_cols, ws, device)

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
