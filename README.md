# Trading Arena

Deep-learning stock trading system for Indian equities (NSE Nifty 50 constituents). Ingests 1-minute candlestick data, engineers 46 technical indicators, trains four neural network architectures to predict **buy / hold / sell**, and backtests each model against a buy-and-hold baseline.

## Dataset

[Stock Market Data (Nifty 50 Stocks) - 1 Min Data](https://www.kaggle.com/datasets/debashis74017/stock-market-data-nifty-50-stocks-1-min-data)

Download the dataset and place the CSV files in `Data/archive/` before running the pipeline.

## Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/Trading-Arena.git
cd Trading-Arena

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch pandas numpy pyarrow scikit-learn
```

## Quick Start — Run Everything

```bash
python run_all.py
```

This runs the full pipeline: **Merge → Features → Train → Backtest**.

## Pipeline Steps

### 1. Merge Raw CSVs

```bash
python run_all.py --skip-features --skip-train
```

Reads CSVs from `Data/archive/`, deduplicates, filters tickers with <50k rows, and writes per-ticker parquet files to `data/processed/`.

### 2. Build Features

```bash
python run_all.py --skip-merge --skip-train
```

Computes 46 features per ticker (log returns, SMA/EMA ratios, RSI, MACD, Stochastic K/D, Williams %R, ADX, Bollinger width, ATR, OBV diff, relative volume, VWAP ratio, price position, dist to high/low) plus rolling z-score normalization. Splits data 70/15/15 (train/val/test) chronologically.

### 3. Train Models

```bash
python run_all.py --skip-merge --skip-features --epochs 30 --window-size 60
```

Or train individually:

```bash
python train.py --models lstm cnn1d --epochs 15 --window-size 60 --batch-size 256
```

Available models: `lstm`, `cnn1d`, `cnn_lstm`, `transformer`

### 4. Backtest

```bash
python run_all.py --skip-merge --skip-features --skip-train
```

Runs sliding-window inference on the test set, applies positions one bar after signal with 7bps transaction cost, and reports CAGR, Sharpe, max drawdown, win rate vs buy-and-hold.

## CLI Arguments

### `run_all.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--skip-merge` | `False` | Skip the merge step |
| `--skip-features` | `False` | Skip the feature engineering step |
| `--skip-train` | `False` | Skip training (only backtest) |
| `--window-size` | `60` | Sliding window size (bars) |
| `--epochs` | `30` | Training epochs |

### `train.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data/processed` | Path to processed data |
| `--ckpt-dir` | `models/checkpoints` | Checkpoint save directory |
| `--window-size` | `60` | Sliding window size |
| `--batch-size` | `256` | Batch size |
| `--epochs` | `15` | Training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--patience` | `4` | Early stopping patience |
| `--stride` | `15` | Window stride |
| `--num-workers` | `0` | DataLoader workers |
| `--models` | all | Model(s) to train |

## Models

| Model | Architecture | Params |
|-------|-------------|--------|
| **lstm** | 2-layer LSTM (hidden=64, dropout=0.2) → MLP head | ~40k |
| **cnn1d** | 3-layer Conv1d → AdaptiveAvgPool → MLP head | ~20k |
| **cnn_lstm** | 2-layer Conv1d → LSTM (64) → MLP head | ~35k |
| **transformer** | TransformerEncoder (2 layers, 4 heads, d=64) → mean pool → MLP head | ~60k |

## Project Structure

```
Trading Arena/
├── run_all.py                 # End-to-end orchestrator
├── train.py                   # Training loop + memory-mapped data loading
├── launch.py                  # TUI launcher (optional)
├── data_pipeline/
│   ├── merge_data.py          # CSV → per-ticker parquet
│   └── build_features.py      # 46 features + labels → train/val/test splits
├── models/
│   ├── lstm_model.py
│   ├── cnn_1d_model.py
│   ├── cnn_lstm_hybrid.py
│   ├── transformer_model.py
│   └── checkpoints/           # Saved .pt files
├── backtest/
│   ├── run_backtest.py        # CLI: historical backtest (portfolio/single modes)
│   ├── run_live.py            # CLI: live paper trading (live/replay)
│   ├── simulator.py           # Core engine: array-indexed equity loop, assertion-validated
│   ├── generate_signals.py    # Model inference → signal parquets
│   ├── analytics.py           # Metrics: Sharpe, MaxDD, CAGR, win rate, etc.
│   ├── monte_carlo.py         # MC simulation on trade sequence
│   ├── visualize.py           # Static chart generation
│   ├── live_data.py           # yfinance polling + ReplayDataStreamer
│   ├── dashboard.py           # Streamlit dashboard (7 tabs)
│   ├── mode_comparison.py     # B&H vs model vs allocation modes
│   └── reports/               # CSVs, comparison tables, walk-forward results
└── data/processed/            # Parquet files + memmap cache (gitignored)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `data/processed` | Processed data directory |
| `CKPT_DIR` | `models/checkpoints` | Checkpoint directory |
| `WINDOW_SIZE` | `60` | Sliding window size |
| `BATCH_SIZE` | `256` | Batch size |
| `EPOCHS` | `15` | Training epochs |
| `LR` | `1e-3` | Learning rate |
| `PATIENCE` | `4` | Early stopping patience |
| `STRIDE` | `15` | Window stride |
| `MERGE_RAW_DIR` | `Data/archive` | Raw CSV location |

## Design Notes

- **Memory-safe**: Parquet data is memory-mapped to `.npy` arrays. Peak RAM is only `batch_size × window_size × 46 × 4 bytes`.
- **No lookahead**: Chronological 70/15/15 split. Features use only backward-looking windows. Positions applied one bar after signal.
- **Resumable**: Feature pipeline skips already-processed tickers on re-run.
- **3-class labels**: buy (forward return > 0.1%), sell (< -0.1%), hold (in between) over 10-bar horizon.

## Backtest Usage

```bash
# Full portfolio backtest (all models, equal-weight)
python backtest/run_backtest.py

# Single-stock mode
python backtest/run_backtest.py --mode single --ticker RELIANCE

# Portfolio with allocation modes
python backtest/run_backtest.py --allocation confidence-weighted --max-positions 20 --max-position-pct 0.10
python backtest/run_backtest.py --allocation top-N --max-positions 20

# Mode comparison (B&H vs model vs allocation)
python backtest/mode_comparison.py

# Live paper trading (replay mode)
python backtest/run_live.py --mode replay --tickers RELIANCE TCS INFY

# Launch dashboard
streamlit run backtest/dashboard.py
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `portfolio` | `portfolio` (multi-stock) or `single` (one ticker) |
| `--ticker` | — | Ticker symbol (required for single mode) |
| `--allocation` | `equal` | `equal`, `confidence-weighted`, or `top-N` |
| `--max-positions` | `0` | Max concurrent positions (0=unlimited) |
| `--max-position-pct` | `0.0` | Max fraction of capital per position |
| `--confidence-threshold` | `0.65` | Min confidence to act on Buy signal |
| `--min-holding-bars` | `10` | Min bars to hold before allowing exit |

## Key Findings

### Walk-Forward Validation (4 periods, conf=0.90/hold=75)

| Period | LSTM | CNN1D | CNN-LSTM |
|--------|------|-------|----------|
| P1: Q1'23 | -7.76% | -6.91% | -8.22% |
| P2: Q2-Q3'23 | +3.06% | +5.03% | +4.24% |
| P3: Q4'23-Q1'24 | +3.83% | +3.48% | +3.83% |
| P4: Q2-Q3'24 | +2.83% | +2.86% | +3.55% |
| **Average** | **+0.49%** | **+1.12%** | **+0.85%** |

3/4 periods profitable. P1 loses ~8% consistently — likely regime-specific.

### B&H vs Model vs Allocation Modes

| Mode | Model | Return | Sharpe | MaxDD |
|------|-------|--------|--------|-------|
| B&H | — | +119.33% | 1.300 | -26.96% |
| Equal-weight | LSTM | +52.43% | 1.446 | -9.25% |
| Conf-weighted | LSTM | +52.30% | 1.445 | -9.25% |
| Top-N | LSTM | +52.43% | 1.446 | -9.25% |

**B&H beats model on raw return (+119% vs +52%).** Model wins on risk-adjusted basis (Sharpe 1.45 vs 1.30) and drawdown (-9.3% vs -27%). Allocation modes produce nearly identical results — model doesn't trade enough tickers simultaneously for max_positions to bind.

### Single-Stock: Model vs Buy-and-Hold

| Ticker | Model | B&H | Alpha |
|--------|-------|-----|-------|
| BHARTIARTL | +0.7% | +72.1% | -71.4pp |
| LT | +0.9% | +68.9% | -68.0pp |
| ITC | +0.3% | +45.1% | -44.8pp |
| SBIN | +0.3% | +34.5% | -34.1pp |
| KOTAKBANK | -0.5% | -8.7% | +8.3pp |

**Model has negative alpha on individual stocks.** It's essentially market-neutral (flat ~0.3%) while B&H captures the bull market. Model's edge is risk management, not return generation.
