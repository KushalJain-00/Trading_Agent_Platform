# Model Architectures Compared

## What It Is

This project trains four neural network architectures for time-series
classification. Each processes a window of 60 bars × 23 features differently,
capturing different types of patterns.

## The Four Architectures

### LSTM (Long Short-Term Memory)

**What it does:** Processes the window one bar at a time, maintaining a
"memory" vector that accumulates information across the sequence. At each step,
gates decide what to remember, what to forget, and what to output.

**In this project** (`models/lstm_model.py`):
- 2-layer LSTM, hidden size 64, dropout 0.2
- Takes the final hidden state `h_n[-1]` (last layer's output)
- Classification head: Linear(64→32) → ReLU → Linear(32→3)
- Processes each window sequentially (60 timesteps)

**Strengths:** Good at capturing long-range temporal dependencies. The gates
allow it to remember important events from many bars ago.

**Weaknesses:** Slow to train (sequential processing, can't parallelize across
timesteps). May forget very early information despite the gating mechanism.

### 1D CNN (Convolutional Neural Network)

**What it does:** Slides small filters (kernel size 3) across the window,
detecting local patterns. Multiple convolutional layers stack to capture
patterns at different scales.

**In this project** (`models/cnn_1d_model.py`):
- Transposes input from (batch, time, features) to (batch, features, time)
- Conv1d(23→32, kernel=3, padding=1) → ReLU → MaxPool1d(2)
- Conv1d(32→64, kernel=3, padding=1) → ReLU → MaxPool1d(2)
- Conv1d(64→64, kernel=3, padding=1) → ReLU
- AdaptiveAvgPool1d(1) → Flatten → Linear(64→32) → ReLU → Linear(32→3)

**Strengths:** Fast (convolutions are highly parallelizable). Good at detecting
local patterns (e.g., a specific 3-bar candlestick formation).

**Weaknesses:** Limited receptive field — even with 3 layers, it sees patterns
of ~12 bars at most. May miss longer-range dependencies that LSTM captures.

### CNN-LSTM Hybrid

**What it does:** Uses CNN layers to extract local features from the window,
then feeds those features into an LSTM for temporal reasoning.

**In this project** (`models/cnn_lstm_hybrid.py`):
- CNN backbone: 2 × Conv1d(input_dim→32, kernel=3, padding=1) → ReLU
- Transposes back to (batch, time, features) for LSTM
- LSTM(32→64, batch_first=True)
- Classification head: Linear(64→32) → ReLU → Linear(32→3)

**Strengths:** Combines CNN's local pattern detection with LSTM's temporal
memory. The CNN reduces the feature dimensionality before the LSTM processes it.

**Weaknesses:** More complex architecture, harder to debug. May not outperform
either component alone if the data doesn't have both local patterns and
long-range dependencies.

### Transformer

**What it does:** Uses self-attention to weigh the importance of every bar in
the window relative to every other bar, without sequential processing.

**In this project** (`models/transformer_model.py`):
- Projects 23 features to d_model=64 via Linear layer
- Learned positional encoding (not sinusoidal)
- 2 TransformerEncoderLayer (d_model=64, nhead=4, dim_feedforward=128)
- Mean pooling over sequence → Linear(64→32) → ReLU → Linear(32→3)

**Strengths:** Parallel processing (like CNN), but captures long-range
dependencies (like LSTM). Self-attention can learn which past bars matter most
for the current prediction.

**Weaknesses:** Most memory-intensive (attention matrix is O(n²)). May need more
data to learn attention patterns effectively. Not used in the final backtest
pipeline.

## Comparative Results

From the full corrected sweep (confidence=0.85, holding=75):

| Model | Return | Sharpe | MaxDD | Win Rate |
|-------|--------|--------|-------|----------|
| LSTM | +85.15% | 1.447 | -17.88% | 53.0% |
| CNN1D | +85.05% | 1.453 | -17.99% | 52.8% |
| CNN-LSTM | +83.87% | 1.416 | -18.40% | 53.2% |

All three models perform similarly at this config. The differences (1-2% in
return) are within the noise margin — the edge comes more from the
configuration (confidence threshold, holding period) than the architecture.

## Tunable Parameters

| Parameter | LSTM | CNN1D | CNN-LSTM | Transformer |
|-----------|------|-------|----------|-------------|
| Hidden/channels | 64 | 32→64→64 | 32→64 | d_model=64 |
| Layers | 2 | 3 conv | 2 conv + 1 LSTM | 2 |
| Dropout | 0.2 | 0 | 0.2 (LSTM part) | 0 |
| Kernel size | — | 3 | 3 | — |
| Heads | — | — | — | 4 |

These are not tuned — they're reasonable defaults that train in reasonable
time on CPU.

## Common Pitfalls

1. **Choosing architecture by complexity.** More complex ≠ better. Our CNN1D
   (simplest) performs within 0.1% of LSTM (more complex) at the best config.

2. **Ignoring training time.** On CPU-only hardware, LSTM takes ~2x longer to
   train than CNN1D. If you're iterating quickly, start with CNN1D.

3. **Not comparing on the same data with the same config.** Our sweep tests all
   three models with identical configs (same confidence threshold, holding
   period, cost parameters). This is the only fair comparison.
