# Execution Latency

## What It Is

**Execution latency** is the delay between when a signal is generated and when
the trade is actually executed. In the real world, this includes:

1. **Signal computation time**: How long the model takes to process the window
2. **Network latency**: Time for the order to reach the exchange
3. **Order matching**: Time for the exchange to find a counterparty
4. **Confirmation**: Time for the fill to come back

In a backtest, latency is simulated by delaying the trade by a fixed number of
bars.

## Why It Matters for This Project

**Lookahead bias** is the most dangerous backtesting error. It happens when you
use information that wouldn't be available at the time of the trade decision.

For example, if you receive a "Buy" signal at bar 100 and execute at bar 100's
close price, you're cheating — the close price at bar 100 is only known AFTER
bar 100 ends. In reality, you'd execute at bar 101's open (or close), which
could be significantly different.

This project avoids lookahead bias by shifting positions forward by
`latency_bars=1`:

```python
if latency_bars > 0:
    merged["position"] = merged.groupby("ticker")["raw_pos"]\
        .shift(latency_bars).fillna(0).astype(int)
```

This means: if the model predicts Buy at bar 100, the position starts at bar
101. The trade uses bar 101's close price (adjusted for spread), which is the
earliest realistic execution point.

## How It's Implemented Here

In `simulator.py`, the position shift happens after signal generation:

```python
# Raw position: 1 if Buy signal, 0 otherwise
merged["raw_pos"] = (merged["predicted_signal"] == "Buy").astype(int)

# Shifted position: delayed by latency_bars
merged["position"] = merged.groupby("ticker")["raw_pos"]\
    .shift(latency_bars).fillna(0).astype(int)
```

The `LiveSimulator` class in `simulator.py` uses a `pending_signals` queue for
the same purpose:
```python
def process_bar(self, bar, signal):
    # Execute signals from latency_bars ago
    while self.pending_signals and self.bars_processed >= self.pending_signals[0][1]:
        sig, _, b = self.pending_signals.pop(0)
        self._execute_fill(sig, b)
    # Queue current signal for future execution
    self.pending_signals.append((signal, self.bars_processed + self.latency_bars, bar))
```

## Tunable Parameters

| Parameter | Default | What It Does | Effect of Increasing | Effect of Decreasing |
|-----------|---------|--------------|---------------------|---------------------|
| `latency_bars` | 1 | Bars between signal and execution | More conservative, avoids lookahead, lower returns | More optimistic, may include lookahead, higher returns |

**Common values:**
- `0`: No latency (dangerous — likely includes lookahead bias)
- `1`: Next-bar execution (our default — conservative and realistic)
- `2-5`: Multi-bar delay (for slow execution or illiquid markets)

## Common Pitfalls

1. **Using latency=0.** This is the most common backtesting mistake. The model
   "sees" bar 100's close and trades at bar 100's close — impossible in
   reality. Our default of 1 prevents this.

2. **Inconsistent latency between backtest and live.** If the backtest uses
   latency=1 but live trading has 5-second latency (which is < 1 minute bar),
   the backtest is more optimistic. For minute-level data, latency=1 is
   realistic.

3. **Forgetting latency in the equity curve.** The equity curve must use the
   delayed position, not the raw signal. Our implementation shifts the position
   BEFORE computing the equity curve, so the curve reflects realistic execution.
