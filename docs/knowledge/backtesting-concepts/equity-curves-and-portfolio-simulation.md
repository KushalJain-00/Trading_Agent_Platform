# Equity Curves and Portfolio Simulation

## What It Is

An **equity curve** is a graph of your portfolio's total value over time. It
starts at your initial capital and goes up (profit) or down (loss) with each
trade. The equity curve is the single most important output of a backtest — it
shows not just the final result, but the path taken (including drawdowns,
recovery periods, and volatility).

**Portfolio simulation** is the process of tracking all positions, cash flows,
and mark-to-market valuations to produce the equity curve.

## Why It Matters for This Project

The equity curve is how we measure whether the strategy works. But computing
it correctly is harder than it looks — and this project hit a real bug.

### The Per-Ticker Compounding Bug

The original equity curve implementation used a per-ticker cumulative product:

```python
# BUGGY — DO NOT USE
df["strat_ret"] = df.groupby("ticker")["position"].shift(1) * df["close"].pct_change()
df["equity"] = df.groupby("ticker")["strat_ret"].transform(
    lambda x: capital * (1 + x).cumprod()
)
```

This computed equity independently for each ticker, as if each had its own
capital pool. In reality, all tickers share a single capital pool — entering a
position in RELIANCE reduces the cash available for TCS.

**The result:** The equity curve showed +2.19% return while the trade log
showed -58.59% return. The comparison table's "return" metric was reading from
the buggy equity curve, showing phantom profits.

**How it was diagnosed:** The assertion check was added:
```python
expected_final = capital + closed_pnl + open_unrealized
actual_final = float(eq_df["equity"].iloc[-1])
assert abs(actual_final - expected_final) < 1.0
```
This revealed a ₹6 billion discrepancy between the equity curve and the trade
log.

**The fix:** The equity curve was rewritten as a proper **multi-asset portfolio
ledger** in `simulator.py`:

```python
# State arrays indexed by integer ticker ID
current_shares = np.zeros(n_tickers)  # shares held per ticker
current_ep = np.zeros(n_tickers)      # entry price per ticker
last_price_arr = np.zeros(n_tickers)  # last known close per ticker

cash = float(capital)
mtm_sum = 0.0  # running mark-to-market of all open positions

for idx in range(n_bars):
    if pos == 1 and prev == 0:  # Entry
        cash -= entry_price * size
        mtm_sum += size * close
    elif pos == 0 and prev == 1:  # Exit
        cash += exit_price * size - costs
        mtm_sum -= shares * last_price
    elif pos == 1 and prev == 1:  # Hold
        mtm_sum += shares * (close - last_price)

    equity = cash + mtm_sum
```

After the fix: both the equity curve and trade log agree on -58.59% return
(for the unfiltered config). After applying filters, the corrected numbers
show +85% return.

## How It's Implemented Here

**Historical backtest** — `_run_backtest_core()` in `simulator.py`:
1. Sort all bars by timestamp (interleaving all tickers)
2. Maintain per-ticker state: current shares, entry price, last price
3. At each bar: update cash on entries/exits, update mark-to-market on holds
4. Record equity = cash + mtm_sum
5. Deduplicate timestamps (keep last value per timestamp)
6. Assert: final equity == capital + closed P&L + unrealized P&L

**Live paper trading** — `LiveSimulator` class in `simulator.py`:
- Same logic but bar-by-bar processing
- Uses `pending_signals` queue for latency simulation
- `process_bar(bar, signal)` updates state and appends to equity curve

## The Assertion Check

The assertion at the end of `_run_backtest_core()` verifies mathematical
consistency:

```python
closed_pnl = sum of net_pnl for all completed trades
open_unrealized = sum of (shares × (current_price - entry_price)) for open positions
expected_final = capital + closed_pnl + open_unrealized
actual_final = equity_curve[-1]

assert abs(actual_final - expected_final) < 1.0
```

This catches any bug in the equity curve computation. Every run in the full
sweep (60 runs) passes this assertion.

## Common Pitfalls

1. **Ignoring open positions.** The equity at the end of the backtest includes
   unrealized P&L on still-open positions. If you only count closed trades,
   you understate (or overstate) the final equity.

2. **Per-ticker compounding.** Treating each ticker as having its own capital
   pool inflates returns because it ignores capital constraints. Our fix tracks
   a single shared cash pool.

3. **Not verifying.** Without the assertion, bugs like the per-ticker
   compounding error can go unnoticed for months. The assertion is cheap
   insurance.
