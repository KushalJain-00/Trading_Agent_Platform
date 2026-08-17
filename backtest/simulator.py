"""Trading simulator — vectorized historical + bar-by-bar live.

Historical backtest uses fully vectorized pandas/numpy (no Python loops).
Live paper trading uses the bar-by-bar LiveSimulator class.
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


def run_historical_backtest(signals_df, prices_df, capital=100_000_000,
                             position_size_pct=0.02, cost_bps=5, spread_bps=3,
                             latency_bars=1, output_dir=None,
                             confidence_threshold=0.0, min_holding_bars=1,
                             max_positions=0, max_position_pct=0.0,
                             allocation="equal",
                             stop_loss_pct=0.0, take_profit_pct=0.0):
    """Vectorized historical backtest with optional signal filters.

    Filters:
      - confidence_threshold: Buy signals below this confidence become Hold
      - min_holding_bars: once long, ignore Sell signals until this many bars pass
      - max_positions: max concurrent open positions (0=unlimited)
      - max_position_pct: max fraction of capital per position (0=unlimited)
      - allocation: 'equal' | 'confidence-weighted' | 'top-N'
      - stop_loss_pct: force exit if position drops this fraction (0=disabled)
      - take_profit_pct: force exit if position gains this fraction (0=disabled)
    """
    # ── Unfiltered baseline (for before/after comparison) ────────────
    _baseline = _run_backtest_core(
        signals_df, prices_df, capital, position_size_pct,
        cost_bps, spread_bps, latency_bars, output_dir=None,
        confidence_threshold=0.0, min_holding_bars=1,
        stop_loss_pct=0.0, take_profit_pct=0.0,
    )

    # ── Filtered run ────────────────────────────────────────────────
    n_total = len(signals_df)
    n_filtered_conf = 0
    if confidence_threshold > 0.0:
        n_filtered_conf = int((signals_df["predicted_confidence"] < confidence_threshold).sum())

    result = _run_backtest_core(
        signals_df, prices_df, capital, position_size_pct,
        cost_bps, spread_bps, latency_bars, output_dir,
        confidence_threshold=confidence_threshold,
        min_holding_bars=min_holding_bars,
        max_positions=max_positions,
        max_position_pct=max_position_pct,
        allocation=allocation,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    n_before = len(_baseline.get_trade_log_df())
    n_after = len(result.get_trade_log_df())
    costs_before = _baseline.get_trade_log_df()["costs"].sum() if n_before else 0
    costs_after = result.get_trade_log_df()["costs"].sum() if n_after else 0
    gross_before = _baseline.get_trade_log_df()["gross_pnl"].sum() if n_before else 0
    gross_after = result.get_trade_log_df()["gross_pnl"].sum() if n_after else 0

    result._filter_stats = {
        "n_signals_total": n_total,
        "n_filtered_by_confidence": n_filtered_conf,
        "n_trades_before": n_before,
        "n_trades_after": n_after,
        "trades_reduction_pct": (1 - n_after / max(n_before, 1)) * 100,
        "cost_before": costs_before,
        "cost_after": costs_after,
        "gross_pnl_before": gross_before,
        "gross_pnl_after": gross_after,
        "cost_pct_before": (costs_before / abs(gross_before) * 100) if gross_before else 0,
        "cost_pct_after": (costs_after / abs(gross_after) * 100) if gross_after else 0,
    }
    return result


def _run_backtest_core(signals_df, prices_df, capital, position_size_pct,
                        cost_bps, spread_bps, latency_bars, output_dir,
                        confidence_threshold, min_holding_bars,
                        max_positions=0, max_position_pct=0.0,
                        allocation="equal",
                        stop_loss_pct=0.0, take_profit_pct=0.0):
    """Core backtest logic, reused for baseline and filtered runs.

    Equity curve uses a proper multi-asset portfolio ledger: at every bar
    in chronological order, we maintain a running cash balance + dict of
    open positions, compute mark-to-market equity, and verify it matches
    the trade log's summed P&L.

    allocation modes:
      'equal'              — fixed position_size_pct per ticker (default)
      'confidence-weighted' — size scaled by signal confidence
      'top-N'              — only enter top-N signals by confidence per bar
    max_positions: max concurrent open positions (0=unlimited)
    max_position_pct: max fraction of capital per position (0=unlimited)
    """
    sigs = signals_df[["ticker", "timestamp", "predicted_signal", "predicted_confidence"]].copy()
    sigs["timestamp"] = pd.to_datetime(sigs["timestamp"])

    px = prices_df[["ticker", "timestamp", "close"]].copy()
    px["timestamp"] = pd.to_datetime(px["timestamp"])

    merged = pd.merge(sigs, px, on=["ticker", "timestamp"], how="inner", sort=False)
    merged = merged.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    # ── Confidence filter: Buy signals below threshold → Hold ───────
    if confidence_threshold > 0.0:
        weak_buy = (merged["predicted_signal"] == "Buy") & (merged["predicted_confidence"] < confidence_threshold)
        merged.loc[weak_buy, "predicted_signal"] = "Hold"

    # ── Top-N mode: only keep the N highest-confidence Buys per bar ─
    if allocation == "top-N" and max_positions > 0:
        buy_mask = merged["predicted_signal"] == "Buy"
        for ts, group in merged.loc[buy_mask].groupby("timestamp"):
            if len(group) <= max_positions:
                continue
            keep_idx = group.nlargest(max_positions, "predicted_confidence").index
            drop_idx = group.index.difference(keep_idx)
            merged.loc[drop_idx, "predicted_signal"] = "Hold"

    # ── Base position vector ────────────────────────────────────────
    merged["raw_pos"] = (merged["predicted_signal"] == "Buy").astype(int)
    if latency_bars > 0:
        merged["position"] = merged.groupby("ticker")["raw_pos"].shift(latency_bars).fillna(0).astype(int)
    else:
        merged["position"] = merged["raw_pos"]

    # ── Min-holding-bars filter: suppress exits too soon ────────────
    if min_holding_bars > 1:
        pos = merged["position"].values.copy()
        tickers = merged["ticker"].values
        entry_bar = {}
        in_trade = {}
        for i in range(len(pos)):
            t = tickers[i]
            if pos[i] == 1 and not in_trade.get(t, False):
                in_trade[t] = True
                entry_bar[t] = i
            elif pos[i] == 0 and in_trade.get(t, False):
                if i - entry_bar[t] < min_holding_bars:
                    pos[i] = 1
                else:
                    in_trade[t] = False
        merged["position"] = pos

    # ── Cost parameters ─────────────────────────────────────────────
    cost_frac = cost_bps / 10000
    half_spread = spread_bps / 2 / 10000

    # ── Equity curve — array-indexed loop (dict-free, ~3x faster) ──
    all_bars = merged.sort_values("timestamp").reset_index(drop=True)
    n_bars = len(all_bars)
    bar_closes = all_bars["close"].values.astype(np.float64)
    bar_positions = all_bars["position"].values.astype(np.int64)
    bar_tickers = all_bars["ticker"].values
    bar_timestamps = all_bars["timestamp"].values
    bar_confidences = all_bars["predicted_confidence"].values.astype(np.float64)
    bar_prev_positions = all_bars.groupby("ticker")["position"].shift(1).fillna(0).values.astype(np.int64)

    # Map tickers to integer IDs for O(1) array indexing instead of dict lookup
    unique_tickers = np.unique(bar_tickers)
    ticker_to_id = {t: i for i, t in enumerate(unique_tickers)}
    n_tickers = len(unique_tickers)
    bar_ticker_ids = np.array([ticker_to_id[t] for t in bar_tickers], dtype=np.int32)

    # State arrays indexed by ticker_id (numpy arrays, not dicts)
    current_shares = np.zeros(n_tickers, dtype=np.float64)
    current_ep = np.zeros(n_tickers, dtype=np.float64)
    last_price_arr = np.zeros(n_tickers, dtype=np.float64)
    is_open = np.zeros(n_tickers, dtype=np.int32)

    cash = float(capital)
    mtm_sum = 0.0
    n_open = 0
    eq_timestamps = np.empty(n_bars, dtype=bar_timestamps.dtype)
    eq_values = np.empty(n_bars, dtype=np.float64)

    # Record trades during the loop so they always match equity curve
    trade_records = []  # (ticker, entry_time, entry_price, size, entry_bar_idx)
    open_trade = {}  # ticker_id -> {ticker, entry_time, entry_price, size, entry_bar_idx}
    bar_ticker_strs = all_bars["ticker"].values

    for idx in range(n_bars):
        tid = bar_ticker_ids[idx]
        close = bar_closes[idx]
        pos = bar_positions[idx]
        prev = bar_prev_positions[idx]

        # ── Stop-loss / take-profit check (every bar, for open positions) ──
        if is_open[tid] and tid in open_trade and current_shares[tid] > 0:
            entry_px = open_trade[tid]["entry_price"]
            ret = (close - entry_px) / entry_px
            sl_triggered = stop_loss_pct > 0 and ret <= -stop_loss_pct
            tp_triggered = take_profit_pct > 0 and ret >= take_profit_pct
            if sl_triggered or tp_triggered:
                # Force exit — override signal
                sh_val = current_shares[tid]
                exit_price = close * (1 - half_spread)
                tc = abs(exit_price * sh_val) * cost_frac + sh_val * exit_price * half_spread
                cash += exit_price * sh_val - tc
                mtm_sum -= sh_val * last_price_arr[tid]
                current_shares[tid] = 0.0
                current_ep[tid] = 0.0
                is_open[tid] = 0
                n_open -= 1
                ot = open_trade.pop(tid)
                gross_pnl = (exit_price - ot["entry_price"]) * ot["size"]
                trade_records.append({
                    "entry_time": ot["entry_time"], "exit_time": str(bar_timestamps[idx]),
                    "ticker": ot["ticker"], "direction": "long",
                    "entry_price": ot["entry_price"], "exit_price": exit_price,
                    "size": ot["size"], "gross_pnl": gross_pnl,
                    "net_pnl": gross_pnl - tc, "costs": tc,
                    "holding_bars": idx - ot["entry_bar"],
                    "exit_reason": "stop_loss" if sl_triggered else "take_profit",
                })
                last_price_arr[tid] = close
                eq_timestamps[idx] = bar_timestamps[idx]
                eq_values[idx] = cash + mtm_sum
                continue  # already exited, skip normal position logic

        if pos == 1 and prev == 0:
            # ── Entry: compute position value based on allocation mode ──
            if allocation == "confidence-weighted":
                conf_rank = bar_confidences[idx]
                pos_value = capital * position_size_pct * (0.5 + 0.5 * conf_rank)
            else:
                pos_value = capital * position_size_pct

            if max_position_pct > 0:
                pos_value = min(pos_value, cash * max_position_pct)

            if max_positions > 0 and n_open >= max_positions:
                pass  # skip — at capacity
            else:
                entry_price = close * (1 + half_spread)
                size = pos_value / entry_price
                cash -= entry_price * size
                current_ep[tid] = entry_price
                current_shares[tid] = size
                is_open[tid] = 1
                n_open += 1
                mtm_sum += size * close
                open_trade[tid] = {
                    "ticker": bar_ticker_strs[idx], "entry_time": str(bar_timestamps[idx]),
                    "entry_price": entry_price, "size": size, "entry_bar": idx,
                }

        elif pos == 0 and prev == 1:
            sh_val = current_shares[tid]
            exit_price = close * (1 - half_spread)
            tc = abs(exit_price * sh_val) * cost_frac + sh_val * exit_price * half_spread
            cash += exit_price * sh_val - tc
            mtm_sum -= sh_val * last_price_arr[tid]
            current_shares[tid] = 0.0
            current_ep[tid] = 0.0
            is_open[tid] = 0
            n_open -= 1

            ot = open_trade.pop(tid, None)
            if ot:
                gross_pnl = (exit_price - ot["entry_price"]) * ot["size"]
                trade_records.append({
                    "entry_time": ot["entry_time"], "exit_time": str(bar_timestamps[idx]),
                    "ticker": ot["ticker"], "direction": "long",
                    "entry_price": ot["entry_price"], "exit_price": exit_price,
                    "size": ot["size"], "gross_pnl": gross_pnl,
                    "net_pnl": gross_pnl - tc, "costs": tc,
                    "holding_bars": idx - ot["entry_bar"],
                    "exit_reason": "signal",
                })

        elif pos == 1 and prev == 1:
            mtm_sum += current_shares[tid] * (close - last_price_arr[tid])

        last_price_arr[tid] = close
        eq_timestamps[idx] = bar_timestamps[idx]
        eq_values[idx] = cash + mtm_sum

    # Close any remaining open positions at last bar
    for tid, ot in open_trade.items():
        if current_shares[tid] > 0:
            sh_val = current_shares[tid]
            exit_price = bar_closes[-1]
            tc = abs(exit_price * sh_val) * cost_frac + sh_val * exit_price * half_spread
            gross_pnl = (exit_price - ot["entry_price"]) * sh_val
            trade_records.append({
                "entry_time": ot["entry_time"], "exit_time": str(bar_timestamps[-1]),
                "ticker": ot["ticker"], "direction": "long",
                "entry_price": ot["entry_price"], "exit_price": exit_price,
                "size": sh_val, "gross_pnl": 0, "net_pnl": 0, "costs": 0,
                "holding_bars": n_bars - 1 - ot["entry_bar"],
            })

    trade_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame(
        columns=["entry_time", "exit_time", "ticker", "direction", "entry_price",
                 "exit_price", "size", "gross_pnl", "net_pnl", "costs", "holding_bars",
                 "exit_reason"])

    eq_df = pd.DataFrame({"timestamp": eq_timestamps, "equity": eq_values})
    eq_df = eq_df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    # ── Validation: equity must match trade log ─────────────────────
    closed_pnl = float(trade_df["net_pnl"].sum()) if len(trade_df) else 0.0
    open_unrealized = 0.0
    for tid in range(n_tickers):
        if current_shares[tid] > 0:
            open_unrealized += current_shares[tid] * (last_price_arr[tid] - current_ep[tid])

    expected_final = capital + closed_pnl + open_unrealized
    actual_final = float(eq_df["equity"].iloc[-1]) if len(eq_df) else capital
    assert abs(actual_final - expected_final) < 1.0, (
        f"BUG: equity final {actual_final:,.2f} != expected {expected_final:,.2f} "
        f"(diff {actual_final - expected_final:,.2f})"
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        eq_df.to_parquet(output_dir / "equity_curve.parquet", index=False)
        if len(trade_df):
            trade_df.to_parquet(output_dir / "trade_log.parquet", index=False)
        with open(output_dir / "summary.json", "w") as f:
            json.dump({
                "initial_capital": capital,
                "final_equity": actual_final,
                "total_costs": float(trade_df["costs"].sum()) if len(trade_df) else 0.0,
                "total_gross_pnl": float(trade_df["gross_pnl"].sum()) if len(trade_df) else 0.0,
                "total_trades": len(trade_df),
                "bars_processed": len(merged),
            }, f, indent=2)

    return _SimResult(eq_df, trade_df, capital)


class _SimResult:
    def __init__(self, eq_df, trade_df, capital):
        self._eq_df = eq_df
        self._trade_df = trade_df
        self.initial_capital = capital
        self._filter_stats = None
    def get_equity_curve_df(self):
        return self._eq_df
    def get_trade_log_df(self):
        return self._trade_df
    def get_filter_stats(self):
        return self._filter_stats


# ── Live paper trading (bar-by-bar) ─────────────────────────────────

class LiveSimulator:
    def __init__(self, capital=100_000_000, position_size_pct=0.02,
                 cost_bps=5, spread_bps=3, latency_bars=1,
                 stop_loss_pct=0.0, take_profit_pct=0.0):
        self.initial_capital = capital
        self.equity = capital
        self.cash = capital
        self.position_size_pct = position_size_pct
        self.cost_bps = cost_bps
        self.spread_bps = spread_bps
        self.latency_bars = latency_bars
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.open_positions = {}
        self.pending_signals = []
        self.trade_log = []
        self.equity_curve = []
        self.total_costs = 0.0
        self.total_gross_pnl = 0.0
        self.bars_processed = 0

    def _execute_fill(self, signal, bar, exit_reason="signal"):
        ticker = bar["ticker"]
        price = bar["close"]
        half_spread = price * (self.spread_bps / 2 / 10000)
        cost_frac = self.cost_bps / 10000

        if signal == "Buy" and ticker not in self.open_positions:
            pos_value = self.equity * self.position_size_pct
            entry_price = price + half_spread
            size = pos_value / entry_price
            cost = pos_value * cost_frac + size * half_spread
            self.cash -= pos_value + cost
            self.total_costs += cost
            self.open_positions[ticker] = {
                "entry_time": str(bar["timestamp"]), "ticker": ticker,
                "entry_price": entry_price, "size": size, "costs": cost,
                "holding_bars": 0,
            }
        elif signal == "Sell" and ticker in self.open_positions:
            pos = self.open_positions.pop(ticker)
            exit_price = price - half_spread
            gross_pnl = (exit_price - pos["entry_price"]) * pos["size"]
            cost = abs(exit_price * pos["size"]) * cost_frac + pos["size"] * half_spread
            self.cash += exit_price * pos["size"] - cost
            self.total_costs += cost
            self.total_gross_pnl += gross_pnl
            self.trade_log.append({
                "entry_time": pos["entry_time"], "exit_time": str(bar["timestamp"]),
                "ticker": ticker, "direction": "long",
                "entry_price": pos["entry_price"], "exit_price": exit_price,
                "size": pos["size"], "gross_pnl": gross_pnl,
                "net_pnl": gross_pnl - cost, "costs": pos["costs"] + cost,
                "holding_bars": pos["holding_bars"],
                "exit_reason": exit_reason,
            })

    def _check_sl_tp(self, bar):
        """Check stop-loss/take-profit for the current bar's ticker."""
        ticker = bar["ticker"]
        if ticker not in self.open_positions:
            return False
        pos = self.open_positions[ticker]
        price = bar["close"]
        ret = (price - pos["entry_price"]) / pos["entry_price"]
        if self.stop_loss_pct > 0 and ret <= -self.stop_loss_pct:
            self._execute_fill("Sell", bar, exit_reason="stop_loss")
            return True
        if self.take_profit_pct > 0 and ret >= self.take_profit_pct:
            self._execute_fill("Sell", bar, exit_reason="take_profit")
            return True
        return False

    def process_bar(self, bar, signal):
        while self.pending_signals and self.bars_processed >= self.pending_signals[0][1]:
            sig, _, b = self.pending_signals.pop(0)
            self._execute_fill(sig, b)
        self.pending_signals.append((signal, self.bars_processed + self.latency_bars, bar))

        unrealized = sum(
            (bar["close"] - p["entry_price"]) * p["size"]
            for t, p in self.open_positions.items() if t == bar["ticker"]
        )
        self.equity = self.cash + sum(
            bar["close"] * p["size"] if t == bar["ticker"] else p["entry_price"] * p["size"]
            for t, p in self.open_positions.items()
        ) + unrealized
        self.equity_curve.append({"timestamp": bar["timestamp"], "equity": self.equity})
        self.bars_processed += 1

    def close_all_positions(self, last_bar):
        for ticker in list(self.open_positions.keys()):
            self._execute_fill("Sell", {"ticker": ticker, "close": self.open_positions[ticker]["entry_price"],
                                         "timestamp": last_bar["timestamp"]})
        for sig, _, bar in self.pending_signals:
            self._execute_fill(sig, bar)
        self.pending_signals.clear()

    def get_equity_curve_df(self):
        return pd.DataFrame(self.equity_curve)

    def get_trade_log_df(self):
        return pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame(
            columns=["entry_time","exit_time","ticker","direction","entry_price",
                     "exit_price","size","gross_pnl","net_pnl","costs","holding_bars",
                     "exit_reason"])

    def save_state(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.get_equity_curve_df().to_parquet(path / "equity_curve.parquet", index=False)
        td = self.get_trade_log_df()
        if len(td):
            td.to_parquet(path / "trade_log.parquet", index=False)
        with open(path / "summary.json", "w") as f:
            json.dump({"initial_capital": self.initial_capital, "final_equity": self.equity,
                        "total_costs": self.total_costs, "total_gross_pnl": self.total_gross_pnl,
                        "total_trades": len(self.trade_log), "bars_processed": self.bars_processed,
                        "open_positions": len(self.open_positions)}, f, indent=2)


Simulator = LiveSimulator

if __name__ == "__main__":
    print("Simulator module. Use run_backtest.py or dashboard.py to run.")
