"""Vectorized backtester. No lookahead by construction."""
import numpy as np
import pandas as pd
import torch

DEFAULT_COST_BPS = 7


def model_strategy(model, feature_cols, window_size, device):
    model.eval()

    def strategy(df_slice):
        if len(df_slice) < window_size + 1:
            return pd.Series(0, index=df_slice.index)
        feats = df_slice[feature_cols].values.astype(np.float32)
        n = len(feats)
        positions = np.zeros(n, dtype=int)
        windows, valid_idx = [], []
        for i in range(window_size, n):
            w = feats[i - window_size:i]
            if not np.any(np.isnan(w)):
                windows.append(w)
                valid_idx.append(i)
        if windows:
            batch = torch.from_numpy(np.array(windows)).to(device)
            with torch.no_grad():
                preds = model(batch).argmax(1).cpu().numpy()
            for idx, pred in zip(valid_idx, preds):
                if pred == 0:
                    positions[idx] = 1
                elif pred == 2:
                    positions[idx] = -1
        return pd.Series(positions, index=df_slice.index)

    return strategy


def buy_and_hold_strategy(df):
    return pd.Series(1, index=df.index)


def run_backtest(prices, positions, cost_bps=DEFAULT_COST_BPS):
    returns = prices.pct_change().fillna(0)
    exposure = positions.shift(1).fillna(0)
    strat_returns = exposure * returns
    pos_change = exposure.diff().fillna(0).abs()
    strat_returns -= pos_change * (cost_bps / 10000)

    cum = (1 + strat_returns).cumprod()
    total_return = cum.iloc[-1] - 1.0
    n = len(strat_returns)
    years = max(n / (252 * 390), 0.01)
    cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1.0
    sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(252 * 390) if strat_returns.std() > 0 else 0.0
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    trades = strat_returns[pos_change > 0]
    return {
        "total_return": total_return, "cagr": cagr, "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": (trades > 0).mean() if len(trades) else 0.0,
        "num_trades": int(pos_change.gt(0).sum()),
    }


def backtest_model(model, test_df, feature_cols, window_size, device, cost_bps=DEFAULT_COST_BPS):
    strategy = model_strategy(model, feature_cols, window_size, device)
    results = []
    for _, g in test_df.groupby("ticker"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        r = run_backtest(g["close"], strategy(g), cost_bps)
        results.append(r)
    return {
        "total_return": np.mean([r["total_return"] for r in results]),
        "sharpe": np.mean([r["sharpe"] for r in results]),
        "max_drawdown": np.mean([r["max_drawdown"] for r in results]),
        "win_rate": np.mean([r["win_rate"] for r in results]),
        "num_trades": sum(r["num_trades"] for r in results),
    }


def run_all_backtests(model_configs, test_df, feature_cols, window_size, device, cost_bps=DEFAULT_COST_BPS):
    results = {}
    for _, g in test_df.groupby("ticker"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        r = run_backtest(g["close"], buy_and_hold_strategy(g), cost_bps)
        if "baseline" not in results:
            results["baseline"] = {k: [] for k in r}
        for k in r:
            results["baseline"][k].append(r[k])

    baseline = {k: np.mean(v) if isinstance(v, list) and k != "num_trades" else sum(v) for k, v in results.pop("baseline").items()}
    results["baseline"] = baseline

    for name, (model, _) in model_configs.items():
        model.eval().to(device)
        results[name] = backtest_model(model, test_df, feature_cols, window_size, device, cost_bps)

    print("\n" + "=" * 80)
    print(f"{'Model':<14} {'Return':>9} {'Sharpe':>8} {'MaxDD':>9} {'WinRate':>9} {'Trades':>8}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<14} {r['total_return']:9.4f} {r['sharpe']:8.3f} "
              f"{r['max_drawdown']:9.4f} {r['win_rate']:9.4f} {r['num_trades']:8d}")
    print("=" * 80)
    return results
