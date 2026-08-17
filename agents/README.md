# Multi-Agent Trading System — Architecture

## Overview

Five communicating agents with a shared SQLite state store for traceability.

```
Agent 1 (Signals) ──→ signals table ──→ Agent 4 (Portfolio) ──→ portfolio table ──→ Agent 5 (Execution)
                                                         ↑
Agent 2 (Regime) ──→ regime table ───────────────────────┘
```

## Agents

### Agent 1: Signal Generation
- **Input**: Raw price data + trained model checkpoints
- **Output**: `signals` table (ticker, timestamp, signal, confidence, probabilities)
- **File**: `agents/agent1_signal.py`
- Wraps LSTM/CNN1D/CNN-LSTM with temperature calibration

### Agent 2: Regime Detection
- **Input**: All 113 tickers' close prices
- **Output**: `regime` table (timestamp, regime_label, regime_confidence, features)
- **File**: `agents/agent2_regime.py`
- Rules-based: volatility percentile, MA crossover, drawdown detection
- Regimes: calm-trending, calm-choppy, volatile-trending, volatile-choppy, drawdown

### Agent 3: Stock Selection (NOT BUILT — see below)
- Would rank tickers by trade-worthiness
- **Not built** because models' confidence doesn't vary meaningfully across tickers (std ~0.14, 63% >90% confidence)

### Agent 4: Portfolio Optimization
- **Input**: signals + regime tables
- **Output**: `portfolio` table (ticker, timestamp, target_weight, signal, size_pct, reason)
- **File**: `agents/agent4_portfolio.py`
- Regime-conditional exposure scaling:
  - calm-trending: 100% exposure
  - calm-choppy: 70%
  - volatile-trending: 60%
  - volatile-choppy: 30%
  - drawdown: 20%

### Agent 5: Execution & Risk Management
- **Input**: Agent 1 signals + Agent 4 portfolio targets
- **Output**: Equity curve + trade log (via existing simulator)
- **File**: `agents/agent5_execution.py`
- Merges filtered Buy signals with full signal stream (preserves Sell exits)
- Applies: costs, spread, latency, stop-loss, take-profit

## Shared State (SQLite)

Database: `backtest/multiagent_state.db`

### Tables

| Table | Written by | Key columns |
|-------|-----------|-------------|
| `signals` | Agent 1 | bar_idx, ticker, timestamp, signal, confidence, prob_buy/hold/sell, model |
| `regime` | Agent 2 | bar_idx, timestamp, regime_label, regime_confidence, features_json |
| `portfolio` | Agent 4 | bar_idx, ticker, timestamp, target_weight, signal, size_pct, reason |
| `executions` | Agent 5 | bar_idx, ticker, timestamp, action, price, size, cost, pnl, exit_reason |
| `agent_log` | All agents | bar_idx, agent, event, details_json |

### Schema definition
See `agents/schema.py` for CREATE TABLE statements and indexes.

## Running

### Full backtest
```bash
python -m backtest.run_multiagent --model lstm --stop-loss 0.05 --take-profit 0.10
```

### Individual agents
```bash
python -m agents.agent1_signal --model lstm
python -m agents.agent2_regime
python -m agents.agent4_portfolio
python -m agents.agent5_execution
```

### Inspect state
```bash
sqlite3 backtest/multiagent_state.db "SELECT * FROM regime LIMIT 10;"
sqlite3 backtest/multiagent_state.db "SELECT regime_label, COUNT(*) FROM regime GROUP BY regime_label;"
```

## Performance (full validation set, LSTM)

| Metric | Single Model + SL/TP | Multi-Agent + SL/TP |
|--------|---------------------|---------------------|
| Return | +74.65% | +44.38% |
| Sharpe | 1.449 | 1.478 |
| Max DD | -16.34% | -7.35% |
| Trades | 13,220 | 13,164 |
| Win Rate | 52.7% | 52.8% |

**Key trade-off**: Multi-agent sacrifices ~30pp return for halved drawdown and slightly better Sharpe.

## Known Limitations

1. **Regime detector doesn't catch P1**: The 10% drawdown threshold is too high for the equal-weight index during Jan-Mar 2023
2. **Stock selection not built**: Model confidence doesn't vary meaningfully across tickers — regime-level risk management is the real edge
3. **Temperature scaling barely helps**: Models are already well-calibrated (ECE ~2.5%) but genuinely overconfident (63% >90% confidence)
4. **~7 min per full pipeline run** on CPU (Agent 1 dominates at ~3 min)
