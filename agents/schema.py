"""Shared state store — SQLite-based inter-agent communication.

Each agent reads from upstream tables and writes to its own output table.
All tables include timestamp for traceability. The DB file is inspectable
with any SQLite client.

Tables:
  signals      — Agent 1 output: model predictions per ticker per bar
  regime       — Agent 2 output: market regime classification
  portfolio    — Agent 4 output: target allocations per ticker
  executions   — Agent 5 output: actual fills and risk events
  agent_log    — All agents log decisions here for traceability
"""
import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent.parent / "backtest" / "multiagent_state.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_idx INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    signal TEXT NOT NULL,           -- Buy/Hold/Sell
    confidence REAL NOT NULL,
    prob_buy REAL,
    prob_hold REAL,
    prob_sell REAL,
    model TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS regime (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_idx INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    regime_label TEXT NOT NULL,     -- calm-trending/calm-choppy/volatile-trending/volatile-choppy/drawdown
    regime_confidence REAL NOT NULL,
    features_json TEXT,             -- raw regime features for debugging
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS portfolio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_idx INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    target_weight REAL NOT NULL,    -- fraction of capital (0 = no position)
    signal TEXT NOT NULL,           -- Buy/Hold/Sell (derived from signal+regime)
    size_pct REAL,                  -- position size as % of capital
    reason TEXT,                    -- why this allocation
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_idx INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,           -- Buy/Sell/StopLoss/TakeProfit
    price REAL NOT NULL,
    size REAL NOT NULL,
    cost REAL,
    pnl REAL,
    exit_reason TEXT,
    equity_after REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS agent_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_idx INTEGER,
    agent TEXT NOT NULL,
    event TEXT NOT NULL,
    details_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS latency_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_idx INTEGER,
    ticker TEXT,
    timestamp TEXT NOT NULL,
    stage TEXT NOT NULL,            -- 'feature_compute' | 'model_inference' | 'signal_decision' | 'full_pipeline'
    latency_ms REAL NOT NULL,
    model TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS manual_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,           -- Buy/Sell
    price REAL NOT NULL,
    size REAL NOT NULL,
    reason TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    headline TEXT NOT NULL,
    summary TEXT,
    source TEXT,
    url TEXT,
    published_at TEXT,
    sentiment_score REAL,           -- -1.0 (bearish) to +1.0 (bullish)
    relevance_score REAL,           -- 0.0 to 1.0
    fetched_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_signals_bar ON signals(bar_idx);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);
CREATE INDEX IF NOT EXISTS idx_regime_bar ON regime(bar_idx);
CREATE INDEX IF NOT EXISTS idx_portfolio_bar ON portfolio(bar_idx);
CREATE INDEX IF NOT EXISTS idx_executions_bar ON executions(bar_idx);
CREATE INDEX IF NOT EXISTS idx_agent_log_bar ON agent_log(bar_idx);
CREATE INDEX IF NOT EXISTS idx_latency_bar ON latency_metrics(bar_idx);
CREATE INDEX IF NOT EXISTS idx_manual_trades_ticker ON manual_trades(ticker);
CREATE INDEX IF NOT EXISTS idx_news_ticker ON news_articles(ticker);
CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at);
"""


def init_db(db_path=None):
    """Create/reset the state database."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA_SQL)
    conn.close()
    return path


@contextmanager
def get_conn(db_path=None):
    """Context manager for database connections."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def log_agent(agent, event, bar_idx=None, details=None, db_path=None):
    """Write an agent log entry."""
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO agent_log (bar_idx, agent, event, details_json) VALUES (?, ?, ?, ?)",
            (bar_idx, agent, event, json.dumps(details) if details else None)
        )


def reset_tables(db_path=None):
    """Clear all data tables (keep schema)."""
    with get_conn(db_path) as conn:
        for t in ["signals", "regime", "portfolio", "executions", "agent_log"]:
            conn.execute(f"DELETE FROM {t}")
