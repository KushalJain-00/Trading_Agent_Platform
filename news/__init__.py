"""News intelligence pipeline — Tavily search + Firecrawl scrape + sentiment.

Fetches recent news for NSE tickers, stores headlines + summaries in SQLite.
Runs on configurable polling interval (default 30 min).

API keys: set TAVILY_API_KEY and FIRECRAWL_API_KEY env vars.
"""
import os
import time
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

try:
    from firecrawl import FirecrawlApp
except ImportError:
    FirecrawlApp = None

DB_PATH = Path(__file__).resolve().parent.parent / "backtest" / "multiagent_state.db"

# ── Config ────────────────────────────────────────────────────────────
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
SEARCH_INTERVAL_MIN = 30  # polling interval in minutes
MAX_ARTICLES_PER_TICKER = 5

# ── Sentiment keywords (simple heuristic) ────────────────────────────
BULLISH_WORDS = {
    "surge", "rally", "gain", "rise", "jump", "bull", "buy", "upgrade",
    "outperform", "beat", "strong", "record", "high", "profit", "growth",
    "expand", "boost", "soar", "climb", "positive", "recovery", "rebound",
    "upside", "momentum", "breakout",
}
BEARISH_WORDS = {
    "fall", "drop", "decline", "crash", "bear", "sell", "downgrade",
    "underperform", "miss", "weak", "loss", "low", "risk", "debt",
    "crisis", "plunge", "tumble", "negative", "slump", "correction",
    "downside", "pressure", "breakdown", "warning",
}


def _simple_sentiment(text):
    """Keyword-based sentiment score: -1.0 (bearish) to +1.0 (bullish)."""
    if not text:
        return 0.0
    words = set(text.lower().split())
    bull = len(words & BULLISH_WORDS)
    bear = len(words & BEARISH_WORDS)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


def _relevance_score(headline, ticker):
    """Simple relevance: does the headline mention the ticker name?"""
    if not headline or not ticker:
        return 0.0
    h = headline.lower()
    t = ticker.lower()
    if t in h:
        return 1.0
    # Partial match (e.g. "Reliance" in "RIL")
    return 0.3


def search_news_tavily(ticker, max_results=MAX_ARTICLES_PER_TICKER):
    """Search for recent news about a ticker using Tavily."""
    if not TAVILY_API_KEY or TavilyClient is None:
        return []

    client = TavilyClient(api_key=TAVILY_API_KEY)
    query = f"{ticker} NSE stock news India"

    try:
        result = client.search(query, max_results=max_results, search_depth="basic")
        articles = []
        for r in result.get("results", []):
            headline = r.get("title", "")
            content = r.get("content", "")
            # Use first 200 chars as summary (don't store full body)
            summary = content[:200] + "..." if len(content) > 200 else content
            articles.append({
                "ticker": ticker,
                "headline": headline,
                "summary": summary,
                "source": r.get("url", "").split("/")[2] if "/" in r.get("url", "") else "",
                "url": r.get("url", ""),
                "published_at": r.get("published_date", datetime.now(timezone.utc).isoformat()),
            })
        return articles
    except Exception as e:
        print(f"Tavily search error for {ticker}: {e}")
        return []


def scrape_article_firecrawl(url):
    """Scrape a full article using Firecrawl. Returns extracted text."""
    if not FIRECRAWL_API_KEY or FirecrawlApp is None:
        return ""

    app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    try:
        result = app.scrape_url(url, params={"formats": ["markdown"]})
        return result.get("markdown", "")[:2000]  # cap at 2000 chars
    except Exception as e:
        print(f"Firecrawl error: {e}")
        return ""


def fetch_and_store_news(tickers, db_path=None):
    """Fetch news for all tickers and store in SQLite.

    Returns total articles fetched.
    """
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    total = 0

    for ticker in tickers:
        articles = search_news_tavily(ticker)
        for art in articles:
            sentiment = _simple_sentiment(art["headline"] + " " + art.get("summary", ""))
            relevance = _relevance_score(art["headline"], ticker)

            # Deduplicate by URL
            existing = conn.execute(
                "SELECT id FROM news_articles WHERE url = ? AND ticker = ?",
                (art["url"], ticker)).fetchone()
            if existing:
                continue

            conn.execute(
                "INSERT INTO news_articles (ticker, headline, summary, source, url, published_at, sentiment_score, relevance_score) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ticker, art["headline"], art["summary"], art["source"],
                 art["url"], art["published_at"], sentiment, relevance))
            total += 1

        # Rate limit: 1 req/sec
        time.sleep(1)

    conn.commit()
    conn.close()
    return total


def get_news_for_ticker(ticker, limit=20, db_path=None):
    """Retrieve recent news for a ticker from the database."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    df = __import__("pandas").read_sql_query(
        "SELECT headline, summary, source, url, published_at, sentiment_score, relevance_score "
        "FROM news_articles WHERE ticker = ? ORDER BY published_at DESC LIMIT ?",
        conn, params=(ticker, limit))
    conn.close()
    return df


def get_sentiment_summary(ticker, days=30, db_path=None):
    """Get average sentiment over time for a ticker."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    df = __import__("pandas").read_sql_query(
        "SELECT DATE(published_at) as date, AVG(sentiment_score) as avg_sentiment, COUNT(*) as n_articles "
        "FROM news_articles WHERE ticker = ? AND published_at >= ? GROUP BY DATE(published_at) ORDER BY date",
        conn, params=(ticker, cutoff))
    conn.close()
    return df
