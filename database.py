"""
Crypto Backtest System - Database Setup
Creates SQLite database with tables for all data types.
Designed for cross-signal analysis from day one.
"""

import sqlite3
import os

DB_PATH = "crypto_backtest.db"


def create_database(db_path=DB_PATH):
    """Create all tables. Safe to run multiple times (IF NOT EXISTS)."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # === DAILY PRICES (CoinGecko) ===
    # One row per coin per day. Foundation table for all backtests.
    c.execute("""
        CREATE TABLE IF NOT EXISTS daily_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coingecko_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            bucket INTEGER NOT NULL,
            date TEXT NOT NULL,
            price_usd REAL,
            market_cap REAL,
            volume_usd REAL,
            UNIQUE(coingecko_id, date)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON daily_prices(date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_prices_coin ON daily_prices(coingecko_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_prices_bucket ON daily_prices(bucket)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker ON daily_prices(ticker)")

    # === FEAR & GREED INDEX (Alternative.me) ===
    # Daily sentiment score 0-100. BTC-centric but applies broadly.
    c.execute("""
        CREATE TABLE IF NOT EXISTS fear_greed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            value INTEGER NOT NULL,
            classification TEXT NOT NULL
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_fg_date ON fear_greed(date)")

    # === ON-CHAIN METRICS (Glassnode free tier) ===
    # Daily metrics for BTC and ETH only on free tier.
    c.execute("""
        CREATE TABLE IF NOT EXISTS onchain_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coingecko_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            value REAL,
            UNIQUE(coingecko_id, date, metric_name)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_onchain_date ON onchain_metrics(date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_onchain_coin ON onchain_metrics(coingecko_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_onchain_metric ON onchain_metrics(metric_name)")

    # === FUNDING RATES (Binance) ===
    # Every 8 hours for perpetual contracts. Key derivatives signal.
    c.execute("""
        CREATE TABLE IF NOT EXISTS funding_rates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            ticker TEXT NOT NULL,
            funding_time TEXT NOT NULL,
            funding_rate REAL NOT NULL,
            UNIQUE(symbol, funding_time)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_funding_time ON funding_rates(funding_time)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_funding_symbol ON funding_rates(symbol)")

    # === COLLECTION LOG ===
    # Track what's been collected to enable incremental updates.
    c.execute("""
        CREATE TABLE IF NOT EXISTS collection_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            coingecko_id TEXT,
            metric TEXT,
            last_collected_date TEXT,
            records_added INTEGER,
            collected_at TEXT NOT NULL,
            status TEXT DEFAULT 'success',
            notes TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database created/verified: {db_path}")
    return db_path


def get_db_stats(db_path=DB_PATH):
    """Print summary of what's in the database."""
    if not os.path.exists(db_path):
        print("Database does not exist yet.")
        return

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    print("=" * 60)
    print("CRYPTO BACKTEST DATABASE SUMMARY")
    print("=" * 60)

    # Daily prices
    c.execute("SELECT COUNT(*) FROM daily_prices")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT coingecko_id) FROM daily_prices")
    coins = c.fetchone()[0]
    c.execute("SELECT MIN(date), MAX(date) FROM daily_prices")
    row = c.fetchone()
    print(f"\nDAILY PRICES: {total:,} records | {coins} coins | {row[0]} to {row[1]}")

    # By bucket
    for bucket in [1, 2, 3]:
        c.execute("""SELECT COUNT(DISTINCT coingecko_id), COUNT(*), MIN(date), MAX(date) 
                     FROM daily_prices WHERE bucket = ?""", (bucket,))
        r = c.fetchone()
        labels = {1: "Blue Chips", 2: "Top 20", 3: "Altcoins"}
        if r[0] > 0:
            print(f"  Bucket {bucket} ({labels[bucket]}): {r[0]} coins, {r[1]:,} rows, {r[2]} to {r[3]}")

    # Fear & Greed
    c.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM fear_greed")
    r = c.fetchone()
    print(f"\nFEAR & GREED: {r[0]:,} days | {r[1]} to {r[2]}")

    # On-chain
    c.execute("SELECT COUNT(*), COUNT(DISTINCT metric_name) FROM onchain_metrics")
    r = c.fetchone()
    print(f"\nON-CHAIN: {r[0]:,} records | {r[1]} unique metrics")

    # Funding rates
    c.execute("SELECT COUNT(*), COUNT(DISTINCT symbol) FROM funding_rates")
    r = c.fetchone()
    print(f"\nFUNDING RATES: {r[0]:,} records | {r[1]} symbols")

    # Collection log
    c.execute("SELECT source, MAX(collected_at), SUM(records_added) FROM collection_log GROUP BY source")
    rows = c.fetchall()
    if rows:
        print(f"\nCOLLECTION LOG:")
        for row in rows:
            print(f"  {row[0]}: last run {row[1]}, {row[2]:,} total records added")

    print("=" * 60)
    conn.close()


if __name__ == "__main__":
    create_database()
    get_db_stats()
