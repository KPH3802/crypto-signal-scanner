"""
Crypto Backtest System - Data Collector
Pulls historical data from free APIs into SQLite.

Prices: yfinance (no API key, full history back to coin launch)
Sentiment: Alternative.me Fear & Greed Index (free, no key)
Derivatives: Binance funding rates (free, no key)

Usage:
    python3 collect_data.py --all           # Collect everything
    python3 collect_data.py --prices        # Only prices
    python3 collect_data.py --fear-greed    # Only Fear & Greed
    python3 collect_data.py --funding       # Only funding rates
    python3 collect_data.py --stats         # Show database stats
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta

import requests
import yfinance as yf

from database import create_database, get_db_stats, DB_PATH

# Try to import config, fall back to config_example
try:
    from config import *
except ImportError:
    from config_example import *


# ============================================================
# YAHOO FINANCE TICKER MAPPING
# ============================================================

YAHOO_TICKERS = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "binancecoin": "BNB-USD",
    "solana": "SOL-USD",
    "ripple": "XRP-USD",
    "cardano": "ADA-USD",
    "dogecoin": "DOGE-USD",
    "tron": "TRX-USD",
    "chainlink": "LINK-USD",
    "avalanche-2": "AVAX-USD",
    "polkadot": "DOT-USD",
    "polygon-ecosystem-token": "POL-USD",
    "shiba-inu": "SHIB-USD",
    "toncoin": "TON-USD",
    "litecoin": "LTC-USD",
    "uniswap": "UNI-USD",
    "stellar": "XLM-USD",
    "near": "NEAR-USD",
    "sui": "SUI-USD",
    "internet-computer": "ICP-USD",
    "aave": "AAVE-USD",
    "arbitrum": "ARB-USD",
    "optimism": "OP-USD",
    "render-token": "RNDR-USD",
    "injective-protocol": "INJ-USD",
    "sei-network": "SEI-USD",
    "celestia": "TIA-USD",
    "jupiter-exchange-solana": "JUP-USD",
    "bonk": "BONK-USD",
    "pepe": "PEPE-USD",
    "floki": "FLOKI-USD",
    "worldcoin-wld": "WLD-USD",
    "kaspa": "KAS-USD",
    "stacks": "STX-USD",
    "maker": "MKR-USD",
    "the-graph": "GRT-USD",
    "immutable-x": "IMX-USD",
    "gala": "GALA-USD",
    "fetch-ai": "FET-USD",
    "pendle": "PENDLE-USD",
}


# ============================================================
# YFINANCE PRICE COLLECTOR
# ============================================================

class PriceCollector:
    """Collect daily OHLCV from Yahoo Finance. No API key needed, full history."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def get_last_collected_date(self, coingecko_id):
        """Check what we already have in the DB for this coin."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT MAX(date) FROM daily_prices WHERE coingecko_id = ?",
                  (coingecko_id,))
        result = c.fetchone()[0]
        conn.close()
        return result

    def collect_coin_prices(self, coingecko_id, ticker, bucket):
        """Collect daily prices for a single coin via yfinance."""
        yahoo_ticker = YAHOO_TICKERS.get(coingecko_id)
        if not yahoo_ticker:
            print(f"  {ticker}: No Yahoo Finance mapping, skipping")
            return 0

        last_date = self.get_last_collected_date(coingecko_id)

        if last_date:
            last_dt = datetime.strptime(last_date, "%Y-%m-%d")
            days_since = (datetime.now() - last_dt).days
            if days_since <= 1:
                print(f"  {ticker}: already up to date ({last_date})")
                return 0
            start_date = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"  {ticker}: updating from {last_date}")
        else:
            start_date = "2013-01-01"
            print(f"  {ticker}: full history pull")

        try:
            df = yf.download(
                yahoo_ticker,
                start=start_date,
                end=datetime.now().strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True
            )
        except Exception as e:
            print(f"  {ticker}: yfinance error: {e}")
            return 0

        if df is None or df.empty:
            print(f"  {ticker}: no data returned from Yahoo Finance")
            return 0

        # Handle multi-level columns from yfinance
        if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
            df.columns = df.columns.get_level_values(0)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        inserted = 0

        for date_idx, row in df.iterrows():
            date_str = date_idx.strftime("%Y-%m-%d")

            if last_date and date_str <= last_date:
                continue

            price = float(row["Close"]) if "Close" in row else None
            volume = float(row["Volume"]) if "Volume" in row else None

            if price is None or price == 0:
                continue

            try:
                c.execute("""
                    INSERT OR IGNORE INTO daily_prices
                    (coingecko_id, ticker, bucket, date, price_usd, market_cap, volume_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (coingecko_id, ticker, bucket, date_str, price, None, volume))
                if c.rowcount > 0:
                    inserted += 1
            except sqlite3.Error as e:
                print(f"  DB error for {ticker}/{date_str}: {e}")

        conn.commit()

        last_row_date = df.index[-1].strftime("%Y-%m-%d") if len(df) > 0 else None
        c.execute("""
            INSERT INTO collection_log (source, coingecko_id, metric, last_collected_date,
                                       records_added, collected_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("yfinance", coingecko_id, "daily_price",
              last_row_date, inserted,
              datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "success"))
        conn.commit()
        conn.close()

        first_date = df.index[0].strftime("%Y-%m-%d") if len(df) > 0 else "N/A"
        print(f"  {ticker}: +{inserted} rows ({first_date} to {last_row_date})")
        return inserted

    def collect_all_prices(self):
        """Collect prices for all coins in the universe."""
        all_coins = get_all_coins()
        total_inserted = 0
        total_coins = len(all_coins)
        failed = []

        print(f"\n{'='*60}")
        print(f"COLLECTING PRICES (yfinance): {total_coins} coins")
        print(f"No API key needed - full history available")
        print(f"{'='*60}")

        for i, (cg_id, ticker) in enumerate(all_coins.items(), 1):
            bucket = get_bucket_for_coin(cg_id)
            bucket_label = {1: "Blue Chip", 2: "Top 20", 3: "Altcoin"}[bucket]
            print(f"\n[{i}/{total_coins}] {ticker} ({bucket_label})")

            inserted = self.collect_coin_prices(cg_id, ticker, bucket)
            total_inserted += inserted
            if inserted == 0 and not self.get_last_collected_date(cg_id):
                failed.append(ticker)

            time.sleep(0.5)

        print(f"\n{'='*60}")
        print(f"PRICE COLLECTION COMPLETE")
        print(f"Total new rows: {total_inserted:,}")
        if failed:
            print(f"Failed/no data: {', '.join(failed)}")
        print(f"{'='*60}")

        return total_inserted


# ============================================================
# FEAR & GREED INDEX COLLECTOR
# ============================================================

class FearGreedCollector:
    """Collect Crypto Fear & Greed Index from Alternative.me."""

    API_URL = "https://api.alternative.me/fng/"

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def collect(self):
        """Pull full history of Fear & Greed Index."""
        print(f"\n{'='*60}")
        print("COLLECTING FEAR & GREED INDEX")
        print(f"{'='*60}")

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT MAX(date) FROM fear_greed")
        last_date = c.fetchone()[0]
        conn.close()

        if last_date:
            last_dt = datetime.strptime(last_date, "%Y-%m-%d")
            days_needed = (datetime.now() - last_dt).days
            if days_needed <= 1:
                print(f"  Already up to date ({last_date})")
                return 0
            print(f"  Last collected: {last_date}. Fetching {days_needed} new days.")
        else:
            print("  No existing data. Fetching full history.")

        params = {"limit": 0, "format": "json"}

        try:
            resp = requests.get(self.API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  FAILED: {e}")
            return 0

        entries = data.get("data", [])
        if not entries:
            print("  No data returned")
            return 0

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        inserted = 0

        for entry in entries:
            ts = int(entry["timestamp"])
            date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            value = int(entry["value"])
            classification = entry["value_classification"]

            if last_date and date_str <= last_date:
                continue

            try:
                c.execute("""
                    INSERT OR IGNORE INTO fear_greed (date, value, classification)
                    VALUES (?, ?, ?)
                """, (date_str, value, classification))
                if c.rowcount > 0:
                    inserted += 1
            except sqlite3.Error as e:
                print(f"  DB error for {date_str}: {e}")

        c.execute("""
            INSERT INTO collection_log (source, metric, records_added, collected_at, status)
            VALUES (?, ?, ?, ?, ?)
        """, ("alternative_me", "fear_greed", inserted,
              datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "success"))

        conn.commit()
        conn.close()

        print(f"  Inserted {inserted:,} new days (total entries received: {len(entries)})")
        return inserted


# ============================================================
# BINANCE FUNDING RATE COLLECTOR
# ============================================================

class FundingRateCollector:
    """Collect perpetual futures funding rates from Binance."""

    BASE_URL = "https://fapi.binance.com"

    SYMBOLS = {
        "BTCUSDT": "BTC", "ETHUSDT": "ETH", "BNBUSDT": "BNB",
        "SOLUSDT": "SOL", "XRPUSDT": "XRP", "ADAUSDT": "ADA",
        "DOGEUSDT": "DOGE", "AVAXUSDT": "AVAX", "DOTUSDT": "DOT",
        "LINKUSDT": "LINK", "SHIBUSDT": "SHIB", "LTCUSDT": "LTC",
        "UNIUSDT": "UNI", "NEARUSDT": "NEAR", "SUIUSDT": "SUI",
        "AAVEUSDT": "AAVE", "ARBUSDT": "ARB", "OPUSDT": "OP",
        "INJUSDT": "INJ", "MKRUSDT": "MKR", "PEPEUSDT": "PEPE",
        "FETUSDT": "FET", "GRTUSDT": "GRT",
    }

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.session = requests.Session()

    def get_last_collected_time(self, symbol):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT MAX(funding_time) FROM funding_rates WHERE symbol = ?",
                  (symbol,))
        result = c.fetchone()[0]
        conn.close()
        return result

    def collect_symbol(self, symbol, ticker):
        last_time = self.get_last_collected_time(symbol)
        params = {"symbol": symbol, "limit": 1000}

        if last_time:
            last_ts = int(datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
            params["startTime"] = last_ts + 1
            print(f"  {ticker}: updating from {last_time}")
        else:
            print(f"  {ticker}: full history pull")

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        total_inserted = 0

        while True:
            try:
                resp = self.session.get(
                    f"{self.BASE_URL}/fapi/v1/fundingRate",
                    params=params, timeout=30
                )
                if resp.status_code == 429:
                    print(f"  Rate limited, waiting 60s...")
                    time.sleep(60)
                    continue
                if resp.status_code != 200:
                    print(f"  HTTP {resp.status_code} for {symbol}")
                    break

                data = resp.json()
                if not data:
                    break

                for entry in data:
                    ft = datetime.utcfromtimestamp(
                        entry["fundingTime"] / 1000
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    rate = float(entry["fundingRate"])
                    try:
                        c.execute("""
                            INSERT OR IGNORE INTO funding_rates
                            (symbol, ticker, funding_time, funding_rate)
                            VALUES (?, ?, ?, ?)
                        """, (symbol, ticker, ft, rate))
                        if c.rowcount > 0:
                            total_inserted += 1
                    except sqlite3.Error:
                        pass

                conn.commit()
                if len(data) < 1000:
                    break
                params["startTime"] = data[-1]["fundingTime"] + 1
                time.sleep(0.5)

            except requests.exceptions.RequestException as e:
                print(f"  Request error: {e}")
                time.sleep(5)
                break

        c.execute("""
            INSERT INTO collection_log (source, coingecko_id, metric, records_added,
                                       collected_at, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("binance_funding", symbol, "funding_rate", total_inserted,
              datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "success"))
        conn.commit()
        conn.close()

        print(f"  {ticker}: +{total_inserted} records")
        return total_inserted

    def collect_all(self):
        print(f"\n{'='*60}")
        print(f"COLLECTING FUNDING RATES: {len(self.SYMBOLS)} symbols")
        print(f"{'='*60}")

        total = 0
        for i, (symbol, ticker) in enumerate(self.SYMBOLS.items(), 1):
            print(f"\n[{i}/{len(self.SYMBOLS)}] {symbol}")
            total += self.collect_symbol(symbol, ticker)
            time.sleep(1)

        print(f"\n{'='*60}")
        print(f"FUNDING RATE COLLECTION COMPLETE: {total:,} new records")
        print(f"{'='*60}")
        return total


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Crypto Backtest Data Collector")
    parser.add_argument("--all", action="store_true", help="Collect everything")
    parser.add_argument("--prices", action="store_true", help="Collect prices (yfinance)")
    parser.add_argument("--fear-greed", action="store_true", help="Collect Fear & Greed")
    parser.add_argument("--funding", action="store_true", help="Collect Binance funding rates")
    parser.add_argument("--stats", action="store_true", help="Show database stats")
    args = parser.parse_args()

    if not any([args.all, args.prices, args.fear_greed, args.funding, args.stats]):
        args.all = True

    create_database()

    if args.stats:
        get_db_stats()
        return

    start_time = time.time()
    print(f"\nCollection started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.all or args.prices:
        pc = PriceCollector()
        pc.collect_all_prices()

    if args.all or args.fear_greed:
        fg = FearGreedCollector()
        fg.collect()

    if args.all or args.funding:
        fr = FundingRateCollector()
        fr.collect_all()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL COLLECTION COMPLETE")
    print(f"Elapsed: {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    get_db_stats()


if __name__ == "__main__":
    main()
