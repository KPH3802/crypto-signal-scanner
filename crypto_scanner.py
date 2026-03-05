"""
Crypto Signal Scanner — Live Daily Alerts
Computes combined signal scores from backtest-validated indicators.
Sends email alert when actionable signals (Score ≥ 2) are detected.

Strategy (validated OOS 2022-2026, +7.27% alpha at 5d, p<0.0001):
  - Long-only, 3-5 day hold
  - Score ≥ 2 = BUY signal (+6.90% alpha, 57.7% win rate)
  - Score ≥ 3 = STRONG BUY (+8.72% alpha, 58.9% win rate)
  - Score ≥ 4 = VERY STRONG (+19.51% alpha, 67.9% win rate)

Scoring:
  +3: 5d drawdown ≥ 25% (crash family)
  +2: Extreme Greed FG ≥ 90 (trend/greed family)
  +1: 5d drawdown 15-25%
  +1: Greed Zone FG 75-89
  +1: Capitulation (3x volume + 5%+ daily drop)
  -1: Low volatility (coin's own bottom 10th percentile)
  -2: Fear Exit (FG crosses above 25 from below)

Schedule: Run daily after market close (e.g., 00:30 UTC)
Requires: Database with ≥20 days of history per coin

Usage:
    python3 crypto_scanner.py              # Full run: update data + scan + email
    python3 crypto_scanner.py --scan-only  # Scan existing data, no update
    python3 crypto_scanner.py --no-email   # Run but skip email
"""

import argparse
import os
import smtplib
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from database import DB_PATH

try:
    from config import *
except ImportError:
    from config_example import *


# ============================================================
# YAHOO FINANCE TICKER MAPPING
# yfinance periodically changes crypto symbols. If a ticker
# fails, check https://finance.yahoo.com for the current symbol.
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
    "toncoin": "TON11419-USD",
    "litecoin": "LTC-USD",
    "uniswap": "UNI7083-USD",
    "stellar": "XLM-USD",
    "near": "NEAR-USD",
    "sui": "SUI20947-USD",
    "internet-computer": "ICP-USD",
    "aave": "AAVE-USD",
    "arbitrum": "ARB11841-USD",
    "optimism": "OP-USD",
    "render-token": "RENDER-USD",
    "injective-protocol": "INJ-USD",
    "sei-network": "SEI-USD",
    "celestia": "TIA22861-USD",
    "jupiter-exchange-solana": "JUP29210-USD",
    "bonk": "BONK-USD",
    "pepe": "PEPE24478-USD",
    "floki": "FLOKI-USD",
    "worldcoin-wld": "WLD-USD",
    "kaspa": "KAS-USD",
    "stacks": "STX4847-USD",
    "maker": "MKR-USD",
    "the-graph": "GRT6719-USD",
    "immutable-x": "IMX10603-USD",
    "gala": "GALA-USD",
    "fetch-ai": "FET-USD",
    "pendle": "PENDLE-USD",
}

# Fallback tickers if primary fails (yfinance changes symbols often)
YAHOO_FALLBACKS = {
    "polygon-ecosystem-token": ["MATIC-USD"],
    "uniswap": ["UNI-USD"],
    "sui": ["SUI-USD"],
    "stacks": ["STX-USD"],
    "the-graph": ["GRT-USD"],
    "immutable-x": ["IMX-USD"],
    "pepe": ["PEPE-USD"],
    "render-token": ["RNDR-USD"],
    "toncoin": ["TON-USD"],
    "arbitrum": ["ARB-USD"],
    "celestia": ["TIA-USD"],
    "jupiter-exchange-solana": ["JUP-USD"],
}

BUCKET_LABELS = {1: "Blue Chip", 2: "Top 20", 3: "Altcoin"}
COST_MAP = {1: 0.20, 2: 0.40, 3: 0.80}


# ============================================================
# DATA UPDATE
# ============================================================

def download_with_fallback(cg_id, primary_ticker, start_date, end_date):
    """Try primary ticker, then fallbacks if it fails."""
    tickers_to_try = [primary_ticker]

    if cg_id in YAHOO_FALLBACKS:
        tickers_to_try.extend(YAHOO_FALLBACKS[cg_id])

    for ticker in tickers_to_try:
        try:
            df = yf.download(ticker, start=start_date, end=end_date,
                             progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue

    return None


def update_prices(db_path=DB_PATH):
    """Pull latest prices from yfinance for all coins in DB."""
    conn = sqlite3.connect(db_path)
    coins = pd.read_sql_query(
        "SELECT DISTINCT coingecko_id, ticker, bucket FROM daily_prices", conn
    )
    conn.close()

    print(f"Updating prices for {len(coins)} coins...")
    total_inserted = 0
    failed = []
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    for _, row in coins.iterrows():
        cg_id = row["coingecko_id"]
        ticker = row["ticker"]
        bucket = row["bucket"]
        yahoo = YAHOO_TICKERS.get(cg_id)

        if not yahoo:
            continue

        df = download_with_fallback(cg_id, yahoo, start_date, end_date)

        if df is None or df.empty:
            failed.append(ticker)
            continue

        if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
            df.columns = df.columns.get_level_values(0)

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        inserted = 0

        for date_idx, price_row in df.iterrows():
            date_str = date_idx.strftime("%Y-%m-%d")
            price = float(price_row["Close"]) if "Close" in price_row else None
            volume = float(price_row["Volume"]) if "Volume" in price_row else None

            if price is None or price == 0:
                continue

            try:
                c.execute("""
                    INSERT OR IGNORE INTO daily_prices
                    (coingecko_id, ticker, bucket, date, price_usd, market_cap, volume_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (cg_id, ticker, bucket, date_str, price, None, volume))
                if c.rowcount > 0:
                    inserted += 1
            except sqlite3.Error:
                pass

        conn.commit()
        conn.close()
        total_inserted += inserted

        if inserted > 0:
            print(f"  {ticker}: +{inserted} rows")

        time.sleep(0.3)

    print(f"Price update complete: +{total_inserted} total rows")
    if failed:
        print(f"  Failed to update ({len(failed)}): {', '.join(failed)}")
    return total_inserted


def update_fear_greed(db_path=DB_PATH):
    """Pull latest Fear & Greed Index."""
    print("Updating Fear & Greed Index...")

    try:
        resp = requests.get("https://api.alternative.me/fng/",
                            params={"limit": 7, "format": "json"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  F&G fetch failed: {e}")
        return 0

    entries = data.get("data", [])
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    inserted = 0

    for entry in entries:
        ts = int(entry["timestamp"])
        date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        value = int(entry["value"])
        classification = entry["value_classification"]

        try:
            c.execute("INSERT OR IGNORE INTO fear_greed (date, value, classification) "
                      "VALUES (?, ?, ?)", (date_str, value, classification))
            if c.rowcount > 0:
                inserted += 1
        except sqlite3.Error:
            pass

    conn.commit()
    conn.close()

    if inserted > 0:
        print(f"  F&G: +{inserted} days")
    else:
        print(f"  F&G: already up to date")

    return inserted


# ============================================================
# SIGNAL COMPUTATION
# ============================================================

def compute_signals(db_path=DB_PATH):
    """
    Load recent data, compute indicators, score every coin for today.
    Returns DataFrame of all coins with scores, sorted by score desc.
    """
    conn = sqlite3.connect(db_path)

    # Load last 45 days of prices (need 20d for vol, 5d for returns)
    cutoff = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")
    prices = pd.read_sql_query(f"""
        SELECT coingecko_id, ticker, bucket, date, price_usd, volume_usd
        FROM daily_prices
        WHERE date >= '{cutoff}'
        ORDER BY coingecko_id, date
    """, conn)

    # Load full vol history for each coin (for time-series percentile)
    # We need each coin's own vol distribution, not cross-sectional
    all_prices = pd.read_sql_query("""
        SELECT coingecko_id, date, price_usd
        FROM daily_prices
        ORDER BY coingecko_id, date
    """, conn)

    # Load Fear & Greed
    fg = pd.read_sql_query("""
        SELECT date, value, classification FROM fear_greed
        ORDER BY date DESC LIMIT 5
    """, conn)
    conn.close()

    if prices.empty:
        print("ERROR: No price data found in database.")
        return pd.DataFrame()

    prices["date"] = pd.to_datetime(prices["date"])
    all_prices["date"] = pd.to_datetime(all_prices["date"])
    fg["date"] = pd.to_datetime(fg["date"])

    # Get most recent date in DB
    max_date = prices["date"].max()
    print(f"Most recent price date: {max_date.strftime('%Y-%m-%d')}")

    # Current F&G
    if not fg.empty:
        fg_today = fg.iloc[0]["value"]
        fg_class = fg.iloc[0]["classification"]
        fg_yesterday = fg.iloc[1]["value"] if len(fg) > 1 else None
        print(f"Fear & Greed: {fg_today} ({fg_class})")
    else:
        fg_today = None
        fg_yesterday = None
        fg_class = "N/A"
        print("WARNING: No Fear & Greed data available")

    # Pre-compute each coin's historical vol distribution for percentile calc
    # This matches the backtest: vol_pctile = rank within coin's own history
    coin_vol_history = {}
    for coin_id, group in all_prices.groupby("coingecko_id"):
        group = group.sort_values("date")
        if len(group) < 25:
            continue
        daily_rets = group["price_usd"].pct_change() * 100
        rolling_vol = daily_rets.rolling(20).std() * np.sqrt(365)
        vol_values = rolling_vol.dropna().values
        if len(vol_values) > 0:
            coin_vol_history[coin_id] = vol_values

    results = []

    for coin_id, group in prices.groupby("coingecko_id"):
        group = group.sort_values("date").copy()

        if len(group) < 6:
            continue

        latest = group.iloc[-1]
        ticker = latest["ticker"]
        bucket = latest["bucket"]
        current_price = latest["price_usd"]
        current_volume = latest["volume_usd"]

        # --- Compute indicators ---

        # 1d return
        if len(group) >= 2:
            ret_1d = (group.iloc[-1]["price_usd"] / group.iloc[-2]["price_usd"] - 1) * 100
        else:
            ret_1d = 0

        # 5d return
        if len(group) >= 6:
            ret_5d = (group.iloc[-1]["price_usd"] / group.iloc[-6]["price_usd"] - 1) * 100
        else:
            ret_5d = 0

        # 10d return
        if len(group) >= 11:
            ret_10d = (group.iloc[-1]["price_usd"] / group.iloc[-11]["price_usd"] - 1) * 100
        else:
            ret_10d = None

        # 20d return
        if len(group) >= 21:
            ret_20d = (group.iloc[-1]["price_usd"] / group.iloc[-21]["price_usd"] - 1) * 100
        else:
            ret_20d = None

        # 20d realized vol (annualized)
        if len(group) >= 21:
            daily_rets = group["price_usd"].pct_change().dropna().tail(20) * 100
            vol_20d = daily_rets.std() * np.sqrt(365)
        else:
            vol_20d = None

        # Volume ratio (today vs 20d avg)
        if len(group) >= 21 and current_volume and current_volume > 0:
            avg_vol_20d = group["volume_usd"].tail(21).iloc[:-1].mean()
            vol_ratio = current_volume / avg_vol_20d if avg_vol_20d > 0 else 0
        else:
            vol_ratio = 0

        # Vol percentile: where does current vol rank in THIS COIN's history?
        # This matches the backtest methodology (time-series, not cross-sectional)
        vol_pctile = None
        if vol_20d is not None and coin_id in coin_vol_history:
            hist = coin_vol_history[coin_id]
            vol_pctile = (hist < vol_20d).sum() / len(hist) * 100

        # --- Compute score ---
        score = 0
        reasons = []

        # Crash family
        if ret_5d <= -25:
            score += 3
            reasons.append(f"5d DD ≥25% ({ret_5d:+.1f}%): +3")
        elif ret_5d <= -15:
            score += 1
            reasons.append(f"5d DD 15-25% ({ret_5d:+.1f}%): +1")

        # Greed/trend family
        if fg_today is not None:
            if fg_today >= 90:
                score += 2
                reasons.append(f"Extreme Greed FG={fg_today}: +2")
            elif fg_today >= 75:
                score += 1
                reasons.append(f"Greed Zone FG={fg_today}: +1")

        # Capitulation (3x volume + 5%+ daily drop)
        if vol_ratio >= 3.0 and ret_1d <= -5:
            score += 1
            reasons.append(f"Capitulation (vol {vol_ratio:.1f}x, 1d {ret_1d:+.1f}%): +1")

        # Anti-signals
        if fg_today is not None and fg_yesterday is not None:
            if fg_yesterday <= 25 and fg_today > 25:
                score -= 2
                reasons.append(f"Fear Exit (FG {fg_yesterday}→{fg_today}): -2")

        # Low vol: coin in its own bottom 10th percentile historically
        if vol_pctile is not None and vol_pctile <= 10:
            score -= 1
            reasons.append(f"Low Vol (own {vol_pctile:.0f}th pctile, "
                           f"vol={vol_20d:.0f}%): -1")

        results.append({
            "ticker": ticker,
            "coingecko_id": coin_id,
            "bucket": bucket,
            "bucket_label": BUCKET_LABELS.get(bucket, "?"),
            "price": current_price,
            "ret_1d": ret_1d,
            "ret_5d": ret_5d,
            "ret_10d": ret_10d,
            "ret_20d": ret_20d,
            "vol_20d": vol_20d,
            "vol_pctile": vol_pctile,
            "vol_ratio": vol_ratio,
            "score": score,
            "reasons": reasons,
            "txn_cost": COST_MAP.get(bucket, 0.80),
        })

    df = pd.DataFrame(results)

    if df.empty:
        return df

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    return df


# ============================================================
# EMAIL ALERT
# ============================================================

def format_email(signals_df, fg_today, fg_class):
    """Format the signal alert email."""
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Filter actionable signals
    buys = signals_df[signals_df["score"] >= 2].copy()
    watches = signals_df[(signals_df["score"] == 1)].copy()
    avoids = signals_df[signals_df["score"] <= -1].copy()

    # Summary line
    n_buy = len(buys)
    n_strong = len(buys[buys["score"] >= 3])
    n_very_strong = len(buys[buys["score"] >= 4])

    subject = f"Crypto Scanner: "
    if n_very_strong > 0:
        subject += f"{n_very_strong} VERY STRONG + {n_buy - n_very_strong} more signals"
    elif n_strong > 0:
        subject += f"{n_strong} STRONG + {n_buy - n_strong} more signals"
    elif n_buy > 0:
        subject += f"{n_buy} BUY signals detected"
    else:
        subject += "No signals today"

    # Build email body
    lines = []
    lines.append("=" * 60)
    lines.append(f"CRYPTO SIGNAL SCANNER — {today_str}")
    lines.append("=" * 60)
    lines.append(f"Fear & Greed: {fg_today} ({fg_class})")
    lines.append(f"Signals: {n_buy} BUY (≥2) | {len(watches)} Watch (1) | "
                 f"{len(avoids)} Avoid (≤-1)")
    lines.append("")

    # Market snapshot — top movers
    lines.append("-" * 60)
    lines.append("MARKET SNAPSHOT (5-day change)")
    lines.append("-" * 60)
    for ticker in ["BTC", "ETH", "SOL", "XRP"]:
        row = signals_df[signals_df["ticker"] == ticker]
        if not row.empty:
            r = row.iloc[0]
            ret_20d_str = f"  20d: {r['ret_20d']:+.1f}%" if r["ret_20d"] is not None else ""
            lines.append(f"  {r['ticker']:<6} ${r['price']:>10,.2f}  "
                         f"1d: {r['ret_1d']:+.1f}%  5d: {r['ret_5d']:+.1f}%{ret_20d_str}")
    lines.append("")

    if n_buy > 0:
        lines.append("-" * 60)
        lines.append("BUY SIGNALS (Score ≥ 2)")
        lines.append("-" * 60)
        lines.append(f"{'Ticker':<8} {'Score':>6} {'Bucket':<10} {'Price':>12} "
                     f"{'1d':>8} {'5d':>8} {'20d':>8}")
        lines.append("-" * 60)

        for _, row in buys.iterrows():
            strength = ""
            if row["score"] >= 4:
                strength = " ★★★"
            elif row["score"] >= 3:
                strength = " ★★"
            else:
                strength = " ★"

            ret_20d_str = f"{row['ret_20d']:+.1f}%" if row["ret_20d"] is not None else "N/A"

            lines.append(
                f"{row['ticker']:<8} {row['score']:>+5.0f}{strength:<4} "
                f"{row['bucket_label']:<10} "
                f"${row['price']:>10,.2f} "
                f"{row['ret_1d']:>+7.1f}% "
                f"{row['ret_5d']:>+7.1f}% "
                f"{ret_20d_str:>8}"
            )

        lines.append("")

        # Detailed breakdown for each buy signal
        lines.append("SIGNAL DETAILS:")
        lines.append("")
        for _, row in buys.iterrows():
            lines.append(f"  {row['ticker']} (Score {row['score']:+.0f}):")
            for reason in row["reasons"]:
                lines.append(f"    • {reason}")
            if row["vol_ratio"] > 0:
                lines.append(f"    Volume ratio: {row['vol_ratio']:.1f}x avg")
            if row["vol_20d"] is not None:
                lines.append(f"    20d realized vol: {row['vol_20d']:.0f}%")
            if row["vol_pctile"] is not None:
                lines.append(f"    Vol percentile (own history): "
                             f"{row['vol_pctile']:.0f}th")
            lines.append("")

    if len(watches) > 0:
        lines.append("-" * 60)
        lines.append(f"WATCHLIST (Score = 1) — {len(watches)} coins")
        lines.append("-" * 60)
        for _, row in watches.head(10).iterrows():
            reasons_short = " | ".join(row["reasons"]) if row["reasons"] else ""
            lines.append(f"  {row['ticker']:<8} 5d: {row['ret_5d']:>+7.1f}%  "
                         f"${row['price']:>10,.2f}  ({reasons_short})")
        if len(watches) > 10:
            lines.append(f"  ... and {len(watches) - 10} more")
        lines.append("")

    if len(avoids) > 0:
        lines.append("-" * 60)
        lines.append(f"AVOID (Score ≤ -1) — {len(avoids)} coins")
        lines.append("-" * 60)
        for _, row in avoids.head(5).iterrows():
            reasons_str = " | ".join(row["reasons"]) if row["reasons"] else "Low vol"
            lines.append(f"  {row['ticker']:<8} Score {row['score']:+.0f}  ({reasons_str})")
        lines.append("")

    # All scores summary
    lines.append("-" * 60)
    lines.append("ALL COIN SCORES")
    lines.append("-" * 60)
    lines.append(f"{'Ticker':<8} {'Score':>6} {'5d':>8} {'Bucket':<10}")
    lines.append("-" * 60)
    for _, row in signals_df.iterrows():
        lines.append(f"  {row['ticker']:<8} {row['score']:>+5.0f}  "
                     f"{row['ret_5d']:>+7.1f}%  {row['bucket_label']}")
    lines.append("")

    # Strategy reminder
    lines.append("=" * 60)
    lines.append("STRATEGY RULES")
    lines.append("=" * 60)
    lines.append("• Long-only, 3-5 day hold period")
    lines.append("• Score ≥2: BUY (+6.9% alpha, 57.7% win, OOS validated)")
    lines.append("• Score ≥3: STRONG BUY (+8.7% alpha, 58.9% win)")
    lines.append("• Score ≥4: VERY STRONG (+19.5% alpha, 67.9% win)")
    lines.append("• Position size: equal weight across signals")
    lines.append("• BTC/ETH: safest (most liquid, +4.6%/+3.7% alpha)")
    lines.append("• Altcoins: highest alpha but wider spreads")
    lines.append("")
    lines.append("Backtest: 2014-2026, 4,650 events, OOS alpha +7.27%")

    body = "\n".join(lines)
    return subject, body


def send_email(subject, body, email_addr=None, app_password=None, recipient=None):
    """Send alert email via Gmail SMTP."""
    email_addr = email_addr or EMAIL_ADDRESS
    app_password = app_password or EMAIL_APP_PASSWORD
    recipient = recipient or ALERT_RECIPIENT

    msg = MIMEMultipart()
    msg["From"] = email_addr
    msg["To"] = recipient
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(email_addr, app_password)
        server.send_message(msg)
        server.quit()
        print(f"Email sent: {subject}")
        return True
    except Exception as e:
        print(f"Email FAILED: {e}")
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Crypto Signal Scanner")
    parser.add_argument("--scan-only", action="store_true",
                        help="Skip data update, scan existing data")
    parser.add_argument("--no-email", action="store_true",
                        help="Run scan but don't send email")
    args = parser.parse_args()

    print("=" * 60)
    print(f"CRYPTO SIGNAL SCANNER")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 1: Update data
    if not args.scan_only:
        print("\n--- UPDATING DATA ---")
        update_fear_greed()
        update_prices()
    else:
        print("\n--- SCAN ONLY (skipping data update) ---")

    # Step 2: Compute signals
    print("\n--- COMPUTING SIGNALS ---")
    df = compute_signals()

    if df.empty:
        print("No data available. Run collect_data.py --all first.")
        return

    # Step 3: Display results
    buys = df[df["score"] >= 2]
    watches = df[df["score"] == 1]
    avoids = df[df["score"] <= -1]
    neutral = df[df["score"] == 0]

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(buys)} BUY | {len(watches)} Watch | "
          f"{len(neutral)} Neutral | {len(avoids)} Avoid")
    print(f"{'='*60}")

    if len(buys) > 0:
        for _, row in buys.iterrows():
            star = "★★★" if row["score"] >= 4 else "★★" if row["score"] >= 3 else "★"
            print(f"  {star} {row['ticker']:<8} Score {row['score']:>+.0f}  "
                  f"${row['price']:>10,.2f}  5d: {row['ret_5d']:>+.1f}%  "
                  f"({row['bucket_label']})")
            for reason in row["reasons"]:
                print(f"       {reason}")
    else:
        print("  No actionable signals today.")

    if len(avoids) > 0:
        print(f"\n  Avoid ({len(avoids)}):")
        for _, row in avoids.iterrows():
            reasons_str = " | ".join(row["reasons"]) if row["reasons"] else ""
            print(f"    {row['ticker']:<8} Score {row['score']:+.0f}  ({reasons_str})")

    # Step 4: Send email
    conn = sqlite3.connect(DB_PATH)
    fg_row = pd.read_sql_query(
        "SELECT value, classification FROM fear_greed ORDER BY date DESC LIMIT 1", conn
    )
    conn.close()

    fg_today = int(fg_row.iloc[0]["value"]) if not fg_row.empty else 0
    fg_class = fg_row.iloc[0]["classification"] if not fg_row.empty else "N/A"

    subject, body = format_email(df, fg_today, fg_class)

    if not args.no_email:
        send_email(subject, body)
    else:
        print(f"\n--- EMAIL PREVIEW (not sent) ---")
        print(f"Subject: {subject}")
        print(body)

    print(f"\nScanner complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
