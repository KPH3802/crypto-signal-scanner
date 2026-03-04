#!/usr/bin/env python3
"""
crypto_autotrader.py
====================
Coinbase Advanced Trade auto-executor for the crypto signal scanner.

Strategy (backtest-validated, OOS +7.27% alpha):
  - BUY when scanner fires Score ≥ 2
  - SELL after 3-day hold
  - Position size scaled by score
  - Trades the 34-coin Coinbase-tradable universe across 3 buckets

Position Sizing ($5,000 portfolio):
  - Score 2: 5% of portfolio  = $250
  - Score 3: 8% of portfolio  = $400
  - Score 4: 12% of portfolio = $600
  - Max open positions: 6 (never more than 60% deployed)
  - Min cash reserve: 40% always

Usage:
  python3 crypto_autotrader.py --check-entries   # Run after scanner fires (00:30 UTC)
  python3 crypto_autotrader.py --check-exits     # Run daily to close 3-day-old trades
  python3 crypto_autotrader.py --status          # Show open positions + P&L
  python3 crypto_autotrader.py --dry-run         # Simulate without executing

Scheduled on PythonAnywhere:
  00:45 UTC — check-entries (15 min after scanner)
  01:00 UTC — check-exits
"""

import argparse
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone

try:
    import requests
except ImportError:
    os.system("pip3 install requests")
    import requests

try:
    import jwt as pyjwt
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
except ImportError:
    os.system("pip3 install PyJWT cryptography")
    import jwt as pyjwt
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

# ── Config ────────────────────────────────────────────────────────────────────
PORTFOLIO_SIZE    = 5000.0   # Total capital in USD
MIN_CASH_RESERVE  = 0.40     # Always keep 40% cash
MAX_POSITIONS     = 6        # Max concurrent open positions
HOLD_DAYS         = 3        # Exit after N days

POSITION_SIZE = {
    2: 0.05,   # Score 2 → 5% of portfolio  = $250
    3: 0.08,   # Score 3 → 8% of portfolio  = $400
    4: 0.12,   # Score 4 → 12% of portfolio = $600
}

# Coins excluded (not on Coinbase)
EXCLUDED = {"GALA", "JUP", "KAS", "MKR", "RNDR", "TRX"}

# Bucket mapping (from config.py)
BUCKET_MAP = {
    "BTC": 1, "ETH": 1,
    "BNB": 2, "SOL": 2, "XRP": 2, "ADA": 2, "DOGE": 2, "LINK": 2,
    "AVAX": 2, "DOT": 2, "POL": 2, "SHIB": 2, "LTC": 2,
    "UNI": 2, "XLM": 2, "NEAR": 2, "SUI": 2, "ICP": 2,
    "AAVE": 3, "ARB": 3, "OP": 3, "INJ": 3, "SEI": 3, "TIA": 3,
    "BONK": 3, "PEPE": 3, "FLOKI": 3, "WLD": 3, "STX": 3, "GRT": 3,
    "IMX": 3, "FET": 3, "PENDLE": 3,
}

BUCKET_LABELS = {1: "Large Cap", 2: "Mid Cap", 3: "Smaller Alt"}

# DB for tracking open positions
TRADER_DB = os.path.join(os.path.dirname(__file__), "autotrader.db")

# CDP key file (fallback if env vars not set)
KEY_FILE = os.path.join(os.path.dirname(__file__), "cdp_api_key.json")

# Email (from config)
try:
    from config import EMAIL_ADDRESS, EMAIL_APP_PASSWORD, ALERT_RECIPIENT
except ImportError:
    EMAIL_ADDRESS = ""
    EMAIL_APP_PASSWORD = ""
    ALERT_RECIPIENT = ""


# ── Coinbase JWT Auth ─────────────────────────────────────────────────────────
def load_cdp_key():
    """
    Load Coinbase CDP credentials.
    Priority 1: Environment variables (preferred — no key file on disk)
      COINBASE_API_KEY_NAME    — the 'name' field from cdp_api_key.json
      COINBASE_API_PRIVATE_KEY — the full PEM private key string
    Priority 2: cdp_api_key.json file (local dev fallback)
    """
    key_name    = os.environ.get("COINBASE_API_KEY_NAME")
    private_key = os.environ.get("COINBASE_API_PRIVATE_KEY")

    if key_name and private_key:
        # Replace literal \n with actual newlines in case the env var was set
        # as a single line (common in some hosting environments)
        private_key = private_key.replace("\\n", "\n")
        return {"name": key_name, "privateKey": private_key}

    # Fall back to JSON file
    if not os.path.exists(KEY_FILE):
        raise FileNotFoundError(
            "Coinbase credentials not found. Set COINBASE_API_KEY_NAME and "
            "COINBASE_API_PRIVATE_KEY environment variables, or place "
            "cdp_api_key.json in the project directory."
        )
    with open(KEY_FILE) as f:
        return json.load(f)

def build_jwt(method, path, cdp):
    private_key = load_pem_private_key(cdp["privateKey"].encode(), password=None)
    now = int(time.time())
    payload = {
        "sub": cdp["name"],
        "iss": "cdp",
        "nbf": now,
        "exp": now + 120,
        "uri": f"{method} api.coinbase.com{path}",
    }
    return pyjwt.encode(
        payload,
        private_key,
        algorithm="ES256",
        headers={"kid": cdp["name"], "nonce": str(now)},
    )

def cb_request(method, path, body=None):
    cdp = load_cdp_key()
    token = build_jwt(method.upper(), path, cdp)
    url = f"https://api.coinbase.com{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if method.upper() == "GET":
        r = requests.get(url, headers=headers, timeout=15)
    else:
        r = requests.post(url, headers=headers, json=body, timeout=15)
    r.raise_for_status()
    return r.json()


# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(TRADER_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker        TEXT NOT NULL,
            bucket        INTEGER,
            bucket_label  TEXT,
            score         INTEGER,
            entry_date    TEXT,
            exit_date     TEXT,
            entry_price   REAL,
            exit_price    REAL,
            quantity      REAL,
            usd_value     REAL,
            order_id      TEXT,
            exit_order_id TEXT,
            status        TEXT DEFAULT 'OPEN',
            pnl_usd       REAL,
            pnl_pct       REAL,
            dry_run       INTEGER DEFAULT 0,
            notes         TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_open_positions():
    conn = sqlite3.connect(TRADER_DB)
    import pandas as pd
    df = pd.read_sql_query(
        "SELECT * FROM positions WHERE status = 'OPEN' ORDER BY entry_date",
        conn
    )
    conn.close()
    return df


def get_deployed_capital():
    conn = sqlite3.connect(TRADER_DB)
    c = conn.cursor()
    c.execute("SELECT SUM(usd_value) FROM positions WHERE status = 'OPEN'")
    result = c.fetchone()[0]
    conn.close()
    return result or 0.0


def record_entry(ticker, bucket, score, entry_price, quantity, usd_value,
                 order_id, dry_run=False):
    conn = sqlite3.connect(TRADER_DB)
    c = conn.cursor()
    c.execute("""
        INSERT INTO positions
        (ticker, bucket, bucket_label, score, entry_date, entry_price,
         quantity, usd_value, order_id, status, dry_run)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
    """, (
        ticker, bucket, BUCKET_LABELS.get(bucket, "?"), score,
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        entry_price, quantity, usd_value, order_id, int(dry_run)
    ))
    conn.commit()
    conn.close()


def record_exit(position_id, exit_price, exit_order_id, dry_run=False):
    conn = sqlite3.connect(TRADER_DB)
    c = conn.cursor()
    c.execute("SELECT entry_price, quantity, usd_value FROM positions WHERE id = ?",
              (position_id,))
    row = c.fetchone()
    entry_price, quantity, usd_value = row
    pnl_usd = (exit_price - entry_price) * quantity
    pnl_pct = ((exit_price / entry_price) - 1) * 100
    c.execute("""
        UPDATE positions
        SET status = 'CLOSED', exit_date = ?, exit_price = ?,
            exit_order_id = ?, pnl_usd = ?, pnl_pct = ?
        WHERE id = ?
    """, (
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        exit_price, exit_order_id, pnl_usd, pnl_pct, position_id
    ))
    conn.commit()
    conn.close()
    return pnl_usd, pnl_pct


# ── Coinbase Market Data ──────────────────────────────────────────────────────
def get_current_price(ticker):
    """Get current mid price from Coinbase."""
    product_id = f"{ticker}-USD"
    try:
        data = cb_request("GET", f"/api/v3/brokerage/best_bid_ask?product_ids={product_id}")
        pricebooks = data.get("pricebooks", [])
        if pricebooks:
            pb = pricebooks[0]
            bids = pb.get("bids", [])
            asks = pb.get("asks", [])
            if bids and asks:
                mid = (float(bids[0]["price"]) + float(asks[0]["price"])) / 2
                return mid
    except Exception as e:
        print(f"  Price fetch failed for {ticker}: {e}")
    return None


def get_min_order_size(ticker):
    """Get minimum order size for a product."""
    product_id = f"{ticker}-USD"
    try:
        data = cb_request("GET", f"/api/v3/brokerage/products/{product_id}")
        base_min = float(data.get("base_min_size", "0.00001"))
        quote_min = float(data.get("quote_min_size", "1.00"))
        return base_min, quote_min
    except Exception:
        return 0.00001, 1.0


# ── Order Execution ───────────────────────────────────────────────────────────
def place_market_buy(ticker, usd_amount, dry_run=False):
    """
    Place a market buy order for $usd_amount of ticker.
    Returns (order_id, fill_price, quantity) or None on failure.
    """
    product_id = f"{ticker}-USD"
    client_order_id = f"autotrader-buy-{ticker}-{int(time.time())}"

    if dry_run:
        price = get_current_price(ticker)
        if price is None:
            return None
        qty = usd_amount / price
        print(f"  [DRY RUN] BUY {qty:.6f} {ticker} @ ${price:,.4f} = ${usd_amount:.2f}")
        return f"dry-run-{client_order_id}", price, qty

    body = {
        "client_order_id": client_order_id,
        "product_id": product_id,
        "side": "BUY",
        "order_configuration": {
            "market_market_ioc": {
                "quote_size": f"{usd_amount:.2f}"
            }
        }
    }

    try:
        resp = cb_request("POST", "/api/v3/brokerage/orders", body)
        order = resp.get("success_response", {})
        order_id = order.get("order_id", "")

        if not order_id:
            print(f"  BUY FAILED {ticker}: {resp}")
            return None

        # Poll for fill
        time.sleep(2)
        fill_data = cb_request("GET", f"/api/v3/brokerage/orders/historical/{order_id}")
        filled_order = fill_data.get("order", {})
        avg_price = float(filled_order.get("average_filled_price", 0))
        filled_size = float(filled_order.get("filled_size", 0))

        if avg_price == 0:
            price = get_current_price(ticker)
            avg_price = price or 0
            filled_size = usd_amount / avg_price if avg_price > 0 else 0

        print(f"  ✅ BUY {filled_size:.6f} {ticker} @ ${avg_price:,.4f} = ${usd_amount:.2f}")
        return order_id, avg_price, filled_size

    except Exception as e:
        print(f"  BUY ERROR {ticker}: {e}")
        return None


def place_market_sell(ticker, quantity, dry_run=False):
    """
    Place a market sell order for quantity of ticker.
    Returns (order_id, fill_price) or None on failure.
    """
    product_id = f"{ticker}-USD"
    client_order_id = f"autotrader-sell-{ticker}-{int(time.time())}"

    if dry_run:
        price = get_current_price(ticker)
        if price is None:
            return None
        usd_value = quantity * price
        print(f"  [DRY RUN] SELL {quantity:.6f} {ticker} @ ${price:,.4f} = ${usd_value:.2f}")
        return f"dry-run-{client_order_id}", price

    body = {
        "client_order_id": client_order_id,
        "product_id": product_id,
        "side": "SELL",
        "order_configuration": {
            "market_market_ioc": {
                "base_size": f"{quantity:.8f}"
            }
        }
    }

    try:
        resp = cb_request("POST", "/api/v3/brokerage/orders", body)
        order = resp.get("success_response", {})
        order_id = order.get("order_id", "")

        if not order_id:
            print(f"  SELL FAILED {ticker}: {resp}")
            return None

        time.sleep(2)
        fill_data = cb_request("GET", f"/api/v3/brokerage/orders/historical/{order_id}")
        filled_order = fill_data.get("order", {})
        avg_price = float(filled_order.get("average_filled_price", 0))

        if avg_price == 0:
            avg_price = get_current_price(ticker) or 0

        usd_value = quantity * avg_price
        print(f"  ✅ SELL {quantity:.6f} {ticker} @ ${avg_price:,.4f} = ${usd_value:.2f}")
        return order_id, avg_price

    except Exception as e:
        print(f"  SELL ERROR {ticker}: {e}")
        return None


# ── Core Logic ────────────────────────────────────────────────────────────────
def check_entries(dry_run=False):
    """
    Read today's scanner signals and execute buys for Score ≥ 2.
    Called at 00:45 UTC (15 min after scanner runs).
    """
    import pandas as pd

    print("=" * 60)
    print(f"CHECK ENTRIES — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 60)

    # Update prices and Fear & Greed before scanning
    from crypto_scanner import compute_signals, update_prices, update_fear_greed
    from database import DB_PATH
    print("Updating data...")
    update_fear_greed()
    update_prices()
    signals = compute_signals()
    if signals.empty:
        print("No signals computed.")
        return

    # Filter: Score ≥ 2, not excluded, not already in open positions
    open_pos = get_open_positions()
    open_tickers = set(open_pos["ticker"].tolist()) if not open_pos.empty else set()

    buys = signals[
        (signals["score"] >= 2) &
        (~signals["ticker"].isin(EXCLUDED)) &
        (~signals["ticker"].isin(open_tickers))
    ].copy()

    if buys.empty:
        print("No new BUY signals today.")
        return

    print(f"\nNew BUY signals: {len(buys)}")
    for _, row in buys.iterrows():
        bucket_label = BUCKET_LABELS.get(row["bucket"], "?")
        print(f"  {row['ticker']:<8} Score {row['score']:+.0f}  {bucket_label}  "
              f"5d: {row['ret_5d']:+.1f}%")

    # Capital check
    deployed = get_deployed_capital()
    available = PORTFOLIO_SIZE - deployed
    max_deployable = PORTFOLIO_SIZE * (1 - MIN_CASH_RESERVE) - deployed
    n_open = len(open_pos)

    print(f"\nCapital: ${PORTFOLIO_SIZE:,.0f} total | "
          f"${deployed:,.0f} deployed | ${available:,.0f} available")
    print(f"Positions: {n_open}/{MAX_POSITIONS} open")

    if n_open >= MAX_POSITIONS:
        print("⚠️  Max positions reached. No new entries.")
        return

    if max_deployable <= 0:
        print("⚠️  Cash reserve limit reached. No new entries.")
        return

    # Execute buys
    executed = []
    for _, row in buys.iterrows():
        ticker = row["ticker"]
        score = int(row["score"])
        bucket = int(row["bucket"])

        # Check position cap
        if len(open_pos) + len(executed) >= MAX_POSITIONS:
            print(f"  {ticker}: skipped — max positions reached")
            break

        # Position size
        size_pct = POSITION_SIZE.get(min(score, 4), POSITION_SIZE[2])
        usd_amount = PORTFOLIO_SIZE * size_pct

        # Don't exceed available deployable capital
        if usd_amount > max_deployable:
            usd_amount = max_deployable
        if usd_amount < 1.0:
            print(f"  {ticker}: skipped — insufficient capital")
            break

        print(f"\n  → Buying {ticker} (Score {score:+d}, "
              f"${usd_amount:.0f}, {BUCKET_LABELS.get(bucket, '?')})")

        result = place_market_buy(ticker, usd_amount, dry_run=dry_run)
        if result:
            order_id, fill_price, quantity = result
            record_entry(ticker, bucket, score, fill_price, quantity,
                        usd_amount, order_id, dry_run=dry_run)
            executed.append(ticker)
            max_deployable -= usd_amount
            time.sleep(0.5)
        else:
            print(f"  {ticker}: order failed")

    print(f"\n✅ Entries complete: {len(executed)} trades executed")
    if executed:
        print(f"   Bought: {', '.join(executed)}")

    return executed


def check_exits(dry_run=False):
    """
    Check open positions and sell any that have held for HOLD_DAYS.
    Called daily at 01:00 UTC.
    """
    print("=" * 60)
    print(f"CHECK EXITS — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 60)

    open_pos = get_open_positions()

    if open_pos.empty:
        print("No open positions.")
        return

    today = datetime.now(timezone.utc).date()
    exits_done = []

    for _, pos in open_pos.iterrows():
        entry_date = datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()
        days_held = (today - entry_date).days

        ticker = pos["ticker"]
        print(f"\n  {ticker}: held {days_held}d "
              f"(entry {pos['entry_date']}, score {pos['score']:+d})")

        if days_held >= HOLD_DAYS:
            print(f"  → SELL signal (held {days_held} days ≥ {HOLD_DAYS})")

            result = place_market_sell(ticker, pos["quantity"], dry_run=dry_run)
            if result:
                exit_order_id, exit_price = result
                pnl_usd, pnl_pct = record_exit(
                    pos["id"], exit_price, exit_order_id, dry_run=dry_run
                )
                exits_done.append({
                    "ticker": ticker,
                    "pnl_usd": pnl_usd,
                    "pnl_pct": pnl_pct,
                    "days_held": days_held,
                })
                print(f"  ✅ Closed: P&L ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)")
                time.sleep(0.5)
            else:
                print(f"  ❌ SELL FAILED for {ticker} — manual review required")
        else:
            days_left = HOLD_DAYS - days_held
            current_price = get_current_price(ticker)
            if current_price:
                unrealized = (current_price - pos["entry_price"]) * pos["quantity"]
                unreal_pct = (current_price / pos["entry_price"] - 1) * 100
                print(f"  Holding ({days_left}d left): "
                      f"${pos['entry_price']:,.4f} → ${current_price:,.4f}  "
                      f"P&L ${unrealized:+.2f} ({unreal_pct:+.2f}%)")

    if exits_done:
        total_pnl = sum(e["pnl_usd"] for e in exits_done)
        print(f"\n✅ Exits complete: {len(exits_done)} trades closed")
        print(f"   Total P&L: ${total_pnl:+.2f}")
    else:
        print("\nNo exits today.")

    return exits_done


def show_status():
    """Display current portfolio status and P&L summary."""
    import pandas as pd

    print("=" * 60)
    print(f"PORTFOLIO STATUS — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    open_pos = get_open_positions()
    today = datetime.now(timezone.utc).date()

    if open_pos.empty:
        print("No open positions.")
    else:
        deployed = 0
        print(f"\nOPEN POSITIONS ({len(open_pos)}/{MAX_POSITIONS}):")
        print(f"{'#':<3} {'Ticker':<8} {'Score':>6} {'Bucket':<12} "
              f"{'Entry $':>10} {'Current $':>10} {'P&L $':>10} {'P&L %':>8} {'Days':>5}")
        print("-" * 75)

        for i, (_, pos) in enumerate(open_pos.iterrows(), 1):
            current = get_current_price(pos["ticker"])
            entry_date = datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()
            days_held = (today - entry_date).days

            if current:
                unreal_usd = (current - pos["entry_price"]) * pos["quantity"]
                unreal_pct = (current / pos["entry_price"] - 1) * 100
                current_str = f"${current:,.4f}"
                pnl_str = f"${unreal_usd:+.2f}"
                pnl_pct_str = f"{unreal_pct:+.2f}%"
            else:
                current_str = "N/A"
                pnl_str = "N/A"
                pnl_pct_str = "N/A"

            dry_tag = " [DRY]" if pos.get("dry_run") else ""
            print(f"{i:<3} {pos['ticker']:<8} {pos['score']:>+5d}  "
                  f"{pos['bucket_label']:<12} "
                  f"${pos['entry_price']:>9,.4f} "
                  f"{current_str:>10} "
                  f"{pnl_str:>10} "
                  f"{pnl_pct_str:>8} "
                  f"{days_held:>5}{dry_tag}")
            deployed += pos["usd_value"]

        available = PORTFOLIO_SIZE - deployed
        print(f"\nCapital: ${PORTFOLIO_SIZE:,.0f} total | "
              f"${deployed:,.0f} deployed | ${available:,.0f} available")

    # Closed trade summary
    conn = sqlite3.connect(TRADER_DB)
    df_closed = pd.read_sql_query(
        "SELECT * FROM positions WHERE status = 'CLOSED' ORDER BY exit_date DESC LIMIT 20",
        conn
    )
    conn.close()

    if not df_closed.empty:
        total_pnl = df_closed["pnl_usd"].sum()
        wins = len(df_closed[df_closed["pnl_usd"] > 0])
        losses = len(df_closed[df_closed["pnl_usd"] <= 0])
        win_rate = wins / len(df_closed) * 100

        print(f"\nCLOSED TRADES (last 20):")
        print(f"{'Ticker':<8} {'Entry':>10} {'Exit':>10} {'P&L $':>10} "
              f"{'P&L %':>8} {'Days':>5} {'Bucket':<12}")
        print("-" * 65)
        for _, row in df_closed.iterrows():
            entry_d = datetime.strptime(row["entry_date"], "%Y-%m-%d").date()
            exit_d = datetime.strptime(row["exit_date"], "%Y-%m-%d").date()
            days = (exit_d - entry_d).days
            dry_tag = " [DRY]" if row.get("dry_run") else ""
            print(f"{row['ticker']:<8} ${row['entry_price']:>9,.4f} "
                  f"${row['exit_price']:>9,.4f} "
                  f"${row['pnl_usd']:>+9.2f} "
                  f"{row['pnl_pct']:>+7.2f}% "
                  f"{days:>5} "
                  f"{row['bucket_label']:<12}{dry_tag}")

        print(f"\nSummary: {len(df_closed)} trades | "
              f"Win rate {win_rate:.1f}% ({wins}W/{losses}L) | "
              f"Total P&L ${total_pnl:+.2f}")


# ── Email Notification ────────────────────────────────────────────────────────
def send_trade_email(subject, body):
    if not EMAIL_ADDRESS or not EMAIL_APP_PASSWORD:
        return
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = ALERT_RECIPIENT
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print(f"Email failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_db()

    parser = argparse.ArgumentParser(description="Crypto Auto-Trader")
    parser.add_argument("--check-entries", action="store_true",
                        help="Execute buys for today's signals")
    parser.add_argument("--check-exits", action="store_true",
                        help="Close positions held 3+ days")
    parser.add_argument("--status", action="store_true",
                        help="Show open positions and P&L")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without placing real orders")
    args = parser.parse_args()

    try:
        if args.check_entries:
            check_entries(dry_run=args.dry_run)
        elif args.check_exits:
            check_exits(dry_run=args.dry_run)
        elif args.status:
            show_status()
        else:
            parser.print_help()
    except Exception as e:
        import traceback
        mode = "DRY RUN" if args.dry_run else "LIVE"
        task = "check-entries" if args.check_entries else "check-exits" if args.check_exits else "unknown"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        subject = f"⚠️ Crypto Auto-Trader FAILED [{task}] {timestamp}"
        body = (
            f"The crypto auto-trader crashed during {task} at {timestamp}.\n"
            f"Mode: {mode}\n\n"
            f"Error: {str(e)}\n\n"
            f"Full traceback:\n{traceback.format_exc()}"
        )
        print(subject)
        print(body)
        send_trade_email(subject, body)
        raise  # Re-raise so PA logs the error too


if __name__ == "__main__":
    main()
