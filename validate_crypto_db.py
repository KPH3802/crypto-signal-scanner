#!/usr/bin/env python3
"""
validate_crypto_db.py
=====================
Proactive data corruption scanner for crypto_backtest.db.

Checks the daily_prices table for:
  1. Single-day price moves > SPIKE_THRESHOLD (default 200%)
     - LAUNCH_ARTIFACT: spike caused by a near-zero prev_price placeholder row
     - SPIKE / DROP: genuine large move (real market event or corrupt data)
  2. Near-zero prices (< $0.001) — pre-launch placeholder rows inserted by yfinance
  3. Zero or negative prices
  4. Duplicate (coin, date) rows
  5. Stale coins — no new data in > 3 days

Run this after every major DB update or manually at any time.
Reactive detection (TON, STX, TIA, UNI, JUP) found Mar 2026 — this makes it proactive.

Usage:
    python3 validate_crypto_db.py                     # Scan, print report
    python3 validate_crypto_db.py --purge             # Scan + auto-purge flagged rows
    python3 validate_crypto_db.py --db /path/to/db    # Custom DB path
    python3 validate_crypto_db.py --threshold 300     # Custom spike threshold (%)

Exit codes:
    0 — Clean. No issues found.
    1 — Issues found (rows flagged but not purged).
    2 — Issues found and purged.
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta

# ── Thresholds ─────────────────────────────────────────────────────────────────
SPIKE_THRESHOLD  = 200.0   # Flag if 1-day gain > 200%
DROP_THRESHOLD   = 80.0    # Flag if 1-day drop > 80%
STALE_DAYS       = 3       # Flag if coin has no data within this many days of DB max date
NEAR_ZERO_LIMIT  = 0.001   # Flag any price below this as a launch artifact placeholder

# ── Default DB path ────────────────────────────────────────────────────────────
DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crypto_backtest.db")


def get_db_connection(db_path):
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)
    return sqlite3.connect(db_path)


def check_price_spikes(conn, spike_pct, drop_pct):
    """
    Find rows where the 1-day price change exceeds thresholds.
    Returns list of dicts: {coin, ticker, date, prev_price, curr_price, pct_change, row_id}
    """
    c = conn.cursor()
    c.execute("""
        SELECT id, coingecko_id, ticker, date, prev_price, curr_price, pct_change
        FROM (
            SELECT
                a.id,
                a.coingecko_id,
                a.ticker,
                a.date,
                b.price_usd AS prev_price,
                a.price_usd AS curr_price,
                ROUND(((a.price_usd - b.price_usd) / b.price_usd) * 100, 2) AS pct_change
            FROM daily_prices a
            JOIN daily_prices b
              ON a.coingecko_id = b.coingecko_id
             AND b.date = (
                 SELECT MAX(date) FROM daily_prices
                 WHERE coingecko_id = a.coingecko_id
                   AND date < a.date
             )
            WHERE a.price_usd IS NOT NULL
              AND b.price_usd IS NOT NULL
              AND b.price_usd > 0
        ) sub
        WHERE ABS(pct_change) > 0
        ORDER BY ABS(pct_change) DESC
    """)
    rows = c.fetchall()

    # Also look up the row_id of the prev_price row so we can purge the right one
    flagged = []
    for row in rows:
        row_id, coin, ticker, date, prev_price, curr_price, pct = row
        if pct > spike_pct or pct < -drop_pct:
            # Determine if this spike is caused by a near-zero placeholder prev_price
            is_artifact = (prev_price < NEAR_ZERO_LIMIT and pct > 0)
            # Look up the ID of the prev_price row (the corrupt one for artifacts)
            prev_row_id = None
            if is_artifact:
                c2 = conn.cursor()
                c2.execute("""
                    SELECT id FROM daily_prices
                    WHERE coingecko_id = ? AND date = (
                        SELECT MAX(date) FROM daily_prices
                        WHERE coingecko_id = ? AND date < ?
                    )
                """, (coin, coin, date))
                r = c2.fetchone()
                if r:
                    prev_row_id = r[0]
            flagged.append({
                "row_id":      row_id,       # The spike day row
                "prev_row_id": prev_row_id,  # The corrupt near-zero row (artifacts only)
                "coin":        coin,
                "ticker":      ticker,
                "date":        date,
                "prev_price":  prev_price,
                "curr_price":  curr_price,
                "pct_change":  pct,
                "type":        "LAUNCH_ARTIFACT" if is_artifact else ("SPIKE" if pct > 0 else "DROP"),
            })
    return flagged


def check_near_zero_prices(conn):
    """
    Flag rows where price < 0.1% of that coin's median price.
    Coin-relative threshold handles sub-penny coins like SHIB/BONK/PEPE correctly:
      - SHIB median ~$0.00001 → threshold ~$0.00000001 → real SHIB data not flagged
      - ARB median ~$0.80    → threshold ~$0.0008     → pre-launch placeholders flagged
    """
    c = conn.cursor()
    # Get median price per coin using SQLite percentile approximation
    c.execute("""
        SELECT id, coingecko_id, ticker, date, price_usd
        FROM daily_prices
        WHERE price_usd > 0
        ORDER BY coingecko_id, date
    """)
    all_rows = c.fetchall()

    # Build per-coin median
    from statistics import median
    coin_prices = {}
    for row in all_rows:
        coin = row[1]
        price = row[4]
        coin_prices.setdefault(coin, []).append(price)
    coin_median = {coin: median(prices) for coin, prices in coin_prices.items()}

    flagged = []
    for row in all_rows:
        row_id, coin, ticker, date, price = row
        med = coin_median.get(coin, 0)
        threshold = med * 0.001  # 0.1% of median
        if threshold > 0 and price < threshold:
            flagged.append({"row_id": row_id, "coin": coin, "ticker": ticker,
                            "date": date, "price": price, "median": med,
                            "threshold": threshold})
    return flagged


def check_bad_prices(conn):
    """Find rows with zero, negative, or NULL prices."""
    c = conn.cursor()
    c.execute("""
        SELECT id, coingecko_id, ticker, date, price_usd
        FROM daily_prices
        WHERE price_usd IS NULL OR price_usd <= 0
        ORDER BY coingecko_id, date
    """)
    rows = c.fetchall()
    return [
        {"row_id": r[0], "coin": r[1], "ticker": r[2], "date": r[3], "price": r[4]}
        for r in rows
    ]


def check_duplicates(conn):
    """Find (coin, date) pairs with more than one row."""
    c = conn.cursor()
    c.execute("""
        SELECT coingecko_id, ticker, date, COUNT(*) as cnt
        FROM daily_prices
        GROUP BY coingecko_id, date
        HAVING cnt > 1
        ORDER BY coingecko_id, date
    """)
    rows = c.fetchall()
    return [
        {"coin": r[0], "ticker": r[1], "date": r[2], "count": r[3]}
        for r in rows
    ]


def check_stale_coins(conn, stale_days):
    """Find coins whose latest data is more than stale_days behind the DB max date."""
    c = conn.cursor()
    c.execute("SELECT MAX(date) FROM daily_prices")
    max_date_str = c.fetchone()[0]
    if not max_date_str:
        return [], None

    max_date = datetime.strptime(max_date_str, "%Y-%m-%d")
    cutoff   = max_date - timedelta(days=stale_days)

    c.execute("""
        SELECT coingecko_id, ticker, MAX(date) as last_date
        FROM daily_prices
        GROUP BY coingecko_id
        HAVING last_date < ?
        ORDER BY last_date
    """, (cutoff.strftime("%Y-%m-%d"),))
    rows = c.fetchall()
    return [
        {"coin": r[0], "ticker": r[1], "last_date": r[2]}
        for r in rows
    ], max_date_str


def purge_rows(conn, row_ids, label):
    """Delete rows by ID. Returns count deleted."""
    if not row_ids:
        return 0
    c = conn.cursor()
    placeholders = ",".join("?" * len(row_ids))
    c.execute(f"DELETE FROM daily_prices WHERE id IN ({placeholders})", row_ids)
    conn.commit()
    print(f"  ✓ Purged {c.rowcount} {label} rows.")
    return c.rowcount


def print_section(title):
    print()
    print("─" * 60)
    print(f"  {title}")
    print("─" * 60)


def run_scan(db_path, spike_pct, drop_pct, stale_days, auto_purge, output_path=None):
    import sys
    # If output_path set, tee all print() output to a file
    if output_path:
        _outfile = open(output_path, "w")
        class Tee:
            def write(self, msg): _outfile.write(msg)
            def flush(self): _outfile.flush()
        orig_stdout = sys.stdout
        sys.stdout = Tee()

    conn = get_db_connection(db_path)
    total_issues = 0
    total_purged = 0

    print("=" * 60)
    print("  CRYPTO DB VALIDATION REPORT")
    print(f"  DB:         {db_path}")
    print(f"  Run at:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Spike threshold:  >{spike_pct}%  |  Drop threshold: >{drop_pct}%")
    print(f"  Stale threshold:  >{stale_days} days behind DB max date")
    print("=" * 60)

    # ── 1. Price spikes / drops ────────────────────────────────────────────────
    print_section("1. PRICE SPIKES / DROPS")
    spikes = check_price_spikes(conn, spike_pct, drop_pct)
    if spikes:
        artifacts = [f for f in spikes if f["type"] == "LAUNCH_ARTIFACT"]
        genuine   = [f for f in spikes if f["type"] != "LAUNCH_ARTIFACT"]
        total_issues += len(spikes)
        print(f"  FOUND {len(spikes)} flagged row(s)  "
              f"({len(artifacts)} launch artifacts, {len(genuine)} genuine spikes/drops)\n")
        print(f"  {'Ticker':<8} {'Date':<12} {'Prev Price':>12} {'Curr Price':>12} {'% Change':>10}  Type")
        print(f"  {'------':<8} {'----------':<12} {'----------':>12} {'----------':>12} {'--------':>10}  ----")
        for f in spikes:
            print(f"  {f['ticker']:<8} {f['date']:<12} ${f['prev_price']:>11.6f} ${f['curr_price']:>11.4f} {f['pct_change']:>+9.1f}%  {f['type']}")

        if artifacts:
            print()
            print("  LAUNCH_ARTIFACT = spike caused by a near-zero yfinance placeholder row.")
            print("  --purge will delete the corrupt near-zero prev_price row, not the spike day.")
        if genuine:
            print()
            print("  SPIKE/DROP = large genuine move. Review manually before purging.")
            print("  Could be: real market event (SHIB/DOGE 2021) OR corrupt data (GRT $46 in 2020).")

        if auto_purge:
            # For launch artifacts: purge the near-zero prev_price row
            artifact_prev_ids = [f["prev_row_id"] for f in artifacts if f["prev_row_id"]]
            if artifact_prev_ids:
                total_purged += purge_rows(conn, artifact_prev_ids, "launch-artifact near-zero")
            # For genuine spikes/drops: do NOT auto-purge — require manual review
            if genuine:
                print(f"  ⚠️  {len(genuine)} genuine SPIKE/DROP row(s) NOT auto-purged — manual review required.")
    else:
        print("  ✓ No price spikes or drops detected.")

    # ── 2. Near-zero prices ───────────────────────────────────────────────────
    print_section(f"2. NEAR-ZERO PRICES (< ${NEAR_ZERO_LIMIT} — likely launch artifacts)")
    near_zero = check_near_zero_prices(conn)
    if near_zero:
        total_issues += len(near_zero)
        print(f"  FOUND {len(near_zero)} near-zero price row(s) (price < 0.1% of coin median):\n")
        print(f"  {'Ticker':<8} {'Date':<12} {'Price':>14} {'Coin Median':>14} {'Threshold':>14}")
        print(f"  {'------':<8} {'----------':<12} {'-----':>14} {'-----------':>14} {'---------':>14}")
        for n in near_zero:
            print(f"  {n['ticker']:<8} {n['date']:<12} ${n['price']:>13.8f} ${n['median']:>13.8f} ${n['threshold']:>13.8f}")
        if auto_purge:
            ids = [n["row_id"] for n in near_zero]
            total_purged += purge_rows(conn, ids, "near-zero")
    else:
        print(f"  ✓ No near-zero prices found (coin-relative check).")

    # ── 3. Zero / negative / NULL prices ──────────────────────────────────────
    print_section("3. BAD PRICES (zero, negative, NULL)")
    bad = check_bad_prices(conn)
    if bad:
        total_issues += len(bad)
        print(f"  FOUND {len(bad)} bad price row(s):\n")
        for b in bad:
            print(f"  {b['ticker']:<8} {b['date']}  price={b['price']}")

        if auto_purge:
            ids = [b["row_id"] for b in bad]
            total_purged += purge_rows(conn, ids, "bad-price")
    else:
        print("  ✓ No zero/negative/NULL prices found.")

    # ── 4. Duplicate rows ─────────────────────────────────────────────────────
    print_section("4. DUPLICATE (COIN, DATE) ROWS")
    dupes = check_duplicates(conn)
    if dupes:
        total_issues += len(dupes)
        print(f"  FOUND {len(dupes)} duplicate pair(s):\n")
        for d in dupes:
            print(f"  {d['ticker']:<8} {d['date']}  ({d['count']} rows)")
        if auto_purge:
            print("  NOTE: Duplicate cleanup requires manual review — skipping auto-purge.")
    else:
        print("  ✓ No duplicate rows found.")

    # ── 5. Stale coins ────────────────────────────────────────────────────────
    print_section("5. STALE COINS (lagging DB max date)")
    stale, max_date = check_stale_coins(conn, stale_days)
    if stale:
        total_issues += len(stale)
        print(f"  DB max date: {max_date}")
        print(f"  FOUND {len(stale)} stale coin(s):\n")
        for s in stale:
            print(f"  {s['ticker']:<8} last data: {s['last_date']}")
        print()
        print("  ACTION: Check collect_data.py — yfinance may have failed for these tickers.")
        print("          If last_date is >7 days stale, consider removing from universe.")
    else:
        print(f"  ✓ All coins current within {stale_days} days of {max_date}.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if total_issues == 0:
        print("  ✅ DATABASE IS CLEAN. No issues found.")
        conn.close()
        return 0
    else:
        print(f"  ⚠️  {total_issues} issue(s) found across all checks.")
        if auto_purge and total_purged > 0:
            print(f"  🗑️  {total_purged} row(s) purged.")
            print()
            print("  RECOMMENDATION: Re-run without --purge to confirm clean state.")
            conn.close()
            return 2
        elif not auto_purge:
            print()
            print("  NEXT STEP: Review flagged rows above.")
            print("  To auto-purge spike/drop and bad-price rows, re-run with --purge.")
            conn.close()
            return 1
    conn.close()
    if output_path:
        sys.stdout = orig_stdout
        _outfile.close()
        print(f"Report saved to: {output_path}")
    return 1


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Proactive data corruption scanner for crypto_backtest.db"
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"Path to SQLite database (default: {DEFAULT_DB})"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SPIKE_THRESHOLD,
        help=f"Single-day spike threshold %% (default: {SPIKE_THRESHOLD})"
    )
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=DROP_THRESHOLD,
        help=f"Single-day drop threshold %% (default: {DROP_THRESHOLD})"
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=STALE_DAYS,
        help=f"Days behind DB max date to flag as stale (default: {STALE_DAYS})"
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Auto-purge spike/drop and bad-price rows after reporting"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save report to a text file instead of printing (e.g. --output report.txt)"
    )
    args = parser.parse_args()

    exit_code = run_scan(
        db_path=args.db,
        spike_pct=args.threshold,
        drop_pct=args.drop_threshold,
        stale_days=args.stale_days,
        auto_purge=args.purge,
        output_path=args.output,
    )
    sys.exit(exit_code)
