#!/usr/bin/env python3
"""
purge_corrupt_rows.py
=====================
One-shot targeted purge of known corrupt rows that require manual price
knowledge to identify (not detectable by the general validator alone).

Targeted purges:
  1. GRT (The Graph) — all rows where price > $3.00
     GRT all-time high was $2.88 (Feb 2021). Any row above $3.00 is physically
     impossible and is corrupt yfinance data.

  2. POL (Polygon / ex-MATIC) — all rows
     Rebranded from MATIC in Oct 2023. yfinance ticker broken post-rebrand.
     Last data point: 2023-10-31. Stale and unrecoverable. Excluded from
     autotrader universe — purging from DB for cleanliness.

  3. 2026-02-24 bad pull — specific coin/date rows
     Multiple coins show extreme drops on exactly 2026-02-24, a known bad
     yfinance pull date. Purge the specific rows listed below.

Usage:
    python3 purge_corrupt_rows.py           # Dry run — shows what would be deleted
    python3 purge_corrupt_rows.py --execute # Actually delete the rows
"""

import argparse
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crypto_backtest.db")

# ── Targeted purge definitions ─────────────────────────────────────────────────

# 1. GRT price cap — ATH was $2.88, cap at $3.00 for safety margin
GRT_PRICE_CAP = 3.00

# 2. POL — purge all rows (stale coin, broken ticker)
POL_COINGECKO_ID = "polygon-ecosystem-token"

# 3. Bad yfinance date range — coins with corrupt data from a bad pull window
# Rather than whack-a-mole one day at a time, purge the full corrupt range
# and let collect_data.py refill nightly.
BAD_PULL_DATE_RANGES = [
    # (coingecko_id, ticker, from_date_inclusive, to_date_inclusive)
    # All show corrupt prices starting 2026-02-24 through DB max date (2026-03-04)
    ("the-graph",    "GRT",  "2026-02-24", "2026-03-04"),
    ("render-token", "RNDR", "2026-02-24", "2026-03-04"),
    ("immutable-x",  "IMX",  "2026-02-24", "2026-03-04"),
    ("sui",          "SUI",  "2026-02-24", "2026-03-04"),
    ("arbitrum",     "ARB",  "2026-02-24", "2026-03-04"),
]

# 4. Specific bad rows not covered by ranges above
BAD_PULL_ROWS = [
    # ARB 2023-12-07/08 — ARB did not go to zero; cascading corrupt rows
    ("arbitrum", "ARB", "2023-12-07"),
    ("arbitrum", "ARB", "2023-12-08"),
]


def dry_run(conn):
    c = conn.cursor()
    total = 0

    print("=" * 60)
    print("  DRY RUN — No rows will be deleted")
    print("  Re-run with --execute to apply changes")
    print("=" * 60)

    # 1. GRT impossible prices
    c.execute("""
        SELECT COUNT(*), MIN(price_usd), MAX(price_usd), MIN(date), MAX(date)
        FROM daily_prices
        WHERE coingecko_id = 'the-graph' AND price_usd > ?
    """, (GRT_PRICE_CAP,))
    r = c.fetchone()
    count = r[0] or 0
    print(f"\n1. GRT rows with price > ${GRT_PRICE_CAP}:")
    if count:
        print(f"   Would delete {count} rows  |  price range: ${r[1]:.4f}–${r[2]:.4f}  |  dates: {r[3]} to {r[4]}")
    else:
        print("   None found.")
    total += count

    # 2. POL all rows
    c.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM daily_prices WHERE coingecko_id = ?",
              (POL_COINGECKO_ID,))
    r = c.fetchone()
    count = r[0] or 0
    print(f"\n2. POL (polygon-ecosystem-token) all rows:")
    if count:
        print(f"   Would delete {count} rows  |  dates: {r[1]} to {r[2]}")
    else:
        print("   None found.")
    total += count

    # 3. Bad pull date ranges
    print(f"\n3. Bad yfinance date range purges:")
    range_count = 0
    for cg_id, ticker, from_date, to_date in BAD_PULL_DATE_RANGES:
        c.execute("""
            SELECT COUNT(*), MIN(date), MAX(date)
            FROM daily_prices
            WHERE coingecko_id = ? AND date >= ? AND date <= ?
        """, (cg_id, from_date, to_date))
        r = c.fetchone()
        if r[0]:
            print(f"   {ticker:<8} {from_date} to {to_date}: {r[0]} rows ({r[1]} to {r[2]})  → would delete")
            range_count += r[0]
        else:
            print(f"   {ticker:<8} {from_date} to {to_date}: not found (already purged)")
    total += range_count

    # 4. Specific bad rows
    print(f"\n4. Specific bad rows:")
    bad_count = 0
    for cg_id, ticker, date in BAD_PULL_ROWS:
        c.execute("""
            SELECT COUNT(*), price_usd FROM daily_prices
            WHERE coingecko_id = ? AND date = ?
        """, (cg_id, date))
        r = c.fetchone()
        if r[0]:
            print(f"   {ticker:<8} {date}  price=${r[1]:.6f}  → would delete")
            bad_count += r[0]
        else:
            print(f"   {ticker:<8} {date}  → not found (already purged)")
    total += bad_count

    print(f"\n{'─'*60}")
    print(f"  TOTAL: {total} rows would be deleted.")
    print(f"  Run with --execute to apply.")


def execute_purge(conn):
    c = conn.cursor()
    total = 0

    print("=" * 60)
    print("  EXECUTING PURGE")
    print("=" * 60)

    # 1. GRT impossible prices
    c.execute("""
        DELETE FROM daily_prices
        WHERE coingecko_id = 'the-graph' AND price_usd > ?
    """, (GRT_PRICE_CAP,))
    n = c.rowcount
    print(f"\n1. GRT rows > ${GRT_PRICE_CAP}: deleted {n} rows")
    total += n

    # 2. POL all rows
    c.execute("DELETE FROM daily_prices WHERE coingecko_id = ?", (POL_COINGECKO_ID,))
    n = c.rowcount
    print(f"2. POL all rows: deleted {n} rows")
    total += n

    # 3. Bad pull date ranges
    print(f"3. Bad yfinance date range purges:")
    range_count = 0
    for cg_id, ticker, from_date, to_date in BAD_PULL_DATE_RANGES:
        c.execute("""
            DELETE FROM daily_prices
            WHERE coingecko_id = ? AND date >= ? AND date <= ?
        """, (cg_id, from_date, to_date))
        n = c.rowcount
        if n:
            print(f"   {ticker:<8} {from_date} to {to_date}: deleted {n} row(s)")
            range_count += n
        else:
            print(f"   {ticker:<8} {from_date} to {to_date}: not found (already purged)")
    total += range_count

    # 4. Specific bad rows
    print(f"4. Specific bad rows:")
    bad_count = 0
    for cg_id, ticker, date in BAD_PULL_ROWS:
        c.execute("DELETE FROM daily_prices WHERE coingecko_id = ? AND date = ?", (cg_id, date))
        n = c.rowcount
        if n:
            print(f"   {ticker:<8} {date}: deleted {n} row(s)")
            bad_count += n
        else:
            print(f"   {ticker:<8} {date}: not found (already purged)")
    total += bad_count

    conn.commit()
    print(f"\n{'─'*60}")
    print(f"  ✅ Done. {total} rows deleted total.")
    print(f"  Run validate_crypto_db.py to confirm clean state.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Targeted purge of known corrupt DB rows")
    parser.add_argument("--execute", action="store_true",
                        help="Actually delete rows (default is dry run)")
    parser.add_argument("--db", default=DB_PATH, help="Path to database")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: DB not found at {args.db}")
        raise SystemExit(1)

    conn = sqlite3.connect(args.db)
    if args.execute:
        execute_purge(conn)
    else:
        dry_run(conn)
    conn.close()
