#!/usr/bin/env python3
"""
backtest_autotrader.py
======================
Simulates the crypto_autotrader running daily from Jan 1, 2026 to Mar 2, 2026.
Uses actual historical price data and Fear & Greed from crypto_backtest.db.

Strategy:
  - $5,000 starting capital
  - Score >= 2 = BUY
  - 3-day hold, then SELL at close
  - Score-scaled position sizing
  - Max 6 positions, 40% cash reserve
  - Bucket analysis: Large Cap / Mid Cap / Smaller Alt

Usage:
    python3 backtest_autotrader.py
    python3 backtest_autotrader.py --start 2026-01-01 --end 2026-03-02
"""

import argparse
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

DB_PATH = "crypto_backtest.db"

PORTFOLIO_SIZE   = 5000.0
MIN_CASH_RESERVE = 0.40
MAX_POSITIONS    = 6
HOLD_DAYS        = 3

POSITION_SIZE = {2: 0.05, 3: 0.08, 4: 0.12}

EXCLUDED = {"GALA", "JUP", "KAS", "MKR", "RNDR", "TRX"}

BUCKET_LABELS = {1: "Large Cap", 2: "Mid Cap", 3: "Smaller Alt"}


def load_data(start_date, end_date):
    conn = sqlite3.connect(DB_PATH)
    history_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
    prices = pd.read_sql_query(f"""
        SELECT coingecko_id, ticker, bucket, date, price_usd, volume_usd
        FROM daily_prices WHERE date >= '{history_start}'
        ORDER BY coingecko_id, date
    """, conn)
    fg = pd.read_sql_query("SELECT date, value, classification FROM fear_greed ORDER BY date", conn)
    conn.close()
    prices["date"] = pd.to_datetime(prices["date"])
    fg["date"] = pd.to_datetime(fg["date"])
    return prices, fg


def compute_signals_for_date(prices_all, fg_all, as_of_date):
    cutoff = as_of_date - timedelta(days=60)
    prices = prices_all[(prices_all["date"] <= as_of_date) & (prices_all["date"] >= cutoff)].copy()
    fg_row = fg_all[fg_all["date"] <= as_of_date].tail(2)
    fg_today = int(fg_row.iloc[-1]["value"]) if len(fg_row) >= 1 else None
    fg_yesterday = int(fg_row.iloc[-2]["value"]) if len(fg_row) >= 2 else None

    results = []
    for coin_id, group in prices.groupby("coingecko_id"):
        group = group.sort_values("date").copy()
        if len(group) < 6:
            continue
        if group.iloc[-1]["date"].date() != as_of_date.date():
            continue
        latest = group.iloc[-1]
        ticker = latest["ticker"]
        bucket = int(latest["bucket"])
        if ticker in EXCLUDED:
            continue

        current_price = latest["price_usd"]
        current_volume = latest["volume_usd"]

        ret_1d = (group.iloc[-1]["price_usd"] / group.iloc[-2]["price_usd"] - 1) * 100 if len(group) >= 2 else 0
        ret_5d = (group.iloc[-1]["price_usd"] / group.iloc[-6]["price_usd"] - 1) * 100 if len(group) >= 6 else 0

        vol_ratio = 0
        if len(group) >= 21 and current_volume and current_volume > 0:
            avg_vol = group["volume_usd"].tail(21).iloc[:-1].mean()
            vol_ratio = current_volume / avg_vol if avg_vol > 0 else 0

        vol_20d = None
        vol_pctile = None
        if len(group) >= 21:
            daily_rets = group["price_usd"].pct_change().dropna().tail(20) * 100
            vol_20d = daily_rets.std() * np.sqrt(365)
            all_vols = []
            for i in range(20, len(group)):
                sub = group.iloc[i-20:i]["price_usd"].pct_change().dropna() * 100
                if len(sub) >= 15:
                    all_vols.append(sub.std() * np.sqrt(365))
            if all_vols:
                vol_pctile = (np.array(all_vols) < vol_20d).sum() / len(all_vols) * 100

        score = 0
        reasons = []

        if ret_5d <= -25:
            score += 3
            reasons.append(f"5d DD>=25% ({ret_5d:+.1f}%): +3")
        elif ret_5d <= -15:
            score += 1
            reasons.append(f"5d DD 15-25% ({ret_5d:+.1f}%): +1")

        if fg_today is not None:
            if fg_today >= 90:
                score += 2
                reasons.append(f"Extreme Greed FG={fg_today}: +2")
            elif fg_today >= 75:
                score += 1
                reasons.append(f"Greed Zone FG={fg_today}: +1")

        if vol_ratio >= 3.0 and ret_1d <= -5:
            score += 1
            reasons.append(f"Capitulation (vol {vol_ratio:.1f}x, 1d {ret_1d:+.1f}%): +1")

        if fg_today is not None and fg_yesterday is not None:
            if fg_yesterday <= 25 and fg_today > 25:
                score -= 2
                reasons.append(f"Fear Exit (FG {fg_yesterday}->{fg_today}): -2")

        if vol_pctile is not None and vol_pctile <= 10:
            score -= 1
            reasons.append(f"Low Vol ({vol_pctile:.0f}th pctile): -1")

        results.append({
            "ticker": ticker, "bucket": bucket, "score": score,
            "price": current_price, "ret_5d": ret_5d, "reasons": reasons,
        })

    return pd.DataFrame(results).sort_values("score", ascending=False) if results else pd.DataFrame()


def get_price_on_date(prices_all, ticker, date):
    rows = prices_all[(prices_all["ticker"] == ticker) & (prices_all["date"].dt.date == date.date())]
    if not rows.empty:
        return float(rows.iloc[0]["price_usd"])
    rows = prices_all[
        (prices_all["ticker"] == ticker) &
        (prices_all["date"].dt.date <= date.date()) &
        (prices_all["date"].dt.date >= (date - timedelta(days=2)).date())
    ].sort_values("date", ascending=False)
    return float(rows.iloc[0]["price_usd"]) if not rows.empty else None


def run_backtest(start_date, end_date):
    print("=" * 70)
    print("CRYPTO AUTO-TRADER HISTORICAL SIMULATION")
    print(f"Period:  {start_date} to {end_date}")
    print(f"Capital: ${PORTFOLIO_SIZE:,.0f} | Hold: {HOLD_DAYS}d | Max positions: {MAX_POSITIONS}")
    print("=" * 70)

    prices_all, fg_all = load_data(start_date, end_date)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")

    all_dates = sorted(prices_all[
        (prices_all["date"] >= start_dt) & (prices_all["date"] <= end_dt)
    ]["date"].dt.date.unique())

    cash = PORTFOLIO_SIZE
    open_positions = []
    closed_trades  = []
    daily_equity   = []

    print(f"\nSimulating {len(all_dates)} trading days...\n")

    for date in all_dates:
        date_dt = datetime.combine(date, datetime.min.time())

        # EXITS
        still_open = []
        for pos in open_positions:
            days_held = (date - pos["entry_date"]).days
            if days_held >= HOLD_DAYS:
                exit_price = get_price_on_date(prices_all, pos["ticker"], date_dt)
                if exit_price is None:
                    still_open.append(pos)
                    continue
                pnl_usd = (exit_price - pos["entry_price"]) * pos["quantity"]
                pnl_pct = (exit_price / pos["entry_price"] - 1) * 100
                cash += pos["usd_value"] + pnl_usd
                closed_trades.append({**pos, "exit_date": date, "exit_price": exit_price,
                                       "pnl_usd": pnl_usd, "pnl_pct": pnl_pct, "days_held": days_held})
            else:
                still_open.append(pos)
        open_positions = still_open

        # ENTRIES
        signals = compute_signals_for_date(prices_all, fg_all, date_dt)
        open_tickers = {p["ticker"] for p in open_positions}

        if not signals.empty:
            buys = signals[(signals["score"] >= 2) & (~signals["ticker"].isin(open_tickers))]
            for _, row in buys.iterrows():
                if len(open_positions) >= MAX_POSITIONS:
                    break
                deployed = sum(p["usd_value"] for p in open_positions)
                max_deployable = PORTFOLIO_SIZE * (1 - MIN_CASH_RESERVE) - deployed
                if max_deployable <= 0:
                    break
                score = int(row["score"])
                size_pct = POSITION_SIZE.get(min(score, 4), POSITION_SIZE[2])
                usd_amount = min(PORTFOLIO_SIZE * size_pct, max_deployable, cash)
                if usd_amount < 1.0:
                    break
                entry_price = row["price"]
                quantity = usd_amount / entry_price
                cash -= usd_amount
                open_positions.append({
                    "ticker": row["ticker"], "bucket": int(row["bucket"]),
                    "score": score, "entry_date": date, "entry_price": entry_price,
                    "quantity": quantity, "usd_value": usd_amount, "reasons": row["reasons"],
                })

        # Daily equity
        unrealized = 0
        for pos in open_positions:
            cur = get_price_on_date(prices_all, pos["ticker"], date_dt)
            if cur:
                unrealized += (cur - pos["entry_price"]) * pos["quantity"]

        equity = cash + sum(p["usd_value"] for p in open_positions) + unrealized
        fg_row = fg_all[fg_all["date"].dt.date <= date].tail(1)
        fg_val = int(fg_row.iloc[0]["value"]) if not fg_row.empty else None

        prev_equity = daily_equity[-1]["equity"] if daily_equity else PORTFOLIO_SIZE
        daily_equity.append({
            "date": date, "equity": equity, "cash": cash,
            "n_positions": len(open_positions), "fg": fg_val,
            "daily_pnl": equity - prev_equity,
        })

    eq_df     = pd.DataFrame(daily_equity)
    trades_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()

    final_equity = eq_df.iloc[-1]["equity"]
    total_return = (final_equity / PORTFOLIO_SIZE - 1) * 100
    total_pnl    = final_equity - PORTFOLIO_SIZE

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Period:           {start_date} to {end_date} ({len(all_dates)} days)")
    print(f"Starting capital: ${PORTFOLIO_SIZE:,.2f}")
    print(f"Ending equity:    ${final_equity:,.2f}")
    print(f"Total P&L:        ${total_pnl:+,.2f}")
    print(f"Total return:     {total_return:+.2f}%")

    if not trades_df.empty:
        wins   = trades_df[trades_df["pnl_usd"] > 0]
        losses = trades_df[trades_df["pnl_usd"] <= 0]
        win_rate = len(wins) / len(trades_df) * 100
        print(f"\nTrades:           {len(trades_df)} closed")
        print(f"Win rate:         {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
        print(f"Avg win:          {wins['pnl_pct'].mean():+.2f}%" if len(wins) > 0 else "Avg win: N/A")
        print(f"Avg loss:         {losses['pnl_pct'].mean():+.2f}%" if len(losses) > 0 else "Avg loss: N/A")
        print(f"Best trade:       {trades_df['pnl_pct'].max():+.2f}% ({trades_df.loc[trades_df['pnl_pct'].idxmax(),'ticker']})")
        print(f"Worst trade:      {trades_df['pnl_pct'].min():+.2f}% ({trades_df.loc[trades_df['pnl_pct'].idxmin(),'ticker']})")
        print(f"Avg hold:         {trades_df['days_held'].mean():.1f} days")

    eq_df["peak"] = eq_df["equity"].cummax()
    eq_df["dd"]   = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"] * 100
    print(f"Max drawdown:     {eq_df['dd'].min():.2f}%")

    btc_start = get_price_on_date(prices_all, "BTC", datetime.strptime(start_date, "%Y-%m-%d"))
    btc_end   = get_price_on_date(prices_all, "BTC", datetime.strptime(end_date,   "%Y-%m-%d"))
    if btc_start and btc_end:
        btc_ret = (btc_end / btc_start - 1) * 100
        print(f"\nBTC buy-and-hold: {btc_ret:+.2f}% (${PORTFOLIO_SIZE*(1+btc_ret/100):,.2f})")
        print(f"Alpha vs BTC:     {total_return - btc_ret:+.2f}%")

    if not trades_df.empty:
        print(f"\n{'=' * 70}")
        print("PERFORMANCE BY BUCKET")
        print(f"{'=' * 70}")
        print(f"{'Bucket':<15} {'Trades':>7} {'Win%':>7} {'Avg%':>9} {'Total$':>10} {'Best':>8} {'Worst':>8}")
        print("-" * 70)
        for bid, label in BUCKET_LABELS.items():
            b = trades_df[trades_df["bucket"] == bid]
            if b.empty:
                continue
            bw = b[b["pnl_usd"] > 0]
            print(f"{label:<15} {len(b):>7} {len(bw)/len(b)*100:>6.1f}% "
                  f"{b['pnl_pct'].mean():>+8.2f}% ${b['pnl_usd'].sum():>+9.2f} "
                  f"{b['pnl_pct'].max():>+7.2f}% {b['pnl_pct'].min():>+7.2f}%")

    print(f"\n{'=' * 70}")
    print("DAILY EQUITY & P&L")
    print(f"{'=' * 70}")
    print(f"{'Date':<12} {'Equity':>10} {'Daily P&L':>12} {'Cumul%':>9} {'Cash':>10} {'Pos':>5} {'F&G':>5}")
    print("-" * 70)
    for _, row in eq_df.iterrows():
        cumul = (row["equity"] / PORTFOLIO_SIZE - 1) * 100
        fg_s  = str(int(row["fg"])) if row["fg"] is not None else "N/A"
        print(f"{str(row['date']):<12} ${row['equity']:>9,.2f} ${row['daily_pnl']:>+10.2f} "
              f"{cumul:>+8.2f}% ${row['cash']:>9,.2f} {int(row['n_positions']):>5} {fg_s:>5}")

    if not trades_df.empty:
        print(f"\n{'=' * 70}")
        print("FULL TRADE LOG")
        print(f"{'=' * 70}")
        print(f"{'Ticker':<8} {'Bucket':<12} {'Sc':>3} {'Entry Date':>12} {'Exit Date':>12} "
              f"{'Entry$':>10} {'Exit$':>10} {'P&L$':>9} {'P&L%':>8} {'Days':>5}")
        print("-" * 95)
        for _, t in trades_df.sort_values("entry_date").iterrows():
            print(f"{t['ticker']:<8} {BUCKET_LABELS.get(t['bucket'],'?'):<12} {t['score']:>+3d} "
                  f"{str(t['entry_date']):>12} {str(t['exit_date']):>12} "
                  f"${t['entry_price']:>9,.4f} ${t['exit_price']:>9,.4f} "
                  f"${t['pnl_usd']:>+8.2f} {t['pnl_pct']:>+7.2f}% {t['days_held']:>5}")

    print(f"\n{'=' * 70}")
    print("Simulation complete.")
    print(f"{'=' * 70}")
    return eq_df, trades_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end",   default="2026-03-02")
    args = parser.parse_args()
    run_backtest(args.start, args.end)
