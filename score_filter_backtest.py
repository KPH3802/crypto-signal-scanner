#!/usr/bin/env python3
"""
score_filter_backtest.py
========================
Tests the proposed Score 4 quality filter:

  CURRENT RULE:  Score ≥ 4 = VERY STRONG → 12% position
  PROPOSED RULE: Score ≥ 4 requires severe crash (≥-25%) +
                 capitulation OR extreme greed (FG ≥ 90)
                 Otherwise downgrades to Score 3 (8% position)

The 42 "severe crash alone" events that score 4 via:
  severe crash (+3) + greed zone FG 75-89 (+1)
...show only 47.6% win rate (below coin flip).

This script compares:
  A) Old rules: all Score ≥ 4 treated the same
  B) New rules: "severe crash + greed zone only" capped at Score 3

Usage:
    python3 score_filter_backtest.py
    python3 score_filter_backtest.py --output filter_results.txt
    python3 score_filter_backtest.py --recent 180  # last N days only
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats

from database import DB_PATH

WINSOR_LOW  = 1
WINSOR_HIGH = 99

COST_BLUE_CHIP = 0.20
COST_TOP20     = 0.40
COST_ALTCOIN   = 0.80

PORTFOLIO      = 5000.0
POSITION_SIZE  = {2: 0.05, 3: 0.08, 4: 0.12}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _winsorize(s):
    lo = np.nanpercentile(s, WINSOR_LOW)
    hi = np.nanpercentile(s, WINSOR_HIGH)
    return s.clip(lower=lo, upper=hi)


def alpha_vs_base(sig_rets, base_rets):
    if len(sig_rets) < 5 or len(base_rets) < 5:
        return float("nan"), float("nan"), float("nan")
    alpha = sig_rets.mean() - base_rets.mean()
    t, p  = stats.ttest_ind(sig_rets, base_rets, equal_var=False)
    return alpha, t, p


def fmt(alpha, p_val, width=9):
    if np.isnan(alpha):
        return f"{'N/A':>{width}}"
    star = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "  "
    return f"{alpha:>+{width-2}.2f}%{star}"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(db_path=DB_PATH, since_days=None):
    conn = sqlite3.connect(db_path)
    prices = pd.read_sql_query("""
        SELECT coingecko_id, ticker, bucket, date, price_usd, volume_usd
        FROM daily_prices ORDER BY coingecko_id, date
    """, conn)
    fg = pd.read_sql_query(
        "SELECT date, value AS fg_value FROM fear_greed ORDER BY date", conn)
    conn.close()

    prices["date"] = pd.to_datetime(prices["date"])
    fg["date"]     = pd.to_datetime(fg["date"])

    if since_days:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=since_days)
        prices = prices[prices["date"] >= cutoff]

    all_coins = []
    for coin_id, group in prices.groupby("coingecko_id"):
        group = group.sort_values("date").copy()
        group["fwd_3d"]  = (group["price_usd"].shift(-3)  / group["price_usd"] - 1) * 100
        group["fwd_5d"]  = (group["price_usd"].shift(-5)  / group["price_usd"] - 1) * 100
        group["ret_1d"]  = group["price_usd"].pct_change() * 100
        group["ret_5d"]  = (group["price_usd"] / group["price_usd"].shift(5)  - 1) * 100
        group["vol_1d"]  = group["price_usd"].pct_change() * 100
        group["hv_20d"]  = group["vol_1d"].rolling(20).std() * np.sqrt(365)
        group["vol_pctile"] = group["hv_20d"].rank(pct=True) * 100
        group["vol_ratio"]  = group["volume_usd"] / group["volume_usd"].rolling(20).mean()
        all_coins.append(group)

    df = pd.concat(all_coins, ignore_index=True)
    df["fwd_3d"] = _winsorize(df["fwd_3d"])
    df["fwd_5d"] = _winsorize(df["fwd_5d"])
    df["ret_1d"] = _winsorize(df["ret_1d"])
    df["ret_5d"] = _winsorize(df["ret_5d"])

    df = df.merge(fg, on="date", how="left")

    df["txn_cost"] = COST_ALTCOIN
    df.loc[df["bucket"] == 1, "txn_cost"] = COST_BLUE_CHIP
    df.loc[df["bucket"] == 2, "txn_cost"] = COST_TOP20

    return df


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_scores(df):
    df = df.copy()
    df["score_raw"] = 0.0

    # Crash family
    df.loc[df["ret_5d"] <= -25, "score_raw"] += 3
    df.loc[(df["ret_5d"] <= -15) & (df["ret_5d"] > -25), "score_raw"] += 1

    # Greed family
    df.loc[df["fg_value"] >= 90,  "score_raw"] += 2
    df.loc[(df["fg_value"] >= 75) & (df["fg_value"] < 90), "score_raw"] += 1

    # Capitulation
    df.loc[(df["vol_ratio"] >= 3.0) & (df["ret_1d"] <= -5), "score_raw"] += 1

    # Anti-signals
    df_s = df.sort_values(["coingecko_id", "date"])
    fg_prev = df_s.groupby("coingecko_id")["fg_value"].shift(1)
    fear_exit = (fg_prev <= 25) & (df_s["fg_value"] > 25)
    df.loc[fear_exit.values, "score_raw"] -= 2
    df.loc[df["vol_pctile"] <= 10, "score_raw"] -= 1

    # ── Component flags ────────────────────────────────────────────────────
    df["comp_severe_crash"] = df["ret_5d"] <= -25
    df["comp_ext_greed"]    = df["fg_value"] >= 90
    df["comp_capitulation"] = (df["vol_ratio"] >= 3.0) & (df["ret_1d"] <= -5)

    # ── FILTERED SCORE ─────────────────────────────────────────────────────
    # "Severe crash alone" = severe crash, no extreme greed, no capitulation
    # These score exactly 4 via crash(+3) + greed zone(+1)
    # Filter: if score_raw == 4 and this exact combo → cap at 3
    severe_crash_alone = (
        df["comp_severe_crash"] &
        ~df["comp_ext_greed"] &
        ~df["comp_capitulation"]
    )
    df["score_filtered"] = df["score_raw"].copy()
    df.loc[
        (df["score_raw"] == 4) & severe_crash_alone,
        "score_filtered"
    ] = 3

    return df


# ── Analysis ──────────────────────────────────────────────────────────────────

def compare_rules(df, label="All time"):
    base = df["fwd_3d"].dropna()
    years = (df["date"].max() - df["date"].min()).days / 365.25

    print(f"\n  {'Rule':<30} {'n':>6} {'n/yr':>6} {'3d Alpha':>9} "
          f"{'t-stat':>7} {'p-val':>7} {'Win%':>6} {'Net/Yr$':>9}")
    print(f"  {'-'*75}")

    # Old: all score ≥ 4
    old4 = df[df["score_raw"] >= 4]["fwd_3d"].dropna()
    old4_cost = df[df["score_raw"] >= 4]["txn_cost"].mean()
    a, t, p = alpha_vs_base(old4, base)
    win = (old4 > 0).mean() * 100
    nyr = len(old4) / years
    annual_net = nyr * ((a - old4_cost) / 100) * (POSITION_SIZE[4] * PORTFOLIO)
    print(f"  {'OLD: Score ≥ 4 (all)':<30} {len(old4):>6,} {nyr:>5.1f} "
          f"{fmt(a, p)} {t:>7.2f} {p:>7.4f} {win:>5.1f}% ${annual_net:>+8.0f}")

    # New: filtered score ≥ 4 (severe crash + cap or extreme greed)
    new4 = df[df["score_filtered"] >= 4]["fwd_3d"].dropna()
    new4_cost = df[df["score_filtered"] >= 4]["txn_cost"].mean()
    a2, t2, p2 = alpha_vs_base(new4, base)
    win2 = (new4 > 0).mean() * 100
    nyr2 = len(new4) / years
    annual_net2 = nyr2 * ((a2 - new4_cost) / 100) * (POSITION_SIZE[4] * PORTFOLIO)
    print(f"  {'NEW: Score ≥ 4 (filtered)':<30} {len(new4):>6,} {nyr2:>5.1f} "
          f"{fmt(a2, p2)} {t2:>7.2f} {p2:>7.4f} {win2:>5.1f}% ${annual_net2:>+8.0f}")

    # Downgraded events (now Score 3)
    downgraded = df[(df["score_raw"] == 4) & (df["score_filtered"] == 3)]["fwd_3d"].dropna()
    dg_cost = df[(df["score_raw"] == 4) & (df["score_filtered"] == 3)]["txn_cost"].mean()
    if len(downgraded) >= 5:
        a3, t3, p3 = alpha_vs_base(downgraded, base)
        win3 = (downgraded > 0).mean() * 100
        nyr3 = len(downgraded) / years
        # These now get Score 3 sizing (8% vs 12%)
        annual_net3 = nyr3 * ((a3 - dg_cost) / 100) * (POSITION_SIZE[3] * PORTFOLIO)
        print(f"  {'DOWNGRADED to Score 3':<30} {len(downgraded):>6,} {nyr3:>5.1f} "
              f"{fmt(a3, p3)} {t3:>7.2f} {p3:>7.4f} {win3:>5.1f}% ${annual_net3:>+8.0f}")

    print(f"\n  ** p<0.01   * p<0.05")
    return len(old4), len(new4), len(downgraded)


def year_by_year(df):
    print(f"\n{'='*72}")
    print(f"YEAR-BY-YEAR: OLD vs NEW RULE")
    print(f"{'='*72}")
    base_all = df["fwd_3d"].dropna()

    print(f"\n  {'Year':<6} {'Old n':>6} {'Old α':>8} {'Old Win':>8} "
          f"{'New n':>6} {'New α':>8} {'New Win':>8} {'Dropped':>8}")
    print(f"  {'-'*64}")

    for year in range(2018, 2027):
        yr = df[df["date"].dt.year == year]
        yr_base = yr["fwd_3d"].dropna()
        if len(yr_base) < 10:
            continue

        old4 = yr[yr["score_raw"] >= 4]["fwd_3d"].dropna()
        new4 = yr[yr["score_filtered"] >= 4]["fwd_3d"].dropna()
        drop = yr[(yr["score_raw"] == 4) & (yr["score_filtered"] == 3)]["fwd_3d"].dropna()

        if len(old4) < 2:
            print(f"  {year:<6} {'—':>6}")
            continue

        a_old, _, p_old = alpha_vs_base(old4, yr_base)
        win_old = (old4 > 0).mean() * 100

        if len(new4) >= 2:
            a_new, _, p_new = alpha_vs_base(new4, yr_base)
            win_new = (new4 > 0).mean() * 100
        else:
            a_new, p_new, win_new = float("nan"), 1.0, float("nan")

        print(f"  {year:<6} {len(old4):>6} {fmt(a_old, p_old, 8)} {win_old:>7.1f}% "
              f"{len(new4):>6} {fmt(a_new, p_new, 8)} "
              f"{win_new:>7.1f}% {len(drop):>7}")


def what_would_change(df, days_back=90):
    """Show which recent signals would have been downgraded."""
    print(f"\n{'='*72}")
    print(f"RECENT SIGNALS THAT WOULD BE DOWNGRADED (last {days_back} days)")
    print(f"{'='*72}")

    cutoff = df["date"].max() - pd.Timedelta(days=days_back)
    recent = df[df["date"] >= cutoff].copy()

    downgraded = recent[
        (recent["score_raw"] == 4) & (recent["score_filtered"] == 3)
    ].copy()

    if len(downgraded) == 0:
        print(f"\n  No signals would have been downgraded in the last {days_back} days.")
        print(f"  (No severe crash + greed zone only signals fired recently)")
        return

    downgraded = downgraded.sort_values("date", ascending=False)
    print(f"\n  {len(downgraded)} signals downgraded from Score 4 → 3:\n")
    print(f"  {'Date':<12} {'Ticker':<8} {'5d Ret':>8} {'FG':>5} {'Cap?':>5} "
          f"{'3d Fwd':>8} {'Win?':>6}")
    print(f"  {'-'*56}")

    for _, row in downgraded.iterrows():
        cap = "Yes" if row["comp_capitulation"] else "No"
        fwd = f"{row['fwd_3d']:+.2f}%" if not np.isnan(row.get("fwd_3d", float("nan"))) else "N/A"
        win = "✓" if not np.isnan(row.get("fwd_3d", float("nan"))) and row["fwd_3d"] > 0 else "✗"
        print(f"  {str(row['date'].date()):<12} {row['ticker']:<8} "
              f"{row['ret_5d']:>+7.1f}% {row['fg_value']:>5.0f} {cap:>5} "
              f"{fwd:>8} {win:>6}")


def summary_verdict(df):
    print(f"\n{'='*72}")
    print(f"VERDICT")
    print(f"{'='*72}")

    base = df["fwd_3d"].dropna()
    years = (df["date"].max() - df["date"].min()).days / 365.25

    old4 = df[df["score_raw"] >= 4]["fwd_3d"].dropna()
    new4 = df[df["score_filtered"] >= 4]["fwd_3d"].dropna()
    drop = df[(df["score_raw"] == 4) & (df["score_filtered"] == 3)]["fwd_3d"].dropna()

    a_old, _, p_old = alpha_vs_base(old4, base)
    a_new, _, p_new = alpha_vs_base(new4, base)

    win_old = (old4 > 0).mean() * 100 if len(old4) > 0 else 0
    win_new = (new4 > 0).mean() * 100 if len(new4) > 0 else 0

    old_cost = df[df["score_raw"] >= 4]["txn_cost"].mean()
    new_cost = df[df["score_filtered"] >= 4]["txn_cost"].mean()

    nyr_old = len(old4) / years
    nyr_new = len(new4) / years

    annual_old = nyr_old * ((a_old - old_cost) / 100) * (POSITION_SIZE[4] * PORTFOLIO)
    annual_new = nyr_new * ((a_new - new_cost) / 100) * (POSITION_SIZE[4] * PORTFOLIO)

    # Downgraded events as Score 3
    if len(drop) >= 2:
        a_drop, _, _ = alpha_vs_base(drop, base)
        drop_cost = df[(df["score_raw"] == 4) & (df["score_filtered"] == 3)]["txn_cost"].mean()
        nyr_drop = len(drop) / years
        annual_drop_as3 = nyr_drop * ((a_drop - drop_cost) / 100) * (POSITION_SIZE[3] * PORTFOLIO)
    else:
        annual_drop_as3 = 0

    total_old = annual_old
    total_new = annual_new + annual_drop_as3

    print(f"""
  OLD RULE (Score ≥ 4 = all):
    n={len(old4)} events ({nyr_old:.1f}/yr), alpha {a_old:+.2f}%, win rate {win_old:.1f}%
    Annual net: ${annual_old:+.0f}/yr from Score 4 trades

  NEW RULE (Score ≥ 4 = filtered):
    n={len(new4)} events ({nyr_new:.1f}/yr), alpha {a_new:+.2f}%, win rate {win_new:.1f}%
    Annual net from Score 4: ${annual_new:+.0f}/yr
    Downgraded {len(drop)} events now trade as Score 3: ${annual_drop_as3:+.0f}/yr
    Combined: ${total_new:+.0f}/yr

  CHANGE: ${total_new - total_old:+.0f}/yr vs old rule

  RECOMMENDATION:""")

    alpha_improved = a_new > a_old
    win_improved   = win_new > win_old
    net_improved   = total_new > total_old

    if alpha_improved and win_improved:
        print(f"""
  ✅ IMPLEMENT THE FILTER
     Alpha improved: {a_old:+.2f}% → {a_new:+.2f}%
     Win rate improved: {win_old:.1f}% → {win_new:.1f}%
     The 'severe crash + greed zone only' events are genuinely weaker signals.
     Downgrading them to Score 3 (smaller position) is justified by the data.""")
    elif alpha_improved and not win_improved:
        print(f"""
  ⚠️  CONSIDER IMPLEMENTING
     Alpha improved but win rate did not.
     Review year-by-year results above before deciding.""")
    else:
        print(f"""
  ❌ DO NOT IMPLEMENT
     Alpha or win rate did not improve with the filter.
     The current rule is already optimal.""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score 4 filter backtest")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--recent", type=int, default=None,
                        help="Restrict backtest to last N days")
    args = parser.parse_args()

    if args.output:
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

    print(f"{'='*72}")
    print(f"  SCORE 4 FILTER BACKTEST")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Filter: Score 4 requires severe crash + (capitulation OR extreme greed)")
    print(f"  Without this: downgrade to Score 3")
    print(f"{'='*72}")

    df = load_data(since_days=args.recent)
    df = compute_scores(df)

    n_total = len(df)
    date_range = f"{df['date'].min().date()} → {df['date'].max().date()}"
    print(f"\nData: {n_total:,} rows | {date_range}")
    print(f"Score ≥ 4 (raw):      {(df['score_raw'] >= 4).sum():,} events")
    print(f"Score ≥ 4 (filtered): {(df['score_filtered'] >= 4).sum():,} events")
    print(f"Downgraded to Score 3: {((df['score_raw'] == 4) & (df['score_filtered'] == 3)).sum():,} events")

    print(f"\n{'='*72}")
    print(f"OVERALL COMPARISON")
    print(f"{'='*72}")
    compare_rules(df)

    year_by_year(df)
    what_would_change(df, days_back=180)
    summary_verdict(df)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.output:
        sys.stdout = old_stdout
        text = buf.getvalue()
        with open(args.output, "w") as f:
            f.write(text)
        print(text)
        print(f"\n[Saved to {args.output}]")


if __name__ == "__main__":
    main()
