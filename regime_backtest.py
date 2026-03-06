#!/usr/bin/env python3
"""
regime_backtest.py
==================
Phase 2 research: Does 20-day historical volatility (HV) at signal time
predict whether a 1-day or 3-day hold captures more alpha?

Goal: Determine if the autotrader should switch hold windows dynamically
based on the coin's current volatility regime, instead of using a fixed
HOLD_DAYS = 3.

Logic under test:
  LOW HV regime  → market trending smoothly → 1-day hold may capture alpha faster
  HIGH HV regime → choppy / bear market    → 3-day hold needed to let trade develop

Tests:
  1. 1d vs 3d alpha split by coin-level HV percentile (low / mid / high tercile)
  2. 1d vs 3d alpha split by absolute HV threshold (below / above median)
  3. Year-by-year: which hold window won each year?
  4. Score tier interaction: does HV effect vary by score 2 vs 3 vs 4?
  5. Optimal HV threshold search (where does the crossover happen?)

Usage:
    python3 regime_backtest.py
    python3 regime_backtest.py --output regime_results.txt
"""

import argparse
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from database import DB_PATH

WINSOR_LOW  = 1
WINSOR_HIGH = 99

COST_BLUE_CHIP = 0.20
COST_TOP20     = 0.40
COST_ALTCOIN   = 0.80


# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_prepare(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    prices = pd.read_sql_query("""
        SELECT coingecko_id, ticker, bucket, date, price_usd, volume_usd
        FROM daily_prices ORDER BY coingecko_id, date
    """, conn)
    fg = pd.read_sql_query("""
        SELECT date, value AS fg_value FROM fear_greed ORDER BY date
    """, conn)
    conn.close()

    prices["date"] = pd.to_datetime(prices["date"])
    fg["date"]     = pd.to_datetime(fg["date"])

    all_coins = []
    for coin_id, group in prices.groupby("coingecko_id"):
        group = group.sort_values("date").copy()

        # Forward returns: 1d, 3d, 5d, 7d, 10d
        for days in [1, 3, 5, 7, 10]:
            group[f"fwd_{days}d"] = (
                group["price_usd"].shift(-days) / group["price_usd"] - 1
            ) * 100

        # Lookback returns
        group["ret_1d"]  = group["price_usd"].pct_change() * 100
        group["ret_5d"]  = (group["price_usd"] / group["price_usd"].shift(5)  - 1) * 100
        group["ret_20d"] = (group["price_usd"] / group["price_usd"].shift(20) - 1) * 100

        # 20-day historical volatility (annualized)
        group["hv_20d"] = group["ret_1d"].rolling(20).std() * np.sqrt(365)

        # HV percentile rank within this coin's own history (0–100)
        group["hv_pctile"] = group["hv_20d"].rank(pct=True) * 100

        # Volume ratio
        group["vol_ratio"] = (
            group["volume_usd"] / group["volume_usd"].rolling(20).mean()
        )

        # Vol percentile (used for scoring)
        group["vol_pctile"] = group["hv_20d"].rank(pct=True) * 100

        all_coins.append(group)

    df = pd.concat(all_coins, ignore_index=True)

    # Winsorize forward returns
    for days in [1, 3, 5, 7, 10]:
        df[f"fwd_{days}d"] = _winsorize(df[f"fwd_{days}d"])
    for col in ["ret_1d", "ret_5d", "ret_20d"]:
        df[col] = _winsorize(df[col])

    # Merge Fear & Greed
    df = df.merge(fg, on="date", how="left")

    # Transaction cost column
    df["txn_cost"] = COST_ALTCOIN
    df.loc[df["bucket"] == 1, "txn_cost"] = COST_BLUE_CHIP
    df.loc[df["bucket"] == 2, "txn_cost"] = COST_TOP20

    return df


def _winsorize(s, low=WINSOR_LOW, high=WINSOR_HIGH):
    lo = np.nanpercentile(s, low)
    hi = np.nanpercentile(s, high)
    return s.clip(lower=lo, upper=hi)


def compute_scores(df):
    """Replicate the scoring system from signal_refine.py."""
    df = df.copy()
    df["score"] = 0.0

    df.loc[df["ret_5d"] <= -25, "score"] += 3
    df.loc[(df["ret_5d"] <= -15) & (df["ret_5d"] > -25), "score"] += 1
    df.loc[df["fg_value"] >= 90,  "score"] += 2
    df.loc[(df["fg_value"] >= 75) & (df["fg_value"] < 90), "score"] += 1
    df.loc[(df["vol_ratio"] >= 3.0) & (df["ret_1d"] <= -5), "score"] += 1

    df_s = df.sort_values(["coingecko_id", "date"])
    fg_prev = df_s.groupby("coingecko_id")["fg_value"].shift(1)
    fear_exit = (fg_prev <= 25) & (df_s["fg_value"] > 25)
    df.loc[fear_exit.values, "score"] -= 2
    df.loc[df["vol_pctile"] <= 10, "score"] -= 1

    return df


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_alpha(alpha, p_val):
    star = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    return f"{alpha:>+7.2f}%{star}"


def alpha_vs_base(sig_rets, base_rets):
    """Return (alpha, t_stat, p_val). Returns (nan,nan,nan) if insufficient data."""
    if len(sig_rets) < 10 or len(base_rets) < 10:
        return float("nan"), float("nan"), float("nan")
    alpha = sig_rets.mean() - base_rets.mean()
    t, p  = stats.ttest_ind(sig_rets, base_rets, equal_var=False)
    return alpha, t, p


# ── Test 1: HV tercile breakdown ──────────────────────────────────────────────

def test_hv_tercile(df, sig):
    print(f"\n{'='*72}")
    print(f"TEST 1: 1d vs 3d ALPHA BY HV REGIME (coin-relative HV percentile)")
    print(f"{'='*72}")
    print(f"  Coin's 20d HV split into LOW / MID / HIGH tercile of its own history.")
    print(f"  Each coin defines its own threshold — avoids cross-coin scale differences.")
    print()

    tercile_defs = [
        ("LOW  HV (bottom 33%)",   0,  33),
        ("MID  HV (33–67%)",      33,  67),
        ("HIGH HV (top 33%)",     67, 100),
    ]

    base_1d = df["fwd_1d"].dropna()
    base_3d = df["fwd_3d"].dropna()

    print(f"  {'Regime':<26} {'n':>6}  {'1d alpha':>9}  {'3d alpha':>9}  {'Winner':>8}")
    print(f"  {'-'*68}")

    for label, lo, hi in tercile_defs:
        mask     = (sig["hv_pctile"] >= lo) & (sig["hv_pctile"] < hi)
        sub      = sig[mask]
        if len(sub) < 10:
            print(f"  {label:<26} {len(sub):>6}  — insufficient data")
            continue

        a1, t1, p1 = alpha_vs_base(sub["fwd_1d"].dropna(), base_1d)
        a3, t3, p3 = alpha_vs_base(sub["fwd_3d"].dropna(), base_3d)

        winner = "3d ✓" if (not np.isnan(a3) and not np.isnan(a1) and a3 > a1) else "1d ✓"
        if np.isnan(a1) or np.isnan(a3):
            winner = "—"

        s1 = fmt_alpha(a1, p1) if not np.isnan(a1) else "   N/A  "
        s3 = fmt_alpha(a3, p3) if not np.isnan(a3) else "   N/A  "
        print(f"  {label:<26} {len(sub):>6}  {s1}  {s3}  {winner:>8}")

    print(f"\n  ** p<0.01   * p<0.05   Alpha = signal mean − baseline mean (winsorized)")


# ── Test 2: Absolute HV threshold (below/above median) ───────────────────────

def test_hv_median_split(df, sig):
    print(f"\n{'='*72}")
    print(f"TEST 2: 1d vs 3d ALPHA — BELOW vs ABOVE COIN'S MEDIAN HV")
    print(f"{'='*72}")
    print(f"  Coin's 20d HV compared to its own median (50th pctile).")
    print()

    base_1d = df["fwd_1d"].dropna()
    base_3d = df["fwd_3d"].dropna()

    print(f"  {'Regime':<28} {'n':>6}  {'1d alpha':>9}  {'3d alpha':>9}  "
          f"{'1d net':>8}  {'3d net':>8}  {'Winner':>8}")
    print(f"  {'-'*82}")

    for label, lo, hi in [("BELOW median HV (calm)", 0, 50),
                           ("ABOVE median HV (volatile)", 50, 100)]:
        mask = (sig["hv_pctile"] >= lo) & (sig["hv_pctile"] < hi)
        sub  = sig[mask]
        if len(sub) < 10:
            continue

        a1, t1, p1 = alpha_vs_base(sub["fwd_1d"].dropna(), base_1d)
        a3, t3, p3 = alpha_vs_base(sub["fwd_3d"].dropna(), base_3d)
        cost       = sub["txn_cost"].mean()

        net1 = a1 - cost if not np.isnan(a1) else float("nan")
        net3 = a3 - cost if not np.isnan(a3) else float("nan")
        winner = "3d ✓" if (not np.isnan(net3) and not np.isnan(net1)
                            and net3 > net1) else "1d ✓"

        s1 = fmt_alpha(a1, p1) if not np.isnan(a1) else "   N/A  "
        s3 = fmt_alpha(a3, p3) if not np.isnan(a3) else "   N/A  "
        print(f"  {label:<28} {len(sub):>6}  {s1}  {s3}  "
              f"{net1:>+7.2f}%  {net3:>+7.2f}%  {winner:>8}")

    print(f"\n  Net = alpha − avg transaction cost ({COST_ALTCOIN:.2f}% altcoin / "
          f"{COST_TOP20:.2f}% mid / {COST_BLUE_CHIP:.2f}% large)")


# ── Test 3: Year-by-year hold window winner ───────────────────────────────────

def test_yearly_winner(df, sig):
    print(f"\n{'='*72}")
    print(f"TEST 3: YEAR-BY-YEAR — WHICH HOLD WINDOW WON?")
    print(f"{'='*72}")
    print(f"  Shows whether 1d or 3d captured more alpha in each calendar year.")
    print()

    base = df.copy()
    print(f"  {'Year':<6}  {'n':>6}  {'1d alpha':>9}  {'3d alpha':>9}  "
          f"{'5d alpha':>9}  {'Winner':>8}  {'BTC 20d avg':>12}")
    print(f"  {'-'*72}")

    # Get BTC average 20d return per year as regime proxy
    btc = df[df["ticker"] == "BTC"][["date", "ret_20d"]].copy()
    btc["year"] = btc["date"].dt.year

    for year in sorted(sig["date"].dt.year.unique()):
        mask  = sig["date"].dt.year == year
        sub   = sig[mask]
        base_y = base[base["date"].dt.year == year]
        if len(sub) < 10:
            continue

        alphas = {}
        pvals  = {}
        for days in [1, 3, 5]:
            col    = f"fwd_{days}d"
            a, t, p = alpha_vs_base(sub[col].dropna(), base_y[col].dropna())
            alphas[days] = a
            pvals[days]  = p

        valid = {d: a for d, a in alphas.items() if not np.isnan(a)}
        if not valid:
            continue
        best = max(valid, key=valid.get)
        winner = f"{best}d ✓"

        btc_yr   = btc[btc["year"] == year]["ret_20d"].mean()
        btc_str  = f"{btc_yr:>+8.1f}%" if not np.isnan(btc_yr) else "   N/A"

        s1 = fmt_alpha(alphas[1], pvals[1]) if not np.isnan(alphas.get(1, float("nan"))) else "   N/A  "
        s3 = fmt_alpha(alphas[3], pvals[3]) if not np.isnan(alphas.get(3, float("nan"))) else "   N/A  "
        s5 = fmt_alpha(alphas[5], pvals[5]) if not np.isnan(alphas.get(5, float("nan"))) else "   N/A  "

        print(f"  {year:<6}  {len(sub):>6}  {s1}  {s3}  {s5}  {winner:>8}  {btc_str:>12}")

    print(f"\n  BTC 20d avg = average of BTC's trailing 20-day return across the year")
    print(f"  (positive = bull trend, negative = bear trend)")


# ── Test 4: Score tier interaction ────────────────────────────────────────────

def test_score_tier_hv(df, sig):
    print(f"\n{'='*72}")
    print(f"TEST 4: SCORE TIER × HV REGIME INTERACTION")
    print(f"{'='*72}")
    print(f"  Does HV regime effect differ by signal strength?")
    print()

    base_1d = df["fwd_1d"].dropna()
    base_3d = df["fwd_3d"].dropna()

    for score_min in [2, 3, 4]:
        tier = sig[sig["score"] >= score_min]
        if len(tier) < 20:
            continue
        print(f"  === Score ≥ {score_min} (n={len(tier):,}) ===")
        print(f"  {'Regime':<26} {'n':>6}  {'1d alpha':>9}  {'3d alpha':>9}  {'Winner':>8}")
        print(f"  {'-'*62}")

        for label, lo, hi in [("LOW  HV (<33rd pctile)",  0, 33),
                               ("HIGH HV (>67th pctile)", 67, 100)]:
            mask = (tier["hv_pctile"] >= lo) & (tier["hv_pctile"] < hi)
            sub  = tier[mask]
            if len(sub) < 10:
                print(f"  {label:<26} {len(sub):>6}  — insufficient")
                continue

            a1, t1, p1 = alpha_vs_base(sub["fwd_1d"].dropna(), base_1d)
            a3, t3, p3 = alpha_vs_base(sub["fwd_3d"].dropna(), base_3d)
            winner = "3d ✓" if (not np.isnan(a3) and not np.isnan(a1)
                                and a3 > a1) else "1d ✓"
            s1 = fmt_alpha(a1, p1) if not np.isnan(a1) else "   N/A  "
            s3 = fmt_alpha(a3, p3) if not np.isnan(a3) else "   N/A  "
            print(f"  {label:<26} {len(sub):>6}  {s1}  {s3}  {winner:>8}")
        print()


# ── Test 5: HV threshold sweep ────────────────────────────────────────────────

def test_hv_threshold_sweep(df, sig):
    print(f"\n{'='*72}")
    print(f"TEST 5: HV THRESHOLD SWEEP")
    print(f"{'='*72}")
    print(f"  Where does the 1d→3d crossover happen?")
    print(f"  Sweep HV percentile cutoff from 20th to 80th in steps of 10.")
    print(f"  'Low HV' = coin's HV in bottom X% of its own history.")
    print()

    base_1d = df["fwd_1d"].dropna()
    base_3d = df["fwd_3d"].dropna()

    print(f"  {'Cutoff':<12}  {'Low HV n':>9}  {'1d alpha':>9}  {'3d alpha':>9}  "
          f"{'High HV n':>10}  {'1d alpha':>9}  {'3d alpha':>9}")
    print(f"  {'-'*84}")

    for cutoff in range(20, 85, 10):
        lo_mask = sig["hv_pctile"] < cutoff
        hi_mask = sig["hv_pctile"] >= cutoff

        lo_sub  = sig[lo_mask]
        hi_sub  = sig[hi_mask]

        if len(lo_sub) < 10 or len(hi_sub) < 10:
            continue

        a1_lo, _, p1_lo = alpha_vs_base(lo_sub["fwd_1d"].dropna(), base_1d)
        a3_lo, _, p3_lo = alpha_vs_base(lo_sub["fwd_3d"].dropna(), base_3d)
        a1_hi, _, p1_hi = alpha_vs_base(hi_sub["fwd_1d"].dropna(), base_1d)
        a3_hi, _, p3_hi = alpha_vs_base(hi_sub["fwd_3d"].dropna(), base_3d)

        lo_winner = "<1d" if (not np.isnan(a1_lo) and not np.isnan(a3_lo) and a1_lo > a3_lo) else "3d "
        hi_winner = "<1d" if (not np.isnan(a1_hi) and not np.isnan(a3_hi) and a1_hi > a3_hi) else "3d "

        s1l = fmt_alpha(a1_lo, p1_lo) if not np.isnan(a1_lo) else "   N/A  "
        s3l = fmt_alpha(a3_lo, p3_lo) if not np.isnan(a3_lo) else "   N/A  "
        s1h = fmt_alpha(a1_hi, p1_hi) if not np.isnan(a1_hi) else "   N/A  "
        s3h = fmt_alpha(a3_hi, p3_hi) if not np.isnan(a3_hi) else "   N/A  "

        print(f"  <{cutoff:>2}th pctile   {len(lo_sub):>9,}  {s1l}  {s3l}  "
              f"{len(hi_sub):>10,}  {s1h}  {s3h}")

    print(f"\n  Look for the row where low-HV switches from '3d>' to '1d>'.")
    print(f"  That cutoff is the optimal dynamic switch threshold.")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_verdict(df, sig):
    print(f"\n{'='*72}")
    print(f"VERDICT & IMPLEMENTATION RECOMMENDATION")
    print(f"{'='*72}")

    base_1d = df["fwd_1d"].dropna()
    base_3d = df["fwd_3d"].dropna()

    # Overall 1d vs 3d
    a1, t1, p1 = alpha_vs_base(sig["fwd_1d"].dropna(), base_1d)
    a3, t3, p3 = alpha_vs_base(sig["fwd_3d"].dropna(), base_3d)

    print(f"\n  Overall (Score ≥ 2, all regimes):")
    print(f"    1d alpha: {fmt_alpha(a1, p1)}  (t={t1:.2f}, p={p1:.4f})")
    print(f"    3d alpha: {fmt_alpha(a3, p3)}  (t={t3:.2f}, p={p3:.4f})")

    # Low HV
    lo  = sig[sig["hv_pctile"] < 33]
    a1l, _, p1l = alpha_vs_base(lo["fwd_1d"].dropna(), base_1d)
    a3l, _, p3l = alpha_vs_base(lo["fwd_3d"].dropna(), base_3d)

    # High HV
    hi  = sig[sig["hv_pctile"] >= 67]
    a1h, _, p1h = alpha_vs_base(hi["fwd_1d"].dropna(), base_1d)
    a3h, _, p3h = alpha_vs_base(hi["fwd_3d"].dropna(), base_3d)

    print(f"\n  Low HV regime (bottom 33% of coin's own HV history):")
    print(f"    1d alpha: {fmt_alpha(a1l, p1l)}")
    print(f"    3d alpha: {fmt_alpha(a3l, p3l)}")

    print(f"\n  High HV regime (top 33% of coin's own HV history):")
    print(f"    1d alpha: {fmt_alpha(a1h, p1h)}")
    print(f"    3d alpha: {fmt_alpha(a3h, p3h)}")

    print(f"\n  RECOMMENDATION:")
    if (not np.isnan(a1l) and not np.isnan(a3l) and
            not np.isnan(a1h) and not np.isnan(a3h)):
        low_winner  = "1d" if a1l > a3l else "3d"
        high_winner = "1d" if a1h > a3h else "3d"
        if low_winner != high_winner:
            print(f"    ✅ HV regime is predictive: use {low_winner} hold in LOW HV, "
                  f"{high_winner} hold in HIGH HV")
            print(f"    → Dynamic switching JUSTIFIED by data")
        else:
            print(f"    ⚠️  HV regime does NOT flip hold window winner ({low_winner} wins in both)")
            print(f"    → Dynamic switching NOT justified; stick with fixed {low_winner} hold")
    else:
        print(f"    → Insufficient data to determine recommendation")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Regime detection backtest")
    parser.add_argument("--output", type=str, default=None,
                        help="Save report to file instead of printing to terminal")
    args = parser.parse_args()

    if args.output:
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

    print(f"{'='*72}")
    print(f"  REGIME DETECTION BACKTEST")
    print(f"  Question: Does 20-day HV predict 1d vs 3d hold window?")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*72}")

    print(f"\nLoading data...")
    df = load_and_prepare()
    df = compute_scores(df)

    sig = df[df["score"] >= 2].copy()
    print(f"Signal events (score ≥ 2): {len(sig):,}")
    print(f"Date range: {sig['date'].min().date()} to {sig['date'].max().date()}")
    print(f"Unique coins: {sig['ticker'].nunique()}")

    test_hv_tercile(df, sig)
    test_hv_median_split(df, sig)
    test_yearly_winner(df, sig)
    test_score_tier_hv(df, sig)
    test_hv_threshold_sweep(df, sig)
    print_verdict(df, sig)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.output:
        sys.stdout = old_stdout
        output_text = buf.getvalue()
        with open(args.output, "w") as f:
            f.write(output_text)
        print(output_text)
        print(f"\n[Report saved to {args.output}]")


if __name__ == "__main__":
    main()
