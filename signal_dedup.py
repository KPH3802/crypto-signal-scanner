"""
Crypto Signal Deduplication & Combined Backtest
Phase 3: Which signals are independent? What's the combined edge?

Part 1: Signal Overlap Analysis
  - Correlation matrix of signal triggers (do they fire together?)
  - Conditional alpha (does signal X add value after controlling for Y?)
  - Jaccard similarity between signal pairs

Part 2: Combined Backtest
  - Multi-signal scoring system
  - Position sizing based on signal strength
  - Transaction cost modeling (spreads + slippage)
  - Time-period stability (bull vs bear vs sideways)
  - Equity curve and drawdown analysis

Usage:
    python3 signal_dedup.py              # Full analysis
    python3 signal_dedup.py --overlap    # Only overlap analysis
    python3 signal_dedup.py --combined   # Only combined backtest
"""

import argparse
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from database import DB_PATH

# Winsorization bounds
WINSOR_LOW = 1
WINSOR_HIGH = 99

# Transaction cost assumptions (round trip)
COST_BLUE_CHIP = 0.20   # 0.20% for BTC/ETH (tight spreads)
COST_TOP20 = 0.40       # 0.40% for top 20
COST_ALTCOIN = 0.80     # 0.80% for altcoins (wider spreads, slippage)


# ============================================================
# DATA LOADING (same as backtest_engine.py)
# ============================================================

def winsorize_series(s, low_pct=WINSOR_LOW, high_pct=WINSOR_HIGH):
    low_val = np.nanpercentile(s, low_pct)
    high_val = np.nanpercentile(s, high_pct)
    return s.clip(lower=low_val, upper=high_val)


def load_and_prepare(db_path=DB_PATH):
    """Load all data, compute indicators, winsorize, merge Fear & Greed."""
    conn = sqlite3.connect(db_path)
    prices = pd.read_sql_query("""
        SELECT coingecko_id, ticker, bucket, date, price_usd, volume_usd
        FROM daily_prices ORDER BY coingecko_id, date
    """, conn)
    fg = pd.read_sql_query("""
        SELECT date, value, classification FROM fear_greed ORDER BY date
    """, conn)
    conn.close()

    prices["date"] = pd.to_datetime(prices["date"])
    fg["date"] = pd.to_datetime(fg["date"])

    print(f"Loaded {len(prices):,} price records, {len(fg):,} Fear & Greed days")

    all_coins = []
    for coin_id, group in prices.groupby("coingecko_id"):
        group = group.sort_values("date").copy()

        # Forward returns
        for days in [3, 5, 10, 20]:
            group[f"fwd_{days}d"] = (
                group["price_usd"].shift(-days) / group["price_usd"] - 1
            ) * 100

        # Lookback indicators
        group["ret_1d"] = group["price_usd"].pct_change() * 100
        group["ret_5d"] = (group["price_usd"] / group["price_usd"].shift(5) - 1) * 100
        group["ret_10d"] = (group["price_usd"] / group["price_usd"].shift(10) - 1) * 100
        group["ret_20d"] = (group["price_usd"] / group["price_usd"].shift(20) - 1) * 100
        group["vol_20d"] = group["ret_1d"].rolling(20).std() * np.sqrt(365)
        group["vol_ratio"] = group["volume_usd"] / group["volume_usd"].rolling(20).mean()
        group["vol_pctile"] = group["vol_20d"].rank(pct=True) * 100

        all_coins.append(group)

    df = pd.concat(all_coins, ignore_index=True)

    # Winsorize
    for days in [3, 5, 10, 20]:
        df[f"fwd_{days}d"] = winsorize_series(df[f"fwd_{days}d"])
    for col in ["ret_1d", "ret_5d", "ret_10d", "ret_20d"]:
        df[col] = winsorize_series(df[col])

    # Merge Fear & Greed
    df = df.merge(fg[["date", "value", "classification"]],
                  on="date", how="left")
    df.rename(columns={"value": "fg_value", "classification": "fg_class"}, inplace=True)

    print(f"Prepared: {len(df):,} rows")
    return df


# ============================================================
# PART 1: SIGNAL DEFINITIONS & OVERLAP
# ============================================================

def define_signals(df):
    """
    Define all signals as boolean columns.
    Returns dict of {signal_name: column_name} and modifies df in place.
    """
    # Need percentile thresholds
    ret10_p10 = df["ret_10d"].quantile(0.10)
    ret10_p90 = df["ret_10d"].quantile(0.90)
    ret20_p10 = df["ret_20d"].quantile(0.10)
    ret20_p90 = df["ret_20d"].quantile(0.90)

    signals = {}

    # Fear & Greed
    df["sig_extreme_fear"] = df["fg_value"] <= 10
    signals["Extreme Fear (FG≤10)"] = "sig_extreme_fear"

    df["sig_fear_zone"] = df["fg_value"] <= 25
    signals["Fear Zone (FG≤25)"] = "sig_fear_zone"

    df["sig_extreme_greed"] = df["fg_value"] >= 90
    signals["Extreme Greed (FG≥90)"] = "sig_extreme_greed"

    df["sig_greed_zone"] = df["fg_value"] >= 75
    signals["Greed Zone (FG≥75)"] = "sig_greed_zone"

    # Fear exit
    df_sorted = df.sort_values(["coingecko_id", "date"])
    fg_prev = df_sorted.groupby("coingecko_id")["fg_value"].shift(1)
    df["sig_fear_exit"] = ((fg_prev <= 25) & (df_sorted["fg_value"] > 25)).values
    signals["Fear Exit"] = "sig_fear_exit"

    # Momentum
    df["sig_strong_mom_10d"] = df["ret_10d"] >= ret10_p90
    signals["Strong Mom 10d"] = "sig_strong_mom_10d"

    df["sig_strong_mom_20d"] = df["ret_20d"] >= ret20_p90
    signals["Strong Mom 20d"] = "sig_strong_mom_20d"

    df["sig_weak_mom_10d"] = df["ret_10d"] <= ret10_p10
    signals["Weak Mom 10d"] = "sig_weak_mom_10d"

    df["sig_weak_mom_20d"] = df["ret_20d"] <= ret20_p10
    signals["Weak Mom 20d"] = "sig_weak_mom_20d"

    # Mean reversion
    df["sig_dd_15"] = df["ret_5d"] <= -15
    signals["5d DD ≥15%"] = "sig_dd_15"

    df["sig_dd_25"] = df["ret_5d"] <= -25
    signals["5d DD ≥25%"] = "sig_dd_25"

    df["sig_surge_20"] = df["ret_5d"] >= 20
    signals["5d Surge ≥20%"] = "sig_surge_20"

    df["sig_surge_30"] = df["ret_5d"] >= 30
    signals["5d Surge ≥30%"] = "sig_surge_30"

    df["sig_crash_1d"] = df["ret_1d"] <= -10
    signals["1d Crash ≥10%"] = "sig_crash_1d"

    df["sig_pump_1d"] = df["ret_1d"] >= 15
    signals["1d Pump ≥15%"] = "sig_pump_1d"

    # Volatility
    df["sig_low_vol_10"] = df["vol_pctile"] <= 10
    signals["Low Vol p10"] = "sig_low_vol_10"

    df["sig_high_vol_90"] = df["vol_pctile"] >= 90
    signals["High Vol p90"] = "sig_high_vol_90"

    # Volume
    df["sig_vol_3x"] = df["vol_ratio"] >= 3.0
    signals["Volume 3x"] = "sig_vol_3x"

    df["sig_vol_5x"] = df["vol_ratio"] >= 5.0
    signals["Volume 5x"] = "sig_vol_5x"

    df["sig_capitulation"] = (df["vol_ratio"] >= 3.0) & (df["ret_1d"] <= -5)
    signals["Capitulation"] = "sig_capitulation"

    df["sig_breakout"] = (df["vol_ratio"] >= 3.0) & (df["ret_1d"] >= 5)
    signals["Breakout"] = "sig_breakout"

    return signals


def overlap_analysis(df, signals):
    """Compute pairwise overlap between signals."""
    print(f"\n{'#'*70}")
    print(f"# PART 1: SIGNAL OVERLAP ANALYSIS")
    print(f"{'#'*70}")

    # Focus on the top signals from v2 results
    top_signals = [
        "5d DD ≥25%", "5d DD ≥15%", "1d Crash ≥10%",
        "Extreme Greed (FG≥90)", "Greed Zone (FG≥75)",
        "Weak Mom 10d", "Weak Mom 20d",
        "High Vol p90", "Capitulation",
        "Strong Mom 10d", "Strong Mom 20d",
        "Volume 3x", "Breakout",
    ]

    # Only keep signals that exist
    top_signals = [s for s in top_signals if s in signals]

    sig_cols = [signals[s] for s in top_signals]

    # -- Jaccard Similarity Matrix --
    print(f"\n  === JACCARD SIMILARITY (overlap / union) ===")
    print(f"  Higher = more overlap, signals fire together")
    print(f"  Pairs > 0.30 are likely redundant\n")

    # Build matrix
    n = len(top_signals)
    jaccard = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a = df[sig_cols[i]].fillna(False)
            b = df[sig_cols[j]].fillna(False)
            intersection = (a & b).sum()
            union = (a | b).sum()
            jaccard[i, j] = intersection / union if union > 0 else 0

    # Print abbreviated names
    short_names = [s[:18] for s in top_signals]

    # Print header
    print(f"  {'':>20}", end="")
    for i, name in enumerate(short_names):
        print(f" {i+1:>5}", end="")
    print()

    for i, name in enumerate(short_names):
        print(f"  {i+1:>2}. {name:<16}", end="")
        for j in range(n):
            if i == j:
                print(f"  ---", end="")
            elif jaccard[i, j] >= 0.30:
                print(f" {jaccard[i,j]:>4.2f}*", end="")
            elif jaccard[i, j] >= 0.10:
                print(f" {jaccard[i,j]:>5.2f}", end="")
            else:
                print(f" {jaccard[i,j]:>5.2f}", end="")
        print()

    # -- High overlap pairs --
    print(f"\n  === HIGH OVERLAP PAIRS (Jaccard > 0.15) ===\n")
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if jaccard[i, j] > 0.15:
                pairs.append((top_signals[i], top_signals[j], jaccard[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)
    for a, b, j in pairs:
        print(f"    {j:.2f}  {a}  <->  {b}")

    # -- Conditional Alpha Test --
    print(f"\n\n  === CONDITIONAL ALPHA: Does Signal A add value GIVEN Signal B? ===")
    print(f"  Tests whether Signal A's alpha persists when Signal B is also active\n")

    # Test the crash/drawdown family
    crash_family = [
        ("5d DD ≥25%", "5d DD ≥15%"),
        ("5d DD ≥25%", "Weak Mom 10d"),
        ("5d DD ≥25%", "High Vol p90"),
        ("1d Crash ≥10%", "5d DD ≥15%"),
        ("1d Crash ≥10%", "Weak Mom 10d"),
        ("1d Crash ≥10%", "High Vol p90"),
        ("Capitulation", "1d Crash ≥10%"),
        ("Capitulation", "5d DD ≥15%"),
    ]

    # Test the greed/momentum family
    greed_family = [
        ("Extreme Greed (FG≥90)", "Greed Zone (FG≥75)"),
        ("Extreme Greed (FG≥90)", "Strong Mom 10d"),
        ("Extreme Greed (FG≥90)", "Strong Mom 20d"),
        ("Greed Zone (FG≥75)", "Strong Mom 10d"),
        ("Strong Mom 10d", "Strong Mom 20d"),
        ("Breakout", "Volume 3x"),
        ("Breakout", "Strong Mom 10d"),
    ]

    all_pairs = crash_family + greed_family

    print(f"  {'Signal A':<25} {'given B':<25} {'A only':>8} {'A+B':>8} "
          f"{'B only':>8} {'Neither':>8} {'A adds?':>8}")
    print(f"  {'-'*105}")

    for sig_a_name, sig_b_name in all_pairs:
        if sig_a_name not in signals or sig_b_name not in signals:
            continue

        col_a = signals[sig_a_name]
        col_b = signals[sig_b_name]

        a = df[col_a].fillna(False)
        b = df[col_b].fillna(False)
        fwd = df["fwd_5d"]

        # Four groups
        both = a & b
        a_only = a & ~b
        b_only = ~a & b
        neither = ~a & ~b

        both_ret = fwd[both].dropna()
        a_only_ret = fwd[a_only].dropna()
        b_only_ret = fwd[b_only].dropna()
        neither_ret = fwd[neither].dropna()

        # Does A add value on top of B?
        # Compare (A+B) vs (B only)
        if len(both_ret) >= 10 and len(b_only_ret) >= 10:
            t_add, p_add = stats.ttest_ind(both_ret, b_only_ret, equal_var=False)
            adds = "YES" if p_add < 0.05 and both_ret.mean() > b_only_ret.mean() else "no"
        else:
            adds = "n/a"

        a_only_str = f"{a_only_ret.mean():+.2f}%" if len(a_only_ret) >= 5 else "n/a"
        both_str = f"{both_ret.mean():+.2f}%" if len(both_ret) >= 5 else "n/a"
        b_only_str = f"{b_only_ret.mean():+.2f}%" if len(b_only_ret) >= 5 else "n/a"
        neither_str = f"{neither_ret.mean():+.2f}%" if len(neither_ret) >= 5 else "n/a"

        print(f"  {sig_a_name:<25} {sig_b_name:<25} {a_only_str:>8} {both_str:>8} "
              f"{b_only_str:>8} {neither_str:>8} {adds:>8}")

    # -- Identify independent signal families --
    print(f"\n\n  === SIGNAL FAMILY SUMMARY ===")
    print(f"  Grouping highly overlapping signals into families\n")

    print(f"  CRASH FAMILY (mean reversion after selloffs):")
    print(f"    Primary: 5d DD ≥25% (+8.51% alpha, best in class)")
    print(f"    Overlaps: 5d DD ≥15%, 1d Crash, Weak Mom 10d/20d, High Vol, Capitulation")
    print(f"    Recommendation: Use 5d DD ≥25% as representative")

    print(f"\n  TREND FAMILY (momentum/greed continuation):")
    print(f"    Primary: Extreme Greed FG≥90 (+5.56% alpha)")
    print(f"    Overlaps: Greed Zone FG≥75, Strong Mom 10d/20d")
    print(f"    Recommendation: Use Extreme Greed as representative")

    print(f"\n  VOLUME FAMILY (unusual activity):")
    print(f"    Primary: Capitulation (+6.74% alpha)")
    print(f"    Overlaps: Volume 3x, Volume 5x, Breakout")
    print(f"    Recommendation: Capitulation may be subset of crash family")

    return top_signals


# ============================================================
# PART 2: COMBINED BACKTEST
# ============================================================

def combined_backtest(df, signals):
    """
    Build a scoring system from independent signals and simulate trading.
    """
    print(f"\n\n{'#'*70}")
    print(f"# PART 2: COMBINED BACKTEST")
    print(f"{'#'*70}")

    # ---- Define composite signal ----
    # Based on deduplication, use representative signals from each family
    # Crash signal: 5d drawdown severity
    # Greed signal: Fear & Greed level
    # Volume signal: Volume spike + direction

    # Score: +1 for each bullish signal, -1 for bearish
    df["score"] = 0.0

    # CRASH FAMILY: Buy after big drops (strongest signal)
    df.loc[df["ret_5d"] <= -25, "score"] += 3   # Severe drawdown
    df.loc[(df["ret_5d"] <= -15) & (df["ret_5d"] > -25), "score"] += 1  # Moderate drawdown

    # GREED/TREND FAMILY: Stay long during greed (second strongest)
    df.loc[df["fg_value"] >= 90, "score"] += 2  # Extreme greed
    df.loc[(df["fg_value"] >= 75) & (df["fg_value"] < 90), "score"] += 1  # Greed zone

    # VOLUME CONFIRMATION: Extra point for volume spike during crash
    df.loc[(df["vol_ratio"] >= 3.0) & (df["ret_1d"] <= -5), "score"] += 1  # Capitulation

    # ANTI-SIGNALS: Subtract for proven losers
    # Fear exit is worst signal: -2.01% alpha
    df_sorted = df.sort_values(["coingecko_id", "date"])
    fg_prev = df_sorted.groupby("coingecko_id")["fg_value"].shift(1)
    fear_exit = ((fg_prev <= 25) & (df_sorted["fg_value"] > 25))
    df.loc[fear_exit.values, "score"] -= 2  # Strong negative signal

    # Low vol is negative
    df.loc[df["vol_pctile"] <= 10, "score"] -= 1

    # Transaction costs by bucket
    df["txn_cost"] = COST_ALTCOIN  # Default
    df.loc[df["bucket"] == 1, "txn_cost"] = COST_BLUE_CHIP
    df.loc[df["bucket"] == 2, "txn_cost"] = COST_TOP20

    print(f"\n  SCORING SYSTEM:")
    print(f"    +3: 5d drawdown ≥ 25%")
    print(f"    +2: Extreme Greed (FG ≥ 90)")
    print(f"    +1: 5d drawdown 15-25%")
    print(f"    +1: Greed Zone (FG 75-89)")
    print(f"    +1: Capitulation (3x volume + 5%+ down)")
    print(f"    -1: Low volatility (bottom 10%)")
    print(f"    -2: Fear Exit (FG crosses above 25)")
    print(f"")
    print(f"    Transaction costs: BTC/ETH {COST_BLUE_CHIP}%, "
          f"Top 20 {COST_TOP20}%, Altcoins {COST_ALTCOIN}%")

    # ---- Score distribution ----
    print(f"\n  === SCORE DISTRIBUTION ===")
    score_counts = df["score"].value_counts().sort_index()
    total = len(df)
    for score, count in score_counts.items():
        pct = count / total * 100
        bar = "#" * int(pct)
        print(f"    Score {score:>+3.0f}: {count:>7,} ({pct:>5.1f}%) {bar}")

    # ---- Returns by score ----
    print(f"\n  === FORWARD RETURNS BY SCORE (5-day) ===")
    print(f"  {'Score':>6} {'n':>8} {'Mean':>8} {'Median':>8} {'Net':>8} "
          f"{'t vs 0':>8} {'p-val':>7} {'Win%':>6}")
    print(f"  {'-'*70}")

    baseline_5d = df["fwd_5d"].dropna().mean()
    score_results = []

    for score in sorted(df["score"].unique()):
        subset = df[df["score"] == score]
        rets = subset["fwd_5d"].dropna()
        if len(rets) < 10:
            continue

        mean_ret = rets.mean()
        med_ret = rets.median()
        avg_cost = subset["txn_cost"].mean()
        net_ret = mean_ret - avg_cost  # Net of costs (one way, would be 2x for round trip on exit)
        win = (rets > 0).mean() * 100

        # t-test vs baseline
        t_stat, p_val = stats.ttest_ind(rets, df["fwd_5d"].dropna(), equal_var=False)

        star = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        print(f"  {score:>+5.0f}  {len(rets):>8,} {mean_ret:>+7.2f}% {med_ret:>+7.2f}% "
              f"{net_ret:>+7.2f}% {t_stat:>8.2f} {p_val:>7.4f} {win:>5.1f}% {star}")

        score_results.append({
            "score": score,
            "n": len(rets),
            "mean_5d": mean_ret,
            "median_5d": med_ret,
            "net_5d": net_ret,
            "t_stat": t_stat,
            "p_value": p_val,
            "win_rate": win,
        })

    # ---- Returns by score across all windows ----
    print(f"\n  === MULTI-WINDOW VIEW (scores ≥ 2 vs scores ≤ -1) ===")

    high_score = df[df["score"] >= 2]
    low_score = df[df["score"] <= -1]
    mid_score = df[(df["score"] >= 0) & (df["score"] <= 1)]

    for label, subset in [("Score ≥ 2 (BUY)", high_score),
                           ("Score 0-1 (HOLD)", mid_score),
                           ("Score ≤ -1 (AVOID)", low_score)]:
        print(f"\n  {label} (n={len(subset):,})")
        print(f"  {'Window':<8} {'Mean':>8} {'Median':>8} {'Alpha':>8} "
              f"{'t-stat':>8} {'p-val':>7} {'Win%':>6}")
        print(f"  {'-'*55}")

        for days in [3, 5, 10, 20]:
            col = f"fwd_{days}d"
            sig_rets = subset[col].dropna()
            base_rets = df[col].dropna()
            if len(sig_rets) < 10:
                continue

            alpha = sig_rets.mean() - base_rets.mean()
            t_stat, p_val = stats.ttest_ind(sig_rets, base_rets, equal_var=False)
            win = (sig_rets > 0).mean() * 100
            star = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

            print(f"  {days}d{'':<5} {sig_rets.mean():>+7.2f}% {sig_rets.median():>+7.2f}% "
                  f"{alpha:>+7.2f}% {t_stat:>8.2f} {p_val:>7.4f} {win:>5.1f}% {star}")

    # ---- By bucket ----
    print(f"\n  === SCORE ≥ 2 ALPHA BY BUCKET (5-day) ===")
    print(f"  {'Bucket':<14} {'n':>7} {'Signal':>8} {'Baseline':>9} {'Alpha':>8} "
          f"{'Net':>8} {'p-val':>7}")
    print(f"  {'-'*65}")

    for bucket, bname in [(1, "Blue Chips"), (2, "Top 20"), (3, "Altcoins")]:
        sig_sub = high_score[high_score["bucket"] == bucket]["fwd_5d"].dropna()
        base_sub = df[df["bucket"] == bucket]["fwd_5d"].dropna()
        if len(sig_sub) < 10:
            print(f"  {bname:<14} {len(sig_sub):>7} — insufficient data")
            continue

        alpha = sig_sub.mean() - base_sub.mean()
        cost = {1: COST_BLUE_CHIP, 2: COST_TOP20, 3: COST_ALTCOIN}[bucket]
        net = alpha - cost
        t_stat, p_val = stats.ttest_ind(sig_sub, base_sub, equal_var=False)
        star = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        print(f"  {bname:<14} {len(sig_sub):>7,} {sig_sub.mean():>+7.2f}% "
              f"{base_sub.mean():>+8.2f}% {alpha:>+7.2f}% {net:>+7.2f}% "
              f"{p_val:>7.4f} {star}")


    # ---- TIME PERIOD STABILITY ----
    print(f"\n\n  === TIME PERIOD STABILITY (Score ≥ 2, 5-day alpha) ===")
    print(f"  Testing whether edge persists across different market regimes\n")

    # Define periods
    periods = [
        ("2018 Bear", "2018-01-01", "2018-12-31"),
        ("2019 Recovery", "2019-01-01", "2019-12-31"),
        ("2020 COVID+Bull", "2020-01-01", "2020-12-31"),
        ("2021 Bull Run", "2021-01-01", "2021-12-31"),
        ("2022 Crypto Winter", "2022-01-01", "2022-12-31"),
        ("2023 Recovery", "2023-01-01", "2023-12-31"),
        ("2024 Bull", "2024-01-01", "2024-12-31"),
        ("2025-Present", "2025-01-01", "2026-12-31"),
    ]

    print(f"  {'Period':<20} {'Sig n':>7} {'Sig Mean':>9} {'Base Mean':>10} "
          f"{'Alpha':>8} {'p-val':>7}")
    print(f"  {'-'*65}")

    for pname, start, end in periods:
        mask = (df["date"] >= start) & (df["date"] <= end)
        period_df = df[mask]
        sig_sub = period_df[period_df["score"] >= 2]["fwd_5d"].dropna()
        base_sub = period_df["fwd_5d"].dropna()

        if len(sig_sub) < 5:
            print(f"  {pname:<20} {len(sig_sub):>7} — insufficient signal events")
            continue

        alpha = sig_sub.mean() - base_sub.mean()
        if len(sig_sub) >= 10 and len(base_sub) >= 10:
            t_stat, p_val = stats.ttest_ind(sig_sub, base_sub, equal_var=False)
        else:
            p_val = 1.0

        star = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        print(f"  {pname:<20} {len(sig_sub):>7,} {sig_sub.mean():>+8.2f}% "
              f"{base_sub.mean():>+9.2f}% {alpha:>+7.2f}% {p_val:>7.4f} {star}")


    # ---- LONG/SHORT SIMULATION ----
    print(f"\n\n  === LONG/SHORT SIMULATION ===")
    print(f"  Long Score ≥ 2, Short Score ≤ -1, Flat otherwise")
    print(f"  Using 5-day holding period, equal weight per signal event\n")

    # Group by date, compute daily portfolio returns
    daily_results = []

    for date, day_group in df.groupby("date"):
        longs = day_group[day_group["score"] >= 2]["fwd_5d"].dropna()
        shorts = day_group[day_group["score"] <= -1]["fwd_5d"].dropna()

        if len(longs) == 0 and len(shorts) == 0:
            continue

        # Long return (buy signal coins)
        long_ret = longs.mean() if len(longs) > 0 else 0
        long_cost = day_group[day_group["score"] >= 2]["txn_cost"].mean() if len(longs) > 0 else 0

        # Short return (sell signal coins — profit if they go down)
        short_ret = -shorts.mean() if len(shorts) > 0 else 0
        short_cost = day_group[day_group["score"] <= -1]["txn_cost"].mean() if len(shorts) > 0 else 0

        # Combined
        n_positions = len(longs) + len(shorts)
        combined = (long_ret + short_ret) / 2 if (len(longs) > 0 and len(shorts) > 0) else (long_ret or short_ret)
        total_cost = (long_cost + short_cost) / 2

        daily_results.append({
            "date": date,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "combined": combined,
            "net_combined": combined - total_cost,
            "n_longs": len(longs),
            "n_shorts": len(shorts),
            "n_total": n_positions,
        })

    sim = pd.DataFrame(daily_results).sort_values("date")

    # Convert 5-day returns to approximate daily for equity curve
    # (each position is held 5 days, so divide by 5 for daily contribution)
    sim["daily_approx"] = sim["net_combined"] / 5

    # Summary stats
    total_days = len(sim)
    long_only_days = (sim["n_longs"] > 0).sum()
    short_only_days = (sim["n_shorts"] > 0).sum()

    print(f"  Trading days: {total_days:,}")
    print(f"  Days with long positions: {long_only_days:,}")
    print(f"  Days with short positions: {short_only_days:,}")
    print(f"  Avg positions per signal day: {sim['n_total'].mean():.1f}")

    # Average returns
    print(f"\n  5-Day Returns (per signal event):")
    print(f"    Long side:     {sim['long_ret'].mean():+.3f}% avg")
    print(f"    Short side:    {sim['short_ret'].mean():+.3f}% avg")
    print(f"    Combined:      {sim['combined'].mean():+.3f}% avg")
    print(f"    Net of costs:  {sim['net_combined'].mean():+.3f}% avg")

    # Rough annualization (252 trading days / 5 day holding = ~50 turns per year)
    ann_gross = sim["combined"].mean() * 50
    ann_net = sim["net_combined"].mean() * 50
    print(f"\n  Rough Annualized (50 turns/year):")
    print(f"    Gross: {ann_gross:+.1f}%")
    print(f"    Net:   {ann_net:+.1f}%")

    # Win rate
    wins = (sim["net_combined"] > 0).sum()
    print(f"\n  Win rate: {wins/total_days*100:.1f}% of signal days profitable (net)")

    return sim


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Crypto Signal Dedup & Combined Backtest")
    parser.add_argument("--overlap", action="store_true", help="Only run overlap analysis")
    parser.add_argument("--combined", action="store_true", help="Only run combined backtest")
    args = parser.parse_args()

    run_all = not args.overlap and not args.combined

    print(f"{'='*70}")
    print(f"CRYPTO SIGNAL DEDUPLICATION & COMBINED BACKTEST")
    print(f"{'='*70}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = load_and_prepare()
    signals = define_signals(df)

    if run_all or args.overlap:
        overlap_analysis(df, signals)

    if run_all or args.combined:
        combined_backtest(df, signals)

    print(f"\n\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
