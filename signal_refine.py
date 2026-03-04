"""
Crypto Signal Refinement — Phase 3b
Builds on dedup findings to optimize the combined signal for real trading.

Tests:
1. Holding period optimization (3d/5d/10d/20d) by score tier
2. Long-only analysis (short side was weak at -0.50%)
3. Walk-forward out-of-sample validation (train 2014-2021, test 2022-2026)
4. Risk metrics: max drawdown, Sharpe, worst streaks
5. Position concentration: how many coins fire at once?
6. BTC/ETH-only tradeable subset
7. Score threshold optimization

Usage:
    python3 signal_refine.py
"""

import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from database import DB_PATH

WINSOR_LOW = 1
WINSOR_HIGH = 99

COST_BLUE_CHIP = 0.20
COST_TOP20 = 0.40
COST_ALTCOIN = 0.80


def winsorize_series(s, low_pct=WINSOR_LOW, high_pct=WINSOR_HIGH):
    low_val = np.nanpercentile(s, low_pct)
    high_val = np.nanpercentile(s, high_pct)
    return s.clip(lower=low_val, upper=high_val)


def load_and_prepare(db_path=DB_PATH):
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

    all_coins = []
    for coin_id, group in prices.groupby("coingecko_id"):
        group = group.sort_values("date").copy()
        for days in [3, 5, 7, 10, 14, 20]:
            group[f"fwd_{days}d"] = (
                group["price_usd"].shift(-days) / group["price_usd"] - 1
            ) * 100
        group["ret_1d"] = group["price_usd"].pct_change() * 100
        group["ret_5d"] = (group["price_usd"] / group["price_usd"].shift(5) - 1) * 100
        group["ret_10d"] = (group["price_usd"] / group["price_usd"].shift(10) - 1) * 100
        group["ret_20d"] = (group["price_usd"] / group["price_usd"].shift(20) - 1) * 100
        group["vol_20d"] = group["ret_1d"].rolling(20).std() * np.sqrt(365)
        group["vol_ratio"] = group["volume_usd"] / group["volume_usd"].rolling(20).mean()
        group["vol_pctile"] = group["vol_20d"].rank(pct=True) * 100
        all_coins.append(group)

    df = pd.concat(all_coins, ignore_index=True)

    for days in [3, 5, 7, 10, 14, 20]:
        df[f"fwd_{days}d"] = winsorize_series(df[f"fwd_{days}d"])
    for col in ["ret_1d", "ret_5d", "ret_10d", "ret_20d"]:
        df[col] = winsorize_series(df[col])

    df = df.merge(fg[["date", "value", "classification"]], on="date", how="left")
    df.rename(columns={"value": "fg_value", "classification": "fg_class"}, inplace=True)

    return df


def compute_scores(df):
    """Apply the scoring system from signal_dedup.py."""
    df["score"] = 0.0

    # Crash family
    df.loc[df["ret_5d"] <= -25, "score"] += 3
    df.loc[(df["ret_5d"] <= -15) & (df["ret_5d"] > -25), "score"] += 1

    # Greed/trend family
    df.loc[df["fg_value"] >= 90, "score"] += 2
    df.loc[(df["fg_value"] >= 75) & (df["fg_value"] < 90), "score"] += 1

    # Volume confirmation
    df.loc[(df["vol_ratio"] >= 3.0) & (df["ret_1d"] <= -5), "score"] += 1

    # Anti-signals
    df_sorted = df.sort_values(["coingecko_id", "date"])
    fg_prev = df_sorted.groupby("coingecko_id")["fg_value"].shift(1)
    fear_exit = ((fg_prev <= 25) & (df_sorted["fg_value"] > 25))
    df.loc[fear_exit.values, "score"] -= 2

    df.loc[df["vol_pctile"] <= 10, "score"] -= 1

    # Transaction costs
    df["txn_cost"] = COST_ALTCOIN
    df.loc[df["bucket"] == 1, "txn_cost"] = COST_BLUE_CHIP
    df.loc[df["bucket"] == 2, "txn_cost"] = COST_TOP20

    return df


# ============================================================
# TEST 1: HOLDING PERIOD OPTIMIZATION
# ============================================================

def holding_period_optimization(df):
    print(f"\n{'#'*70}")
    print(f"# TEST 1: HOLDING PERIOD OPTIMIZATION")
    print(f"{'#'*70}")
    print(f"  Which holding period captures the most alpha per day held?")

    windows = [3, 5, 7, 10, 14, 20]

    for threshold, label in [(2, "Score ≥ 2"), (3, "Score ≥ 3")]:
        sig = df[df["score"] >= threshold]
        base = df

        print(f"\n  === {label} (n={len(sig):,}) ===")
        print(f"  {'Window':<8} {'Alpha':>8} {'Alpha/Day':>10} {'Median':>8} "
              f"{'t-stat':>8} {'p-val':>7} {'Win%':>6} {'Sharpe':>7}")
        print(f"  {'-'*68}")

        best_per_day = 0
        best_window = 0

        for days in windows:
            col = f"fwd_{days}d"
            if col not in df.columns:
                continue

            sig_rets = sig[col].dropna()
            base_rets = base[col].dropna()
            if len(sig_rets) < 10:
                continue

            alpha = sig_rets.mean() - base_rets.mean()
            alpha_per_day = alpha / days
            med = sig_rets.median()
            t_stat, p_val = stats.ttest_ind(sig_rets, base_rets, equal_var=False)
            win = (sig_rets > 0).mean() * 100
            sharpe = sig_rets.mean() / sig_rets.std() * np.sqrt(252 / days) if sig_rets.std() > 0 else 0

            star = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

            if alpha_per_day > best_per_day:
                best_per_day = alpha_per_day
                best_window = days

            print(f"  {days}d{'':<5} {alpha:>+7.2f}% {alpha_per_day:>+9.3f}% "
                  f"{med:>+7.2f}% {t_stat:>8.2f} {p_val:>7.4f} {win:>5.1f}% "
                  f"{sharpe:>6.2f} {star}")

        print(f"\n  >> Best alpha/day: {best_window}d at {best_per_day:+.3f}%/day")


# ============================================================
# TEST 2: LONG-ONLY ANALYSIS
# ============================================================

def long_only_analysis(df):
    print(f"\n\n{'#'*70}")
    print(f"# TEST 2: LONG-ONLY ANALYSIS")
    print(f"{'#'*70}")
    print(f"  Short side was weak (-0.50%/event). Is long-only better?")

    print(f"\n  === 5-DAY RETURNS COMPARISON ===")
    print(f"  {'Strategy':<35} {'Mean':>8} {'Median':>8} {'Net':>8} {'Win%':>6}")
    print(f"  {'-'*70}")

    long2 = df[df["score"] >= 2]["fwd_5d"].dropna()
    cost2 = df[df["score"] >= 2]["txn_cost"].mean()
    print(f"  {'Long Score ≥ 2':<35} {long2.mean():>+7.2f}% {long2.median():>+7.2f}% "
          f"{long2.mean() - cost2:>+7.2f}% {(long2 > 0).mean()*100:>5.1f}%")

    long3 = df[df["score"] >= 3]["fwd_5d"].dropna()
    cost3 = df[df["score"] >= 3]["txn_cost"].mean()
    print(f"  {'Long Score ≥ 3':<35} {long3.mean():>+7.2f}% {long3.median():>+7.2f}% "
          f"{long3.mean() - cost3:>+7.2f}% {(long3 > 0).mean()*100:>5.1f}%")

    long4 = df[df["score"] >= 4]["fwd_5d"].dropna()
    if len(long4) >= 10:
        cost4 = df[df["score"] >= 4]["txn_cost"].mean()
        print(f"  {'Long Score ≥ 4':<35} {long4.mean():>+7.2f}% {long4.median():>+7.2f}% "
              f"{long4.mean() - cost4:>+7.2f}% {(long4 > 0).mean()*100:>5.1f}%")

    short = df[df["score"] <= -1]["fwd_5d"].dropna()
    short_cost = df[df["score"] <= -1]["txn_cost"].mean()
    short_profit = -short
    print(f"  {'Short Score ≤ -1':<35} {short_profit.mean():>+7.2f}% "
          f"{short_profit.median():>+7.2f}% "
          f"{short_profit.mean() - short_cost:>+7.2f}% "
          f"{(short_profit > 0).mean()*100:>5.1f}%")

    short2 = df[df["score"] <= -2]["fwd_5d"].dropna()
    if len(short2) >= 10:
        short2_cost = df[df["score"] <= -2]["txn_cost"].mean()
        short2_profit = -short2
        print(f"  {'Short Score ≤ -2':<35} {short2_profit.mean():>+7.2f}% "
              f"{short2_profit.median():>+7.2f}% "
              f"{short2_profit.mean() - short2_cost:>+7.2f}% "
              f"{(short2_profit > 0).mean()*100:>5.1f}%")

    print(f"\n  Verdict: Short side adds {short_profit.mean():+.2f}% gross per event")
    print(f"  After costs ({short_cost:.2f}% avg): {short_profit.mean() - short_cost:+.2f}% net")
    if short_profit.mean() - short_cost < 0.5:
        print(f"  >> RECOMMENDATION: Long-only. Short alpha doesn't justify costs + complexity.")
    else:
        print(f"  >> Short side contributes meaningfully. Keep long/short.")


# ============================================================
# TEST 3: WALK-FORWARD OUT-OF-SAMPLE
# ============================================================

def walk_forward_validation(df):
    print(f"\n\n{'#'*70}")
    print(f"# TEST 3: WALK-FORWARD OUT-OF-SAMPLE VALIDATION")
    print(f"{'#'*70}")
    print(f"  Train: 2014-2021 (in-sample) | Test: 2022-2026 (out-of-sample)")
    print(f"  Scoring rules were derived from full-sample analysis.")
    print(f"  If OOS alpha is similar magnitude, signal is likely real.\n")

    train_mask = df["date"] < "2022-01-01"
    test_mask = df["date"] >= "2022-01-01"

    for label, mask in [("IN-SAMPLE (2014-2021)", train_mask),
                        ("OUT-OF-SAMPLE (2022-2026)", test_mask)]:
        subset = df[mask]
        sig = subset[subset["score"] >= 2]
        anti = subset[subset["score"] <= -1]

        print(f"  === {label} ===")
        print(f"  Total rows: {len(subset):,} | Signal events (≥2): {len(sig):,} "
              f"| Anti-signal (≤-1): {len(anti):,}")

        for days in [5, 10, 20]:
            col = f"fwd_{days}d"
            sig_rets = sig[col].dropna()
            base_rets = subset[col].dropna()
            anti_rets = anti[col].dropna()

            if len(sig_rets) < 10:
                continue

            sig_alpha = sig_rets.mean() - base_rets.mean()
            anti_alpha = anti_rets.mean() - base_rets.mean()
            t_sig, p_sig = stats.ttest_ind(sig_rets, base_rets, equal_var=False)
            t_anti, p_anti = stats.ttest_ind(anti_rets, base_rets, equal_var=False)

            sig_star = "**" if p_sig < 0.01 else "*" if p_sig < 0.05 else ""
            anti_star = "**" if p_anti < 0.01 else "*" if p_anti < 0.05 else ""

            print(f"    {days:>2}d: Long alpha {sig_alpha:>+7.2f}% (t={t_sig:.2f}, "
                  f"p={p_sig:.4f}) {sig_star}  |  "
                  f"Avoid alpha {anti_alpha:>+7.2f}% (t={t_anti:.2f}) {anti_star}")

        print()

    print(f"  === YEAR-BY-YEAR OUT-OF-SAMPLE (Score ≥ 2, 5d alpha) ===")
    print(f"  {'Year':<8} {'n':>6} {'Signal':>8} {'Base':>8} {'Alpha':>8} {'p-val':>7}")
    print(f"  {'-'*50}")

    for year in range(2022, 2027):
        mask = (df["date"] >= f"{year}-01-01") & (df["date"] < f"{year+1}-01-01")
        sub = df[mask]
        sig_sub = sub[sub["score"] >= 2]["fwd_5d"].dropna()
        base_sub = sub["fwd_5d"].dropna()

        if len(sig_sub) < 5:
            continue

        alpha = sig_sub.mean() - base_sub.mean()
        if len(sig_sub) >= 10 and len(base_sub) >= 10:
            _, p_val = stats.ttest_ind(sig_sub, base_sub, equal_var=False)
        else:
            p_val = 1.0

        star = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {year:<8} {len(sig_sub):>6} {sig_sub.mean():>+7.2f}% "
              f"{base_sub.mean():>+7.2f}% {alpha:>+7.2f}% {p_val:>7.4f} {star}")


# ============================================================
# TEST 4: RISK METRICS
# ============================================================

def risk_metrics(df):
    print(f"\n\n{'#'*70}")
    print(f"# TEST 4: RISK METRICS")
    print(f"{'#'*70}")

    sig = df[df["score"] >= 2].copy()
    rets = sig["fwd_5d"].dropna()

    print(f"\n  === RETURN DISTRIBUTION (Score ≥ 2, 5-day) ===")
    print(f"  Events: {len(rets):,}")
    print(f"  Mean: {rets.mean():+.2f}%")
    print(f"  Median: {rets.median():+.2f}%")
    print(f"  Std: {rets.std():.2f}%")
    print(f"  Skew: {rets.skew():.2f}")
    print(f"  Kurtosis: {rets.kurtosis():.2f}")

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(rets, p)
        print(f"    {p:>3}th: {val:>+8.2f}%")

    print(f"\n  === WORST 10 INDIVIDUAL EVENTS ===")
    worst = sig.nsmallest(10, "fwd_5d")[["date", "ticker", "bucket", "score", "fwd_5d", "ret_5d", "fg_value"]]
    print(f"  {'Date':<12} {'Ticker':<8} {'Bucket':>7} {'Score':>6} {'5d Ret':>8} "
          f"{'Prior 5d':>9} {'FG':>4}")
    print(f"  {'-'*60}")
    for _, row in worst.iterrows():
        bname = {1: "Blue", 2: "Top20", 3: "Alt"}[row["bucket"]]
        print(f"  {str(row['date'])[:10]:<12} {row['ticker']:<8} {bname:>7} "
              f"{row['score']:>+5.0f} {row['fwd_5d']:>+7.2f}% "
              f"{row['ret_5d']:>+8.2f}% {row['fg_value']:>4.0f}")

    print(f"\n  === BEST 10 INDIVIDUAL EVENTS ===")
    best = sig.nlargest(10, "fwd_5d")[["date", "ticker", "bucket", "score", "fwd_5d", "ret_5d", "fg_value"]]
    print(f"  {'Date':<12} {'Ticker':<8} {'Bucket':>7} {'Score':>6} {'5d Ret':>8} "
          f"{'Prior 5d':>9} {'FG':>4}")
    print(f"  {'-'*60}")
    for _, row in best.iterrows():
        bname = {1: "Blue", 2: "Top20", 3: "Alt"}[row["bucket"]]
        print(f"  {str(row['date'])[:10]:<12} {row['ticker']:<8} {bname:>7} "
              f"{row['score']:>+5.0f} {row['fwd_5d']:>+7.2f}% "
              f"{row['ret_5d']:>+8.2f}% {row['fg_value']:>4.0f}")

    print(f"\n  === STREAK ANALYSIS (daily signal avg) ===")
    daily_avg = sig.groupby("date")["fwd_5d"].mean().sort_index()

    max_loss_streak = 0
    current_streak = 0
    worst_streak_end = None

    for date, ret in daily_avg.items():
        if ret < 0:
            current_streak += 1
            if current_streak > max_loss_streak:
                max_loss_streak = current_streak
                worst_streak_end = date
        else:
            current_streak = 0

    print(f"  Signal days total: {len(daily_avg)}")
    print(f"  Winning days: {(daily_avg > 0).sum()} ({(daily_avg > 0).mean()*100:.1f}%)")
    print(f"  Losing days: {(daily_avg < 0).sum()} ({(daily_avg < 0).mean()*100:.1f}%)")
    print(f"  Max consecutive losing days: {max_loss_streak} (ended {worst_streak_end})")

    print(f"\n  === WORST CALENDAR MONTHS (Score ≥ 2, avg 5d return) ===")
    sig_monthly = sig.copy()
    sig_monthly["month"] = sig_monthly["date"].dt.to_period("M")
    monthly = sig_monthly.groupby("month").agg(
        n=("fwd_5d", "count"),
        mean_ret=("fwd_5d", "mean"),
        median_ret=("fwd_5d", "median"),
    ).sort_values("mean_ret")

    print(f"  {'Month':<10} {'n':>6} {'Mean':>8} {'Median':>8}")
    print(f"  {'-'*35}")
    for month, row in monthly.head(10).iterrows():
        print(f"  {str(month):<10} {row['n']:>6.0f} {row['mean_ret']:>+7.2f}% "
              f"{row['median_ret']:>+7.2f}%")

    print(f"\n  === BEST CALENDAR MONTHS ===")
    print(f"  {'Month':<10} {'n':>6} {'Mean':>8} {'Median':>8}")
    print(f"  {'-'*35}")
    for month, row in monthly.tail(10).iterrows():
        print(f"  {str(month):<10} {row['n']:>6.0f} {row['mean_ret']:>+7.2f}% "
              f"{row['median_ret']:>+7.2f}%")


# ============================================================
# TEST 5: POSITION CONCENTRATION
# ============================================================

def position_concentration(df):
    print(f"\n\n{'#'*70}")
    print(f"# TEST 5: POSITION CONCENTRATION & COIN ANALYSIS")
    print(f"{'#'*70}")

    sig = df[df["score"] >= 2]

    daily_counts = sig.groupby("date").size()

    print(f"\n  === POSITIONS PER SIGNAL DAY ===")
    print(f"  Signal days: {len(daily_counts)}")
    print(f"  Mean coins per day: {daily_counts.mean():.1f}")
    print(f"  Median coins per day: {daily_counts.median():.0f}")
    print(f"  Max coins per day: {daily_counts.max()}")
    print(f"  Min coins per day: {daily_counts.min()}")

    print(f"\n  Distribution:")
    for bucket_label, low, high in [("1-3 coins", 1, 3), ("4-10 coins", 4, 10),
                                     ("11-20 coins", 11, 20), ("21+ coins", 21, 999)]:
        count = ((daily_counts >= low) & (daily_counts <= high)).sum()
        pct = count / len(daily_counts) * 100
        print(f"    {bucket_label:<15} {count:>5} days ({pct:.1f}%)")

    print(f"\n  === TOP COINS BY SIGNAL FREQUENCY (Score ≥ 2) ===")
    coin_stats = sig.groupby(["ticker", "bucket"]).agg(
        n=("fwd_5d", "count"),
        mean_ret=("fwd_5d", "mean"),
        median_ret=("fwd_5d", "median"),
        win_rate=("fwd_5d", lambda x: (x > 0).mean() * 100),
    ).sort_values("n", ascending=False)

    print(f"  {'Ticker':<8} {'Bucket':>7} {'n':>6} {'Mean 5d':>9} {'Med 5d':>8} {'Win%':>6}")
    print(f"  {'-'*50}")

    for (ticker, bucket), row in coin_stats.head(20).iterrows():
        bname = {1: "Blue", 2: "Top20", 3: "Alt"}[bucket]
        print(f"  {ticker:<8} {bname:>7} {row['n']:>6.0f} {row['mean_ret']:>+8.2f}% "
              f"{row['median_ret']:>+7.2f}% {row['win_rate']:>5.1f}%")

    print(f"\n  === BTC & ETH ONLY (most liquid, most tradeable) ===")
    for ticker in ["BTC", "ETH"]:
        coin_sig = sig[sig["ticker"] == ticker]
        if len(coin_sig) < 5:
            print(f"  {ticker}: insufficient signal events ({len(coin_sig)})")
            continue

        rets = coin_sig["fwd_5d"].dropna()
        base = df[df["ticker"] == ticker]["fwd_5d"].dropna()
        alpha = rets.mean() - base.mean()
        t_stat, p_val = stats.ttest_ind(rets, base, equal_var=False)
        star = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        print(f"  {ticker}: {len(rets)} events, mean {rets.mean():+.2f}%, "
              f"median {rets.median():+.2f}%, alpha {alpha:+.2f}%, "
              f"win {(rets > 0).mean()*100:.1f}%, p={p_val:.4f} {star}")

        for threshold in [2, 3]:
            tier = coin_sig[coin_sig["score"] >= threshold]
            tier_rets = tier["fwd_5d"].dropna()
            if len(tier_rets) >= 5:
                print(f"    Score ≥{threshold}: n={len(tier_rets)}, "
                      f"mean {tier_rets.mean():+.2f}%, "
                      f"median {tier_rets.median():+.2f}%")


# ============================================================
# TEST 6: SCORE THRESHOLD OPTIMIZATION
# ============================================================

def threshold_optimization(df):
    print(f"\n\n{'#'*70}")
    print(f"# TEST 6: SCORE THRESHOLD OPTIMIZATION")
    print(f"{'#'*70}")
    print(f"  Finding the sweet spot between selectivity and sample size")

    print(f"\n  === 5-DAY ALPHA BY MINIMUM SCORE ===")
    print(f"  {'Min Score':>10} {'n':>8} {'Mean':>8} {'Median':>8} {'Alpha':>8} "
          f"{'Net':>8} {'Win%':>6} {'t-stat':>8} {'Events/Yr':>10}")
    print(f"  {'-'*80}")

    base_rets = df["fwd_5d"].dropna()
    years = (df["date"].max() - df["date"].min()).days / 365.25

    for threshold in range(-2, 6):
        sig = df[df["score"] >= threshold]
        rets = sig["fwd_5d"].dropna()
        if len(rets) < 10:
            continue

        alpha = rets.mean() - base_rets.mean()
        avg_cost = sig["txn_cost"].mean()
        net = alpha - avg_cost
        win = (rets > 0).mean() * 100
        t_stat, p_val = stats.ttest_ind(rets, base_rets, equal_var=False)
        events_per_year = len(rets) / years

        star = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        print(f"  {threshold:>+9}  {len(rets):>8,} {rets.mean():>+7.2f}% "
              f"{rets.median():>+7.2f}% {alpha:>+7.2f}% {net:>+7.2f}% "
              f"{win:>5.1f}% {t_stat:>8.2f} {events_per_year:>9.0f} {star}")

    print(f"\n  === ALPHA × FREQUENCY (annual alpha capture potential) ===")
    print(f"  {'Min Score':>10} {'Alpha':>8} {'Events/Yr':>10} {'Alpha×Freq':>11}")
    print(f"  {'-'*45}")

    for threshold in range(1, 6):
        sig = df[df["score"] >= threshold]
        rets = sig["fwd_5d"].dropna()
        if len(rets) < 10:
            continue

        alpha = rets.mean() - base_rets.mean() - sig["txn_cost"].mean()
        events_per_year = len(rets) / years
        capture = alpha * events_per_year / 100

        print(f"  {threshold:>+9}  {alpha:>+7.2f}% {events_per_year:>9.0f} "
              f"{capture:>+10.1f}%")


# ============================================================
# TEST 7: SIGNAL PERFORMANCE BY MARKET REGIME
# ============================================================

def regime_analysis(df):
    print(f"\n\n{'#'*70}")
    print(f"# TEST 7: SIGNAL PERFORMANCE BY MARKET REGIME")
    print(f"{'#'*70}")

    sig = df[df["score"] >= 2].copy()

    # Use BTC 20d return as market regime proxy
    btc = df[df["ticker"] == "BTC"][["date", "ret_20d"]].rename(
        columns={"ret_20d": "btc_20d"}).drop_duplicates("date")

    # Merge BTC regime into both signal and full dataset
    sig = sig.merge(btc, on="date", how="left")
    df_with_btc = df.merge(btc, on="date", how="left")

    print(f"\n  === SCORE ≥ 2 ALPHA BY BTC 20D REGIME ===")
    print(f"  {'Regime':<25} {'n':>7} {'Sig Mean':>9} {'Base Mean':>10} "
          f"{'Alpha':>8} {'Win%':>6}")
    print(f"  {'-'*70}")

    regime_defs = [
        ("BTC down >20%", -999, -20),
        ("BTC down 10-20%", -20, -10),
        ("BTC down 0-10%", -10, 0),
        ("BTC up 0-10%", 0, 10),
        ("BTC up 10-20%", 10, 20),
        ("BTC up >20%", 20, 999),
    ]

    for rname, low, high in regime_defs:
        sig_sub = sig[(sig["btc_20d"] > low) & (sig["btc_20d"] <= high)]["fwd_5d"].dropna()
        base_sub = df_with_btc[(df_with_btc["btc_20d"] > low) & (df_with_btc["btc_20d"] <= high)]["fwd_5d"].dropna()

        if len(sig_sub) < 10:
            print(f"  {rname:<25} {len(sig_sub):>7} — insufficient data")
            continue

        alpha = sig_sub.mean() - base_sub.mean()
        win = (sig_sub > 0).mean() * 100

        print(f"  {rname:<25} {len(sig_sub):>7,} {sig_sub.mean():>+8.2f}% "
              f"{base_sub.mean():>+9.2f}% {alpha:>+7.2f}% {win:>5.1f}%")

    # Signal type breakdown
    print(f"\n  === WHAT TRIGGERS IN EACH REGIME? ===")
    print(f"  {'Regime':<25} {'Crash (DD)':>10} {'Greed (FG)':>11} {'Both':>6}")
    print(f"  {'-'*55}")

    sig["is_crash"] = sig["ret_5d"] <= -15
    sig["is_greed"] = sig["fg_value"] >= 75

    for rname, low, high in regime_defs:
        sub = sig[(sig["btc_20d"] > low) & (sig["btc_20d"] <= high)]
        if len(sub) < 10:
            continue
        crash_pct = sub["is_crash"].mean() * 100
        greed_pct = sub["is_greed"].mean() * 100
        both_pct = (sub["is_crash"] & sub["is_greed"]).mean() * 100

        print(f"  {rname:<25} {crash_pct:>9.1f}% {greed_pct:>10.1f}% {both_pct:>5.1f}%")


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"{'='*70}")
    print(f"CRYPTO SIGNAL REFINEMENT — Phase 3b")
    print(f"{'='*70}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nLoading data...")
    df = load_and_prepare()
    print(f"Loaded {len(df):,} rows")

    print(f"\nComputing scores...")
    df = compute_scores(df)
    sig_count = (df["score"] >= 2).sum()
    print(f"Score ≥ 2 events: {sig_count:,}")

    holding_period_optimization(df)
    long_only_analysis(df)
    walk_forward_validation(df)
    risk_metrics(df)
    position_concentration(df)
    threshold_optimization(df)
    regime_analysis(df)

    print(f"\n\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
