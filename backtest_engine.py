"""
Crypto Backtest Engine v2
Tests trading signals against forward returns.
Fixed: Winsorized returns to handle altcoin outliers.

Signals tested:
  1. Fear & Greed extremes (buy Extreme Fear, sell Extreme Greed)
  2. Momentum (N-day returns as predictor of next N days)
  3. Mean reversion (oversold/overbought bounces)
  4. Volatility regime (low vol breakouts, high vol mean reversion)
  5. Volume spikes (unusual volume as signal)

Methodology:
  - For each signal, identify trigger dates
  - Measure forward returns at 3d, 5d, 10d, 20d
  - Winsorize at 1st/99th percentile to cap outliers
  - Compare to baseline (all-days average return)
  - Calculate alpha = signal return - baseline return
  - t-test for statistical significance
  - Report both mean and median returns
  - Break down by bucket (Blue Chips, Top 20, Altcoins)

Usage:
    python3 backtest_engine.py                # Run all signal tests
    python3 backtest_engine.py --signal fear   # Run only Fear & Greed
    python3 backtest_engine.py --signal momentum
    python3 backtest_engine.py --signal reversion
    python3 backtest_engine.py --signal volatility
    python3 backtest_engine.py --signal volume
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats

from database import DB_PATH

# Winsorization bounds — cap returns at these percentiles
WINSOR_LOW = 1    # 1st percentile
WINSOR_HIGH = 99  # 99th percentile


# ============================================================
# DATA LOADING
# ============================================================

def load_prices(db_path=DB_PATH):
    """Load all daily prices into a DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT coingecko_id, ticker, bucket, date, price_usd, volume_usd
        FROM daily_prices
        ORDER BY coingecko_id, date
    """, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded {len(df):,} price records for {df['ticker'].nunique()} coins")
    return df


def load_fear_greed(db_path=DB_PATH):
    """Load Fear & Greed Index."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT date, value, classification FROM fear_greed ORDER BY date
    """, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded {len(df):,} Fear & Greed days ({df['date'].min().date()} to {df['date'].max().date()})")
    return df


def winsorize_series(s, low_pct=WINSOR_LOW, high_pct=WINSOR_HIGH):
    """Cap values at percentile bounds to remove outlier influence."""
    low_val = np.nanpercentile(s, low_pct)
    high_val = np.nanpercentile(s, high_pct)
    return s.clip(lower=low_val, upper=high_val)


def compute_forward_returns(coin_df):
    """
    Add forward return columns to a single-coin DataFrame.
    coin_df must be sorted by date with price_usd column.
    Returns DataFrame with fwd_3d, fwd_5d, fwd_10d, fwd_20d columns.
    """
    for days in [3, 5, 10, 20]:
        coin_df[f"fwd_{days}d"] = (
            coin_df["price_usd"].shift(-days) / coin_df["price_usd"] - 1
        ) * 100  # As percentage
    return coin_df


def prepare_data(db_path=DB_PATH):
    """Load prices, compute forward returns for each coin, merge Fear & Greed."""
    prices = load_prices(db_path)
    fg = load_fear_greed(db_path)

    # Compute forward returns per coin
    all_coins = []
    for coin_id, group in prices.groupby("coingecko_id"):
        group = group.sort_values("date").copy()
        group = compute_forward_returns(group)

        # Add technical indicators
        group["ret_1d"] = group["price_usd"].pct_change() * 100
        group["ret_5d"] = (group["price_usd"] / group["price_usd"].shift(5) - 1) * 100
        group["ret_10d"] = (group["price_usd"] / group["price_usd"].shift(10) - 1) * 100
        group["ret_20d"] = (group["price_usd"] / group["price_usd"].shift(20) - 1) * 100

        # Rolling volatility (20-day std of daily returns, annualized)
        group["vol_20d"] = group["ret_1d"].rolling(20).std() * np.sqrt(365)

        # Volume ratio (today vs 20-day avg)
        group["vol_ratio"] = group["volume_usd"] / group["volume_usd"].rolling(20).mean()

        # Volatility percentile rank within this coin's history
        group["vol_pctile"] = group["vol_20d"].rank(pct=True) * 100

        all_coins.append(group)

    df = pd.concat(all_coins, ignore_index=True)

    # Winsorize forward returns globally to cap outliers
    print(f"\nWinsorizing forward returns at {WINSOR_LOW}th/{WINSOR_HIGH}th percentile...")
    for days in [3, 5, 10, 20]:
        col = f"fwd_{days}d"
        before_min = df[col].min()
        before_max = df[col].max()
        df[col] = winsorize_series(df[col])
        after_min = df[col].min()
        after_max = df[col].max()
        print(f"  {col}: [{before_min:+.1f}%, {before_max:+.1f}%] -> [{after_min:+.1f}%, {after_max:+.1f}%]")

    # Also winsorize lookback indicators used for signal thresholds
    for col in ["ret_1d", "ret_5d", "ret_10d", "ret_20d"]:
        df[col] = winsorize_series(df[col])

    # Merge Fear & Greed
    df = df.merge(fg[["date", "value", "classification"]],
                  on="date", how="left",
                  suffixes=("", "_fg"))
    df.rename(columns={"value": "fg_value", "classification": "fg_class"}, inplace=True)

    print(f"\nPrepared dataset: {len(df):,} rows with winsorized forward returns")

    # Print baseline sanity check
    print(f"\n  BASELINE SANITY CHECK (mean daily forward returns):")
    for bucket in [0, 1, 2, 3]:
        label = {0: "ALL", 1: "Blue Chips", 2: "Top 20", 3: "Altcoins"}[bucket]
        subset = df if bucket == 0 else df[df["bucket"] == bucket]
        fwd5 = subset["fwd_5d"].dropna()
        print(f"    {label}: 5d mean={fwd5.mean():+.3f}%, median={fwd5.median():+.3f}%, "
              f"std={fwd5.std():.2f}%, n={len(fwd5):,}")

    return df


# ============================================================
# BACKTEST FRAMEWORK
# ============================================================

FORWARD_WINDOWS = [3, 5, 10, 20]
BUCKET_LABELS = {1: "Blue Chips", 2: "Top 20", 3: "Altcoins", 0: "ALL"}


def analyze_signal(df, signal_mask, signal_name, show_coins=False):
    """
    Core backtest function. Given a boolean mask identifying signal days,
    measure forward returns and compare to baseline.
    Reports both mean and median. Uses winsorized data.
    """
    print(f"\n{'='*70}")
    print(f"SIGNAL: {signal_name}")
    print(f"{'='*70}")

    signal_df = df[signal_mask].dropna(subset=["fwd_5d"])
    baseline_df = df.dropna(subset=["fwd_5d"])

    if len(signal_df) < 10:
        print(f"  Insufficient signal events: {len(signal_df)} (need 10+)")
        return None

    print(f"  Signal events: {len(signal_df):,}")
    print(f"  Baseline days: {len(baseline_df):,}")
    print(f"  Date range: {signal_df['date'].min().date()} to {signal_df['date'].max().date()}")
    print(f"  Unique coins in signal: {signal_df['ticker'].nunique()}")

    results = []

    # Overall + by bucket
    for bucket in [0, 1, 2, 3]:
        if bucket == 0:
            s_df = signal_df
            b_df = baseline_df
            label = "ALL"
        else:
            s_df = signal_df[signal_df["bucket"] == bucket]
            b_df = baseline_df[baseline_df["bucket"] == bucket]
            label = BUCKET_LABELS[bucket]

        if len(s_df) < 5:
            continue

        print(f"\n  --- {label} (n={len(s_df):,}) ---")
        print(f"  {'Window':<8} {'Sig Mean':>9} {'Sig Med':>9} {'Base Mean':>10} "
              f"{'Alpha':>8} {'t-stat':>7} {'p-val':>7} {'Win%':>6}")
        print(f"  {'-'*68}")

        for days in FORWARD_WINDOWS:
            col = f"fwd_{days}d"
            sig_rets = s_df[col].dropna()
            base_rets = b_df[col].dropna()

            if len(sig_rets) < 5:
                continue

            sig_mean = sig_rets.mean()
            sig_median = sig_rets.median()
            base_mean = base_rets.mean()
            alpha = sig_mean - base_mean

            # t-test: is signal mean different from baseline mean?
            t_stat, p_val = stats.ttest_ind(sig_rets, base_rets, equal_var=False)
            win_rate = (sig_rets > 0).mean() * 100

            # Significance stars
            star = ""
            if p_val < 0.01:
                star = " **"
            elif p_val < 0.05:
                star = " *"

            print(f"  {days}d{'':<5} {sig_mean:>+8.2f}% {sig_median:>+8.2f}% "
                  f"{base_mean:>+9.2f}% {alpha:>+7.2f}% {t_stat:>7.2f} "
                  f"{p_val:>7.4f} {win_rate:>5.1f}%{star}")

            results.append({
                "signal": signal_name,
                "bucket": label,
                "window": f"{days}d",
                "n": len(sig_rets),
                "signal_mean": sig_mean,
                "signal_median": sig_median,
                "baseline_mean": base_mean,
                "alpha": alpha,
                "t_stat": t_stat,
                "p_value": p_val,
                "win_rate": win_rate,
            })

    # Show top coins in signal if requested
    if show_coins and len(signal_df) > 0:
        print(f"\n  Top coins by signal frequency:")
        coin_counts = signal_df["ticker"].value_counts().head(10)
        for ticker, count in coin_counts.items():
            coin_5d = signal_df[signal_df["ticker"] == ticker]["fwd_5d"]
            print(f"    {ticker}: {count} signals, 5d mean: {coin_5d.mean():+.2f}%, "
                  f"median: {coin_5d.median():+.2f}%")

    return results


# ============================================================
# SIGNAL DEFINITIONS
# ============================================================

def test_fear_greed(df):
    """Test Fear & Greed extremes as entry signals."""
    print(f"\n{'#'*70}")
    print(f"# FEAR & GREED INDEX SIGNALS")
    print(f"{'#'*70}")

    all_results = []

    # Signal 1: Extreme Fear (value <= 10)
    r = analyze_signal(df, df["fg_value"] <= 10,
                       "EXTREME FEAR (FG <= 10) — Buy signal", show_coins=True)
    if r: all_results.extend(r)

    # Signal 2: Fear zone (value <= 25)
    r = analyze_signal(df, df["fg_value"] <= 25,
                       "FEAR ZONE (FG <= 25) — Buy signal")
    if r: all_results.extend(r)

    # Signal 3: Extreme Greed (value >= 90)
    r = analyze_signal(df, df["fg_value"] >= 90,
                       "EXTREME GREED (FG >= 90) — Sell signal")
    if r: all_results.extend(r)

    # Signal 4: Greed zone (value >= 75)
    r = analyze_signal(df, df["fg_value"] >= 75,
                       "GREED ZONE (FG >= 75) — Sell signal")
    if r: all_results.extend(r)

    # Signal 5: Fear-to-neutral transition (yesterday <= 25, today > 25)
    df_sorted = df.sort_values(["coingecko_id", "date"])
    fg_prev = df_sorted.groupby("coingecko_id")["fg_value"].shift(1)
    transition_mask = (fg_prev <= 25) & (df_sorted["fg_value"] > 25)
    r = analyze_signal(df_sorted, transition_mask,
                       "FEAR EXIT (FG crosses above 25) — Buy signal")
    if r: all_results.extend(r)

    return all_results


def test_momentum(df):
    """Test momentum signals — does recent trend continue?"""
    print(f"\n{'#'*70}")
    print(f"# MOMENTUM SIGNALS")
    print(f"{'#'*70}")

    all_results = []

    # Strong 10d momentum (top 10% of 10d returns)
    pctile_90 = df["ret_10d"].quantile(0.90)
    r = analyze_signal(df, df["ret_10d"] >= pctile_90,
                       f"STRONG MOMENTUM (10d ret >= {pctile_90:.1f}%)")
    if r: all_results.extend(r)

    # Strong 20d momentum
    pctile_90_20 = df["ret_20d"].quantile(0.90)
    r = analyze_signal(df, df["ret_20d"] >= pctile_90_20,
                       f"STRONG MOMENTUM (20d ret >= {pctile_90_20:.1f}%)")
    if r: all_results.extend(r)

    # Weak momentum (bottom 10%)
    pctile_10 = df["ret_10d"].quantile(0.10)
    r = analyze_signal(df, df["ret_10d"] <= pctile_10,
                       f"WEAK MOMENTUM (10d ret <= {pctile_10:.1f}%)")
    if r: all_results.extend(r)

    # Negative 20d momentum
    pctile_10_20 = df["ret_20d"].quantile(0.10)
    r = analyze_signal(df, df["ret_20d"] <= pctile_10_20,
                       f"WEAK MOMENTUM (20d ret <= {pctile_10_20:.1f}%)")
    if r: all_results.extend(r)

    return all_results


def test_mean_reversion(df):
    """Test mean reversion — do sharp drops bounce back?"""
    print(f"\n{'#'*70}")
    print(f"# MEAN REVERSION SIGNALS")
    print(f"{'#'*70}")

    all_results = []

    # 5-day drawdown > 15%
    r = analyze_signal(df, df["ret_5d"] <= -15,
                       "5D DRAWDOWN >= 15% — Buy")
    if r: all_results.extend(r)

    # 5-day drawdown > 25%
    r = analyze_signal(df, df["ret_5d"] <= -25,
                       "5D DRAWDOWN >= 25% — Buy")
    if r: all_results.extend(r)

    # 5-day surge > 20%
    r = analyze_signal(df, df["ret_5d"] >= 20,
                       "5D SURGE >= 20% — Sell")
    if r: all_results.extend(r)

    # 5-day surge > 30%
    r = analyze_signal(df, df["ret_5d"] >= 30,
                       "5D SURGE >= 30% — Sell")
    if r: all_results.extend(r)

    # 1-day crash > 10%
    r = analyze_signal(df, df["ret_1d"] <= -10,
                       "1D CRASH >= 10% — Buy")
    if r: all_results.extend(r)

    # 1-day pump > 15%
    r = analyze_signal(df, df["ret_1d"] >= 15,
                       "1D PUMP >= 15% — Sell")
    if r: all_results.extend(r)

    return all_results


def test_volatility(df):
    """Test volatility regime signals."""
    print(f"\n{'#'*70}")
    print(f"# VOLATILITY REGIME SIGNALS")
    print(f"{'#'*70}")

    all_results = []

    # Low volatility (bottom 10th percentile for that coin)
    r = analyze_signal(df, df["vol_pctile"] <= 10,
                       "LOW VOL (bottom 10% for coin) — Breakout setup")
    if r: all_results.extend(r)

    # Low volatility (bottom 20th percentile)
    r = analyze_signal(df, df["vol_pctile"] <= 20,
                       "LOW VOL (bottom 20% for coin)")
    if r: all_results.extend(r)

    # High volatility (top 10th percentile)
    r = analyze_signal(df, df["vol_pctile"] >= 90,
                       "HIGH VOL (top 10% for coin) — Vol mean reversion")
    if r: all_results.extend(r)

    # High volatility (top 20th percentile)
    r = analyze_signal(df, df["vol_pctile"] >= 80,
                       "HIGH VOL (top 20% for coin)")
    if r: all_results.extend(r)

    return all_results


def test_volume(df):
    """Test volume spike signals."""
    print(f"\n{'#'*70}")
    print(f"# VOLUME SPIKE SIGNALS")
    print(f"{'#'*70}")

    all_results = []

    # Volume > 3x 20-day average
    r = analyze_signal(df, df["vol_ratio"] >= 3.0,
                       "VOLUME SPIKE 3x (vs 20d avg)")
    if r: all_results.extend(r)

    # Volume > 5x 20-day average
    r = analyze_signal(df, df["vol_ratio"] >= 5.0,
                       "VOLUME SPIKE 5x (vs 20d avg)")
    if r: all_results.extend(r)

    # Volume spike + down day (capitulation?)
    cap_mask = (df["vol_ratio"] >= 3.0) & (df["ret_1d"] <= -5)
    r = analyze_signal(df, cap_mask,
                       "CAPITULATION (3x volume + 5%+ down day)")
    if r: all_results.extend(r)

    # Volume spike + up day (breakout?)
    breakout_mask = (df["vol_ratio"] >= 3.0) & (df["ret_1d"] >= 5)
    r = analyze_signal(df, breakout_mask,
                       "BREAKOUT (3x volume + 5%+ up day)")
    if r: all_results.extend(r)

    return all_results


# ============================================================
# SUMMARY
# ============================================================

def print_summary(all_results):
    """Print ranked summary of all signals by alpha."""
    if not all_results:
        print("\nNo results to summarize.")
        return

    print(f"\n\n{'#'*70}")
    print(f"# SIGNAL RANKING — 5-DAY ALPHA (ALL COINS)")
    print(f"{'#'*70}")

    # Filter to 5d window, ALL bucket
    five_day = [r for r in all_results if r["window"] == "5d" and r["bucket"] == "ALL"]
    five_day.sort(key=lambda x: x["alpha"], reverse=True)

    print(f"\n{'Signal':<50} {'n':>6} {'Alpha':>8} {'Median':>8} "
          f"{'t-stat':>7} {'p-val':>7} {'Win%':>6}")
    print(f"{'-'*94}")

    for r in five_day:
        sig = r["signal"][:49]
        star = ""
        if r["p_value"] < 0.01:
            star = " **"
        elif r["p_value"] < 0.05:
            star = " *"
        print(f"{sig:<50} {r['n']:>6} {r['alpha']:>+7.2f}% {r['signal_median']:>+7.2f}% "
              f"{r['t_stat']:>7.2f} {r['p_value']:>7.4f} {r['win_rate']:>5.1f}%{star}")

    print(f"\n  * = p < 0.05   ** = p < 0.01")
    print(f"  Alpha = signal mean - baseline mean (winsorized)")

    # Count significant
    sig_05 = sum(1 for r in five_day if r["p_value"] < 0.05)
    sig_01 = sum(1 for r in five_day if r["p_value"] < 0.01)
    print(f"\n  {sig_05}/{len(five_day)} significant at p<0.05, "
          f"{sig_01}/{len(five_day)} at p<0.01")

    # ---- MULTI-WINDOW VIEW FOR TOP SIGNALS ----
    print(f"\n\n{'#'*70}")
    print(f"# TOP SIGNALS — FULL WINDOW BREAKDOWN")
    print(f"{'#'*70}")

    # Get top 8 signals by lowest p-value
    interesting = sorted(five_day, key=lambda x: x["p_value"])[:8]
    interesting_names = [r["signal"] for r in interesting]
    # Preserve order, dedupe
    seen = set()
    unique_names = []
    for name in interesting_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    for sig_name in unique_names:
        print(f"\n  {sig_name}")
        print(f"  {'Bucket':<12} {'3d':>14} {'5d':>14} {'10d':>14} {'20d':>14}")
        print(f"  {'-'*70}")

        for bucket in ["ALL", "Blue Chips", "Top 20", "Altcoins"]:
            row_parts = []
            for window in ["3d", "5d", "10d", "20d"]:
                matches = [r for r in all_results
                           if r["signal"] == sig_name
                           and r["bucket"] == bucket
                           and r["window"] == window]
                if matches:
                    r = matches[0]
                    star = "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
                    cell = f"{r['alpha']:>+6.2f}%{star}"
                    row_parts.append(f"{cell:>14}")
                else:
                    row_parts.append(f"{'---':>14}")
            print(f"  {bucket:<12} {''.join(row_parts)}")

    # ---- BUCKET BREAKDOWN ----
    print(f"\n\n{'#'*70}")
    print(f"# SIGNAL RANKING BY BUCKET — 5-DAY ALPHA")
    print(f"{'#'*70}")

    for bucket_name in ["Blue Chips", "Top 20", "Altcoins"]:
        bucket_5d = [r for r in all_results
                     if r["window"] == "5d" and r["bucket"] == bucket_name]
        if not bucket_5d:
            continue
        bucket_5d.sort(key=lambda x: x["alpha"], reverse=True)

        print(f"\n  === {bucket_name} ===")
        print(f"  {'Signal':<50} {'n':>5} {'Alpha':>8} {'p-val':>7} {'Win%':>6}")
        print(f"  {'-'*80}")

        for r in bucket_5d:
            sig = r["signal"][:49]
            star = ""
            if r["p_value"] < 0.01:
                star = " **"
            elif r["p_value"] < 0.05:
                star = " *"
            print(f"  {sig:<50} {r['n']:>5} {r['alpha']:>+7.2f}% "
                  f"{r['p_value']:>7.4f} {r['win_rate']:>5.1f}%{star}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Crypto Backtest Engine v2")
    parser.add_argument("--signal", type=str, default="all",
                        choices=["all", "fear", "momentum", "reversion",
                                 "volatility", "volume"],
                        help="Which signal group to test")
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"CRYPTO BACKTEST ENGINE v2 — Winsorized Returns")
    print(f"{'='*70}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Outlier handling: returns capped at {WINSOR_LOW}th/{WINSOR_HIGH}th percentile")

    # Load and prepare data
    df = prepare_data()

    all_results = []

    if args.signal in ["all", "fear"]:
        r = test_fear_greed(df)
        if r: all_results.extend(r)

    if args.signal in ["all", "momentum"]:
        r = test_momentum(df)
        if r: all_results.extend(r)

    if args.signal in ["all", "reversion"]:
        r = test_mean_reversion(df)
        if r: all_results.extend(r)

    if args.signal in ["all", "volatility"]:
        r = test_volatility(df)
        if r: all_results.extend(r)

    if args.signal in ["all", "volume"]:
        r = test_volume(df)
        if r: all_results.extend(r)

    # Print ranked summary
    print_summary(all_results)

    print(f"\n\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
