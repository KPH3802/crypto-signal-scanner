#!/usr/bin/env python3
"""
score_tier_analysis.py
======================
Phase 2C: Score tier deep-dive.

Score 4 shows +19.51% alpha vs Score 2 at +6.90%. Before sizing up Score 4
positions or changing its exit rules, we need to answer:

  1. Are Score 4 signals statistically valid? (sample size, consistency)
  2. Do they warrant a longer hold window? (1d/3d/5d/7d/10d breakdown)
  3. Do they warrant a larger position size? (risk-adjusted return)
  4. Are they concentrated in specific coins or time periods? (fragility check)
  5. What actually triggers a Score 4? (component breakdown)
  6. What's the current position sizing — is it already capturing this?

Current sizing:
  Score 2 → 5% of portfolio  = $250
  Score 3 → 8% of portfolio  = $400
  Score 4 → 12% of portfolio = $600

Usage:
    python3 score_tier_analysis.py
    python3 score_tier_analysis.py --output score_tier_results.txt
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

PORTFOLIO      = 5000.0
POSITION_SIZE  = {2: 0.05, 3: 0.08, 4: 0.12}


# ── Data loading ──────────────────────────────────────────────────────────────

def _winsorize(s, low=WINSOR_LOW, high=WINSOR_HIGH):
    lo = np.nanpercentile(s, low)
    hi = np.nanpercentile(s, high)
    return s.clip(lower=lo, upper=hi)


def load_and_prepare(db_path=DB_PATH):
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

    all_coins = []
    for coin_id, group in prices.groupby("coingecko_id"):
        group = group.sort_values("date").copy()
        for days in [1, 3, 5, 7, 10, 20]:
            group[f"fwd_{days}d"] = (
                group["price_usd"].shift(-days) / group["price_usd"] - 1) * 100
        group["ret_1d"]  = group["price_usd"].pct_change() * 100
        group["ret_5d"]  = (group["price_usd"] / group["price_usd"].shift(5)  - 1) * 100
        group["ret_20d"] = (group["price_usd"] / group["price_usd"].shift(20) - 1) * 100
        group["hv_20d"]  = group["ret_1d"].rolling(20).std() * np.sqrt(365)
        group["vol_pctile"] = group["hv_20d"].rank(pct=True) * 100
        group["vol_ratio"]  = group["volume_usd"] / group["volume_usd"].rolling(20).mean()
        all_coins.append(group)

    df = pd.concat(all_coins, ignore_index=True)

    for days in [1, 3, 5, 7, 10, 20]:
        df[f"fwd_{days}d"] = _winsorize(df[f"fwd_{days}d"])
    for col in ["ret_1d", "ret_5d", "ret_20d"]:
        df[col] = _winsorize(df[col])

    df = df.merge(fg, on="date", how="left")

    df["txn_cost"] = COST_ALTCOIN
    df.loc[df["bucket"] == 1, "txn_cost"] = COST_BLUE_CHIP
    df.loc[df["bucket"] == 2, "txn_cost"] = COST_TOP20

    return df


def compute_scores(df):
    df = df.copy()
    df["score"] = 0.0

    # Crash family
    df.loc[df["ret_5d"] <= -25, "score"] += 3
    df.loc[(df["ret_5d"] <= -15) & (df["ret_5d"] > -25), "score"] += 1

    # Greed family
    df.loc[df["fg_value"] >= 90,  "score"] += 2
    df.loc[(df["fg_value"] >= 75) & (df["fg_value"] < 90), "score"] += 1

    # Capitulation
    df.loc[(df["vol_ratio"] >= 3.0) & (df["ret_1d"] <= -5), "score"] += 1

    # Anti-signals
    df_s = df.sort_values(["coingecko_id", "date"])
    fg_prev = df_s.groupby("coingecko_id")["fg_value"].shift(1)
    fear_exit = (fg_prev <= 25) & (df_s["fg_value"] > 25)
    df.loc[fear_exit.values, "score"] -= 2
    df.loc[df["vol_pctile"] <= 10, "score"] -= 1

    # Score component flags (for breakdown analysis)
    df["comp_severe_crash"]  = df["ret_5d"] <= -25
    df["comp_mild_crash"]    = (df["ret_5d"] <= -15) & (df["ret_5d"] > -25)
    df["comp_ext_greed"]     = df["fg_value"] >= 90
    df["comp_greed"]         = (df["fg_value"] >= 75) & (df["fg_value"] < 90)
    df["comp_capitulation"]  = (df["vol_ratio"] >= 3.0) & (df["ret_1d"] <= -5)

    return df


def alpha_vs_base(sig_rets, base_rets):
    if len(sig_rets) < 5 or len(base_rets) < 5:
        return float("nan"), float("nan"), float("nan")
    alpha = sig_rets.mean() - base_rets.mean()
    t, p  = stats.ttest_ind(sig_rets, base_rets, equal_var=False)
    return alpha, t, p


def fmt(alpha, p_val, width=9):
    if np.isnan(alpha):
        return f"{'N/A':>{width}}"
    star = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    return f"{alpha:>+{width-2}.2f}%{star:<2}"


# ── Test 1: Sample size and consistency ───────────────────────────────────────

def test_sample_size(df):
    print(f"\n{'='*72}")
    print(f"TEST 1: SAMPLE SIZE & STATISTICAL VALIDITY BY SCORE TIER")
    print(f"{'='*72}")

    years = (df["date"].max() - df["date"].min()).days / 365.25
    base  = df["fwd_3d"].dropna()

    print(f"\n  {'Tier':<12} {'n':>7} {'Events/Yr':>10} {'3d Mean':>9} "
          f"{'3d Alpha':>9} {'t-stat':>8} {'p-val':>7} {'Win%':>6}")
    print(f"  {'-'*70}")

    for score in [2, 3, 4, 5]:
        sig = df[df["score"] >= score]["fwd_3d"].dropna()
        if len(sig) < 5:
            continue
        alpha, t, p = alpha_vs_base(sig, base)
        win = (sig > 0).mean() * 100
        evy = len(sig) / years
        star = "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  Score ≥{score:<5} {len(sig):>7,} {evy:>9.0f} {sig.mean():>+8.2f}% "
              f"{fmt(alpha, p)} {t:>8.2f} {p:>7.4f} {win:>5.1f}%{star}")

    # Exact score tiers (score == N)
    print(f"\n  {'Exact Tier':<12} {'n':>7} {'Events/Yr':>10} {'3d Mean':>9} "
          f"{'3d Alpha':>9} {'t-stat':>8} {'p-val':>7} {'Win%':>6}")
    print(f"  {'-'*70}")

    for score in [2, 3, 4, 5]:
        sig = df[df["score"] == score]["fwd_3d"].dropna()
        if len(sig) < 5:
            print(f"  Score =={score:<5} {len(sig):>7,} — insufficient")
            continue
        alpha, t, p = alpha_vs_base(sig, base)
        win = (sig > 0).mean() * 100
        evy = len(sig) / years
        star = "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  Score =={score:<5} {len(sig):>7,} {evy:>9.0f} {sig.mean():>+8.2f}% "
              f"{fmt(alpha, p)} {t:>8.2f} {p:>7.4f} {win:>5.1f}%{star}")

    print(f"\n  ** p<0.01   * p<0.05")


# ── Test 2: Hold window by score tier ─────────────────────────────────────────

def test_hold_windows(df):
    print(f"\n{'='*72}")
    print(f"TEST 2: OPTIMAL HOLD WINDOW BY SCORE TIER")
    print(f"{'='*72}")
    print(f"  Does Score 4 warrant a longer hold than Score 2/3?")
    print()

    windows = [1, 3, 5, 7, 10, 20]

    for score in [2, 3, 4]:
        sig  = df[df["score"] >= score]
        base = df
        n    = len(sig[sig["fwd_3d"].notna()])
        if n < 5:
            continue

        print(f"  === Score ≥ {score} (n={n:,}) ===")
        print(f"  {'Window':<8} {'Alpha':>9} {'Alpha/Day':>11} {'Net/Day':>10} "
              f"{'t-stat':>8} {'p-val':>7} {'Win%':>6} {'Sharpe':>8}")
        print(f"  {'-'*74}")

        best_net_per_day = -999
        best_window = 0

        for days in windows:
            col = f"fwd_{days}d"
            sig_r  = sig[col].dropna()
            base_r = base[col].dropna()
            if len(sig_r) < 5:
                continue

            alpha = sig_r.mean() - base_r.mean()
            cost  = sig["txn_cost"].mean()
            net   = alpha - cost
            apd   = alpha / days
            npd   = net / days
            t, p  = stats.ttest_ind(sig_r, base_r, equal_var=False)
            win   = (sig_r > 0).mean() * 100
            sharpe = (sig_r.mean() / sig_r.std() * np.sqrt(252 / days)
                      if sig_r.std() > 0 else 0)
            star = "**" if p < 0.01 else "*" if p < 0.05 else ""

            if npd > best_net_per_day:
                best_net_per_day = npd
                best_window = days

            print(f"  {days}d{'':<5} {fmt(alpha, p)} {apd:>+10.3f}% {npd:>+9.3f}% "
                  f"{t:>8.2f} {p:>7.4f} {win:>5.1f}% {sharpe:>7.2f} {star}")

        print(f"\n  >> Best net/day: {best_window}d hold\n")


# ── Test 3: Position sizing analysis ─────────────────────────────────────────

def test_position_sizing(df):
    print(f"\n{'='*72}")
    print(f"TEST 3: POSITION SIZING — IS CURRENT ALLOCATION APPROPRIATE?")
    print(f"{'='*72}")
    print(f"  Current: Score 2→5% ($250), Score 3→8% ($400), Score 4→12% ($600)")
    print(f"  Question: Does the alpha/risk ratio justify the step-ups?")
    print()

    base = df["fwd_3d"].dropna()
    years = (df["date"].max() - df["date"].min()).days / 365.25

    print(f"  {'Tier':<12} {'n/yr':>6} {'Alpha':>9} {'Std':>7} {'Sharpe':>8} "
          f"{'Win%':>6} {'Pos%':>6} {'$Pos':>6} {'Annual $':>9}")
    print(f"  {'-'*73}")

    for score in [2, 3, 4]:
        sig = df[df["score"] >= score]
        sig_r = sig["fwd_3d"].dropna()
        if len(sig_r) < 5:
            continue

        alpha, t, p = alpha_vs_base(sig_r, base)
        std  = sig_r.std()
        sharpe = (alpha / std * np.sqrt(252 / 3)) if std > 0 else 0
        win  = (sig_r > 0).mean() * 100
        nyr  = len(sig_r) / years
        pos_pct = POSITION_SIZE[score] * 100
        pos_usd = POSITION_SIZE[score] * PORTFOLIO
        # Annual dollar alpha: events/yr × alpha% × position size
        annual_alpha = nyr * (alpha / 100) * pos_usd
        cost = sig["txn_cost"].mean()
        annual_net = nyr * ((alpha - cost) / 100) * pos_usd

        print(f"  Score ≥{score:<5} {nyr:>5.0f} {fmt(alpha, p)} {std:>6.2f}% "
              f"{sharpe:>7.2f} {win:>5.1f}% {pos_pct:>5.0f}% ${pos_usd:>5.0f} "
              f"${annual_net:>+8.0f}/yr")

    # What if Score 4 got a bigger position?
    print(f"\n  === SCORE 4 SENSITIVITY: WHAT IF WE SIZE UP? ===")
    sig4  = df[df["score"] >= 4]
    sig4r = sig4["fwd_3d"].dropna()
    if len(sig4r) >= 5:
        alpha4, _, _ = alpha_vs_base(sig4r, base)
        cost4  = sig4["txn_cost"].mean()
        net4   = alpha4 - cost4
        nyr4   = len(sig4r) / years

        print(f"\n  {'Position %':<12} {'$ Size':>8} {'Annual Net $':>14} "
              f"{'vs Current':>12}")
        current_annual = nyr4 * (net4 / 100) * (POSITION_SIZE[4] * PORTFOLIO)

        for pct in [0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
            pos_usd   = pct * PORTFOLIO
            annual    = nyr4 * (net4 / 100) * pos_usd
            vs_curr   = annual - current_annual
            print(f"  {pct*100:.0f}%{'':<9} ${pos_usd:>7.0f} ${annual:>+13.0f} "
                  f"${vs_curr:>+11.0f}")

        print(f"\n  Note: Score 4 fires ~{nyr4:.0f}x/yr — low frequency limits "
              f"absolute dollar impact of sizing up")


# ── Test 4: Coin and time concentration ───────────────────────────────────────

def test_concentration(df):
    print(f"\n{'='*72}")
    print(f"TEST 4: CONCENTRATION CHECK — IS SCORE 4 FRAGILE?")
    print(f"{'='*72}")
    print(f"  Is Score 4 alpha concentrated in a few coins or time periods,")
    print(f"  or is it broad-based?")

    sig4 = df[df["score"] >= 4].copy()
    base = df["fwd_3d"].dropna()

    if len(sig4) < 5:
        print(f"  Insufficient Score 4 events ({len(sig4)})")
        return

    # By coin
    print(f"\n  === TOP COINS IN SCORE ≥ 4 SIGNALS ===")
    print(f"  {'Ticker':<8} {'n':>5} {'3d Mean':>9} {'3d Alpha':>9} {'Win%':>6}")
    print(f"  {'-'*42}")

    coin_stats = (sig4.groupby("ticker")
                      .apply(lambda g: pd.Series({
                          "n": g["fwd_3d"].notna().sum(),
                          "mean": g["fwd_3d"].dropna().mean(),
                          "win": (g["fwd_3d"].dropna() > 0).mean() * 100,
                      }))
                      .sort_values("n", ascending=False))

    total_events = coin_stats["n"].sum()
    for ticker, row in coin_stats.iterrows():
        if row["n"] < 2:
            continue
        coin_base = df[df["ticker"] == ticker]["fwd_3d"].dropna()
        alpha = row["mean"] - coin_base.mean() if len(coin_base) > 0 else float("nan")
        pct_of_total = row["n"] / total_events * 100
        print(f"  {ticker:<8} {row['n']:>5.0f} {row['mean']:>+8.2f}% "
              f"{alpha:>+8.2f}% {row['win']:>5.1f}%  ({pct_of_total:.1f}% of signals)")

    # By year
    print(f"\n  === SCORE ≥ 4 ALPHA BY YEAR ===")
    print(f"  {'Year':<6} {'n':>5} {'3d Alpha':>9} {'Win%':>6} {'Fired?':>8}")
    print(f"  {'-'*38}")

    for year in range(2018, 2027):
        yr_sig  = sig4[sig4["date"].dt.year == year]["fwd_3d"].dropna()
        yr_base = df[df["date"].dt.year == year]["fwd_3d"].dropna()
        if len(yr_base) < 5:
            continue
        if len(yr_sig) < 2:
            print(f"  {year:<6} {len(yr_sig):>5} — no signals")
            continue
        alpha, t, p = alpha_vs_base(yr_sig, yr_base)
        win = (yr_sig > 0).mean() * 100
        star = "**" if p < 0.01 else "*" if p < 0.05 else "  "
        print(f"  {year:<6} {len(yr_sig):>5} {fmt(alpha, p)} {win:>5.1f}%  ✓ {star}")

    # How many coins fire simultaneously on Score 4 days?
    print(f"\n  === SIMULTANEOUS POSITIONS ON SCORE 4 SIGNAL DAYS ===")
    daily_counts = sig4.groupby("date").size()
    print(f"  Signal days: {len(daily_counts)}")
    print(f"  Avg coins per day: {daily_counts.mean():.1f}")
    print(f"  Max coins per day: {daily_counts.max()}")
    print(f"  Days with 1 coin:  {(daily_counts == 1).sum()}")
    print(f"  Days with 2+ coins: {(daily_counts >= 2).sum()}")


# ── Test 5: Score 4 component breakdown ───────────────────────────────────────

def test_components(df):
    print(f"\n{'='*72}")
    print(f"TEST 5: WHAT TRIGGERS SCORE 4?")
    print(f"{'='*72}")
    print(f"  Score 4 requires multiple simultaneous conditions.")
    print(f"  Breaking down which component combinations fire most.")

    sig4 = df[df["score"] >= 4].copy()
    base = df["fwd_3d"].dropna()

    if len(sig4) < 5:
        print(f"  Insufficient data")
        return

    # Minimum required for score 4:
    # Severe crash (+3) + capitulation (+1) = 4
    # Severe crash (+3) + mild crash impossible (same condition)
    # Severe crash (+3) + extreme greed (+2) = 5 (captured above)
    # Mild crash (+1) + extreme greed (+2) + cap (+1) = 4

    combos = {
        "Severe crash (≥-25%) alone":
            sig4["comp_severe_crash"] & ~sig4["comp_ext_greed"] & ~sig4["comp_capitulation"],
        "Severe crash + capitulation":
            sig4["comp_severe_crash"] & sig4["comp_capitulation"],
        "Severe crash + extreme greed":
            sig4["comp_severe_crash"] & sig4["comp_ext_greed"],
        "Mild crash + extreme greed + cap":
            sig4["comp_mild_crash"] & sig4["comp_ext_greed"] & sig4["comp_capitulation"],
        "Extreme greed dominated":
            sig4["comp_ext_greed"] & ~sig4["comp_severe_crash"],
    }

    print(f"\n  {'Combo':<40} {'n':>5} {'3d Alpha':>9} {'Win%':>6}")
    print(f"  {'-'*64}")

    for label, mask in combos.items():
        sub = sig4[mask]["fwd_3d"].dropna()
        if len(sub) < 3:
            print(f"  {label:<40} {len(sub):>5} — insufficient")
            continue
        alpha, t, p = alpha_vs_base(sub, base)
        win = (sub > 0).mean() * 100
        print(f"  {label:<40} {len(sub):>5} {fmt(alpha, p)} {win:>5.1f}%")

    # FG value distribution in Score 4 signals
    print(f"\n  === FEAR & GREED INDEX WHEN SCORE ≥ 4 FIRES ===")
    fg_dist = sig4["fg_value"].describe()
    print(f"  Min: {fg_dist['min']:.0f}  |  25th: {fg_dist['25%']:.0f}  |  "
          f"Median: {fg_dist['50%']:.0f}  |  75th: {fg_dist['75%']:.0f}  |  "
          f"Max: {fg_dist['max']:.0f}")
    print(f"  Extreme Fear (≤10): {(sig4['fg_value'] <= 10).sum()} signals "
          f"({(sig4['fg_value'] <= 10).mean()*100:.0f}%)")
    print(f"  Extreme Greed (≥90): {(sig4['fg_value'] >= 90).sum()} signals "
          f"({(sig4['fg_value'] >= 90).mean()*100:.0f}%)")

    # Prior 5d return distribution
    print(f"\n  === PRIOR 5-DAY RETURN WHEN SCORE ≥ 4 FIRES ===")
    ret_dist = sig4["ret_5d"].describe()
    print(f"  Min: {ret_dist['min']:+.1f}%  |  25th: {ret_dist['25%']:+.1f}%  |  "
          f"Median: {ret_dist['50%']:+.1f}%  |  75th: {ret_dist['75%']:+.1f}%  |  "
          f"Max: {ret_dist['max']:+.1f}%")


# ── Verdict ───────────────────────────────────────────────────────────────────

def print_verdict(df):
    print(f"\n{'='*72}")
    print(f"VERDICT: SCORE TIER RECOMMENDATIONS")
    print(f"{'='*72}")

    base  = df["fwd_3d"].dropna()
    years = (df["date"].max() - df["date"].min()).days / 365.25

    for score in [2, 3, 4]:
        sig   = df[df["score"] >= score]
        sig_r = sig["fwd_3d"].dropna()
        if len(sig_r) < 5:
            continue
        alpha, t, p = alpha_vs_base(sig_r, base)
        cost  = sig["txn_cost"].mean()
        net   = alpha - cost
        nyr   = len(sig_r) / years
        pos   = POSITION_SIZE[score]
        pos_usd = pos * PORTFOLIO
        annual = nyr * (net / 100) * pos_usd

        print(f"\n  Score ≥ {score}: alpha {alpha:+.2f}%, net {net:+.2f}%, "
              f"{nyr:.0f}/yr, pos {pos*100:.0f}% (${pos_usd:.0f}), "
              f"annual net ${annual:+.0f}")

    print(f"""
  RECOMMENDATIONS:

  Score 2 (5% / $250):
    → KEEP as-is. Solid signal, high frequency (~{len(df[df['score']>=2])/years:.0f}/yr),
      generates the bulk of annual alpha from volume alone.

  Score 3 (8% / $400):
    → KEEP as-is. Clear step-up in alpha and win rate over Score 2.

  Score 4 (12% / $600):
    → Review the concentration check (Test 4).
    → If alpha is broad-based (not 1-2 coins driving it), consider sizing up
      to 15-20%. Low frequency limits absolute impact of any change.
    → Hold window: check Test 2. If 5d shows significantly more alpha/day
      than 3d for Score 4, a tier-specific hold extension may be warranted.
    → DO NOT change until you have 10+ live Score 4 signals to validate.

  Bottom line: The scoring system is working as designed. Score 4 events
  are rare but powerful. The bigger lever is ensuring the system stays live
  and catches every signal — not over-optimizing the sizing.
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score tier deep-dive analysis")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output:
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

    print(f"{'='*72}")
    print(f"  SCORE TIER ANALYSIS — Phase 2C")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*72}")

    df = load_and_prepare()
    df = compute_scores(df)

    print(f"\nTotal rows: {len(df):,}")
    for s in [2, 3, 4, 5]:
        n = (df["score"] >= s).sum()
        print(f"  Score ≥ {s}: {n:,} events")

    test_sample_size(df)
    test_hold_windows(df)
    test_position_sizing(df)
    test_concentration(df)
    test_components(df)
    print_verdict(df)

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
