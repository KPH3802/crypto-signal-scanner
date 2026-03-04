# Crypto Signal Scanner & Auto-Trader

Daily signal scanner and automated execution system for cryptocurrency markets using a composite scoring system validated across 12 years of data (2014–2026). Identifies high-probability long entries based on crash signals, sentiment extremes, and volume anomalies — and executes trades automatically via Coinbase Advanced Trade API.

## Strategy Summary

| Signal | Score | Backtest Alpha (5d) | Win Rate |
|--------|-------|-------------------|----------|
| Score ≥ 2 (BUY) | +2 | +6.90% | 57.7% |
| Score ≥ 3 (STRONG) | +3 | +8.72% | 58.9% |
| Score ≥ 4 (VERY STRONG) | +4 | +19.51% | 67.9% |

- **Long-only**, 3-day hold period
- **Out-of-sample validated**: +7.27% alpha (2022–2026), significant every year
- **33-coin tradable universe**: Coinbase-listed coins across 3 buckets

## Scoring System

| Component | Points | Condition |
|-----------|--------|-----------|
| Severe crash | +3 | 5-day drawdown ≥ 25% |
| Extreme Greed | +2 | Fear & Greed Index ≥ 90 |
| Moderate crash | +1 | 5-day drawdown 15–25% |
| Greed zone | +1 | Fear & Greed 75–89 |
| Capitulation | +1 | 3x volume + 5%+ daily drop |
| Low volatility | -1 | Coin in own bottom 10th percentile |
| Fear exit | -2 | Fear & Greed crosses above 25 |

## Auto-Trader: Historical Simulation Results

Simulated performance starting with $5,000, running the full execution system against historical data:

| Period | Return | BTC Buy-and-Hold | Alpha vs BTC | Win Rate | Max DD |
|--------|--------|-----------------|--------------|----------|--------|
| YTD (Jan–Mar 2026, 61d) | +2.93% | -22.49% | +25.42% | 66.7% | -2.00% |
| 6 Month (Sep 2025–Mar 2026, 182d) | +5.75% | -38.15% | +43.90% | 68.4% | -4.02% |
| 1 Year (Mar 2025–Mar 2026, 366d) | +16.07% | -27.03% | +43.10% | 67.9% | -30.25% |

**Win rate is consistent across all three windows (67–68%).**

## Position Sizing

| Score | Position Size | $ on $5K |
|-------|-------------|----------|
| 2 | 5% | $250 |
| 3 | 8% | $400 |
| 4 | 12% | $600 |

- Max 6 concurrent positions
- Minimum 40% cash reserve at all times
- Market orders via Coinbase Advanced Trade API

## Coin Universe (33 Coins — Coinbase Tradable)

**Large Cap (2):** BTC, ETH

**Mid Cap (16):** BNB, SOL, XRP, ADA, DOGE, LINK, AVAX, DOT, SHIB, LTC, UNI, XLM, NEAR, SUI, ICP, POL

**Smaller Alt (15):** AAVE, ARB, OP, INJ, SEI, TIA, BONK, PEPE, FLOKI, WLD, STX, GRT, IMX, FET, PENDLE

## Project Structure

```
crypto_backtest/
├── crypto_scanner.py        # Live daily scanner with email alerts (00:30 UTC)
├── crypto_autotrader.py     # Automated execution via Coinbase API (00:45/01:00 UTC)
├── backtest_autotrader.py   # Historical simulation of full auto-trader system
├── collect_data.py          # Historical data collection (yfinance, F&G)
├── backtest_engine.py       # Individual signal backtester
├── signal_dedup.py          # Signal overlap & combined scoring
├── signal_refine.py         # Holding period, OOS, risk analysis
├── check_coinbase_universe.py # Validates tradable coin universe via CDP API
├── database.py              # SQLite schema and utilities
├── config_example.py        # Configuration template (copy to config.py)
└── config.py                # Local credentials — NOT tracked
```

## Setup

### 1. Clone and configure

```bash
git clone https://github.com/KPH3802/crypto-signal-scanner.git
cd crypto-signal-scanner
cp config_example.py config.py
```

Edit `config.py` with your Gmail App Password and email address.

### 2. Install dependencies

```bash
pip install yfinance pandas numpy scipy requests PyJWT cryptography
```

### 3. Build the database

```bash
python3 collect_data.py --all
```

Pulls full price history (yfinance) and Fear & Greed Index (alternative.me). No paid API keys required for the scanner.

### 4. Run the scanner

```bash
python3 crypto_scanner.py              # Full run: update + scan + email
python3 crypto_scanner.py --scan-only  # Scan existing data only
python3 crypto_scanner.py --no-email   # Run without sending email
```

### 5. Run the auto-trader (requires Coinbase CDP API key)

```bash
python3 crypto_autotrader.py --dry-run --check-entries   # Simulate entries (no real trades)
python3 crypto_autotrader.py --dry-run --check-exits     # Simulate exits (no real trades)
python3 crypto_autotrader.py --check-entries             # Live entry execution
python3 crypto_autotrader.py --check-exits               # Live exit execution
python3 crypto_autotrader.py --status                    # Show open positions + P&L
```

Place `cdp_api_key.json` (Coinbase Developer Platform key with trade + view permissions) in the project root. This file is excluded from version control.

### 6. Run historical simulation

```bash
python3 backtest_autotrader.py                              # Default: Mar 2025 – Mar 2026
python3 backtest_autotrader.py --start 2026-01-01 --end 2026-03-02
```

### 7. Deploy to PythonAnywhere (scheduled tasks)

```
00:30 UTC — python3 crypto_scanner.py
00:45 UTC — python3 crypto_autotrader.py --check-entries
01:00 UTC — python3 crypto_autotrader.py --check-exits
```

## Backtest Methodology

- **Data**: 83,615+ daily observations across 40 coins (2014–2026)
- **Winsorization**: 1st/99th percentile to limit outlier influence
- **Transaction costs**: BTC/ETH 0.20%, Top 20 0.40%, Altcoins 0.80%
- **Walk-forward validation**: Trained on 2014–2021, tested on 2022–2026
- **Signal independence**: Jaccard similarity and conditional alpha tests confirm scoring components provide additive value

## Key Signal Results

- **4,650 signal events** over 12 years (~406/year)
- **Out-of-sample alpha exceeds in-sample** (+7.27% vs +5.81%)
- **Works in every market regime**: bear (+2.95% to +8.14%), bull (+4.09% to +6.89%), sideways (+7.22% to +9.46%)

## Data Sources

| Source | Data | Cost |
|--------|------|------|
| Yahoo Finance (yfinance) | Daily OHLCV prices | Free |
| Alternative.me | Crypto Fear & Greed Index | Free |
| Coinbase Advanced Trade API | Live order execution | Free (CDP) |

## Disclaimer

This project is for educational and research purposes only. Past backtest performance does not guarantee future results. Cryptocurrency trading involves substantial risk of loss. Always do your own research before making investment decisions.

---

MIT License
