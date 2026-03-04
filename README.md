# Crypto Signal Scanner

Daily signal scanner for cryptocurrency markets using a composite scoring system validated across 12 years of data (2014–2026). Identifies high-probability long entries based on crash signals, sentiment extremes, and volume anomalies.

## Strategy Summary

| Signal | Score | Backtest Alpha (5d) | Win Rate |
|--------|-------|-------------------|----------|
| Score ≥ 2 (BUY) | +2 | +6.90% | 57.7% |
| Score ≥ 3 (STRONG) | +3 | +8.72% | 58.9% |
| Score ≥ 4 (VERY STRONG) | +4 | +19.51% | 67.9% |

- **Long-only**, 3–5 day hold period
- **Out-of-sample validated**: +7.27% alpha (2022–2026), significant every year
- **40-coin universe**: BTC, ETH, Top 20, select altcoins

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

## Project Structure

```
crypto_backtest/
├── crypto_scanner.py      # Live daily scanner with email alerts
├── collect_data.py        # Historical data collection (yfinance, F&G)
├── backtest_engine.py     # Phase 2: individual signal backtester
├── signal_dedup.py        # Phase 3: signal overlap & combined scoring
├── signal_refine.py       # Phase 3b: holding period, OOS, risk analysis
├── database.py            # SQLite schema and utilities
├── config_example.py      # Configuration template
└── config.py              # Local credentials (not tracked)
```

## Setup

### 1. Clone and configure

```bash
git clone https://github.com/yourusername/crypto-signal-scanner.git
cd crypto-signal-scanner
cp config_example.py config.py
```

Edit `config.py` with your Gmail App Password for email alerts.

### 2. Install dependencies

```bash
pip install yfinance pandas numpy scipy requests
```

### 3. Build the database

```bash
python3 collect_data.py --all
```

Pulls full price history (yfinance) and Fear & Greed Index (alternative.me). No paid API keys required.

### 4. Run the scanner

```bash
python3 crypto_scanner.py              # Full run: update + scan + email
python3 crypto_scanner.py --scan-only  # Scan existing data only
python3 crypto_scanner.py --no-email   # Run without sending email
```

### 5. Run backtests (optional)

```bash
python3 backtest_engine.py    # Individual signal tests
python3 signal_dedup.py       # Signal overlap & combined scoring
python3 signal_refine.py      # Holding period, OOS validation, risk
```

## Backtest Methodology

- **Data**: 83,615 daily observations across 40 coins (2014–2026)
- **Winsorization**: 1st/99th percentile to limit outlier influence
- **Transaction costs**: BTC/ETH 0.20%, Top 20 0.40%, Altcoins 0.80%
- **Walk-forward validation**: Trained on 2014–2021, tested on 2022–2026
- **Signal independence**: Jaccard similarity and conditional alpha tests confirm scoring components provide additive value

## Key Results

- **4,650 signal events** over 12 years (~406/year)
- **Out-of-sample alpha exceeds in-sample** (+7.27% vs +5.81%)
- **Works in every market regime**: bear (+2.95% to +8.14%), bull (+4.09% to +6.89%), sideways (+7.22% to +9.46%)
- **BTC-only subset**: +4.57% alpha, 66.7% win rate, p=0.0003

## Data Sources

| Source | Data | Cost |
|--------|------|------|
| Yahoo Finance (yfinance) | Daily OHLCV prices | Free |
| Alternative.me | Crypto Fear & Greed Index | Free |
| Binance | Perpetual funding rates | Free |

## Deployment

Designed for PythonAnywhere scheduled tasks. Set daily run at 00:30 UTC:

```
cd /path/to/crypto_backtest && python3 crypto_scanner.py
```

## Disclaimer

This project is for educational and research purposes only. Past backtest performance does not guarantee future results. Cryptocurrency trading involves substantial risk of loss. Always do your own research before making investment decisions.
