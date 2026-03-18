# Crypto Backtest System — Phase 1: Data Collection

## What This Does
Collects historical data from free APIs into a SQLite database for crypto signal backtesting.

**40 coins across 3 buckets:**
- Bucket 1 (Blue Chips): BTC, ETH
- Bucket 2 (Top 20): BNB, SOL, XRP, ADA, DOGE, TRX, LINK, AVAX, DOT, POL, SHIB, TON, LTC, UNI, XLM, NEAR, SUI, ICP
- Bucket 3 (Altcoins): AAVE, ARB, OP, RNDR, INJ, SEI, TIA, JUP, BONK, PEPE, FLOKI, WLD, KAS, STX, MKR, GRT, IMX, GALA, FET, PENDLE

**3 data sources:**
1. **CoinGecko** — Daily prices, market cap, volume. Free tier, 30 calls/min. Full history back to coin launch.
2. **Alternative.me** — Crypto Fear & Greed Index. Free, no key needed. Daily since Feb 2018.
3. **Binance** — Perpetual futures funding rates (every 8 hours). Free, no key needed. 23 symbols.

## Setup

```bash
# 1. Create project directory
mkdir -p ~/Desktop/Claude_Programs/Trading_Programs/crypto_backtest
cd ~/Desktop/Claude_Programs/Trading_Programs/crypto_backtest

# 2. Copy all files into this directory

# 3. Copy config_example.py to config.py
cp config_example.py config.py

# 4. Get your FREE CoinGecko API key:
#    Go to https://www.coingecko.com/en/api/pricing
#    Click "Create Free Account" → Developer Dashboard → "+ Add New Key"
#    Paste the key into config.py on the COINGECKO_API_KEY line

# 5. Edit config.py with your email credentials (for future alerts)

# 6. Install dependencies (just requests, likely already installed)
pip3 install requests
```

## Running

```bash
cd ~/Desktop/Claude_Programs/Trading_Programs/crypto_backtest

# Collect everything (prices + fear/greed + funding rates)
# Prices: ~40 coins × ~2.4 sec each = ~2 minutes
# Funding: ~23 symbols = ~2 minutes
# Total: ~5 minutes first run
python3 collect_data.py --all

# Or collect individually:
python3 collect_data.py --prices        # CoinGecko daily prices only
python3 collect_data.py --fear-greed    # Fear & Greed only
python3 collect_data.py --funding       # Binance funding rates only

# Check what's in the database:
python3 collect_data.py --stats
```

## Incremental Updates
The collector is incremental — running it again only fetches new data since the last collection. Safe to run daily.

## Expected Output (First Run)
```
DAILY PRICES: ~50,000-80,000 records (varies by coin age)
FEAR & GREED: ~2,500+ days (since Feb 2018)
FUNDING RATES: ~50,000+ records (since Binance perpetuals launch)
```

## File Structure
```
crypto_backtest/
├── config_example.py      # Configuration template (safe for GitHub)
├── config.py              # Your actual config (DO NOT commit)
├── database.py            # Database schema and stats
├── collect_data.py        # Main data collector
├── crypto_backtest.db     # SQLite database (created on first run)
└── README.md              # This file
```

## What's Next (Phase 2)
Once data is collected, we build the backtest engine to test:
- Momentum / mean reversion signals by bucket
- Fear & Greed extreme signals (buy at Extreme Fear?)
- Funding rate extremes as contrarian signals
- Cross-signal combinations

Same methodology as your equity backtests: define signal, measure forward returns at 3d/5d/10d/20d, calculate alpha vs. buy-and-hold, statistical significance.
