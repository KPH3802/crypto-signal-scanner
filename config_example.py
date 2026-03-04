"""
Crypto Backtest System - Configuration
Copy this to config.py and fill in your credentials.
"""

# === Email Settings (for future scanner alerts) ===
EMAIL_ADDRESS = "your_email@gmail.com"
EMAIL_APP_PASSWORD = "your_app_password"
ALERT_RECIPIENT = "your_email@gmail.com"

# === API Keys ===
# CoinGecko: REQUIRED (free). Sign up at https://www.coingecko.com/en/api/pricing
# Click "Create Free Account" → Developer Dashboard → "+ Add New Key"
COINGECKO_API_KEY = ""  # Paste your demo API key here
GLASSNODE_API_KEY = ""  # Free tier: register at studio.glassnode.com

# === Database ===
DB_PATH = "crypto_backtest.db"

# === Rate Limits ===
COINGECKO_CALLS_PER_MIN = 25  # Conservative (limit is 30)
COINGECKO_DELAY = 60 / 25     # ~2.4 seconds between calls

# === Coin Universe ===
# Bucket 1: Blue chips
BUCKET_1_BLUE_CHIPS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
}

# Bucket 2: Top 20 (excluding BTC/ETH) - will be validated against current top 20
BUCKET_2_TOP20 = {
    "binancecoin": "BNB",
    "solana": "SOL",
    "ripple": "XRP",
    "cardano": "ADA",
    "dogecoin": "DOGE",
    "tron": "TRX",
    "chainlink": "LINK",
    "avalanche-2": "AVAX",
    "polkadot": "DOT",
    "polygon-ecosystem-token": "POL",
    "shiba-inu": "SHIB",
    "toncoin": "TON",
    "litecoin": "LTC",
    "uniswap": "UNI",
    "stellar": "XLM",
    "near": "NEAR",
    "sui": "SUI",
    "internet-computer": "ICP",
}

# Bucket 3: Altcoins / smaller projects with trading history
BUCKET_3_ALTCOINS = {
    "aave": "AAVE",
    "arbitrum": "ARB",
    "optimism": "OP",
    "render-token": "RNDR",
    "injective-protocol": "INJ",
    "sei-network": "SEI",
    "celestia": "TIA",
    "jupiter-exchange-solana": "JUP",
    "bonk": "BONK",
    "pepe": "PEPE",
    "floki": "FLOKI",
    "worldcoin-wld": "WLD",
    "kaspa": "KAS",
    "stacks": "STX",
    "maker": "MKR",
    "the-graph": "GRT",
    "immutable-x": "IMX",
    "gala": "GALA",
    "fetch-ai": "FET",
    "pendle": "PENDLE",
}

# Combined for easy iteration
def get_all_coins():
    """Return dict of all coins: {coingecko_id: ticker}"""
    all_coins = {}
    all_coins.update(BUCKET_1_BLUE_CHIPS)
    all_coins.update(BUCKET_2_TOP20)
    all_coins.update(BUCKET_3_ALTCOINS)
    return all_coins

def get_bucket_for_coin(coingecko_id):
    """Return which bucket a coin belongs to."""
    if coingecko_id in BUCKET_1_BLUE_CHIPS:
        return 1
    elif coingecko_id in BUCKET_2_TOP20:
        return 2
    elif coingecko_id in BUCKET_3_ALTCOINS:
        return 3
    return None
