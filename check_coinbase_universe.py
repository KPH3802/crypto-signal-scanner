#!/usr/bin/env python3
"""
check_coinbase_universe.py
==========================
Pulls all tradable USD pairs from Coinbase Advanced Trade and
cross-references against our 40 tested coins.

Usage:
    python3 check_coinbase_universe.py

Output:
    - Coins available on Coinbase (can trade)
    - Coins NOT available (excluded from auto-trader)
"""

import json
import time
import hashlib
import hmac
import os
from datetime import datetime

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system("pip3 install requests --break-system-packages")
    import requests

# ── Load CDP API key ──────────────────────────────────────────────────────────
KEY_FILE = os.path.join(os.path.dirname(__file__), "cdp_api_key.json")

with open(KEY_FILE) as f:
    cdp = json.load(f)

API_KEY_NAME   = cdp["name"]
PRIVATE_KEY_PEM = cdp["privateKey"]

# ── JWT auth for Coinbase Advanced Trade ─────────────────────────────────────
try:
    import jwt as pyjwt
except ImportError:
    os.system("pip3 install PyJWT cryptography")
    import jwt as pyjwt

from cryptography.hazmat.primitives.serialization import load_pem_private_key

def build_jwt(method, path):
    """Build a short-lived JWT for Coinbase CDP auth."""
    private_key = load_pem_private_key(PRIVATE_KEY_PEM.encode(), password=None)
    now = int(time.time())
    payload = {
        "sub": API_KEY_NAME,
        "iss": "cdp",
        "nbf": now,
        "exp": now + 120,
        "uri": f"{method} api.coinbase.com{path}",
    }
    token = pyjwt.encode(
        payload,
        private_key,
        algorithm="ES256",
        headers={"kid": API_KEY_NAME, "nonce": str(now)},
    )
    return token

def coinbase_get(path):
    token = build_jwt("GET", path)
    url = f"https://api.coinbase.com{path}"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=15)
    r.raise_for_status()
    return r.json()

# ── Our tested universe ───────────────────────────────────────────────────────
# Tickers from config.py (all 40 coins)
TESTED_TICKERS = {
    "BTC", "ETH",                                                          # Bucket 1
    "BNB", "SOL", "XRP", "ADA", "DOGE", "TRX", "LINK", "AVAX",           # Bucket 2
    "DOT", "POL", "SHIB", "TON", "LTC", "UNI", "XLM", "NEAR",
    "SUI", "ICP",
    "AAVE", "ARB", "OP", "RNDR", "INJ", "SEI", "TIA", "JUP",             # Bucket 3
    "BONK", "PEPE", "FLOKI", "WLD", "KAS", "STX", "MKR", "GRT",
    "IMX", "GALA", "FET", "PENDLE",
}

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("COINBASE UNIVERSE CHECK")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\nFetching tradable products from Coinbase Advanced Trade...")
    data = coinbase_get("/api/v3/brokerage/products")
    products = data.get("products", [])

    # Keep only active USD spot pairs
    usd_pairs = [
        p for p in products
        if p.get("quote_currency_id") == "USD"
        and p.get("product_type") == "SPOT"
        and not p.get("is_disabled", False)
        and not p.get("trading_disabled", False)
    ]

    coinbase_tickers = {p["base_currency_id"] for p in usd_pairs}

    # Cross-reference
    tradable     = sorted(TESTED_TICKERS & coinbase_tickers)
    not_tradable = sorted(TESTED_TICKERS - coinbase_tickers)

    print(f"\nTotal active USD spot pairs on Coinbase: {len(usd_pairs)}")
    print(f"Our tested universe: {len(TESTED_TICKERS)} coins")

    print(f"\n✅ TRADABLE on Coinbase ({len(tradable)} coins):")
    print("   " + ", ".join(tradable))

    print(f"\n❌ NOT available on Coinbase ({len(not_tradable)} coins):")
    if not_tradable:
        print("   " + ", ".join(not_tradable))
    else:
        print("   None — full universe available!")

    print("\n" + "=" * 60)
    print(f"AUTO-TRADER UNIVERSE: {len(tradable)} coins")
    print("=" * 60)

if __name__ == "__main__":
    main()
