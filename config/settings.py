import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# --- Kalshi credentials (set via env vars, never hardcode) ---
KALSHI_KEY_ID = os.environ.get("KALSHI_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", str(BASE_DIR / "config" / "kalshi_private.pem"))

# --- Environment: "demo" or "prod" ---
KALSHI_ENV = os.environ.get("KALSHI_ENV", "demo")

KALSHI_BASE_URLS = {
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
    "prod": "https://trading-api.kalshi.com/trade-api/v2",
}
KALSHI_WS_URLS = {
    "demo": "wss://demo-api.kalshi.co/trade-api/ws/v2",
    "prod": "wss://trading-api.kalshi.com/trade-api/ws/v2",
}

KALSHI_BASE_URL = KALSHI_BASE_URLS[KALSHI_ENV]
KALSHI_WS_URL = KALSHI_WS_URLS[KALSHI_ENV]

# --- BTC price feed (Binance public, no auth needed) ---
BINANCE_KLINE_URL = "https://api.binance.com/api/v3/klines"
BTC_SYMBOL = "BTCUSDT"

# --- Strategy parameters ---
# Kalshi BTC market series ticker prefix (e.g. "KXBTC")
BTC_SERIES_TICKER = os.environ.get("BTC_SERIES_TICKER", "KXBTC")

# How many past hourly candles to use for feature engineering
LOOKBACK_CANDLES = 24

# Minimum edge (predicted prob - market price) before placing a trade
MIN_EDGE = 0.04  # 4 cents on a $1 contract

# Risk: max fraction of balance to bet on a single trade (Kelly multiplier)
KELLY_FRACTION = 0.10  # 10% of Kelly — very conservative

# Max $ risk per trade (hard cap regardless of Kelly)
MAX_TRADE_USD = 50.0

# Minimum contract quantity to bother placing
MIN_CONTRACTS = 1

# How many minutes before the hour-end to place trades
TRADE_LEAD_MINUTES = 5
