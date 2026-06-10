import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Load .env file if present (so `export` in shell is not required)
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        # strip optional leading "export "
        if _line.startswith("export "):
            _line = _line[7:]
        _k, _, _v = _line.partition("=")
        # strip surrounding quotes (" or ')
        _v = _v.strip().strip('"').strip("'")
        os.environ[_k.strip()] = _v  # .env takes priority over shell env

# --- Kalshi credentials (set via env vars or .env file) ---
KALSHI_KEY_ID = os.environ.get("KALSHI_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", str(BASE_DIR / "config" / "kalshi_private.pem"))

# --- Environment: "demo" or "prod" ---
KALSHI_ENV = os.environ.get("KALSHI_ENV", "demo")

KALSHI_BASE_URLS = {
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
    "prod": "https://api.elections.kalshi.com/trade-api/v2",
}
KALSHI_WS_URLS = {
    "demo": "wss://demo-api.kalshi.co/trade-api/ws/v2",
    "prod": "wss://api.elections.kalshi.com/trade-api/ws/v2",
}

KALSHI_BASE_URL = KALSHI_BASE_URLS[KALSHI_ENV]
KALSHI_WS_URL = KALSHI_WS_URLS[KALSHI_ENV]

# --- BTC price feed (Binance public, no auth needed) ---
BINANCE_KLINE_URL = "https://api.binance.com/api/v3/klines"
BTC_SYMBOL = "BTCUSDT"

# --- Strategy parameters ---
# Kalshi BTC market series ticker prefix (e.g. "KXBTC")
BTC_SERIES_TICKER = os.environ.get("BTC_SERIES_TICKER", "KXBTC")

# How many past hourly candles to use for vol estimation
LOOKBACK_CANDLES = 24

# Only trade YES side (betting BTC stays above strike)
YES_ONLY = True

# Minimum model probability to place a YES trade (high-confidence filter)
MIN_PROB_YES = 0.99

# Minimum edge over market price before placing (model prob - ask price)
MIN_EDGE = 0.03  # 3 cents — tighter since we're already filtering by prob

# BTC must be at least this many dollars above the strike — avoids too-close bets
MIN_BUFFER_USD = 200

# Fixed dollar bet per trade — small, consistent sizing
BET_SIZE_USD = 10.0

# Never spend more than this fraction of balance in a single trade
MAX_BALANCE_FRACTION = 0.10

# Minimum contract quantity to bother placing
MIN_CONTRACTS = 1

# How many minutes before the hour-end to place trades
TRADE_LEAD_MINUTES = 5
