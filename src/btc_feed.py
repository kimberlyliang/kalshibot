"""Pull BTC/USDT candlestick data from Binance (public, no auth)."""
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import BINANCE_KLINE_URL, BTC_SYMBOL, LOOKBACK_CANDLES


def fetch_hourly_candles(n: int = LOOKBACK_CANDLES) -> pd.DataFrame:
    """Return the last `n` completed 1-hour BTC/USDT candles as a DataFrame."""
    params = {
        "symbol": BTC_SYMBOL,
        "interval": "1h",
        "limit": n + 1,  # +1 so we can drop the in-progress candle
    }
    r = httpx.get(BINANCE_KLINE_URL, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()[:-1]  # drop last (incomplete) candle

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df[["open_time", "close_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def current_btc_price() -> float:
    """Spot price right now."""
    r = httpx.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": BTC_SYMBOL}, timeout=5)
    r.raise_for_status()
    return float(r.json()["price"])
