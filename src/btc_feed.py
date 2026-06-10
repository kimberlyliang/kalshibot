"""BTC price feed: hourly candles (Binance → Coinbase fallback) + BRTI approximation."""
from pathlib import Path
import logging
import statistics

import httpx
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import BINANCE_KLINE_URL, BTC_SYMBOL, LOOKBACK_CANDLES

log = logging.getLogger(__name__)


# ── BRTI approximation ────────────────────────────────────────────────────────
# CF Benchmarks BRTI is a volume-weighted median across these constituent venues.
# We pull spot prices from each and take the median — typically within $5-10 of BRTI.

def _price_coinbase() -> float:
    r = httpx.get("https://api.exchange.coinbase.com/products/BTC-USD/ticker", timeout=5)
    r.raise_for_status()
    return float(r.json()["price"])


def _price_kraken() -> float:
    r = httpx.get("https://api.kraken.com/0/public/Ticker", params={"pair": "XBTUSD"}, timeout=5)
    r.raise_for_status()
    data = r.json()
    # Kraken response: {"result": {"XXBTZUSD": {"c": ["price", ...], ...}}}
    pair = next(iter(data["result"].values()))
    return float(pair["c"][0])


def _price_bitstamp() -> float:
    r = httpx.get("https://www.bitstamp.net/api/v2/ticker/btcusd/", timeout=5)
    r.raise_for_status()
    return float(r.json()["last"])


def _price_gemini() -> float:
    r = httpx.get("https://api.gemini.com/v1/pubticker/btcusd", timeout=5)
    r.raise_for_status()
    return float(r.json()["last"])


def _price_binance() -> float:
    r = httpx.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": BTC_SYMBOL}, timeout=5)
    r.raise_for_status()
    return float(r.json()["price"])


_BRTI_SOURCES = {
    "coinbase": _price_coinbase,
    "kraken":   _price_kraken,
    "bitstamp": _price_bitstamp,
    "gemini":   _price_gemini,
    # Binance blocks US IPs (HTTP 451) — excluded
}


def brti_approx() -> dict:
    """
    Fetch spot prices from all BRTI constituent exchanges and return the median.
    Silently drops any exchange that errors so a single outage doesn't break the bot.

    Returns dict: {price, sources_used, individual_prices}
    """
    prices = {}
    for name, fetch in _BRTI_SOURCES.items():
        try:
            prices[name] = fetch()
        except Exception as e:
            log.warning(f"BRTI source {name} failed: {e}")

    if not prices:
        raise RuntimeError("All BRTI price sources failed")

    median = statistics.median(prices.values())
    log.info(f"BRTI approx: ${median:,.2f}  (from {list(prices.keys())}  individual={prices})")
    return {
        "price": median,
        "sources_used": list(prices.keys()),
        "individual_prices": prices,
    }


def current_btc_price() -> float:
    """Best available spot price: BRTI approximation (median of up to 5 exchanges)."""
    return brti_approx()["price"]


# ── Hourly candles ────────────────────────────────────────────────────────────

def fetch_hourly_candles(n: int = LOOKBACK_CANDLES) -> pd.DataFrame:
    """Return the last `n` completed 1-hour BTC/USDT candles as a DataFrame."""
    params = {
        "symbol": BTC_SYMBOL,
        "interval": "1h",
        "limit": n + 1,  # +1 so we can drop the in-progress candle
    }
    try:
        r = httpx.get(BINANCE_KLINE_URL, params=params, timeout=10)
        r.raise_for_status()
        raw = r.json()[:-1]  # drop last (incomplete) candle
    except httpx.HTTPStatusError as e:
        status = getattr(e.response, "status_code", None)
        if status == 451:
            log.warning("Binance returned 451 Unavailable For Legal Reasons — falling back to Coinbase")
            return _fetch_hourly_candles_coinbase(n)
        raise

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


def _fetch_hourly_candles_coinbase(n: int = LOOKBACK_CANDLES) -> pd.DataFrame:
    """Fallback to Coinbase candles when Binance is unavailable."""
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    params = {"granularity": 3600, "limit": n + 1}
    r = httpx.get(url, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()

    df = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df[:-1]  # drop in-progress candle

    df["open_time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["close_time"] = df["open_time"] + pd.Timedelta(hours=1)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df[["open_time", "close_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
