import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from finta import TA
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config.settings  # noqa: F401 — loads .env into os.environ

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # FinTA expects these lowercase column names:
    # open, high, low, close, volume

    df = df.copy()

    df["rsi"] = TA.RSI(df, period=14)
    df["sma_20"] = TA.SMA(df, period=20)
    df["ema_20"] = TA.EMA(df, period=20)
    df["macd"] = TA.MACD(df)["MACD"]
    df["macd_signal"] = TA.MACD(df)["SIGNAL"]
    df["bb_upper"] = TA.BBANDS(df)["BB_UPPER"]
    df["bb_middle"] = TA.BBANDS(df)["BB_MIDDLE"]
    df["bb_lower"] = TA.BBANDS(df)["BB_LOWER"]
    df["atr"] = TA.ATR(df, period=14)
    df["obv"] = TA.OBV(df)

    # Your own useful features
    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return_5"] = np.log(df["close"] / df["close"].shift(5))
    df["log_return_15"] = np.log(df["close"] / df["close"].shift(15))
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["volume_change"] = np.log(df["volume"] + 1).diff()

    # Drop rows where indicators are NaN from rolling windows
    df = df.dropna().reset_index(drop=True)

    return df


def _symbol_prefix(symbol: str) -> str:
    return symbol.lstrip("^")


def _normalize_open_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.as_unit("ns")


def _fetch_yfinance_ticket(symbol: str, target_end_date: pd.Timestamp) -> pd.DataFrame:
    """Fetch 1m bars for the last 30 days (yfinance limit) ending at target_end_date."""
    prefix = _symbol_prefix(symbol)
    close_col = f"{prefix}_close"
    return_col = f"{prefix}_return_1"

    target_end_date = pd.Timestamp(target_end_date)
    if target_end_date.tzinfo is None:
        target_end_date = target_end_date.tz_localize("UTC")
    else:
        target_end_date = target_end_date.tz_convert("UTC")

    now = pd.Timestamp.now(tz="UTC")
    current_download_end = min(target_end_date, now)
    historical_start_limit = current_download_end - pd.Timedelta(days=30)

    chunks: list[pd.DataFrame] = []

    while current_download_end > historical_start_limit:
        current_download_start = current_download_end - pd.Timedelta(days=7)
        if current_download_start < historical_start_limit:
            current_download_start = historical_start_limit

        chunk = yf.download(
            symbol,
            start=current_download_start.strftime("%Y-%m-%d"),
            end=current_download_end.strftime("%Y-%m-%d"),
            interval="1m",
            auto_adjust=False,
            progress=False,
        )
        if not chunk.empty:
            chunks.append(chunk)
        current_download_end = current_download_start

    if not chunks:
        return pd.DataFrame(columns=["open_time", close_col, return_col])

    df = pd.concat(chunks).drop_duplicates(keep="last").sort_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={time_col: "open_time", "Close": close_col})
    df["open_time"] = _normalize_open_time(df["open_time"])
    df[close_col] = df[close_col].astype(float)
    df[return_col] = np.log(df[close_col] / df[close_col].shift(1))

    return df[["open_time", close_col, return_col]].sort_values("open_time")


def _merge_external(df: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    if external.empty:
        return df
    external = external.copy()
    external["open_time"] = _normalize_open_time(external["open_time"])
    return pd.merge_asof(df, external, on="open_time", direction="backward")


def add_external_features(df: pd.DataFrame, api_key: str | None = None) -> pd.DataFrame:
    df = df.copy()
    df["open_time"] = _normalize_open_time(df["open_time"])
    df = df.sort_values("open_time")

    target_end = df["open_time"].max()
    spy = _fetch_yfinance_ticket("SPY", target_end)
    time.sleep(12)
    qqq = _fetch_yfinance_ticket("QQQ", target_end)
    time.sleep(12)
    vix = _fetch_yfinance_ticket("^VIX", target_end)

    df = _merge_external(df, spy)
    df = _merge_external(df, qqq)
    df = _merge_external(df, vix)

    for prefix in ("SPY", "QQQ", "VIX"):
        close_col = f"{prefix}_close"
        return_col = f"{prefix}_return_1"
        if close_col not in df.columns:
            df[close_col] = np.nan
        if return_col not in df.columns:
            df[return_col] = np.nan

    df["is_us_market_data_available"] = df["SPY_return_1"].notna().astype(int)
    df["is_nasdaq_data_available"] = df["QQQ_return_1"].notna().astype(int)
    df["is_vix_data_available"] = df["VIX_return_1"].notna().astype(int)

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/btc_minutes_100000.csv")
    df = add_features(df)
    df = add_external_features(df, ALPHA_VANTAGE_API_KEY)
    df.to_csv("data/btc_minutes_100000_with_features.csv", index=False)

    print(df.head())
    print(df.columns)
    print("Saved:", len(df), "rows")