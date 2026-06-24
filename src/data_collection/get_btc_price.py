"""Collect BTC minute candles, add features, and save to CSV."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import yfinance as yf
from finta import TA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config.settings  # noqa: F401 — loads .env into os.environ

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_CSV = REPO_ROOT / "data" / "btc_minutes_100000.csv"
N_CANDLES = 100_000  # 100000 minutes ~= 70 days
TARGET_COL = "future_log_return_60"
FUTURE_RETURN_COLS = (
    "future_log_return_1",
    "future_log_return_5",
    "future_log_return_15",
    "future_log_return_30",
    TARGET_COL,
)
FUTURE_RETURN_SHIFTS = (
    ("future_log_return_1", 1),
    ("future_log_return_5", 5),
    ("future_log_return_15", 15),
    ("future_log_return_30", 30),
    (TARGET_COL, 60),
)


def _parse_end_time(end: str | pd.Timestamp | None) -> pd.Timestamp:
    if end is None:
        return pd.Timestamp.now("UTC")
    if isinstance(end, pd.Timestamp):
        return end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
    return pd.to_datetime(end, utc=True)


def _fetch_coinbase(n: int, end: str | pd.Timestamp | None = None) -> pd.DataFrame:
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    end_ts = _parse_end_time(end)
    start = end_ts - pd.Timedelta(minutes=n)

    all_rows = []
    cur_start = start
    count = 0

    while cur_start < end_ts:
        cur_end = min(cur_start + pd.Timedelta(minutes=300), end_ts)

        params = {
            "granularity": 60,
            "start": cur_start.isoformat(),
            "end": cur_end.isoformat(),
        }

        response = httpx.get(url, params=params, timeout=15)
        response.raise_for_status()

        batch = response.json()
        all_rows.extend(batch)
        count += len(batch)
        if count and count % 9000 == 0:
            print(f"Collected {count} candles")
        cur_start = cur_end

    df = pd.DataFrame(
        all_rows,
        columns=["time", "low", "high", "open", "close", "volume"],
    )

    if df.empty:
        return df

    df = df.drop_duplicates(subset=["time"])
    df = df.sort_values("time").tail(n).reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["close_time"] = df["open_time"] + pd.Timedelta(minutes=1)

    return df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
        ]
    ]


def collect_candles(
    n: int = N_CANDLES,
    end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, str]:
    """Return candles plus source label."""
    try:
        return _fetch_coinbase(n, end), "coinbase"
    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 451:
            print("Coinbase returned 451 Unavailable for legal reasons")
            raise SystemExit(1) from e
        raise


def add_known_future_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute forward log returns from close; fill unknown tail with expanding mean."""
    df = df.copy()
    for col, minutes in FUTURE_RETURN_SHIFTS:
        df[col] = np.log(df["close"].shift(-minutes) / df["close"])
    for col in FUTURE_RETURN_COLS:
        df[col] = df[col].fillna(df[col].expanding(min_periods=1).mean())
        df[col] = df[col].fillna(0.0)
    return df


def add_features(df: pd.DataFrame, *, for_training: bool = True) -> pd.DataFrame:
    """Add technical and return features. Predict mode skips future return columns."""
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

    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return_5"] = np.log(df["close"] / df["close"].shift(5))
    df["log_return_15"] = np.log(df["close"] / df["close"].shift(15))
    df["log_return_30"] = np.log(df["close"] / df["close"].shift(30))
    df["log_return_60"] = np.log(df["close"] / df["close"].shift(60))
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["volume_change"] = np.log(df["volume"] + 1).diff()

    if for_training:
        df["future_log_return_1"] = np.log(df["close"].shift(-1) / df["close"])
        df["future_log_return_5"] = np.log(df["close"].shift(-5) / df["close"])
        df["future_log_return_15"] = np.log(df["close"].shift(-15) / df["close"])
        df["future_log_return_30"] = np.log(df["close"].shift(-30) / df["close"])
        df[TARGET_COL] = np.log(df["close"].shift(-60) / df["close"])
        df = df.dropna().reset_index(drop=True)
    else:
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

    ext = pd.concat(chunks).drop_duplicates(keep="last").sort_index()

    if isinstance(ext.columns, pd.MultiIndex):
        ext.columns = ext.columns.get_level_values(0)

    ext = ext.reset_index()
    time_col = "Datetime" if "Datetime" in ext.columns else "Date"
    ext = ext.rename(columns={time_col: "open_time", "Close": close_col})
    ext["open_time"] = _normalize_open_time(ext["open_time"])
    ext[close_col] = ext[close_col].astype(float)
    ext[return_col] = np.log(ext[close_col] / ext[close_col].shift(1))

    return ext[["open_time", close_col, return_col]].sort_values("open_time")


def _merge_external(df: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    if external.empty:
        return df
    external = external.copy()
    external["open_time"] = _normalize_open_time(external["open_time"])
    return pd.merge_asof(df, external, on="open_time", direction="backward")


def add_external_features(df: pd.DataFrame) -> pd.DataFrame:
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


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def _filename_label(end: str | None) -> str:
    end_ts = _parse_end_time(end)
    return end_ts.strftime("%Y-%m-%d_%H-%M-%S")


def prepare_btc_data(
    n: int = N_CANDLES,
    end: str | pd.Timestamp | None = None,
    *,
    for_training: bool = True,
    external: bool = False,
) -> tuple[pd.DataFrame, str]:
    candles, source = collect_candles(n, end)
    if candles.empty:
        return candles, source

    df = add_features(candles, for_training=for_training)
    if external:
        df = add_external_features(df)
        if for_training:
            df = df.dropna().reset_index(drop=True)
        else:
            input_cols = [c for c in df.columns if c not in FUTURE_RETURN_COLS]
            df = df.dropna(subset=input_cols).reset_index(drop=True)
    elif not for_training:
        df = add_known_future_returns(df)

    if external and not for_training:
        df = add_known_future_returns(df)

    return df, source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download BTC-USD 1-minute candles from Coinbase and add features",
    )
    parser.add_argument("--n_candles", type=int, default=N_CANDLES)
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help='UTC end of candle window (default: now). Example: "2026-06-23 02:00:00"',
    )
    parser.add_argument(
        "--train_data",
        action="store_true",
        help="Training mode: include future log-return targets and drop tail rows",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Merge SPY/QQQ/VIX features from yfinance (slow)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional explicit output CSV path",
    )
    parser.add_argument(
        "--no_features",
        action="store_true",
        help="Save raw candles only (no indicators)",
    )
    args = parser.parse_args()

    suffix = "train" if args.train_data else "predict"
    if args.output:
        out_csv = Path(args.output)
    elif args.end or args.train_data or args.n_candles != N_CANDLES:
        label = _filename_label(args.end)
        out_csv = REPO_ROOT / "data" / f"btc_minutes_{label}_{args.n_candles}_{suffix}.csv"
    else:
        out_csv = DEFAULT_OUT_CSV
    try:
        if args.no_features:
            df, source = collect_candles(args.n_candles, args.end)
        else:
            df, source = prepare_btc_data(
                args.n_candles,
                args.end,
                for_training=args.train_data,
                external=args.external,
            )

        if df.empty:
            print("FAILURE: no candle data returned.")
            raise SystemExit(1)

        save_csv(df, out_csv)
        mode = "train" if args.train_data else "predict"
        print(
            f"SUCCESS: saved {len(df)} rows ({mode}) from {source} to {out_csv}"
        )
    except SystemExit:
        raise
    except Exception as exc:
        print(f"FAILURE: could not collect/save candles. error={exc}")
        raise SystemExit(1) from exc
