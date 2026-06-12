"""Collect BTC hourly candles and save to CSV automatically."""
from pathlib import Path
import sys

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import BINANCE_KLINE_URL, BTC_SYMBOL


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = REPO_ROOT / "data" / "btc_minutes_100000.csv"
N_CANDLES = 100000 # 100000 minutes = 1666 hours = 70 days

def _fetch_coinbase(n: int) -> pd.DataFrame:
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

    end = pd.Timestamp.now('UTC')
    start = end - pd.Timedelta(minutes=n)

    all_rows = []
    cur_start = start
    count = 0

    while cur_start < end:
        cur_end = min(cur_start + pd.Timedelta(minutes=300), end)

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
        if count % 9000 == 0:
            print(f"Collected {count} candles from")
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
    df["close_time"] = df["open_time"] + pd.Timedelta(hours=1)

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


def collect_candles(n: int = N_CANDLES) -> tuple[pd.DataFrame, str]:
    """Return candles plus source label."""
    try:
        return _fetch_coinbase(n), "coinbase"
    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 451:
            print("Coinbase returned 451 Unavailable for legal reasons")
            raise SystemExit(1)
        raise


def save_csv(df: pd.DataFrame, output_path: Path = OUT_CSV) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    try:
        candles, source = collect_candles(N_CANDLES)
        if candles.empty:
            print("FAILURE: no candle data returned.")
            raise SystemExit(1)

        save_csv(candles, OUT_CSV)
        print(f"SUCCESS: saved {len(candles)} candles from {source} to {OUT_CSV}")
    except Exception as exc:
        print(f"FAILURE: could not collect/save candles. error={exc}")
        raise SystemExit(1)
