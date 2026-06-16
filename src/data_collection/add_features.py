import pandas as pd
from finta import TA

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
    df["log_return_30"] = np.log(df["close"] / df["close"].shift(30))
    df["log_return_60"] = np.log(df["close"] / df["close"].shift(60))
    df["future_log_return_1"] = np.log(df["close"].shift(-1) / df["close"])
    df["future_log_return_5"] = np.log(df["close"].shift(-5) / df["close"])
    df["future_log_return_15"] = np.log(df["close"].shift(-15) / df["close"])
    df["future_log_return_30"] = np.log(df["close"].shift(-30) / df["close"])
    df["future_log_return_60"] = np.log(df["close"].shift(-60) / df["close"])
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["volume_change"] = np.log(df["volume"] + 1).diff()

    # Drop rows where indicators are NaN from rolling windows
    df = df.dropna().reset_index(drop=True)

    return df

if __name__ == "__main__":
    import numpy as np

    df = pd.read_csv("data/btc_minutes_100000.csv")
    df = add_features(df)
    df.to_csv("data/btc_minutes_100000_with_features.csv", index=False)

    print(df.head())
    print(df.columns)
    print("Saved:", len(df), "rows")