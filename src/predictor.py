"""
Price direction predictor.

Strategy: ensemble of three fast, interpretable signals.
Each produces a probability that BTC is ABOVE a threshold at hour-end.
Final probability = weighted average.

Signals
-------
1. Momentum (RSI + rate-of-change)      weight 0.35
2. Mean-reversion (Bollinger band z-score)  weight 0.30
3. Micro-structure (volume imbalance + candle body)  weight 0.35
"""
import math
import numpy as np
import pandas as pd


# ── feature helpers ──────────────────────────────────────────────────────────

def _rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(period).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(period).mean().iloc[-1]
    if loss == 0:
        return 100.0
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def _bollinger_z(closes: pd.Series, period: int = 20) -> float:
    """Z-score of last close relative to BB mid."""
    mid = closes.rolling(period).mean().iloc[-1]
    std = closes.rolling(period).std().iloc[-1]
    if std == 0:
        return 0.0
    return (closes.iloc[-1] - mid) / std


def _roc(closes: pd.Series, period: int = 6) -> float:
    """Rate of change over `period` candles (fraction)."""
    if len(closes) < period + 1:
        return 0.0
    return (closes.iloc[-1] / closes.iloc[-(period + 1)] - 1)


def _volume_momentum(df: pd.DataFrame, period: int = 6) -> float:
    """Fraction of recent volume that is up-candles."""
    recent = df.tail(period)
    up_vol = recent.loc[recent["close"] >= recent["open"], "volume"].sum()
    total_vol = recent["volume"].sum()
    if total_vol == 0:
        return 0.5
    return up_vol / total_vol


def _candle_body_bias(df: pd.DataFrame, period: int = 3) -> float:
    """Average (close-open)/(high-low) over last `period` candles. +1 = all bull."""
    recent = df.tail(period)
    ranges = (recent["high"] - recent["low"]).replace(0, np.nan)
    bodies = (recent["close"] - recent["open"]) / ranges
    return float(bodies.mean())


# ── probability converters ────────────────────────────────────────────────────

def _sigmoid(x: float, scale: float = 1.0) -> float:
    return 1 / (1 + math.exp(-scale * x))


# ── main predictor ────────────────────────────────────────────────────────────

WEIGHTS = {"momentum": 0.35, "mean_rev": 0.30, "micro": 0.35}


def predict_above_threshold(df: pd.DataFrame, threshold: float, current_price: float) -> dict:
    """
    Parameters
    ----------
    df : hourly candles DataFrame (columns: open/high/low/close/volume)
    threshold : the Kalshi contract strike price in USD
    current_price : live BTC spot price

    Returns
    -------
    dict with keys: prob_above, signals, confidence
    """
    closes = df["close"]

    # 1. Momentum signal
    rsi = _rsi(closes)
    roc = _roc(closes)
    # RSI > 50 → bullish; map [0,100] → [-1,1] then sigmoid
    rsi_score = (rsi - 50) / 50
    roc_score = roc / 0.02  # 2% move = 1 SD rough estimate
    momentum_raw = 0.6 * rsi_score + 0.4 * roc_score
    p_momentum = _sigmoid(momentum_raw, scale=1.5)

    # 2. Mean-reversion signal
    bz = _bollinger_z(closes)
    # Extreme high z → likely to fall; invert
    p_mean_rev = _sigmoid(-bz, scale=0.8)

    # Blend: if price well above threshold already, weight momentum less
    price_gap_pct = (current_price - threshold) / threshold
    if abs(price_gap_pct) > 0.02:
        # Strong directional prior from price level
        level_prior = _sigmoid(price_gap_pct * 50)
        p_mean_rev = 0.5 * p_mean_rev + 0.5 * level_prior

    # 3. Micro-structure signal
    vol_mom = _volume_momentum(df)
    body_bias = _candle_body_bias(df)
    micro_raw = 0.5 * (vol_mom - 0.5) * 2 + 0.5 * body_bias
    p_micro = _sigmoid(micro_raw, scale=2.0)

    # Ensemble
    prob = (
        WEIGHTS["momentum"] * p_momentum
        + WEIGHTS["mean_rev"] * p_mean_rev
        + WEIGHTS["micro"] * p_micro
    )

    # Confidence: how far from 0.5 the signals agree
    probs = [p_momentum, p_mean_rev, p_micro]
    agreement = 1 - np.std(probs) * 4  # 0 = total disagreement, 1 = perfect
    confidence = float(np.clip(agreement, 0, 1))

    return {
        "prob_above": float(np.clip(prob, 0.02, 0.98)),
        "signals": {
            "momentum": round(p_momentum, 3),
            "mean_rev": round(p_mean_rev, 3),
            "micro": round(p_micro, 3),
            "rsi": round(rsi, 1),
            "roc_pct": round(roc * 100, 2),
            "boll_z": round(bz, 2),
            "vol_mom": round(vol_mom, 3),
        },
        "confidence": round(confidence, 3),
    }
