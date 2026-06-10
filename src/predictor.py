"""
BTC above-threshold probability estimator.

Core model: binary option pricing (lognormal diffusion) with drift.
  drift=0  → pure vol model (original)
  drift≠0  → drift_forecaster adds mean-reversion signal + GARCH/HAR vol
"""
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ── Volatility ────────────────────────────────────────────────────────────────

def realized_vol_hourly(df: pd.DataFrame, window: int = 12) -> float:
    """
    Annualized realized volatility from the last `window` hourly log-returns.
    Returns a fraction, e.g. 0.60 = 60% annualized vol.
    """
    closes = df["close"].tail(window + 1)
    log_returns = np.log(closes / closes.shift(1)).dropna()
    if len(log_returns) < 2:
        return 0.80  # fallback: high vol assumption (conservative)
    hourly_vol = float(log_returns.std())
    annualized = hourly_vol * math.sqrt(8760)  # 8760 hours/year
    return max(annualized, 0.10)  # floor at 10% annualized


def vol_over_horizon(annualized_vol: float, hours_remaining: float) -> float:
    """Convert annualized vol to vol over the remaining horizon."""
    years = hours_remaining / 8760
    return annualized_vol * math.sqrt(years)


# ── Binary option probability (log-normal model) ─────────────────────────────

def prob_above_lognormal(
    spot: float,
    strike: float,
    sigma_horizon: float,
    drift: float = 0.0,
) -> float:
    """
    P(S_T > K) under log-normal diffusion.

    sigma_horizon: vol scaled to the remaining time horizon (not annualized)
    drift: expected log-return over horizon (default 0 = risk-neutral / no view)

    This is N(d2) from Black-Scholes for a binary call.
    """
    if sigma_horizon <= 0:
        return 1.0 if spot > strike else 0.0
    d2 = (math.log(spot / strike) + drift - 0.5 * sigma_horizon ** 2) / sigma_horizon
    return _norm_cdf(d2)


def _norm_cdf(x: float) -> float:
    return (1 + math.erf(x / math.sqrt(2))) / 2


# ── Momentum adjustment ───────────────────────────────────────────────────────

def momentum_adjustment(df: pd.DataFrame, weight: float = 0.04) -> float:
    """
    Small drift adjustment based on short-term momentum.
    Returns a log-return adjustment to add to the drift parameter.
    Capped at ±weight to keep it from dominating the base probability.
    """
    closes = df["close"]
    if len(closes) < 4:
        return 0.0
    # 3-candle momentum: log return over last 3 hours
    mom = math.log(float(closes.iloc[-1]) / float(closes.iloc[-4]))
    # Dampen heavily — momentum is weak at 1h horizon
    return float(np.clip(mom * 0.15, -weight, weight))


# ── Main entry point ──────────────────────────────────────────────────────────

def predict_above_threshold(
    df: pd.DataFrame,
    strike: float,
    current_price: float,
    expiry_utc: datetime | None = None,
    use_forecaster: bool = True,
) -> dict:
    """
    Estimate P(BTC > strike at hour-end).

    Parameters
    ----------
    df             : hourly OHLCV DataFrame
    strike         : contract strike price in USD
    current_price  : live BTC spot price
    expiry_utc     : contract expiry (UTC); defaults to next hour top
    use_forecaster : if True, use GARCH vol + drift model (recommended)
                     if False, use original rolling vol + tiny momentum nudge

    Returns
    -------
    dict with prob_above, drift, vol, confidence, and full signal breakdown
    """
    from src.drift_forecaster import forecast as drift_forecast

    now = datetime.now(timezone.utc)
    if expiry_utc is None:
        expiry_utc = now.replace(minute=0, second=0, microsecond=0)
        expiry_utc = expiry_utc.replace(hour=(now.hour + 1) % 24)

    hours_remaining = max((expiry_utc - now).total_seconds() / 3600, 1 / 60)

    if use_forecaster:
        fc = drift_forecast(df)
        vol_ann    = fc["vol_ann"]
        # Scale drift to the remaining horizon (drift is per-hour, scale linearly)
        drift      = fc["drift_logret"] * hours_remaining
        drift_conf = fc["confidence"]
        vol_source = fc["vol_source"]
        fc_features = fc["features"]
    else:
        vol_ann    = realized_vol_hourly(df)
        drift      = momentum_adjustment(df)
        drift_conf = 0.0
        vol_source = "rolling"
        fc_features = {}

    sigma_h = vol_over_horizon(vol_ann, hours_remaining)

    prob = prob_above_lognormal(
        spot=current_price,
        strike=strike,
        sigma_horizon=sigma_h,
        drift=drift,
    )

    buffer_pct = (current_price - strike) / strike * 100

    if sigma_h > 0:
        z = math.log(current_price / strike) / sigma_h
    else:
        z = 10.0
    confidence = float(np.clip((z - 0.5) / 2.0, 0.0, 1.0))

    return {
        "prob_above":      round(float(np.clip(prob, 0.01, 0.99)), 4),
        "buffer_pct":      round(buffer_pct, 3),
        "vol_ann_pct":     round(vol_ann * 100, 1),
        "vol_source":      vol_source,
        "hours_remaining": round(hours_remaining, 3),
        "sigma_horizon":   round(sigma_h, 5),
        "drift_logret":    round(drift, 6),
        "drift_pct":       round(drift * 100, 4),
        "drift_confidence":round(drift_conf, 3),
        "confidence":      round(confidence, 3),
        "signals": {
            "z_score":      round(z, 3),
            "buffer_pct":   round(buffer_pct, 3),
            "vol_ann_pct":  round(vol_ann * 100, 1),
            "vol_source":   vol_source,
            "drift_pct":    round(drift * 100, 4),
            "drift_conf":   round(drift_conf, 3),
            **fc_features,
        },
    }
