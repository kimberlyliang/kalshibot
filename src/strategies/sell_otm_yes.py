"""
Sell-YES on far-OTM contracts (vol selling / theta collection)
==============================================================
Posts limit SELL orders on YES contracts for buckets our model
says are very unlikely to win. We collect the ask price as premium
and keep it when the contract expires worthless.

Why sell instead of buy NO:
  Buying NO costs (1 - yes_bid). When yes_bid = 0, no_ask = $1.00 → zero edge.
  Selling YES at the current ask (e.g. 3–9¢) lets us SET the price we receive.
  We post our ask, a retail buyer fills it, we collect the premium.

Risk: if BTC moves into the bucket we're short, we owe $1 - premium collected.
Managed by: (a) only selling on buckets with model prob < MAX_YES_PROB,
            (b) cancelling open sell orders if BTC drifts toward the bucket,
            (c) small position sizes.

Scans BOTH KXBTC (between) and KXBTCD (greater/less) series.
"""
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.predictor import vol_over_horizon, realized_vol_hourly

# ── Config ────────────────────────────────────────────────────────────────────
MAX_YES_PROB      = 0.04    # only sell YES if model says < 4% it wins
MIN_SELL_PRICE    = 0.01    # accept the Kalshi minimum 1¢ ask
MIN_EDGE          = 0.005   # (sell_price - model_prob) must be ≥ 0.5¢
MAX_SELL_USD      = 15.0    # max notional exposure per contract group
MAX_CONTRACTS     = 200     # cap on contracts (each risks $1 if wrong)
MIN_MINUTES       = 8       # don't enter with < 8 min (not enough theta)
MAX_MINUTES       = 52      # don't enter with > 52 min left
# How far the bucket must be from spot (% of spot) before we sell
MIN_DIST_PCT      = 0.008   # ≥ 0.8% away from spot
SERIES            = ["KXBTC", "KXBTCD"]


def find_sell_yes_opportunities(
    markets_by_series: dict[str, list[dict]],
    spot: float,
    candles,
    expiry_utc: datetime | None = None,
) -> list[dict]:
    """
    Scan all markets for far-OTM YES contracts to sell.
    markets_by_series: {"KXBTC": [...], "KXBTCD": [...]}
    Returns list sorted by edge descending.
    """
    now = datetime.now(timezone.utc)

    vol_ann = realized_vol_hourly(candles, window=12)
    candidates = []

    for series, markets in markets_by_series.items():
        if expiry_utc is None:
            exp = _nearest_expiry(markets)
        else:
            exp = expiry_utc
        if exp is None:
            continue

        mins = (exp - now).total_seconds() / 60
        if not (MIN_MINUTES <= mins <= MAX_MINUTES):
            continue

        sigma_h = vol_over_horizon(vol_ann, mins / 60)
        exp_str = exp.strftime("%Y-%m-%dT%H:%M:%SZ")

        for m in markets:
            if m.get("close_time") != exp_str:
                continue

            stype = m.get("strike_type", "")
            ya    = float(m.get("yes_ask_dollars") or 0)

            # We'll sell at the current ask (or slightly below to get filled)
            sell_price = ya
            if sell_price < MIN_SELL_PRICE:
                continue

            model_prob = _model_prob(m, spot, sigma_h, stype)
            if model_prob is None or model_prob > MAX_YES_PROB:
                continue

            # Distance check: bucket must not be too close to spot
            dist_pct = _dist_from_spot(m, spot, stype)
            if dist_pct is None or dist_pct < MIN_DIST_PCT:
                continue

            edge = sell_price - model_prob
            if edge < MIN_EDGE:
                continue

            # Max loss if YES wins = (1 - sell_price) per contract
            max_loss_per = 1.0 - sell_price
            n = min(int(MAX_SELL_USD / max_loss_per), MAX_CONTRACTS)
            if n < 1:
                continue

            candidates.append({
                "ticker":       m["ticker"],
                "action":       "sell",
                "side":         "yes",
                "strategy":     "sell_otm_yes",
                "series":       series,
                "strike_type":  stype,
                "subtitle":     m.get("subtitle", ""),
                "sell_price":   round(sell_price, 4),
                "sell_cents":   round(sell_price * 100),
                "model_prob":   round(model_prob, 5),
                "edge":         round(edge, 4),
                "dist_pct":     round(dist_pct * 100, 2),
                "max_loss_per": round(max_loss_per, 4),
                "n_contracts":  n,
                "premium_collected": round(n * sell_price, 2),
                "max_exposure": round(n * max_loss_per, 2),
                "mins_remaining": round(mins, 1),
            })

    return sorted(candidates, key=lambda x: x["edge"], reverse=True)


def size_sell_yes(sell_price: float, model_prob: float,
                  balance_cents: int, portfolio_dollars: float) -> int:
    """Kelly size for selling YES (collecting premium)."""
    # Selling YES: win prob = 1 - model_prob, profit = sell_price, loss = 1 - sell_price
    p_win     = 1.0 - model_prob
    cost_if_loss = 1.0 - sell_price   # what we pay if YES wins
    premium   = sell_price

    b = premium / cost_if_loss
    q = 1.0 - p_win
    edge = b * p_win - q
    if edge <= 0:
        return 0

    full_kelly = edge / b
    scaled     = full_kelly * 0.20   # 20% fractional Kelly
    kelly_usd  = portfolio_dollars * scaled
    cap        = min(15.0, kelly_usd, (balance_cents / 100) * 0.06)
    n = int(cap / cost_if_loss)
    return min(max(n, 0), MAX_CONTRACTS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _model_prob(m: dict, spot: float, sigma_h: float, stype: str) -> float | None:
    cdf = lambda x: (1 + math.erf(x / math.sqrt(2))) / 2
    if stype == "greater":
        strike = m.get("floor_strike")
        if strike is None:
            return None
        if sigma_h <= 0:
            return 1.0 if spot > float(strike) else 0.0
        d2 = (math.log(spot / float(strike)) - 0.5 * sigma_h**2) / sigma_h
        return cdf(d2)
    elif stype == "less":
        strike = m.get("cap_strike")
        if strike is None:
            return None
        if sigma_h <= 0:
            return 0.0 if spot > float(strike) else 1.0
        d2 = (math.log(spot / float(strike)) - 0.5 * sigma_h**2) / sigma_h
        return 1.0 - cdf(d2)
    elif stype == "between":
        lo = m.get("floor_strike")
        hi = m.get("cap_strike")
        if lo is None or hi is None:
            return None
        lo, hi = float(lo), float(hi)
        if lo <= spot < hi:
            return None   # ATM bucket — don't sell
        if sigma_h <= 0:
            return 0.0
        d_lo = (math.log(lo / spot) + 0.5 * sigma_h**2) / sigma_h
        d_hi = (math.log(hi / spot) + 0.5 * sigma_h**2) / sigma_h
        return max(cdf(d_hi) - cdf(d_lo), 0.0)
    return None


def _dist_from_spot(m: dict, spot: float, stype: str) -> float | None:
    """Fractional distance of the nearest bucket edge from spot."""
    if stype == "greater":
        strike = m.get("floor_strike")
        return abs(float(strike) - spot) / spot if strike else None
    elif stype == "less":
        strike = m.get("cap_strike")
        return abs(float(strike) - spot) / spot if strike else None
    elif stype == "between":
        lo = m.get("floor_strike")
        hi = m.get("cap_strike")
        if lo is None or hi is None:
            return None
        return min(abs(float(lo) - spot), abs(float(hi) - spot)) / spot
    return None


def _nearest_expiry(markets: list[dict]) -> datetime | None:
    now = datetime.now(timezone.utc)
    times = set()
    for m in markets:
        ct = m.get("close_time")
        if ct:
            try:
                times.add(datetime.fromisoformat(ct.replace("Z", "+00:00")))
            except Exception:
                pass
    future = sorted(t for t in times if t > now)
    return future[0] if future else None
