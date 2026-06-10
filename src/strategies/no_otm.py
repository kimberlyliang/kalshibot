"""
OTM scanner — finds edge on both KXBTC and KXBTCD contracts.

The user's strategy in plain English:
  - BTC is at $62,500.
  - "BTC above $64,000" YES is very unlikely → buy NO (if yes_bid > 0)
  - "BTC above $61,000" YES is very likely   → buy YES (if yes_ask is cheap)

Two trade types, evaluated on ALL series:
─────────────────────────────────────────────────────────────────
  NO on far-above contracts  (KXBTCD "greater", strike >> spot)
    Condition: model prob YES < MAX_YES_PROB
    Entry:     buy NO at no_ask = 1 - yes_bid
    Edge:      (1 - model_prob) - no_ask = yes_bid - model_prob

  YES on far-below contracts  (KXBTCD "greater", strike << spot)
    Condition: model prob YES > MIN_YES_PROB_FOR_CHEAP_BUY
    Entry:     buy YES at yes_ask
    Edge:      model_prob - yes_ask

  NO on OTM between buckets  (KXBTC "between", bucket far from spot)
    Same as before — yes_bid > 0 needed
─────────────────────────────────────────────────────────────────

Both require real liquidity: yes_bid > 0 for NO trades, yes_ask priced
below model prob for YES trades. When BTC drifts far from the opening,
yes_bid dries up and these opportunities disappear — that's correct
behavior, not a bug.
"""
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.predictor import vol_over_horizon, realized_vol_hourly

# ── Config ────────────────────────────────────────────────────────────────────
MIN_YES_BID       = 0.01   # YES bid must be ≥ 1¢ to buy NO
MAX_YES_PROB      = 0.10   # model YES prob must be < 10% for NO trades
MIN_YES_PROB_BUY  = 0.85   # model YES prob must be > 85% for cheap YES trades
MAX_YES_ASK_BUY   = 0.97   # YES ask must be < 97¢ (some edge left)
MIN_EDGE          = 0.01   # minimum edge in either direction
MIN_DIST_PCT      = 0.005  # strike must be ≥ 0.5% from spot
MAX_DIST_PCT      = 0.15   # ignore contracts > 15% away (illiquid tail)
MIN_MINUTES       = 5
MAX_MINUTES       = 55


def find_opportunities(
    markets_by_series: dict[str, list[dict]],
    spot: float,
    candles,
    expiry_utc: datetime | None = None,
) -> list[dict]:
    """
    Scan all series for:
      - NO trades (far above spot, yes_bid > 0)
      - YES trades (far below spot, yes_ask cheap relative to model)

    Returns list sorted by edge descending.
    """
    now = datetime.now(timezone.utc)
    vol_ann = realized_vol_hourly(candles, window=12)
    candidates = []

    for series, markets in markets_by_series.items():
        exp = expiry_utc or _nearest_expiry(markets)
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
            yb    = float(m.get("yes_bid_dollars") or 0)
            na    = float(m.get("no_ask_dollars") or 1)

            if stype == "greater":
                strike = float(m.get("floor_strike") or 0)
                dist   = (strike - spot) / spot   # positive = above spot
                model  = _prob_above(spot, strike, sigma_h)

                # ── NO trade: strike well above spot, YES unlikely ────────
                if (dist > MIN_DIST_PCT
                        and abs(dist) < MAX_DIST_PCT
                        and yb >= MIN_YES_BID
                        and model < MAX_YES_PROB):
                    no_ask = 1.0 - yb
                    edge   = (1.0 - model) - no_ask
                    if edge >= MIN_EDGE:
                        candidates.append(_make(
                            m, series, "no", "no_above",
                            no_ask, model, edge, dist, mins, sigma_h,
                            note=f"NO: BTC unlikely to reach ${strike:,.0f}"
                        ))

                # ── YES trade: strike well below spot, YES very likely ────
                if (dist < -MIN_DIST_PCT
                        and abs(dist) < MAX_DIST_PCT
                        and ya > 0
                        and ya < MAX_YES_ASK_BUY
                        and model > MIN_YES_PROB_BUY):
                    edge = model - ya
                    if edge >= MIN_EDGE:
                        candidates.append(_make(
                            m, series, "yes", "yes_below",
                            ya, model, edge, dist, mins, sigma_h,
                            note=f"YES: BTC already ${spot-strike:,.0f} above ${strike:,.0f}"
                        ))

            elif stype == "between":
                lo = float(m.get("floor_strike") or 0)
                hi = float(m.get("cap_strike") or 0)
                mid = (lo + hi) / 2
                dist = (mid - spot) / spot

                if abs(dist) < MIN_DIST_PCT or abs(dist) > MAX_DIST_PCT:
                    continue

                model = _prob_in_bucket(lo, hi, spot, sigma_h)

                # NO on OTM between bucket
                if yb >= MIN_YES_BID and model < MAX_YES_PROB:
                    no_ask = 1.0 - yb
                    edge   = (1.0 - model) - no_ask
                    if edge >= MIN_EDGE:
                        candidates.append(_make(
                            m, series, "no", "no_between",
                            no_ask, model, edge, dist, mins, sigma_h,
                            note=f"NO: bucket far from spot"
                        ))

    return sorted(candidates, key=lambda x: x["edge"], reverse=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prob_above(spot: float, strike: float, sigma_h: float) -> float:
    """P(BTC > strike at expiry) under lognormal diffusion."""
    if sigma_h <= 0:
        return 1.0 if spot > strike else 0.0
    cdf = lambda x: (1 + math.erf(x / math.sqrt(2))) / 2
    d2  = (math.log(spot / strike) - 0.5 * sigma_h**2) / sigma_h
    return cdf(d2)


def _prob_in_bucket(lo: float, hi: float, spot: float, sigma_h: float) -> float:
    if sigma_h <= 0:
        return 1.0 if lo <= spot < hi else 0.0
    cdf  = lambda x: (1 + math.erf(x / math.sqrt(2))) / 2
    d_lo = (math.log(lo / spot) + 0.5 * sigma_h**2) / sigma_h
    d_hi = (math.log(hi / spot) + 0.5 * sigma_h**2) / sigma_h
    return max(cdf(d_hi) - cdf(d_lo), 0.0)


def _make(m, series, side, trade_type, price, model, edge, dist, mins, sigma_h, note):
    return {
        "ticker":       m["ticker"],
        "series":       series,
        "side":         side,
        "trade_type":   trade_type,
        "subtitle":     m.get("subtitle", ""),
        "price":        round(price, 4),
        "price_cents":  round(price * 100),
        "model_prob":   round(model, 5),
        "edge":         round(edge, 4),
        "dist_pct":     round(dist * 100, 2),
        "mins":         round(mins, 1),
        "sigma_h_pct":  round(sigma_h * 100, 3),
        "note":         note,
    }


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
