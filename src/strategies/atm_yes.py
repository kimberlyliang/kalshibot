"""
ATM YES Strategy — Phase 2 (last 5 minutes only)
=================================================
Buys YES on the $100 bucket currently containing the BTC spot price.

WHY it only fires in the last 5 minutes:
  At 5 min remaining the ATM bucket has ~59% probability (model).
  At 3 min remaining it's ~71%. At 1 min it's ~93%.
  Market makers haven't repriced — it still shows at 1-4 cents.
  Edge at 3 min = 67–70 cents per contract.

At 30 min remaining the same bucket is worth 26% — still great edge
vs 1-cent ask, but the trade ties up capital for longer with more
variance. We reserve this slot for near-expiry where it's clearest.
"""
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.predictor import vol_over_horizon, realized_vol_hourly

# ── Config ────────────────────────────────────────────────────────────────────
MAX_MINUTES_REMAINING = 15.0  # enter in last 15 minutes (edge still +0.36 at 15min)
MIN_MODEL_PROB        = 0.30  # model must say ≥30%
MAX_MARKET_ASK        = 0.05  # only if market is still asking ≤5 cents
MIN_EDGE              = 0.25  # model_prob - market_ask ≥ 25¢ (at 15min this is easy to hit)
MAX_ATM_USD           = 20.0
MAX_CONTRACTS         = 1000


def find_atm_opportunity(markets: list[dict], spot: float, candles,
                         expiry_utc: datetime | None = None) -> dict | None:
    """
    Scan for the ATM bucket (containing current spot) near expiry.
    Returns trade dict or None.
    """
    now = datetime.now(timezone.utc)
    if expiry_utc is None:
        expiry_utc = _nearest_expiry(markets)
    if expiry_utc is None:
        return None

    mins_remaining = (expiry_utc - now).total_seconds() / 60
    if mins_remaining > MAX_MINUTES_REMAINING or mins_remaining <= 0:
        return None

    hours_remaining = mins_remaining / 60
    vol_ann = realized_vol_hourly(candles, window=12)
    sigma_h = vol_over_horizon(vol_ann, hours_remaining)

    between = [m for m in markets
               if m.get("strike_type") == "between"
               and m.get("floor_strike") is not None
               and m.get("cap_strike") is not None
               and m.get("close_time") == expiry_utc.strftime("%Y-%m-%dT%H:%M:%SZ")]

    best = None
    best_edge = MIN_EDGE

    for m in between:
        lo = float(m["floor_strike"])
        hi = float(m["cap_strike"])

        if not (lo <= spot < hi):
            continue

        yes_ask = float(m.get("yes_ask_dollars") or 0)
        if yes_ask <= 0 or yes_ask > MAX_MARKET_ASK:
            continue

        model_prob = _prob_in_bucket(lo, hi, spot, sigma_h)
        edge = model_prob - yes_ask

        if edge < best_edge:
            continue

        best_edge = edge
        best = {
            "ticker":          m["ticker"],
            "side":            "yes",
            "strategy":        "atm_yes",
            "floor":           lo,
            "cap":             hi,
            "subtitle":        m.get("subtitle", ""),
            "yes_ask":         yes_ask,
            "yes_ask_cents":   round(yes_ask * 100),
            "model_prob":      round(model_prob, 4),
            "edge":            round(edge, 4),
            "vol_ann_pct":     round(vol_ann * 100, 1),
            "sigma_h_pct":     round(sigma_h * 100, 3),
            "mins_remaining":  round(mins_remaining, 1),
            "hours_remaining": round(hours_remaining, 4),
        }

    return best


def size_atm(yes_ask: float, model_prob: float,
             balance_cents: int, portfolio_dollars: float) -> int:
    """Kelly-sized ATM position."""
    cost_per   = yes_ask
    payout_per = 1.0 - yes_ask
    b          = payout_per / cost_per
    q          = 1.0 - model_prob
    edge       = b * model_prob - q
    if edge <= 0:
        return 0

    full_kelly_f = edge / b
    # Use 20% fractional Kelly — lotto-ticket style, small cost
    scaled_f   = full_kelly_f * 0.20
    kelly_usd  = portfolio_dollars * scaled_f
    max_spend  = min(MAX_ATM_USD, kelly_usd, (balance_cents / 100) * 0.08)
    n = int(max_spend / cost_per)
    return min(max(n, 0), MAX_CONTRACTS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prob_in_bucket(lo: float, hi: float, spot: float, sigma_h: float) -> float:
    if sigma_h <= 0:
        return 1.0 if lo <= spot < hi else 0.0
    cdf  = lambda x: (1 + math.erf(x / math.sqrt(2))) / 2
    d_lo = (math.log(lo / spot) + 0.5 * sigma_h**2) / sigma_h
    d_hi = (math.log(hi / spot) + 0.5 * sigma_h**2) / sigma_h
    return max(cdf(d_hi) - cdf(d_lo), 0.0)


def _nearest_expiry(markets: list[dict]) -> datetime | None:
    times = set()
    for m in markets:
        ct = m.get("close_time")
        if ct:
            try:
                times.add(datetime.fromisoformat(ct.replace("Z", "+00:00")))
            except Exception:
                pass
    if not times:
        return None
    now = datetime.now(timezone.utc)
    future = [t for t in times if t > now]
    return min(future) if future else None
