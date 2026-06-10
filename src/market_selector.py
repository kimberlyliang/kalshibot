"""
Find the best Kalshi BTC hourly YES contract to trade.

Strategy: YES-only, high-confidence (P >= MIN_PROB_YES).
Pick the contract where we have the most edge vs the market ask.
"""
import re
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import BTC_SERIES_TICKER, MIN_EDGE, MIN_PROB_YES, MIN_BUFFER_USD, YES_ONLY
from src.kalshi_client import KalshiClient
from src.risk import compute_edge


def find_best_contract(
    client: KalshiClient,
    current_price: float,
    predictor_fn,   # callable(strike, expiry_utc) -> dict with prob_above
) -> dict | None:
    """
    Scan open BTC hourly markets, filter to high-confidence YES bets,
    return the one with the most edge, or None if nothing qualifies.
    """
    markets = client.get_markets(series_ticker=BTC_SERIES_TICKER, status="open")
    if not markets:
        return None

    best = None
    best_edge = 0.0

    for market in markets:
        ticker = market["ticker"]

        try:
            strike = _parse_strike(market)
            expiry = _parse_expiry(market)
        except (KeyError, ValueError):
            continue

        # Only consider markets where BTC is sufficiently above the strike
        if current_price - strike < MIN_BUFFER_USD:
            continue

        pred = predictor_fn(strike, expiry)
        prob = pred["prob_above"]

        # High-confidence filter
        if prob < MIN_PROB_YES:
            continue

        ob = client.get_orderbook(ticker, depth=3)
        yes_ask = _best_ask(ob, "yes")
        if yes_ask is None:
            continue

        edge = compute_edge(prob, yes_ask)
        if edge < MIN_EDGE:
            continue

        if edge > best_edge:
            best_edge = edge
            best = {
                "ticker": ticker,
                "side": "yes",
                "price_cents": yes_ask,
                "prob": prob,
                "edge": round(edge, 4),
                "confidence": pred["confidence"],
                "strike": strike,
                "buffer_pct": pred["buffer_pct"],
                "vol_ann_pct": pred["vol_ann_pct"],
                "hours_remaining": pred["hours_remaining"],
                "z_score": pred["signals"]["z_score"],
                "expiry": expiry.isoformat() if expiry else None,
            }

    return best


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_strike(market: dict) -> float:
    """Extract the USD strike price from the market title/subtitle."""
    for field in ("subtitle", "title"):
        text = market.get(field, "") or ""
        m = re.search(r"\$([0-9,]+(?:\.[0-9]+)?)", text)
        if m:
            return float(m.group(1).replace(",", ""))
    # Fallback: try the ticker itself e.g. KXBTC-26JUN0917-T72749.99
    m = re.search(r"T([0-9]+(?:\.[0-9]+)?)$", market.get("ticker", ""))
    if m:
        return float(m.group(1))
    raise ValueError(f"Cannot parse strike from market: {market.get('ticker')}")


def _parse_expiry(market: dict) -> datetime | None:
    """Parse expiry timestamp from market data."""
    raw = market.get("close_time") or market.get("expiration_time")
    if not raw:
        return None
    try:
        # ISO format: "2026-06-09T17:00:00Z"
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _best_ask(orderbook: dict, side: str) -> int | None:
    """
    Return the lowest ask price in cents (1-99) for a given side.
    Kalshi orderbook_fp: keys 'yes_dollars'/'no_dollars', prices as decimal strings e.g. "0.91".
    """
    levels = orderbook.get(f"{side}_dollars", [])
    if not levels:
        return None
    return round(float(levels[0][0]) * 100)
