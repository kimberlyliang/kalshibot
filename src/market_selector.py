"""Find the best Kalshi BTC hourly contract to trade."""
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import BTC_SERIES_TICKER, MIN_EDGE
from src.kalshi_client import KalshiClient
from src.risk import compute_edge


def find_best_contract(
    client: KalshiClient,
    current_price: float,
    predictor_fn,         # callable(threshold) -> dict with prob_above
) -> dict | None:
    """
    Scan open BTC hourly markets, score each by edge × liquidity,
    return the best one or None if no edge found.

    Returns dict: {ticker, side, contracts_price_cents, prob, edge, ...}
    """
    markets = client.get_markets(series_ticker=BTC_SERIES_TICKER, status="open")
    if not markets:
        return None

    best = None
    best_score = 0.0

    for market in markets:
        ticker = market["ticker"]
        try:
            threshold = _parse_strike(market)
        except (KeyError, ValueError):
            continue

        pred = predictor_fn(threshold)
        prob = pred["prob_above"]
        confidence = pred["confidence"]

        ob = client.get_orderbook(ticker, depth=3)
        yes_ask = _best_ask(ob, "yes")
        no_ask = _best_ask(ob, "no")

        # YES side
        if yes_ask:
            edge_yes = compute_edge(prob, yes_ask)
            if edge_yes >= MIN_EDGE:
                score = edge_yes * confidence
                if score > best_score:
                    best_score = score
                    best = {
                        "ticker": ticker,
                        "side": "yes",
                        "price_cents": yes_ask,
                        "prob": prob,
                        "edge": round(edge_yes, 4),
                        "confidence": confidence,
                        "threshold": threshold,
                        "signals": pred["signals"],
                    }

        # NO side (prob_below = 1 - prob_above)
        if no_ask:
            edge_no = compute_edge(1 - prob, no_ask)
            if edge_no >= MIN_EDGE:
                score = edge_no * confidence
                if score > best_score:
                    best_score = score
                    best = {
                        "ticker": ticker,
                        "side": "no",
                        "price_cents": no_ask,
                        "prob": 1 - prob,
                        "edge": round(edge_no, 4),
                        "confidence": confidence,
                        "threshold": threshold,
                        "signals": pred["signals"],
                    }

    return best


def _parse_strike(market: dict) -> float:
    """Extract the USD strike price from market subtitle or rules."""
    # Kalshi typically encodes the level in the subtitle like "BTC above $65,000"
    import re
    subtitle = market.get("subtitle", "") or market.get("title", "")
    m = re.search(r"\$([0-9,]+)", subtitle)
    if m:
        return float(m.group(1).replace(",", ""))
    raise ValueError(f"Cannot parse strike from: {subtitle}")


def _best_ask(orderbook: dict, side: str) -> int | None:
    """Return the lowest ask price (in cents) for a given side."""
    key = f"{side}s"  # "yes" → "yess" in some responses; adjust if needed
    # Kalshi orderbook structure: {"yes": [[price, size], ...], "no": [...]}
    levels = orderbook.get(side, [])
    if not levels:
        return None
    # levels sorted ask-first (ascending price)
    return int(levels[0][0])
