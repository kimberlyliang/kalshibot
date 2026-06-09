"""Kelly criterion position sizing with hard caps."""
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import KELLY_FRACTION, MAX_TRADE_USD, MIN_CONTRACTS


def kelly_contracts(
    prob: float,
    market_price_cents: int,
    balance_cents: int,
    confidence: float = 1.0,
) -> int:
    """
    Parameters
    ----------
    prob              : our estimated probability the contract pays out
    market_price_cents: current market price (1–99 cents on a $1 contract)
    balance_cents     : account balance in cents
    confidence        : 0-1 multiplier from the predictor ensemble agreement

    Returns
    -------
    Number of contracts to buy (0 = skip trade)
    """
    p = prob
    q = 1 - p
    b = (100 - market_price_cents) / market_price_cents  # net odds if YES wins

    # Full Kelly: f* = (b*p - q) / b
    edge = b * p - q
    if edge <= 0:
        return 0

    kelly_f = edge / b

    # Fractional Kelly × confidence multiplier
    scaled_f = kelly_f * KELLY_FRACTION * confidence

    # Convert to dollar bet
    cost_per_contract = market_price_cents / 100
    balance_usd = balance_cents / 100
    dollar_bet = balance_usd * scaled_f

    # Hard cap
    dollar_bet = min(dollar_bet, MAX_TRADE_USD)

    contracts = int(dollar_bet / cost_per_contract)
    return max(contracts, 0) if contracts >= MIN_CONTRACTS else 0


def compute_edge(prob: float, market_price_cents: int) -> float:
    """Edge in probability points (positive = we have edge on YES side)."""
    return prob - (market_price_cents / 100)
