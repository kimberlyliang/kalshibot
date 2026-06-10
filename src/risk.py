"""Position sizing: small fixed bet with balance guardrails."""
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import BET_SIZE_USD, MAX_BALANCE_FRACTION, MIN_CONTRACTS


def size_contracts(
    price_cents: int,
    balance_cents: int,
) -> int:
    """
    Buy as many contracts as $BET_SIZE_USD allows,
    capped at MAX_BALANCE_FRACTION of current balance.

    Returns number of contracts (0 = skip).
    """
    cost_per = price_cents / 100          # cost per contract in USD
    balance_usd = balance_cents / 100

    # Hard cap: never risk more than MAX_BALANCE_FRACTION of balance
    max_spend = min(BET_SIZE_USD, balance_usd * MAX_BALANCE_FRACTION)

    if cost_per <= 0 or max_spend <= 0:
        return 0

    contracts = int(max_spend / cost_per)
    return contracts if contracts >= MIN_CONTRACTS else 0


def compute_edge(prob: float, market_price_cents: int) -> float:
    """Edge in probability points (positive = we have edge on YES side)."""
    return prob - (market_price_cents / 100)
