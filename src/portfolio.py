"""
Portfolio sizing — full deployment mode.

Deploys all available cash each hour across whichever strategies
have edge. Kelly fractions at half-Kelly (0.5) per strategy;
the only hard limit is available cash itself.

Allocation priority (run in order, each claiming from the same cash pool):
  1. sell_otm_yes  — highest edge/risk ratio, most capital
  2. no_otm        — second highest, also high capital
  3. atm_yes       — small cost per contract, deploy remaining
  4. high_conf_yes — last, usually near expiry

The strategies compete for the same cash pool. When cash is exhausted,
later strategies get $0. This naturally concentrates capital in the
best-edge opportunities found first.
"""
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Kelly fractions (half-Kelly = 0.50) ──────────────────────────────────────
# Half-Kelly is the standard for real-money trading — theoretically optimal
# growth rate while cutting variance by 75% vs full Kelly.
KELLY_FRACTIONS = {
    "high_conf_yes": 0.50,
    "atm_yes":       0.50,
    "no_otm":        0.50,
    "sell_otm_yes":  0.50,
}

# Keep 2% as a buffer for fees and rounding — otherwise deploy everything
CASH_BUFFER_PCT  = 0.02
MIN_CONTRACTS    = 1


@dataclass
class PortfolioState:
    cash_dollars:      float
    portfolio_dollars: float
    open_risk_dollars: float
    open_positions:    list

    @property
    def available_to_risk(self) -> float:
        """All cash minus a small fee buffer."""
        return max(self.cash_dollars * (1 - CASH_BUFFER_PCT), 0)

    @property
    def per_strategy_cap(self) -> float:
        """No hard per-strategy cap — Kelly math handles allocation."""
        return self.available_to_risk  # effectively uncapped


def build_state(balance_response: dict, positions: list[dict]) -> PortfolioState:
    cash       = float(balance_response.get("balance_dollars", 0) or 0)
    port_value = float(balance_response.get("portfolio_value", 0) or 0) / 100
    if port_value < cash:
        exposure   = sum(float(p.get("market_exposure_dollars", 0) or 0) for p in positions)
        port_value = cash + exposure

    open_risk = sum(
        float(p.get("market_exposure_dollars", 0) or 0)
        for p in positions
        if float(p.get("position_fp", 0) or 0) < 0
    )

    return PortfolioState(
        cash_dollars=cash,
        portfolio_dollars=port_value,
        open_risk_dollars=open_risk,
        open_positions=positions,
    )


# ── Core Kelly sizing ─────────────────────────────────────────────────────────

def kelly_size(
    strategy:   str,
    prob_win:   float,
    cost_per:   float,
    payout_per: float,
    state:      PortfolioState,
    cash_remaining: float | None = None,  # pass to avoid double-spending
) -> tuple[int, dict]:
    """
    Half-Kelly sizing against available cash.
    cash_remaining: if provided, caps bet to this (tracks spend across strategies).
    """
    if prob_win <= 0 or prob_win > 0.9999 or cost_per <= 0 or payout_per <= 0:
        return 0, {"reason": "invalid inputs"}

    b    = payout_per / cost_per
    q    = 1 - prob_win
    edge = b * prob_win - q
    if edge <= 0:
        return 0, {"reason": "no edge", "edge": round(edge, 4)}

    full_kelly_f = edge / b
    frac_kelly   = KELLY_FRACTIONS.get(strategy, 0.50)
    scaled_f     = full_kelly_f * frac_kelly

    kelly_dollars = state.portfolio_dollars * scaled_f

    budget = cash_remaining if cash_remaining is not None else state.available_to_risk
    dollar_bet = min(kelly_dollars, budget)

    n = int(dollar_bet / cost_per)
    actual_cost = n * cost_per

    detail = {
        "portfolio":    round(state.portfolio_dollars, 2),
        "cash":         round(state.cash_dollars, 2),
        "budget":       round(budget, 2),
        "full_kelly_f": round(full_kelly_f, 4),
        "frac_kelly":   frac_kelly,
        "kelly_$":      round(kelly_dollars, 2),
        "bet_$":        round(dollar_bet, 2),
        "n_contracts":  n,
        "cost":         round(actual_cost, 2),
        "max_profit":   round(n * payout_per, 2),
        "ev":           round(n * (prob_win * payout_per - q * cost_per), 2),
    }

    return max(n, 0), detail


# ── Per-strategy wrappers ─────────────────────────────────────────────────────

def size_high_conf_yes(price_dollars: float, prob: float,
                       state: PortfolioState,
                       cash_remaining: float | None = None) -> tuple[int, dict]:
    return kelly_size("high_conf_yes", prob, price_dollars,
                      1.0 - price_dollars, state, cash_remaining)


def size_atm_yes(price_dollars: float, prob: float,
                 state: PortfolioState,
                 cash_remaining: float | None = None) -> tuple[int, dict]:
    return kelly_size("atm_yes", prob, price_dollars,
                      1.0 - price_dollars, state, cash_remaining)


def size_no_otm(no_ask_dollars: float, prob_no_wins: float,
                state: PortfolioState,
                cash_remaining: float | None = None) -> tuple[int, dict]:
    return kelly_size("no_otm", prob_no_wins, no_ask_dollars,
                      1.0 - no_ask_dollars, state, cash_remaining)


def size_sell_yes(sell_price: float, model_prob: float,
                  state: PortfolioState,
                  cash_remaining: float | None = None) -> tuple[int, dict]:
    # Selling YES: win = NO wins = 1-model_prob; profit = sell_price; loss = 1-sell_price
    # Clip so prob_win stays in (0, 0.9999) — avoids hitting the invalid-inputs guard
    # when model_prob rounds to exactly 0.0 (very far OTM contracts)
    prob_win = min(1.0 - model_prob, 0.9999)
    return kelly_size("sell_otm_yes", prob_win, 1.0 - sell_price,
                      sell_price, state, cash_remaining)


# ── Portfolio summary ─────────────────────────────────────────────────────────

def print_portfolio_summary(state: PortfolioState):
    from datetime import datetime, timezone
    utilisation = (state.portfolio_dollars - state.cash_dollars) / state.portfolio_dollars
    print(f"\n  Portfolio @ {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
    print(f"  {'Total value:':<22} ${state.portfolio_dollars:>8.2f}")
    print(f"  {'Free cash:':<22} ${state.cash_dollars:>8.2f}  "
          f"({state.cash_dollars/max(state.portfolio_dollars,1):.0%} of portfolio)")
    print(f"  {'Deployed:':<22} ${state.portfolio_dollars-state.cash_dollars:>8.2f}  "
          f"({utilisation:.0%} utilisation)")
    print(f"  {'Available to risk:':<22} ${state.available_to_risk:>8.2f}")
    print(f"  {'Open positions:':<22} {len(state.open_positions)}")
    print()
