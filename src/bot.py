"""
Two-phase Kalshi BTC bot. Runs every 2 minutes via cron.

Phase 1  (8–52 min into hour):  NO on OTM — sell overpriced vol, collect theta
Phase 2  (last 5 min):          ATM YES — exploit market's failure to reprice
Baseline (last 5 min):          High-conf YES on far-below-strike contracts

Each run the bot detects which phase applies and fires the right strategies.
A per-expiry dedup guard prevents re-entering the same ticker twice.

Usage:
    python src/bot.py --dry-run
    python src/bot.py
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MIN_CONTRACTS, MIN_PROB_YES, MIN_BUFFER_USD
from src.btc_feed import fetch_hourly_candles, brti_approx
from src.kalshi_client import KalshiClient
from src.market_selector import find_best_contract
from src.predictor import predict_above_threshold
from src.portfolio import (
    build_state, print_portfolio_summary,
    size_high_conf_yes, size_atm_yes, size_no_otm, size_sell_yes,
)
from src.strategies.atm_yes import find_atm_opportunity, size_atm
from src.strategies.no_otm import find_opportunities as find_otm_opportunities
from src.strategies.last15_spread import find_spread_opportunities, size_spread_full

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "bot.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Dedup state ───────────────────────────────────────────────────────────────
TRADED_FILE = Path(__file__).parent.parent / "data" / "traded.json"

def _load_traded() -> dict:
    if TRADED_FILE.exists():
        try:
            return json.loads(TRADED_FILE.read_text())
        except Exception:
            pass
    return {}

def _save_traded(d: dict):
    TRADED_FILE.parent.mkdir(exist_ok=True)
    TRADED_FILE.write_text(json.dumps(d, indent=2))

def _already_traded(ticker: str, expiry_str: str) -> bool:
    d = _load_traded()
    return d.get(expiry_str, {}).get(ticker, False)

def _mark_traded(ticker: str, expiry_str: str):
    d = _load_traded()
    d.setdefault(expiry_str, {})[ticker] = True
    # Prune old expiries (keep last 5)
    keys = sorted(d.keys())
    for old in keys[:-5]:
        del d[old]
    _save_traded(d)


# ── Time phase detection ──────────────────────────────────────────────────────

def _phase_info(expiry_utc: datetime) -> dict:
    now = datetime.now(timezone.utc)
    mins_remaining = (expiry_utc - now).total_seconds() / 60
    mins_into_hour = 60 - mins_remaining

    phase = None
    if 8 <= mins_remaining <= 52:
        phase = "no_otm"       # Phase 1: sell vol
    if mins_remaining <= 5:
        phase = "atm_expiry"   # Phase 2: ATM + high_conf

    return {
        "mins_remaining": round(mins_remaining, 1),
        "mins_into_hour": round(mins_into_hour, 1),
        "phase": phase,
        "expiry": expiry_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
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


# ═══════════════════════════════════════════════════════════════
# Strategy runners
# ═══════════════════════════════════════════════════════════════

def run_no_otm(client, markets_by_series, df, spot, state, phase, cash, dry_run):
    """Phase 1: buy NO on far-above contracts, YES on far-below contracts."""
    log.info(f"── Phase 1 · otm_scanner  ({phase['mins_remaining']:.0f} min remaining) ──────")

    candidates = find_otm_opportunities(markets_by_series, spot, df)
    if not candidates:
        log.info("  No qualifying trades (need yes_bid≥1¢ for NO, or cheap YES below spot).")
        return

    expiry_str = phase["expiry"]
    for opp in candidates:
        if _already_traded(opp["ticker"], expiry_str):
            continue
        if cash[0] < opp["price"]:
            log.info(f"  Cash exhausted (${cash[0]:.2f}) — stop.")
            break

        # Size: Kelly based on trade direction
        if opp["side"] == "no":
            prob_win = 1.0 - opp["model_prob"]
            n, detail = size_no_otm(opp["price"], prob_win, state, cash[0])
        else:
            n, detail = size_high_conf_yes(opp["price"], opp["model_prob"], state, cash[0])

        if n < 1:
            log.info(f"  {opp['subtitle']}: Kelly gives 0 — skip.")
            continue

        cost = n * opp["price"]
        log.info(
            f"  [{opp['trade_type']}] {opp['series']} {opp['subtitle']}\n"
            f"    {opp['side'].upper()} @ {opp['price_cents']}¢  model={opp['model_prob']:.3%}"
            f"  edge={opp['edge']:.4f}  dist={opp['dist_pct']:+.1f}%  {opp['note']}\n"
            f"    Kelly: {detail['full_kelly_f']:.3f}×{detail['frac_kelly']}"
            f"  → {n}×  cost=${cost:.2f}  EV=${detail['ev']}"
        )

        if dry_run:
            log.info("    [DRY RUN]")
            _mark_traded(opp["ticker"], expiry_str)
            cash[0] -= cost
            continue

        try:
            r = client.place_order(opp["ticker"], opp["side"], n,
                                   price_dollars=opp["price"])
            log.info(f"    ✓ {r.get('order', {}).get('order_id', r)}")
            _mark_traded(opp["ticker"], expiry_str)
            cash[0] -= cost
        except Exception as e:
            log.error(f"    ✗ {e}")


def run_last15_spread(client, markets, spot, state, phase, cash, dry_run):
    """Phase 2: buy YES on all 7 nearby buckets at 1¢ — empirically calibrated spread."""
    log.info(f"── Phase 2 · last15_spread  ({phase['mins_remaining']:.1f} min remaining) ──")

    expiry_utc = datetime.fromisoformat(phase["expiry"].replace("Z", "+00:00"))
    opps = find_spread_opportunities(markets, spot, expiry_utc)

    if not opps:
        log.info("  No buckets qualify (need ask ≤4¢, within 15 min, contracts exist).")
        return

    expiry_str = phase["expiry"]
    new_opps = [o for o in opps if not _already_traded(o["ticker"], expiry_str)]
    if not new_opps:
        log.info("  All buckets already traded this expiry.")
        return

    # Allocate ALL available cash proportionally across qualifying buckets
    sizes = size_spread_full(new_opps, cash[0])
    total_prob  = sum(o["emp_prob"] for o in new_opps)
    total_cost_ = sum(sizes[o["ticker"]] * o["yes_ask"] for o in new_opps)

    log.info(
        f"  {len(new_opps)} buckets  spot=${spot:,.0f}  "
        f"ATM=[{math.floor(spot/100)*100:.0f},{math.floor(spot/100)*100+100:.0f})  "
        f"{phase['mins_remaining']:.1f} min left\n"
        f"  Deploying ${cash[0]:.2f} → total cost ~${total_cost_:.2f}  "
        f"coverage={total_prob:.1%}"
    )

    total_spent = 0.0
    for opp in new_opps:
        n = sizes.get(opp["ticker"], 0)
        if n < 1:
            continue
        cost = n * opp["yes_ask"]
        log.info(
            f"  offset={opp['offset']:+d}  {opp['subtitle']}"
            f"  ask={opp['ask_cents']}¢  emp={opp['emp_prob']:.1%}"
            f"  edge={opp['edge']:.4f}  → {n:,}×  cost=${cost:.2f}"
        )

        if dry_run:
            _mark_traded(opp["ticker"], expiry_str)
            cash[0] -= cost
            total_spent += cost
            continue

        try:
            r = client.place_order(opp["ticker"], "yes", n, price_dollars=opp["yes_ask"])
            log.info(f"    ✓ {r.get('order', {}).get('order_id', r)}")
            _mark_traded(opp["ticker"], expiry_str)
            cash[0] -= cost
            total_spent += cost
        except Exception as e:
            log.error(f"    ✗ {e}")

    log.info(f"  {'[DRY RUN] ' if dry_run else ''}Total deployed: ${total_spent:.2f}  "
             f"cash remaining: ${cash[0]:.2f}")


def run_atm_yes(client, markets, df, spot, state, phase, cash, dry_run):
    """Phase 2a: ATM YES in last 15 minutes."""
    log.info(f"── Phase 2a · atm_yes  ({phase['mins_remaining']:.1f} min remaining) ──")

    expiry_utc = datetime.fromisoformat(phase["expiry"].replace("Z", "+00:00"))
    opp = find_atm_opportunity(markets, spot, df, expiry_utc)

    if opp is None:
        log.info("  No ATM bucket qualifies.")
        return

    if _already_traded(opp["ticker"], phase["expiry"]):
        log.info(f"  {opp['subtitle']}: already entered — skip.")
        return

    n, detail = size_atm_yes(opp["yes_ask"], opp["model_prob"], state, cash[0])
    if n < 1:
        log.info("  Kelly gives 0 contracts — skip.")
        return

    cost = n * opp["yes_ask"]
    log.info(
        f"  {opp['subtitle']}\n"
        f"    model={opp['model_prob']:.1%}  ask={opp['yes_ask_cents']}¢"
        f"  edge={opp['edge']:.3f}  {opp['mins_remaining']:.1f} min left\n"
        f"    Kelly: {detail['full_kelly_f']:.3f}×{detail['frac_kelly']}"
        f"  → {n}×  cost=${cost:.2f}  EV=${detail['ev']}"
    )

    if dry_run:
        log.info("    [DRY RUN]")
        _mark_traded(opp["ticker"], phase["expiry"])
        cash[0] -= cost
        return

    try:
        r = client.place_order(opp["ticker"], "yes", n, price_dollars=opp["yes_ask"])
        log.info(f"    ✓ {r.get('order', {}).get('order_id', r)}")
        _mark_traded(opp["ticker"], phase["expiry"])
        cash[0] -= cost
    except Exception as e:
        log.error(f"    ✗ {e}")




def run_high_conf_yes(client, markets, df, spot, state, phase, cash, dry_run):
    """Phase 2b: high-confidence YES on far-below-strike contracts (KXBTCD)."""
    log.info(f"── Phase 2b · high_conf_yes  ({phase['mins_remaining']:.1f} min remaining) ──")

    def predictor(strike, expiry_utc):
        return predict_above_threshold(df, strike, spot, expiry_utc)

    best = find_best_contract(client, spot, predictor)
    if best is None:
        log.info("  No qualifying contract (prob≥99%, buffer≥$200).")
        return

    if _already_traded(best["ticker"], phase["expiry"]):
        log.info(f"  {best['ticker']}: already entered — skip.")
        return

    price_d = best["price_cents"] / 100
    n, detail = size_high_conf_yes(price_d, best["prob"], state, cash[0])

    log.info(
        f"  {best['ticker']}  strike=${best['strike']:,.0f}"
        f"  buffer={best['buffer_pct']:+.2f}%  prob={best['prob']:.1%}"
        f"  ask={best['price_cents']}¢  edge={best['edge']:.3f}\n"
        f"    Kelly: {detail['full_kelly_f']:.3f}×{detail['frac_kelly']}"
        f"  → {n}×  cost=${detail['cost']:.2f}  EV=${detail['ev']}"
    )

    if n < MIN_CONTRACTS:
        log.info(f"  Kelly gives {n} contracts — skip.")
        return
    if dry_run:
        log.info("    [DRY RUN]")
        _mark_traded(best["ticker"], phase["expiry"])
        cash[0] -= detail["cost"]
        return

    try:
        r = client.place_order(best["ticker"], "yes", n, price_dollars=price_d)
        log.info(f"    ✓ {r.get('order', {}).get('order_id', r)}")
        _mark_traded(best["ticker"], phase["expiry"])
        cash[0] -= detail["cost"]
    except Exception as e:
        log.error(f"    ✗ {e}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def run(dry_run: bool = False):
    now = datetime.now(timezone.utc)
    log.info("=" * 62)
    log.info(f"Bot  {now.strftime('%Y-%m-%d %H:%M:%S UTC')}  dry={dry_run}")
    log.info("=" * 62)

    # ── Data ──────────────────────────────────────────────────────
    brti = brti_approx()
    spot = brti["price"]
    log.info(f"BRTI: ${spot:,.2f}  spread=${max(brti['individual_prices'].values())-min(brti['individual_prices'].values()):.0f}")

    df = fetch_hourly_candles()

    # ── Portfolio ─────────────────────────────────────────────────
    client = KalshiClient()
    try:
        balance   = client.get_balance()
        positions = client.get_positions()
    except Exception as e:
        log.error(f"Kalshi fetch failed: {e}")
        return

    state = build_state(balance, positions)
    print_portfolio_summary(state)

    if state.cash_dollars < 1.0:
        log.warning("Cash too low — abort.")
        return

    # ── Markets + phase ───────────────────────────────────────────
    markets_kxbtc  = client.get_markets(series_ticker="KXBTC",  status="open")
    markets_kxbtcd = client.get_markets(series_ticker="KXBTCD", status="open")
    markets = markets_kxbtc + markets_kxbtcd  # combined for phase detection
    markets_by_series = {"KXBTC": markets_kxbtc, "KXBTCD": markets_kxbtcd}
    expiry  = _nearest_expiry(markets)

    if expiry is None:
        log.warning("No open markets found.")
        return

    phase = _phase_info(expiry)
    log.info(
        f"Next expiry: {phase['expiry']}  "
        f"{phase['mins_remaining']:.1f} min remaining  "
        f"→ phase={phase['phase'] or 'none (outside window)'}"
    )

    if phase["phase"] is None:
        log.info("Outside trading windows — nothing to do.")
        return

    if state.available_to_risk < 0.50:
        log.warning(f"Run budget ${state.available_to_risk:.2f} too low — abort.")
        return

    # Shared cash pool — strategies draw from this in priority order
    cash = [state.available_to_risk]
    log.info(f"Cash pool: ${cash[0]:.2f} available to deploy")

    # ── Dispatch by phase ─────────────────────────────────────────
    if phase["phase"] == "no_otm":
        run_no_otm(client, markets_by_series, df, spot, state, phase, cash, dry_run)

    elif phase["phase"] == "atm_expiry":
        run_last15_spread(client, markets_kxbtc, spot, state, phase, cash, dry_run)
        run_atm_yes(client, markets_kxbtc, df, spot, state, phase, cash, dry_run)
        run_high_conf_yes(client, markets_kxbtcd, df, spot, state, phase, cash, dry_run)

    log.info(f"Cash remaining after run: ${cash[0]:.2f}")

    log.info("Run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
