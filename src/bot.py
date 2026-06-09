"""
Main bot loop. Runs once per hour (cron or scheduler), 5 minutes before close.

Usage:
    python src/bot.py             # single run (call from cron at :55 each hour)
    python src/bot.py --dry-run   # simulate without placing real orders
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MIN_CONTRACTS
from src.btc_feed import fetch_hourly_candles, current_btc_price
from src.kalshi_client import KalshiClient
from src.market_selector import find_best_contract
from src.predictor import predict_above_threshold
from src.risk import kelly_contracts

# ── logging ───────────────────────────────────────────────────────────────────
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


def run(dry_run: bool = False):
    now = datetime.now(timezone.utc)
    log.info(f"=== Bot run started at {now.isoformat()} (dry_run={dry_run}) ===")

    # 1. Fetch BTC price data
    df = fetch_hourly_candles()
    spot = current_btc_price()
    log.info(f"BTC spot: ${spot:,.2f}  |  {len(df)} candles loaded")

    # 2. Connect to Kalshi
    client = KalshiClient()
    balance = client.get_balance()
    balance_cents = balance.get("balance", 0)
    log.info(f"Kalshi balance: ${balance_cents / 100:.2f}")

    if balance_cents < 100:  # less than $1
        log.warning("Balance too low to trade — aborting.")
        return

    # 3. Find best contract
    def predictor(threshold):
        return predict_above_threshold(df, threshold, spot)

    best = find_best_contract(client, spot, predictor)

    if best is None:
        log.info("No contract with sufficient edge found — sitting out this hour.")
        return

    log.info(
        f"Best contract: {best['ticker']} | side={best['side']} "
        f"@ {best['price_cents']}¢ | edge={best['edge']:.3f} | "
        f"prob={best['prob']:.3f} | confidence={best['confidence']:.2f}"
    )
    log.info(f"Signals: {json.dumps(best['signals'])}")

    # 4. Size position
    n_contracts = kelly_contracts(
        prob=best["prob"],
        market_price_cents=best["price_cents"],
        balance_cents=balance_cents,
        confidence=best["confidence"],
    )

    if n_contracts < MIN_CONTRACTS:
        log.info(f"Kelly sizing returned {n_contracts} contracts — skipping.")
        return

    cost_usd = n_contracts * best["price_cents"] / 100
    log.info(f"Placing order: {n_contracts} × {best['side'].upper()} @ {best['price_cents']}¢  (cost ~${cost_usd:.2f})")

    # 5. Place order
    if dry_run:
        log.info("[DRY RUN] Order not submitted.")
        return

    try:
        result = client.place_order(
            ticker=best["ticker"],
            side=best["side"],
            contracts=n_contracts,
            price_cents=best["price_cents"],
        )
        log.info(f"Order placed: {json.dumps(result)}")
    except Exception as e:
        log.error(f"Order failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
