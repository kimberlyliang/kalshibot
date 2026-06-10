"""
Backtest: Buy NO above / Buy YES below.

  NO_above: BTC at $62,500, buy NO on "above $64,000" (unlikely to rise $1,500)
  YES_below: BTC at $62,500, buy YES on "above $61,000" (already $1,500 above)

Market price simulation:
  Contracts only exist in a realistic range around spot (±8%).
  Spread model: market maker charges half-spread of 2¢.
  For NO: no_ask = 1 - yes_bid = 1 - (model - 0.02). Valid only if model > 0.03.
  For YES: yes_ask = model + 0.02. Valid only if model < 0.98.

Kelly sizing: half-Kelly with hard 10% portfolio cap per trade.

Run:
    python src/backtest.py
    python src/backtest.py --candles 1000 --entry-mins 15
    python src/backtest.py --entry-mins 30 --spread 0.03
"""
import argparse
import math
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.predictor import vol_over_horizon, realized_vol_hourly

# ── Tunable parameters ────────────────────────────────────────────────────────
# Empirically measured hit rates (from 973 candles, Monte Carlo simulation):
# These replace the lognormal model for contracts within 2% of spot.
# The lognormal underestimates fat-tailed BTC moves by 2–9x at short horizons.
#
# Format: (min_dist_pct, max_dist_pct) -> empirical_hit_rate at 30min entry
# For 15min entry, hit rates are ~1.3x higher for 0.5-1% and similar for 1-2%
EMPIRICAL_HIT_RATES_30 = {
    (0.005, 0.010): 0.039,   # 0.5–1.0% away: 3.9% hit rate
    (0.010, 0.020): 0.011,   # 1.0–2.0% away: 1.1% hit rate
    (0.020, 0.040): 0.0005,  # 2.0–4.0% away: ~0% hit rate
    (0.040, 0.150): 0.0001,  # 4%+: essentially 0
}
EMPIRICAL_HIT_RATES_15 = {
    (0.005, 0.010): 0.051,   # 0.5–1.0% away: 5.1% hit rate at 15min
    (0.010, 0.020): 0.012,   # 1.0–2.0% away: 1.2% hit rate
    (0.020, 0.040): 0.0005,
    (0.040, 0.150): 0.0001,
}

HALF_SPREAD       = 0.02    # market maker half-spread (2¢)
MIN_EDGE          = 0.005   # minimum empirical edge to enter
MIN_MARKET_BID    = 0.01    # YES bid must be ≥ 1¢ (contract exists/is liquid)
MIN_DIST_PCT      = 0.005   # ≥ 0.5% from spot
MAX_DIST_NO_PCT   = 0.08    # NO trade: look up to 8% above spot
MAX_DIST_YES_PCT  = 0.04    # YES trade: look up to 4% below spot
STRIKE_STEP       = 100     # $100 increments


def empirical_prob(dist_pct_abs: float, entry_mins: float) -> float:
    """Return empirical hit rate for a given distance (absolute %) and entry time."""
    table = EMPIRICAL_HIT_RATES_15 if entry_mins <= 15 else EMPIRICAL_HIT_RATES_30
    for (lo, hi), rate in table.items():
        if lo <= dist_pct_abs < hi:
            return rate
    return 0.0001

# Kelly
KELLY_FRAC        = 0.50
MAX_TRADE_PCT     = 0.10    # never bet more than 10% of portfolio per trade
STARTING_BAL      = 200.0


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_candles(n: int = 300) -> pd.DataFrame:
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    all_rows, end = [], datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    while len(all_rows) < n:
        batch = min(300, n - len(all_rows))
        start = end - timedelta(hours=batch)
        r = httpx.get(url, params={"granularity": 3600,
                                    "start": start.isoformat(),
                                    "end":   end.isoformat()}, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        all_rows.extend(rows)
        end = start
    df = pd.DataFrame(all_rows, columns=["time","low","high","open","close","volume"])
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    df["open_time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def prob_above(spot: float, strike: float, sigma_h: float) -> float:
    if sigma_h <= 0:
        return 1.0 if spot > strike else 0.0
    cdf = lambda x: (1 + math.erf(x / math.sqrt(2))) / 2
    d2  = (math.log(spot / strike) - 0.5 * sigma_h**2) / sigma_h
    return float(np.clip(cdf(d2), 0.0, 1.0))


def kelly_n(prob_win: float, cost_per: float, payout_per: float,
            portfolio: float) -> tuple[int, float]:
    """Half-Kelly with 10% portfolio hard cap."""
    prob_win = float(np.clip(prob_win, 0.001, 0.999))
    b    = payout_per / cost_per
    edge = b * prob_win - (1.0 - prob_win)
    if edge <= 0 or cost_per <= 0:
        return 0, 0.0
    f          = (edge / b) * KELLY_FRAC
    kelly_usd  = portfolio * f
    cap_usd    = portfolio * MAX_TRADE_PCT      # hard cap
    dollar_bet = min(kelly_usd, cap_usd)
    n          = int(dollar_bet / cost_per)
    return max(n, 0), n * cost_per


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, entry_mins: float = 30.0) -> pd.DataFrame:
    warmup    = 24
    entry_h   = entry_mins / 60
    np.random.seed(42)
    portfolio = STARTING_BAL
    trades    = []

    for i in range(warmup, len(df) - 1):
        hist   = df.iloc[:i + 1]
        candle = df.iloc[i]
        settle = float(df.iloc[i + 1]["open"])

        open_p   = float(candle["open"])
        log_rets = np.log(hist["close"] / hist["close"].shift(1)).dropna().tail(24)
        vol_h    = float(log_rets.std())
        vol_ann  = vol_h * math.sqrt(8760)
        sigma_h  = vol_over_horizon(vol_ann, entry_h)

        # Simulate entry price at (60 - entry_mins) min into the hour
        elapsed  = vol_h * math.sqrt((60 - entry_mins) / 60)
        entry_p  = open_p * math.exp(np.random.normal(-0.5*elapsed**2, elapsed))

        # ── Generate strikes ──────────────────────────────────────────────────
        base = round(entry_p / STRIKE_STEP) * STRIKE_STEP

        candidates = []   # collect all qualifying trades, take best 3

        # ── NO_above ──────────────────────────────────────────────────────────
        for mult in range(1, int(MAX_DIST_NO_PCT / (STRIKE_STEP / entry_p)) + 2):
            strike = base + mult * STRIKE_STEP
            dist   = (strike - entry_p) / entry_p
            if dist < MIN_DIST_PCT or dist > MAX_DIST_NO_PCT:
                continue

            # Use empirical hit rate instead of lognormal model
            emp_hit   = empirical_prob(dist, entry_mins)   # P(BTC reaches strike)
            prob_no   = 1.0 - emp_hit                      # P(NO wins)

            # Simulate market YES bid: market knows there's some probability,
            # bids at emp_hit minus half-spread (but at least MIN_MARKET_BID)
            yes_bid = max(emp_hit - HALF_SPREAD, MIN_MARKET_BID)
            no_ask  = 1.0 - yes_bid
            edge    = prob_no - no_ask   # = yes_bid - emp_hit

            if edge < MIN_EDGE:
                continue

            candidates.append(dict(
                trade_type="NO_above", strike=strike,
                dist_pct=round(dist*100,2),
                our_model=round(prob_no,5), mkt_model=round(1-no_ask,5),
                price=no_ask, edge=edge, won=(settle<=strike),
            ))

        # ── YES_below ─────────────────────────────────────────────────────────
        for mult in range(1, int(MAX_DIST_YES_PCT / (STRIKE_STEP / entry_p)) + 2):
            strike = base - mult * STRIKE_STEP
            dist   = (entry_p - strike) / entry_p
            if dist < MIN_DIST_PCT or dist > MAX_DIST_YES_PCT:
                continue

            # Empirical: P(BTC drops through strike) = same as hitting upside
            emp_drop   = empirical_prob(dist, entry_mins)
            prob_yes   = 1.0 - emp_drop

            if prob_yes < 0.85:
                continue

            # Market asks yes_ask above fair value (conservative: adds spread)
            yes_ask = min(prob_yes + HALF_SPREAD, 0.99)
            edge    = prob_yes - yes_ask

            if edge < MIN_EDGE:
                continue

            candidates.append(dict(
                trade_type="YES_below", strike=strike,
                dist_pct=round(-dist*100,2),
                our_model=round(prob_yes,5), mkt_model=round(yes_ask,5),
                price=yes_ask, edge=edge, won=(settle>strike),
            ))

        # Take top 3 by edge, size with Kelly, track portfolio
        for cand in sorted(candidates, key=lambda x: x["edge"], reverse=True)[:3]:
            ttype = cand["trade_type"]
            p     = cand["price"]
            if ttype == "NO_above":
                n, cost = kelly_n(1.0 - cand["our_model"], p, 1.0 - p, portfolio)
            else:
                n, cost = kelly_n(cand["our_model"], p, 1.0 - p, portfolio)
            if n < 1:
                continue
            pnl = n * (1.0 - p) if cand["won"] else -cost
            portfolio = max(portfolio + pnl, 1.0)
            trades.append(dict(
                open_time=candle["open_time"],
                trade_type=ttype,
                entry_p=round(entry_p,2),
                strike=cand["strike"],
                dist_pct=cand["dist_pct"],
                our_model=round(cand["our_model"],5),
                mkt_model=round(cand["mkt_model"],5),
                price=round(p,4),
                edge=round(cand["edge"],4),
                n=n, cost=round(cost,2),
                settle=round(settle,2),
                won=cand["won"],
                pnl=round(pnl,4),
                portfolio=round(portfolio,2),
                vol_ann=round(vol_ann*100,1),
            ))

    return pd.DataFrame(trades)


# ── Report ────────────────────────────────────────────────────────────────────

def report(trades: pd.DataFrame, entry_mins: float):
    if trades.empty:
        print("No qualifying trades."); return

    print(f"\n{'='*64}")
    print(f"  BACKTEST  entry={entry_mins:.0f}min  half-Kelly  "
          f"10%-cap  spread={HALF_SPREAD*100:.0f}¢")
    print(f"  {df['open_time'].iloc[0].date()} → {df['open_time'].iloc[-1].date()}")
    print(f"{'='*64}")

    for ttype in ["NO_above", "YES_below", "ALL"]:
        t = trades if ttype == "ALL" else trades[trades["trade_type"] == ttype]
        if t.empty: continue
        n    = len(t)
        wr   = t["won"].mean()
        pnl  = t["pnl"].sum()
        cost = t["cost"].sum()
        roi  = pnl / cost * 100 if cost else 0
        cum  = t["pnl"].cumsum()
        dd   = float((cum.cummax() - cum).max())
        sh   = (t["pnl"].mean() / t["pnl"].std() * math.sqrt(8760)
                if t["pnl"].std() > 0 else 0)
        avg_cost = t["cost"].mean()
        avg_edge = t["edge"].mean()
        print(f"\n  [{ttype}]  n={n}  win={wr:.1%}  avg_edge={avg_edge:.4f}  "
              f"P&L=${pnl:+.2f}  ROI={roi:+.1f}%  "
              f"MaxDD=${dd:.2f}  Sharpe={sh:.2f}  avg_cost=${avg_cost:.2f}")

    # Distance breakdown
    print(f"\n  By distance from spot:")
    print(f"  {'Type':<12} {'Dist':>10}  {'N':>5}  {'Win%':>6}  {'ROI':>8}  {'Avg edge':>9}")
    for ttype in ["NO_above", "YES_below"]:
        t = trades[trades["trade_type"] == ttype]
        sign = 1 if ttype == "NO_above" else -1
        for lo, hi in [(0.5,1.0),(1.0,2.0),(2.0,4.0),(4.0,8.0)]:
            g = t[(t["dist_pct"].abs() >= lo) & (t["dist_pct"].abs() < hi)]
            if len(g) < 10: continue
            r = g["pnl"].sum() / g["cost"].sum() * 100
            print(f"  {ttype:<12} {lo:>4.1f}–{hi:.1f}%   "
                  f"{len(g):>5}  {g['won'].mean():>5.1%}  "
                  f"{r:>7.1f}%  {g['edge'].mean():>9.4f}")

    # Vol breakdown
    print(f"\n  By vol regime:")
    print(f"  {'Vol':>10}  {'N':>5}  {'Win%':>6}  {'ROI':>8}")
    for lo, hi in [(0,20),(20,35),(35,60),(60,200)]:
        g = trades[(trades["vol_ann"] >= lo) & (trades["vol_ann"] < hi)]
        if len(g) < 10: continue
        r = g["pnl"].sum() / g["cost"].sum() * 100
        print(f"  {lo:>3}–{hi:<4}%    {len(g):>5}  "
              f"{g['won'].mean():>5.1%}  {r:>7.1f}%")

    start = STARTING_BAL
    end   = trades["portfolio"].iloc[-1]
    print(f"\n  Portfolio: ${start:.2f} → ${end:.2f}  "
          f"({(end/start-1)*100:+.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candles",    type=int,   default=500)
    parser.add_argument("--entry-mins", type=float, default=30.0)
    parser.add_argument("--spread",     type=float, default=HALF_SPREAD)
    args = parser.parse_args()
    HALF_SPREAD = args.spread

    print(f"Fetching {args.candles} candles...")
    df = fetch_candles(args.candles)
    print(f"Got {len(df)} candles  "
          f"{df['open_time'].iloc[0].date()} → {df['open_time'].iloc[-1].date()}")
    print(f"Running backtest (entry={args.entry_mins:.0f} min remaining)...")
    trades = run_backtest(df, entry_mins=args.entry_mins)
    report(trades, args.entry_mins)
    Path("data").mkdir(exist_ok=True)
    trades.to_csv("data/backtest_trades.csv", index=False)
    print(f"\nSaved → data/backtest_trades.csv")
