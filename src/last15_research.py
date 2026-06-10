"""
Last-15-minutes strategy research.

Key empirical findings from Monte Carlo calibration:
  ATM bucket: model=31% vs actual=13%   (model overestimates 2.4x)
  ±1 bucket:  model=21% vs actual=12%   (model overestimates 1.7x)
  ±2 bucket:  model=9%  vs actual=9.4%  (well calibrated)
  ±3 bucket:  model=3%  vs actual=6.7%  (model UNDERESTIMATES 2x)

Strategy: buy YES across multiple nearby buckets simultaneously.
  - All have positive EV at 1-4 cent market asks
  - They're mutually exclusive — at most one wins
  - Diversifying across ±0, ±1, ±2, ±3 covers 70% of settlement probability
  - Total cost ≈ 6-8 cents covers a $1 payout when we win

This script:
  1. Calibrates empirical hit rates from historical data
  2. Fits a 'corrected' probability model
  3. Runs an exhaustive backtest of all portfolio configurations
  4. Finds the optimal bucket selection and sizing
  5. Reports Sharpe, ROI, drawdown, Kelly fractions

Run: python src/last15_research.py
"""
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.backtest import fetch_candles
from src.predictor import vol_over_horizon, realized_vol_hourly


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMPIRICAL CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

BUCKET_W    = 100
OFFSETS     = [0, 1, -1, 2, -2, 3, -3]
N_SIM       = 200          # Monte Carlo draws per candle
ENTRY_MINS  = 15
WARMUP      = 72


def simulate_entries(df: pd.DataFrame, n_sim: int = N_SIM, seed: int = 42) -> pd.DataFrame:
    """
    For each candle, simulate N_SIM possible prices at t=45min (15 min before settle).
    Settlement = next candle's open.
    Returns DataFrame with columns: offset, entry_price, settle, model_p, actual_hit, vol_ann.
    """
    np.random.seed(seed)
    records = []
    cdf = lambda x: (1 + math.erf(x / math.sqrt(2))) / 2

    for i in range(WARMUP, len(df) - 1):
        hist   = df.iloc[:i + 1]
        open_p = float(df.iloc[i]["open"])
        settle = float(df.iloc[i + 1]["open"])

        log_rets = np.log(hist["close"] / hist["close"].shift(1)).dropna().tail(24)
        vol_h    = float(log_rets.std())
        vol_ann  = vol_h * math.sqrt(8760)
        sigma_15 = vol_over_horizon(vol_ann, ENTRY_MINS / 60)
        sigma_45 = vol_h * math.sqrt(45 / 60)

        entry_prices = open_p * np.exp(
            np.random.normal(-0.5 * sigma_45 ** 2, sigma_45, n_sim)
        )

        for ep in entry_prices:
            atm_lo = math.floor(ep / BUCKET_W) * BUCKET_W
            for offset in OFFSETS:
                lo = atm_lo + offset * BUCKET_W
                hi = lo + BUCKET_W
                if sigma_15 > 0:
                    d_lo = (math.log(lo / ep) + 0.5 * sigma_15 ** 2) / sigma_15
                    d_hi = (math.log(hi / ep) + 0.5 * sigma_15 ** 2) / sigma_15
                    model_p = max(cdf(d_hi) - cdf(d_lo), 0.0)
                else:
                    model_p = 1.0 if lo <= ep < hi else 0.0

                records.append({
                    "candle_idx":  i,
                    "offset":      offset,
                    "entry_price": ep,
                    "settle":      settle,
                    "vol_ann":     vol_ann,
                    "sigma_15":    sigma_15,
                    "model_p":     model_p,
                    "actual_hit":  int(lo <= settle < hi),
                })

    return pd.DataFrame(records)


def calibrate(sim: pd.DataFrame) -> pd.DataFrame:
    """Return empirical hit rates and Kelly fractions per offset."""
    rows = []
    for off in OFFSETS:
        g = sim[sim["offset"] == off]
        actual_p  = g["actual_hit"].mean()
        model_p   = g["model_p"].mean()
        # Kelly at 1-cent ask
        kelly_1c  = _kelly(actual_p, ask=0.01)
        kelly_4c  = _kelly(actual_p, ask=0.04)
        rows.append({
            "offset":   off,
            "actual_p": actual_p,
            "model_p":  model_p,
            "ratio":    actual_p / model_p if model_p > 0 else 0,
            "edge_1c":  actual_p - 0.01,
            "edge_4c":  actual_p - 0.04,
            "kelly_1c": kelly_1c,
            "kelly_4c": kelly_4c,
        })
    return pd.DataFrame(rows).set_index("offset")


def _kelly(prob: float, ask: float) -> float:
    """Full Kelly fraction for a YES bet at ask price."""
    if ask <= 0 or prob <= ask:
        return 0.0
    b = (1.0 - ask) / ask   # net odds
    f = (b * prob - (1 - prob)) / b
    return max(f, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PORTFOLIO BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def backtest_portfolio(
    df: pd.DataFrame,
    sim: pd.DataFrame,
    calib: pd.DataFrame,
    offsets_to_trade: list[int],
    ask_cents: int = 1,
    kelly_mult: float = 0.5,
    portfolio_dollars: float = 200.0,
) -> pd.DataFrame:
    """
    For each unique candle simulation, trade YES on the given offsets.
    Size each position by half-Kelly (or provided mult) of calibrated Kelly.

    Returns per-hour P&L DataFrame.
    """
    ask = ask_cents / 100
    trades = []

    # Group by candle_idx, take median entry price per candle
    for candle_idx, group in sim.groupby("candle_idx"):
        ep     = float(group["entry_price"].median())
        settle = float(group["settle"].iloc[0])

        # For this candle, how much cash to deploy total?
        # Each offset gets its Kelly-sized allocation
        hour_cost  = 0.0
        hour_pnl   = 0.0
        positions  = {}

        for off in offsets_to_trade:
            if off not in calib.index:
                continue
            cal_p = calib.loc[off, "actual_p"]
            k     = _kelly(cal_p, ask) * kelly_mult
            if k <= 0:
                continue

            dollar_bet = portfolio_dollars * k
            n_contracts = int(dollar_bet / ask)
            if n_contracts < 1:
                continue

            cost = n_contracts * ask
            atm_lo = math.floor(ep / BUCKET_W) * BUCKET_W
            lo = atm_lo + off * BUCKET_W
            hi = lo + BUCKET_W
            won = (lo <= settle < hi)
            pnl = n_contracts * (1.0 - ask) if won else -cost

            hour_cost += cost
            hour_pnl  += pnl
            positions[off] = {
                "n": n_contracts, "cost": cost, "won": won,
                "lo": lo, "hi": hi,
            }

        if hour_cost > 0:
            trades.append({
                "candle_idx":   candle_idx,
                "entry_price":  ep,
                "settle":       settle,
                "total_cost":   round(hour_cost, 4),
                "total_pnl":    round(hour_pnl, 4),
                "any_won":      any(p["won"] for p in positions.values()),
                "n_buckets":    len(positions),
            })

    return pd.DataFrame(trades)


def summarise(trades: pd.DataFrame, label: str = "") -> dict:
    if trades.empty:
        return {}
    n        = len(trades)
    win_rate = trades["any_won"].mean()
    total_pnl = trades["total_pnl"].sum()
    total_cost = trades["total_cost"].sum()
    roi      = total_pnl / total_cost * 100
    cum      = trades["total_pnl"].cumsum()
    max_dd   = float((cum.cummax() - cum).max())
    hourly_ret = trades["total_pnl"] / trades["total_cost"]
    sharpe   = hourly_ret.mean() / hourly_ret.std() * math.sqrt(8760) if hourly_ret.std() > 0 else 0
    return {
        "label": label, "n": n, "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2), "roi": round(roi, 2),
        "max_dd": round(max_dd, 2), "sharpe": round(sharpe, 2),
        "avg_cost_per_hour": round(trades["total_cost"].mean(), 2),
        "avg_pnl_per_hour": round(trades["total_pnl"].mean(), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPTIMAL PORTFOLIO SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def search_configs(df, sim, calib, portfolio_dollars=200.0):
    """Try all reasonable bucket combinations and Kelly fractions."""
    from itertools import combinations

    results = []
    all_offsets = [0, 1, -1, 2, -2, 3, -3]

    # All non-empty subsets
    for r in range(1, len(all_offsets) + 1):
        for combo in combinations(all_offsets, r):
            for ask_cents in [1, 2, 4]:
                for kelly_mult in [0.25, 0.50, 0.75, 1.0]:
                    trades = backtest_portfolio(
                        df, sim, calib,
                        offsets_to_trade=list(combo),
                        ask_cents=ask_cents,
                        kelly_mult=kelly_mult,
                        portfolio_dollars=portfolio_dollars,
                    )
                    s = summarise(trades,
                                  f"offsets={sorted(combo)} ask={ask_cents}c kelly={kelly_mult}")
                    if s:
                        s["offsets"]    = sorted(combo)
                        s["ask_cents"]  = ask_cents
                        s["kelly_mult"] = kelly_mult
                        results.append(s)

    return pd.DataFrame(results).sort_values("sharpe", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fetching 1000 candles...")
    df = fetch_candles(1000)
    print(f"Got {len(df)} candles  {df['open_time'].iloc[0].date()} → {df['open_time'].iloc[-1].date()}\n")

    print("Running Monte Carlo simulation (200 draws × 926 candles)...")
    sim = simulate_entries(df, n_sim=N_SIM)
    print(f"Simulated {len(sim):,} entry scenarios\n")

    calib = calibrate(sim)

    # ── Calibration table ─────────────────────────────────────────────────────
    print("=" * 70)
    print("  EMPIRICAL CALIBRATION  (15 min before expiry)")
    print("=" * 70)
    print(f"  {'Offset':>8}  {'Actual%':>8}  {'Model%':>8}  {'Ratio':>7}"
          f"  {'Edge@1c':>8}  {'Edge@4c':>8}  {'Kelly@1c':>9}  {'Kelly@4c':>9}")
    print("  " + "─" * 68)
    for off, row in calib.iterrows():
        flag = " ◄ sweet spot" if 2 <= abs(off) <= 3 else ""
        print(f"  {off:>+8}  {row['actual_p']:>8.4f}  {row['model_p']:>8.4f}"
              f"  {row['ratio']:>7.3f}  {row['edge_1c']:>+8.4f}  {row['edge_4c']:>+8.4f}"
              f"  {row['kelly_1c']:>9.4f}  {row['kelly_4c']:>9.4f}{flag}")

    # ── Search all configs ────────────────────────────────────────────────────
    print("\nSearching all bucket combinations and Kelly fractions...")
    print("(this takes ~60 seconds)\n")
    results = search_configs(df, sim, calib)

    print("=" * 80)
    print("  TOP 15 CONFIGURATIONS  (ranked by Sharpe ratio)")
    print("=" * 80)
    print(f"  {'Config':<45}  {'Win%':>6}  {'ROI':>6}  {'Sharpe':>7}  {'MaxDD':>7}  {'$/hr':>6}")
    print("  " + "─" * 78)
    for _, row in results.head(15).iterrows():
        conf = f"ask={row['ask_cents']}c  kelly={row['kelly_mult']}  {row['offsets']}"
        print(f"  {conf:<45}  {row['win_rate']:>6.1%}  {row['roi']:>5.1f}%"
              f"  {row['sharpe']:>7.2f}  ${row['max_dd']:>6.2f}  ${row['avg_pnl_per_hour']:>5.4f}")

    # ── Deep dive on best config ──────────────────────────────────────────────
    best = results.iloc[0]
    print(f"\n{'=' * 60}")
    print(f"  OPTIMAL STRATEGY")
    print(f"{'=' * 60}")
    print(f"  Buckets:      {best['offsets']}")
    print(f"  Market ask:   {best['ask_cents']}¢ per contract")
    print(f"  Kelly mult:   {best['kelly_mult']} (of full Kelly)")
    print(f"  Win rate:     {best['win_rate']:.1%}")
    print(f"  ROI:          {best['roi']:+.1f}%")
    print(f"  Sharpe:       {best['sharpe']:.2f}")
    print(f"  Max drawdown: ${best['max_dd']:.2f}")
    print(f"  Avg P&L/hr:   ${best['avg_pnl_per_hour']:.4f}")
    print(f"  Cost/hr:      ${best['avg_cost_per_hour']:.2f}")

    print(f"\n  Per-bucket Kelly fractions (half-Kelly):")
    for off in best["offsets"]:
        k1 = calib.loc[off, "kelly_1c"] if off in calib.index else 0
        k4 = calib.loc[off, "kelly_4c"] if off in calib.index else 0
        k  = k1 if best["ask_cents"] == 1 else k4
        print(f"    Offset {off:+d}: Kelly={k:.4f}  → "
              f"${200 * k * best['kelly_mult']:.2f} per $200 portfolio")

    # ── Equity curve ─────────────────────────────────────────────────────────
    best_trades = backtest_portfolio(
        df, sim, calib,
        offsets_to_trade=best["offsets"],
        ask_cents=best["ask_cents"],
        kelly_mult=best["kelly_mult"],
    )
    best_trades["cum_pnl"] = best_trades["total_pnl"].cumsum()

    print(f"\n  Period P&L: start=${0:.2f} → end=${best_trades['cum_pnl'].iloc[-1]:.2f}"
          f"  ({best_trades['cum_pnl'].iloc[-1] / best_trades['total_cost'].sum() * 100:.1f}% ROI)")

    print(f"\n  P&L by quartile of vol:")
    best_trades_with_vol = best_trades.merge(
        sim[["candle_idx", "vol_ann"]].groupby("candle_idx").first(),
        on="candle_idx", how="left"
    )
    for label, lo, hi in [("Low vol", 0, 0.25), ("Mid vol", 0.25, 0.50),
                           ("High vol", 0.50, 0.75), ("Extreme", 0.75, 1.00)]:
        q_lo = best_trades_with_vol["vol_ann"].quantile(lo)
        q_hi = best_trades_with_vol["vol_ann"].quantile(hi)
        g = best_trades_with_vol[(best_trades_with_vol["vol_ann"] >= q_lo) &
                                  (best_trades_with_vol["vol_ann"] < q_hi)]
        if g.empty:
            continue
        r = g["total_pnl"].sum() / g["total_cost"].sum() * 100
        print(f"    {label:<12} (vol {q_lo*100:.0f}–{q_hi*100:.0f}%): "
              f"ROI={r:+.1f}%  n={len(g)}")

    # Save
    out = Path("data")
    out.mkdir(exist_ok=True)
    results.to_csv(out / "last15_configs.csv", index=False)
    best_trades.to_csv(out / "last15_best_trades.csv", index=False)
    calib.to_csv(out / "last15_calibration.csv")
    print(f"\n  Results saved to data/")
