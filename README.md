# kalshi-btc-bot

Hourly BTC prediction-market trader on Kalshi. Runs 5 minutes before each hour close, scores open contracts, and sizes positions with fractional Kelly.

## Setup

```bash
pip install -r requirements.txt
```

Place your Kalshi RSA private key at `config/kalshi_private.pem` (or set `KALSHI_PRIVATE_KEY_PATH`).

```bash
export KALSHI_KEY_ID="your-key-id"
export KALSHI_ENV="demo"          # "prod" for live trading
export BTC_SERIES_TICKER="KXBTC" # adjust to actual series ticker
```

## Run

```bash
# Dry run (no orders placed)
python src/bot.py --dry-run

# Live (demo env)
python src/bot.py
```

## Cron (hourly at :55)

```
55 * * * * cd /path/to/kalshi-btc-bot && python src/bot.py >> logs/cron.log 2>&1
```

## Architecture

```
btc_feed.py      ← Binance public API (hourly OHLCV candles + spot price)
predictor.py     ← Ensemble: momentum + mean-reversion + micro-structure
market_selector.py ← Scan Kalshi BTC contracts, compute edge vs market price
risk.py          ← Fractional Kelly sizing with hard $ cap
kalshi_client.py ← RSA-signed REST client
bot.py           ← Main orchestrator
```

## Risk knobs (config/settings.py)

| Setting | Default | Meaning |
|---|---|---|
| `MIN_EDGE` | 0.04 | Skip trade if edge < 4 cents |
| `KELLY_FRACTION` | 0.10 | Bet 10% of full Kelly |
| `MAX_TRADE_USD` | $50 | Hard cap per trade |
