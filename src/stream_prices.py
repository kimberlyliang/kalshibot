"""Stream live BTC prices (BRTI approximation) to the terminal."""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.btc_feed import brti_approx

INTERVAL_SECONDS = 5

def stream():
    print(f"{'Time':<12} {'Price':>12}  {'Coinbase':>12}  {'Kraken':>12}  {'Bitstamp':>12}  {'Gemini':>12}  {'Spread':>10}")
    print("-" * 90)
    prev_price = None
    while True:
        try:
            result = brti_approx()
            price = result["price"]
            ind = result["individual_prices"]

            prices = list(ind.values())
            spread = max(prices) - min(prices)
            arrow = ""
            if prev_price is not None:
                arrow = "▲" if price > prev_price else ("▼" if price < prev_price else " ")
            prev_price = price

            ts = time.strftime("%H:%M:%S")
            print(
                f"{ts:<12} {arrow}${price:>10,.2f}  "
                f"${ind.get('coinbase', 0):>10,.2f}  "
                f"${ind.get('kraken', 0):>10,.2f}  "
                f"${ind.get('bitstamp', 0):>10,.2f}  "
                f"${ind.get('gemini', 0):>10,.2f}  "
                f"${spread:>8,.2f}"
            )
        except Exception as e:
            print(f"  error: {e}")

        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        stream()
    except KeyboardInterrupt:
        print("\nStopped.")
