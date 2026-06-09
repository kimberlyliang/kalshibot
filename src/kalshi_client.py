"""Kalshi REST + WebSocket client with RSA-PSS request signing."""
import base64
import hashlib
import time
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import KALSHI_BASE_URL, KALSHI_KEY_ID, KALSHI_PRIVATE_KEY_PATH


def _load_private_key():
    pem = Path(KALSHI_PRIVATE_KEY_PATH).read_bytes()
    return serialization.load_pem_private_key(pem, password=None)


def _sign_request(method: str, path: str) -> dict:
    """Return the three Kalshi auth headers for a request."""
    ts_ms = str(int(time.time() * 1000))
    message = (ts_ms + method.upper() + path).encode()
    key = _load_private_key()
    sig = key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": KALSHI_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "Content-Type": "application/json",
    }


class KalshiClient:
    def __init__(self):
        self.base = KALSHI_BASE_URL
        self.http = httpx.Client(timeout=10)

    def _get(self, path: str, params: dict | None = None):
        headers = _sign_request("GET", path)
        r = self.http.get(self.base + path, headers=headers, params=params)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict):
        headers = _sign_request("POST", path)
        r = self.http.post(self.base + path, headers=headers, json=body)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str):
        headers = _sign_request("DELETE", path)
        r = self.http.delete(self.base + path, headers=headers)
        r.raise_for_status()
        return r.json()

    # --- Market data ---

    def get_markets(self, series_ticker: str | None = None, status: str = "open") -> list[dict]:
        params = {"status": status}
        if series_ticker:
            params["series_ticker"] = series_ticker
        return self._get("/markets", params)["markets"]

    def get_orderbook(self, ticker: str, depth: int = 5) -> dict:
        return self._get(f"/markets/{ticker}/orderbook", {"depth": depth})["orderbook"]

    def get_candlesticks(self, ticker: str, period_seconds: int = 3600) -> list[dict]:
        return self._get(f"/markets/{ticker}/candlesticks", {"period_seconds": period_seconds})["candlesticks"]

    # --- Account ---

    def get_balance(self) -> dict:
        """Returns balance in cents."""
        return self._get("/portfolio/balance")

    def get_positions(self) -> list[dict]:
        return self._get("/portfolio/positions")["market_positions"]

    # --- Orders ---

    def place_order(
        self,
        ticker: str,
        side: str,        # "yes" or "no"
        contracts: int,
        price_cents: int,  # 1–99
        order_type: str = "limit",
    ) -> dict:
        body = {
            "ticker": ticker,
            "action": "buy",
            "side": side,
            "type": order_type,
            "count": contracts,
            "yes_price": price_cents if side == "yes" else 100 - price_cents,
        }
        return self._post("/orders", body)

    def cancel_order(self, order_id: str) -> dict:
        return self._delete(f"/orders/{order_id}")

    def get_open_orders(self) -> list[dict]:
        return self._get("/orders", {"status": "resting"})["orders"]
