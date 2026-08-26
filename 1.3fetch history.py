"""
fetch_history.py - Pulls historical daily closes from Delta Exchange
and saves them as a CSV ready for backtest.py.

Usage:
    python fetch_history.py BTCUSD 365
    python fetch_history.py PAXGUSD 365
"""

import sys
import time
import requests
import numpy as np

DELTA_BASE_URL = "https://api.delta.exchange"


def fetch_daily_closes(symbol: str, days: int = 365):
    end = int(time.time())
    start = end - days * 86400
    url = f"{DELTA_BASE_URL}/v2/history/candles"
    params = {"resolution": "1D", "symbol": symbol, "start": start, "end": end}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("result", [])
    if not data:
        raise ValueError(f"No data returned for {symbol}. Check the symbol name is correct.")
    # Delta returns newest-first; reverse so oldest is first (required by backtest.py)
    closes = [c["close"] for c in reversed(data)]
    return np.array(closes, dtype=float)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_history.py SYMBOL [days]")
        print("Example: python fetch_history.py BTCUSD 365")
        sys.exit(1)

    symbol = sys.argv[1]
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365

    print(f"Fetching {days} days of {symbol} daily closes...")
    closes = fetch_daily_closes(symbol, days)

    filename = f"{symbol}_history.csv"
    np.savetxt(filename, closes, delimiter=",")
    print(f"Saved {len(closes)} closing prices to {filename}")
    print(f"\nNow run: python backtest.py {filename}")
