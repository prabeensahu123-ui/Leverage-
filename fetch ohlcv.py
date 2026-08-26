"""
fetch_ohlcv.py - Pulls full OHLCV (open/high/low/close/volume) history
from Delta Exchange, not just closing price. Volume and range data
enable much better features than price alone.

Usage:
    python fetch_ohlcv.py BTCUSD 1000
"""

import sys
import time
import csv
import requests

DELTA_BASE_URL = "https://api.delta.exchange"


def fetch_ohlcv(symbol: str, days: int = 1000):
    end = int(time.time())
    start = end - days * 86400
    url = f"{DELTA_BASE_URL}/v2/history/candles"
    params = {"resolution": "1d", "symbol": symbol, "start": start, "end": end}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("result", [])
    if not data:
        raise ValueError(f"No data returned for {symbol}")
    # Delta returns newest-first; reverse so oldest is first
    return list(reversed(data))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_ohlcv.py SYMBOL [days]")
        sys.exit(1)

    symbol = sys.argv[1]
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    print(f"Fetching {days} days of {symbol} OHLCV data...")
    candles = fetch_ohlcv(symbol, days)

    filename = f"{symbol}_ohlcv.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "open", "high", "low", "close", "volume"])
        for c in candles:
            writer.writerow([c["time"], c["open"], c["high"], c["low"], c["close"], c["volume"]])

    print(f"Saved {len(candles)} candles to {filename}")
    print(f"\nUpload this file back to Claude for the next step.")
