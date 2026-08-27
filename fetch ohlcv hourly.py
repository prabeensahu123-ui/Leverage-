"""
fetch_ohlcv_hourly.py - Pulls hourly OHLCV data from Delta Exchange
(India endpoint, which has the fuller product list including PAXG).

Usage:
    python fetch_ohlcv_hourly.py BTCUSD 60
    (fetches 60 days of HOURLY candles = ~1440 bars, much more data
    than daily candles give us for the same wall-clock history)
"""

import sys
import time
import csv
import requests

DELTA_BASE_URL = "https://api.india.delta.exchange"


def fetch_ohlcv(symbol: str, days: int = 60):
    end = int(time.time())
    start = end - days * 86400
    url = f"{DELTA_BASE_URL}/v2/history/candles"
    params = {"resolution": "1h", "symbol": symbol, "start": start, "end": end}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("result", [])
    if not data:
        raise ValueError(f"No data returned for {symbol}")
    return list(reversed(data))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_ohlcv_hourly.py SYMBOL [days]")
        sys.exit(1)

    symbol = sys.argv[1]
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    print(f"Fetching {days} days of HOURLY {symbol} OHLCV data...")
    candles = fetch_ohlcv(symbol, days)

    filename = f"{symbol}_hourly.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "open", "high", "low", "close", "volume"])
        for c in candles:
            writer.writerow([c["time"], c["open"], c["high"], c["low"], c["close"], c["volume"]])

    print(f"Saved {len(candles)} hourly candles to {filename}")
    print("Upload this file back to Claude for the next step.")
