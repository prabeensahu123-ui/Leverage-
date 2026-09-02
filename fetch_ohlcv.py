"""
fetch_ohlcv.py - Download OHLCV history from Delta Exchange

Usage:
    python fetch_ohlcv.py BTCUSD 90
    python fetch_ohlcv.py ETHUSD 180
"""

import sys
import time
import csv
import requests

DELTA_BASE_URL = "https://api.india.delta.exchange"

def fetch_ohlcv(symbol: str, days: int = 90, resolution: str = "1h"):
    end = int(time.time())
    start = end - days * 86400
    url = f"{DELTA_BASE_URL}/v2/history/candles"
    params = {
        "resolution": resolution,
        "symbol": symbol,
        "start": start,
        "end": end
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json().get("result", [])
    if not data:
        raise ValueError(f"No data returned for {symbol}")
    return list(reversed(data))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_ohlcv.py SYMBOL [days] [resolution]")
        print("Example: python fetch_ohlcv.py BTCUSD 90 1h")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    resolution = sys.argv[3] if len(sys.argv) > 3 else "1h"

    print(f"Fetching {days} days of {symbol} ({resolution}) data...")
    candles = fetch_ohlcv(symbol, days, resolution)

    filename = f"{symbol}_{resolution}_ohlcv.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "open", "high", "low", "close", "volume"])
        for c in candles:
            writer.writerow([
                c["time"], c["open"], c["high"],
                c["low"], c["close"], c["volume"]
            ])

    print(f"Saved {len(candles)} candles to {filename}")
