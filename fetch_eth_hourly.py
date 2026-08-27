"""
fetch_eth_hourly.py - Pulls hourly ETH data matching our existing BTC
hourly window, to test cross-asset lead-lag relationships.

Usage:
    python fetch_eth_hourly.py 90
"""

import sys
import time
import csv
import requests

DELTA_BASE_URL = "https://api.india.delta.exchange"


def fetch_ohlcv(symbol: str, days: int = 90):
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
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90

    print(f"Fetching {days} days of hourly ETHUSD OHLCV data...")
    candles = fetch_ohlcv("ETHUSD", days)

    filename = "ETHUSD_hourly.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "open", "high", "low", "close", "volume"])
        for c in candles:
            writer.writerow([c["time"], c["open"], c["high"], c["low"], c["close"], c["volume"]])

    print(f"Saved {len(candles)} hourly candles to {filename}")
    print("Upload this file back to Claude for the next step.")
