"""
pattern_analysis_phase2.py - Phase 2: Add Market Sentiment context to Highs & Lows

Builds on Phase 1 by attaching Fear & Greed Index (market sentiment)
around each significant swing high and low.

This gives us insight into whether tops/bottoms formed during
Extreme Greed, Greed, Neutral, Fear, or Extreme Fear.
"""

import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import Counter

from signal_engine import sma, ema, rsi, macd
from features import atr, volume_zscore

DELTA_BASE = "https://api.india.delta.exchange"
FNG_URL = "https://api.alternative.me/fng/"
SYMBOL = "BTCUSD"
RESOLUTION = "1h"
DAYS = 120

LEFT_BARS = 5
RIGHT_BARS = 5
MIN_MOVE_PCT = 1.5


def fetch_candles(symbol=SYMBOL, resolution=RESOLUTION, days=DAYS):
    end = int(time.time())
    start = end - days * 86400
    url = f"{DELTA_BASE}/v2/history/candles"
    params = {"resolution": resolution, "symbol": symbol, "start": start, "end": end}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json().get("result", [])
    return list(reversed(data))


def fetch_fear_greed(limit=150):
    """Fetch historical Fear & Greed Index."""
    try:
        resp = requests.get(f"{FNG_URL}?limit={limit}&format=json", timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        # Convert to dict: date_str -> (value, classification)
        fng = {}
        for item in data:
            ts = int(item["timestamp"])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            fng[dt] = {
                "value": int(item["value"]),
                "classification": item["value_classification"]
            }
        return fng
    except Exception as e:
        print(f"Warning: Could not fetch Fear & Greed data: {e}")
        return {}


def get_indicators_at(close, high, low, volume, idx):
    c = close[:idx+1]
    h = high[:idx+1]
    l = low[:idx+1]
    v = volume[:idx+1]
    price = c[-1]

    if len(c) < 30:
        return None

    rsi_val = rsi(c, 14)
    macd_line, macd_sig = macd(c)
    macd_hist = macd_line - macd_sig
    ema20 = ema(c, 20)
    sma50 = sma(c, 50) if len(c) >= 50 else sma(c, max(10, len(c)//2))
    atr_val = atr(h, l, c)
    vol_z = volume_zscore(v)

    return {
        "price": price,
        "rsi": round(rsi_val, 1),
        "macd_hist": round(macd_hist, 2),
        "ema20": round(ema20, 1),
        "price_vs_ema20_pct": round((price - ema20) / ema20 * 100, 2),
        "price_vs_sma50_pct": round((price - sma50) / sma50 * 100, 2),
        "atr_pct": round(atr_val / price * 100, 3),
        "volume_z": round(vol_z, 2),
    }


def detect_swings(high, low, close):
    swings = []
    n = len(close)
    for i in range(LEFT_BARS, n - RIGHT_BARS):
        is_high = all(high[i] > high[i-j] for j in range(1, LEFT_BARS+1)) and \
                  all(high[i] > high[i+j] for j in range(1, RIGHT_BARS+1))
        is_low  = all(low[i] < low[i-j] for j in range(1, LEFT_BARS+1)) and \
                  all(low[i] < low[i+j] for j in range(1, RIGHT_BARS+1))
        if is_high:
            swings.append({"idx": i, "type": "HIGH", "price": high[i]})
        elif is_low:
            swings.append({"idx": i, "type": "LOW", "price": low[i]})
    return swings


def analyze(candles, fng_data):
    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)
    volume = np.array([c["volume"] for c in candles], dtype=float)
    times = [c["time"] for c in candles]

    swings = detect_swings(high, low, close)
    results = []

    for i, sw in enumerate(swings):
        idx = sw["idx"]
        sw_type = sw["type"]
        sw_price = sw["price"]

        prev_opposite = None
        for j in range(i-1, -1, -1):
            if swings[j]["type"] != sw_type:
                prev_opposite = swings[j]
                break
        if prev_opposite is None:
            continue

        move_pct = (sw_price - prev_opposite["price"]) / prev_opposite["price"] * 100
        duration_bars = idx - prev_opposite["idx"]
        if abs(move_pct) < MIN_MOVE_PCT:
            continue

        at_turn = get_indicators_at(close, high, low, volume, idx)
        if at_turn is None:
            continue

        before_idx = max(0, idx - 3)
        before_turn = get_indicators_at(close, high, low, volume, before_idx)

        # Fear & Greed on the day of the swing
        dt = datetime.fromtimestamp(times[idx], tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")
        fng = fng_data.get(date_str, {"value": None, "classification": "Unknown"})

        # Also try previous day if exact day missing
        if fng["value"] is None:
            prev_day = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
            fng = fng_data.get(prev_day, {"value": None, "classification": "Unknown"})

        results.append({
            "time": dt.strftime("%Y-%m-%d %H:%M"),
            "date": date_str,
            "type": sw_type,
            "price": round(sw_price, 1),
            "move_pct": round(move_pct, 2),
            "duration_hours": duration_bars,
            "rsi_at": at_turn["rsi"],
            "rsi_before3": before_turn["rsi"] if before_turn else None,
            "macd_hist_at": at_turn["macd_hist"],
            "macd_hist_before3": before_turn["macd_hist"] if before_turn else None,
            "price_vs_ema20_at": at_turn["price_vs_ema20_pct"],
            "volume_z_at": at_turn["volume_z"],
            "fng_value": fng["value"],
            "fng_class": fng["classification"],
        })

    return pd.DataFrame(results)


def report(df):
    if df.empty:
        print("No significant swings found.")
        return

    highs = df[df["type"] == "HIGH"]
    lows = df[df["type"] == "LOW"]

    print("\n" + "="*75)
    print("PHASE 2 REPORT - Highs/Lows + Market Sentiment (Fear & Greed)")
    print("="*75)

    print(f"\nTotal significant swings : {len(df)}")
    print(f"  Highs (Tops)           : {len(highs)}")
    print(f"  Lows  (Bottoms)        : {len(lows)}")

    # --- Sentiment distribution ---
    print("\n----- SENTIMENT AT HIGHS (TOPS) -----")
    if len(highs) > 0:
        high_fng = highs[highs["fng_class"] != "Unknown"]
        if len(high_fng) > 0:
            print(f"Average Fear & Greed at Highs : {high_fng['fng_value'].mean():.1f}")
            counts = Counter(high_fng["fng_class"])
            for cls, cnt in counts.most_common():
                pct = cnt / len(high_fng) * 100
                print(f"  {cls:15s} : {cnt:3d} times ({pct:.1f}%)")
        else:
            print("No Fear & Greed data matched for highs.")

    print("\n----- SENTIMENT AT LOWS (BOTTOMS) -----")
    if len(lows) > 0:
        low_fng = lows[lows["fng_class"] != "Unknown"]
        if len(low_fng) > 0:
            print(f"Average Fear & Greed at Lows  : {low_fng['fng_value'].mean():.1f}")
            counts = Counter(low_fng["fng_class"])
            for cls, cnt in counts.most_common():
                pct = cnt / len(low_fng) * 100
                print(f"  {cls:15s} : {cnt:3d} times ({pct:.1f}%)")
        else:
            print("No Fear & Greed data matched for lows.")

    # --- Combined Technical + Sentiment insight ---
    print("\n----- KEY COMBINED INSIGHTS -----")
    if len(highs) > 0 and len(lows) > 0:
        print(f"Highs form with RSI ≈ {highs['rsi_at'].mean():.0f}  |  F&G ≈ {highs['fng_value'].dropna().mean():.0f}")
        print(f"Lows  form with RSI ≈ {lows['rsi_at'].mean():.0f}  |  F&G ≈ {lows['fng_value'].dropna().mean():.0f}")
        print(f"MACD Hist expands strongly into Highs (avg {highs['macd_hist_at'].mean():+.0f})")
        print(f"MACD Hist is deeply negative at Lows   (avg {lows['macd_hist_at'].mean():+.0f})")

    print("\n----- RECENT SWINGS WITH SENTIMENT -----")
    cols = ["time", "type", "price", "move_pct", "rsi_at", "macd_hist_at", "fng_value", "fng_class"]
    print(df[cols].tail(15).to_string(index=False))

    print("\n" + "="*75)
    print("Phase 2 complete. Sentiment context added to turning points.")
    print("="*75)

    df.to_csv("swing_events_phase2.csv", index=False)
    print("Saved → swing_events_phase2.csv")
    return df


if __name__ == "__main__":
    print("Fetching BTCUSD 1h candles...")
    candles = fetch_candles()
    print(f"Loaded {len(candles)} candles.")

    print("Fetching Fear & Greed Index history...")
    fng = fetch_fear_greed(limit=200)
    print(f"Loaded {len(fng)} days of Fear & Greed data.")

    df = analyze(candles, fng)
    report(df)
