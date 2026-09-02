"""
pattern_analysis.py - Phase 1: Reverse-engineer Highs & Lows

Detects significant swing highs and swing lows, measures move size & duration,
and records key indicator values BEFORE and AT the turning points.
"""

import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from signal_engine import sma, ema, rsi, macd
from features import atr, volume_zscore

DELTA_BASE = "https://api.india.delta.exchange"
SYMBOL = "BTCUSD"
RESOLUTION = "1h"
DAYS = 120

# Swing detection parameters
LEFT_BARS = 5          # bars to the left that must be lower/higher
RIGHT_BARS = 5         # bars to the right that must be lower/higher
MIN_MOVE_PCT = 1.5     # minimum % move to consider significant


def fetch_candles(symbol=SYMBOL, resolution=RESOLUTION, days=DAYS):
    end = int(time.time())
    start = end - days * 86400
    url = f"{DELTA_BASE}/v2/history/candles"
    params = {"resolution": resolution, "symbol": symbol, "start": start, "end": end}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json().get("result", [])
    return list(reversed(data))


def get_indicators_at(close, high, low, volume, idx):
    """Return key indicator values using data up to idx (inclusive)."""
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
    sma20 = sma(c, 20)
    sma50 = sma(c, 50) if len(c) >= 50 else sma(c, len(c)//2)
    atr_val = atr(h, l, c)
    vol_z = volume_zscore(v)

    return {
        "price": price,
        "rsi": round(rsi_val, 1),
        "macd_hist": round(macd_hist, 2),
        "ema20": round(ema20, 1),
        "sma20": round(sma20, 1),
        "sma50": round(sma50, 1),
        "price_vs_ema20_pct": round((price - ema20) / ema20 * 100, 2),
        "price_vs_sma50_pct": round((price - sma50) / sma50 * 100, 2),
        "atr_pct": round(atr_val / price * 100, 3),
        "volume_z": round(vol_z, 2),
    }


def detect_swings(high, low, close):
    """Detect swing highs and swing lows."""
    swings = []
    n = len(close)

    for i in range(LEFT_BARS, n - RIGHT_BARS):
        # Swing High
        is_high = True
        for j in range(1, LEFT_BARS + 1):
            if high[i] <= high[i - j]:
                is_high = False
                break
        if is_high:
            for j in range(1, RIGHT_BARS + 1):
                if high[i] <= high[i + j]:
                    is_high = False
                    break

        # Swing Low
        is_low = True
        for j in range(1, LEFT_BARS + 1):
            if low[i] >= low[i - j]:
                is_low = False
                break
        if is_low:
            for j in range(1, RIGHT_BARS + 1):
                if low[i] >= low[i + j]:
                    is_low = False
                    break

        if is_high:
            swings.append({"idx": i, "type": "HIGH", "price": high[i]})
        elif is_low:
            swings.append({"idx": i, "type": "LOW", "price": low[i]})

    return swings


def analyze_swings(candles):
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

        # Find previous opposite swing to measure the move
        prev_opposite = None
        for j in range(i - 1, -1, -1):
            if swings[j]["type"] != sw_type:
                prev_opposite = swings[j]
                break

        if prev_opposite is None:
            continue

        move_pct = (sw_price - prev_opposite["price"]) / prev_opposite["price"] * 100
        duration_bars = idx - prev_opposite["idx"]

        if abs(move_pct) < MIN_MOVE_PCT:
            continue

        # Indicators at the turning point
        at_turn = get_indicators_at(close, high, low, volume, idx)
        if at_turn is None:
            continue

        # Indicators 3 bars BEFORE the turning point
        before_idx = max(0, idx - 3)
        before_turn = get_indicators_at(close, high, low, volume, before_idx)

        # Indicators 5 bars BEFORE
        before5_idx = max(0, idx - 5)
        before5 = get_indicators_at(close, high, low, volume, before5_idx)

        ts = datetime.fromtimestamp(times[idx], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

        results.append({
            "time": ts,
            "type": sw_type,
            "price": round(sw_price, 1),
            "move_pct": round(move_pct, 2),
            "duration_bars": duration_bars,
            "duration_hours": duration_bars,  # since 1h timeframe
            "rsi_at": at_turn["rsi"],
            "rsi_before3": before_turn["rsi"] if before_turn else None,
            "macd_hist_at": at_turn["macd_hist"],
            "macd_hist_before3": before_turn["macd_hist"] if before_turn else None,
            "price_vs_ema20_at": at_turn["price_vs_ema20_pct"],
            "price_vs_ema20_before3": before_turn["price_vs_ema20_pct"] if before_turn else None,
            "volume_z_at": at_turn["volume_z"],
            "atr_pct_at": at_turn["atr_pct"],
        })

    return pd.DataFrame(results)


def summarize(df):
    if df.empty:
        print("No significant swings found.")
        return

    highs = df[df["type"] == "HIGH"]
    lows = df[df["type"] == "LOW"]

    print("\n" + "="*70)
    print("PHASE 1 REPORT - Swing High / Low Pattern Analysis (BTCUSD 1h)")
    print("="*70)

    print(f"\nTotal significant swings detected: {len(df)}")
    print(f"  → Highs (potential tops): {len(highs)}")
    print(f"  → Lows  (potential bottoms): {len(lows)}")

    if len(highs) > 0:
        print("\n----- HIGH (TOP) CHARACTERISTICS -----")
        print(f"Average move size into High : {highs['move_pct'].mean():+.2f}%")
        print(f"Average duration of move    : {highs['duration_bars'].mean():.1f} hours")
        print(f"Average RSI at High         : {highs['rsi_at'].mean():.1f}")
        print(f"Average RSI 3 bars before   : {highs['rsi_before3'].mean():.1f}")
        print(f"Average MACD Hist at High   : {highs['macd_hist_at'].mean():.1f}")
        print(f"Average MACD Hist 3 before  : {highs['macd_hist_before3'].mean():.1f}")
        print(f"Avg Price vs EMA20 at High  : {highs['price_vs_ema20_at'].mean():+.2f}%")
        print(f"Average Volume Z-score      : {highs['volume_z_at'].mean():.2f}")

    if len(lows) > 0:
        print("\n----- LOW (BOTTOM) CHARACTERISTICS -----")
        print(f"Average move size into Low  : {lows['move_pct'].mean():+.2f}%")
        print(f"Average duration of move    : {lows['duration_bars'].mean():.1f} hours")
        print(f"Average RSI at Low          : {lows['rsi_at'].mean():.1f}")
        print(f"Average RSI 3 bars before   : {lows['rsi_before3'].mean():.1f}")
        print(f"Average MACD Hist at Low    : {lows['macd_hist_at'].mean():.1f}")
        print(f"Average MACD Hist 3 before  : {lows['macd_hist_before3'].mean():.1f}")
        print(f"Avg Price vs EMA20 at Low   : {lows['price_vs_ema20_at'].mean():+.2f}%")
        print(f"Average Volume Z-score      : {lows['volume_z_at'].mean():.2f}")

    print("\n----- RECENT SIGNIFICANT SWINGS (last 12) -----")
    print(df.tail(12).to_string(index=False))

    print("\n" + "="*70)
    print("Phase 1 complete. Data ready for deeper pattern mining (Phase 2).")
    print("="*70)

    return df


if __name__ == "__main__":
    print("Fetching BTCUSD 1h data...")
    candles = fetch_candles()
    print(f"Loaded {len(candles)} candles.")

    df = analyze_swings(candles)
    summarize(df)

    # Save for later phases
    df.to_csv("swing_events_phase1.csv", index=False)
    print("\nSaved detailed results to swing_events_phase1.csv")
