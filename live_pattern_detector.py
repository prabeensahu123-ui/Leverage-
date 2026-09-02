"""
live_pattern_detector.py

Live Pattern Detector based on Phases 1-3 findings.

Checks current market conditions on 1h and 4h timeframes
and scores how closely they match historical Top / Bottom patterns.
"""

import time
import requests
import numpy as np
from datetime import datetime, timezone

from signal_engine import sma, ema, rsi, macd
from features import atr, volume_zscore

DELTA_BASE = "https://api.india.delta.exchange"
SYMBOL = "BTCUSD"

# Pattern profiles learned from Phase 1-3
TOP_PROFILE = {
    "rsi_center": 64.0,
    "rsi_range": 8.0,          # acceptable deviation
    "macd_hist_min": 10.0,     # should be positive / expanding
    "price_vs_ema20_min": 0.3, # preferably extended above EMA20
}

BOTTOM_PROFILE = {
    "rsi_center": 38.0,
    "rsi_range": 8.0,
    "macd_hist_max": -10.0,    # should be negative
    "price_vs_ema20_max": -0.3,
}


def fetch_candles(resolution="1h", days=60):
    end = int(time.time())
    start = end - days * 86400
    # Delta uses "1d" for daily in some versions, try both if needed
    res = resolution
    url = f"{DELTA_BASE}/v2/history/candles"
    params = {"resolution": res, "symbol": SYMBOL, "start": start, "end": end}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("result", [])
        return list(reversed(data))
    except Exception as e:
        print(f"Failed to fetch {resolution}: {e}")
        return []


def get_current_indicators(candles):
    if len(candles) < 50:
        return None

    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)
    volume = np.array([c["volume"] for c in candles], dtype=float)

    price = close[-1]
    rsi_val = rsi(close, 14)
    macd_line, macd_sig = macd(close)
    macd_hist = macd_line - macd_sig
    ema20 = ema(close, 20)
    price_vs_ema20 = (price - ema20) / ema20 * 100
    atr_pct = atr(high, low, close) / price * 100
    vol_z = volume_zscore(volume)

    return {
        "price": price,
        "rsi": rsi_val,
        "macd_hist": macd_hist,
        "price_vs_ema20": price_vs_ema20,
        "atr_pct": atr_pct,
        "volume_z": vol_z,
        "ema20": ema20,
    }


def score_match(indicators, profile, is_top=True):
    """
    Returns a match score from 0 to 100.
    Higher = closer to the historical pattern.
    """
    if indicators is None:
        return 0.0

    score = 0.0
    weights = 0.0

    # RSI match (most important)
    rsi = indicators["rsi"]
    rsi_diff = abs(rsi - profile["rsi_center"])
    rsi_score = max(0, 100 - (rsi_diff / profile["rsi_range"]) * 50)
    score += rsi_score * 0.45
    weights += 0.45

    # MACD Histogram direction + strength
    macd_h = indicators["macd_hist"]
    if is_top:
        if macd_h >= profile["macd_hist_min"]:
            macd_score = min(100, 50 + macd_h / 5)
        else:
            macd_score = max(0, 50 + macd_h)
    else:
        if macd_h <= profile["macd_hist_max"]:
            macd_score = min(100, 50 + abs(macd_h) / 5)
        else:
            macd_score = max(0, 50 - macd_h)
    score += macd_score * 0.35
    weights += 0.35

    # Price vs EMA20 extension
    pve = indicators["price_vs_ema20"]
    if is_top:
        if pve >= profile.get("price_vs_ema20_min", 0):
            ema_score = min(100, 60 + pve * 10)
        else:
            ema_score = max(0, 50 + pve * 20)
    else:
        if pve <= profile.get("price_vs_ema20_max", 0):
            ema_score = min(100, 60 + abs(pve) * 10)
        else:
            ema_score = max(0, 50 - pve * 20)
    score += ema_score * 0.20
    weights += 0.20

    return round(score / weights, 1) if weights > 0 else 0.0


def interpret(score):
    if score >= 75:
        return "STRONG match"
    elif score >= 60:
        return "Moderate match"
    elif score >= 45:
        return "Weak match"
    else:
        return "No significant match"


def run_detection():
    print("=" * 65)
    print("LIVE PATTERN DETECTOR  (based on Phases 1-3)")
    print("=" * 65)
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Symbol: {SYMBOL}\n")

    results = {}

    for tf, days in [("1h", 60), ("4h", 120)]:
        print(f"Analyzing {tf}...")
        candles = fetch_candles(tf, days)
        inds = get_current_indicators(candles)

        if inds is None:
            print(f"  Not enough data for {tf}\n")
            continue

        top_score = score_match(inds, TOP_PROFILE, is_top=True)
        bottom_score = score_match(inds, BOTTOM_PROFILE, is_top=False)

        results[tf] = {
            "indicators": inds,
            "top_score": top_score,
            "bottom_score": bottom_score,
        }

        print(f"  Price          : ${inds['price']:,.1f}")
        print(f"  RSI (14)       : {inds['rsi']:.1f}")
        print(f"  MACD Hist      : {inds['macd_hist']:+.1f}")
        print(f"  Price vs EMA20 : {inds['price_vs_ema20']:+.2f}%")
        print(f"  Volume Z-score : {inds['volume_z']:.2f}")
        print(f"  → Top match    : {top_score:.1f}%  ({interpret(top_score)})")
        print(f"  → Bottom match : {bottom_score:.1f}%  ({interpret(bottom_score)})")
        print()

    # Final combined reading
    print("-" * 65)
    print("COMBINED READING")
    print("-" * 65)

    if not results:
        print("Could not generate reading.")
        return

    # Average scores across available timeframes
    avg_top = np.mean([r["top_score"] for r in results.values()])
    avg_bottom = np.mean([r["bottom_score"] for r in results.values()])

    print(f"Average Top pattern match    : {avg_top:.1f}%")
    print(f"Average Bottom pattern match : {avg_bottom:.1f}%\n")

    if avg_top >= 65 and avg_top > avg_bottom + 10:
        print(">>> Current conditions RESEMBLE historical TOPS")
        print("    Caution advised for new long positions.")
    elif avg_bottom >= 65 and avg_bottom > avg_top + 10:
        print(">>> Current conditions RESEMBLE historical BOTTOMS")
        print("    Potential accumulation / long setup zone.")
    elif max(avg_top, avg_bottom) < 50:
        print(">>> No strong Top or Bottom pattern currently active.")
        print("    Market is likely in a neutral / transitional state.")
    else:
        print(">>> Mixed signals. No clear dominant pattern.")

    print("\n" + "=" * 65)
    return results


if __name__ == "__main__":
    run_detection()
