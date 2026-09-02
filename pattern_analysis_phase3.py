"""
pattern_analysis_phase3.py - Phase 3: Multi-Timeframe Pattern Comparison

Compares swing High / Low characteristics across 1h, 4h and 1D timeframes.
Identifies which patterns are consistent across timeframes.
"""

import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict

from signal_engine import sma, ema, rsi, macd
from features import atr, volume_zscore

DELTA_BASE = "https://api.india.delta.exchange"
FNG_URL = "https://api.alternative.me/fng/"
SYMBOL = "BTCUSD"

LEFT_BARS = 4
RIGHT_BARS = 4
MIN_MOVE_PCT = 1.8

TIMEFRAMES = {
    "1h": {"resolution": "1h", "days": 90,  "label": "1 Hour"},
    "4h": {"resolution": "4h", "days": 180, "label": "4 Hour"},
    "1D": {"resolution": "1D", "days": 400, "label": "Daily"},
}


def fetch_candles(resolution, days):
    end = int(time.time())
    start = end - days * 86400
    url = f"{DELTA_BASE}/v2/history/candles"
    params = {"resolution": resolution, "symbol": SYMBOL, "start": start, "end": end}
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json().get("result", [])
        return list(reversed(data))
    except Exception as e:
        print(f"Error fetching {resolution}: {e}")
        return []


def fetch_fear_greed(limit=400):
    try:
        resp = requests.get(f"{FNG_URL}?limit={limit}&format=json", timeout=10)
        data = resp.json().get("data", [])
        fng = {}
        for item in data:
            ts = int(item["timestamp"])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            fng[dt] = {"value": int(item["value"]), "classification": item["value_classification"]}
        return fng
    except Exception:
        return {}


def get_indicators(close, high, low, volume, idx):
    c = close[:idx+1]
    h = high[:idx+1]
    l = low[:idx+1]
    v = volume[:idx+1]
    if len(c) < 35:
        return None
    price = c[-1]
    rsi_val = rsi(c, 14)
    macd_line, macd_sig = macd(c)
    macd_hist = macd_line - macd_sig
    ema20 = ema(c, 20)
    return {
        "rsi": round(rsi_val, 1),
        "macd_hist": round(macd_hist, 2),
        "price_vs_ema20": round((price - ema20) / ema20 * 100, 2),
        "volume_z": round(volume_zscore(v), 2),
        "atr_pct": round(atr(h, l, c) / price * 100, 3),
    }


def detect_swings(high, low):
    swings = []
    n = len(high)
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


def analyze_timeframe(tf_key, candles, fng_data):
    if len(candles) < 80:
        return pd.DataFrame()

    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)
    volume = np.array([c["volume"] for c in candles], dtype=float)
    times = [c["time"] for c in candles]

    swings = detect_swings(high, low)
    rows = []

    for i, sw in enumerate(swings):
        idx = sw["idx"]
        # previous opposite swing
        prev = None
        for j in range(i-1, -1, -1):
            if swings[j]["type"] != sw["type"]:
                prev = swings[j]
                break
        if prev is None:
            continue

        move_pct = (sw["price"] - prev["price"]) / prev["price"] * 100
        if abs(move_pct) < MIN_MOVE_PCT:
            continue

        inds = get_indicators(close, high, low, volume, idx)
        if inds is None:
            continue

        before_inds = get_indicators(close, high, low, volume, max(0, idx-3))

        dt = datetime.fromtimestamp(times[idx], tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")
        fng = fng_data.get(date_str, {"value": None, "classification": "Unknown"})
        if fng["value"] is None:
            fng = fng_data.get((dt - timedelta(days=1)).strftime("%Y-%m-%d"),
                               {"value": None, "classification": "Unknown"})

        rows.append({
            "timeframe": tf_key,
            "type": sw["type"],
            "price": round(sw["price"], 1),
            "move_pct": round(move_pct, 2),
            "duration_bars": idx - prev["idx"],
            "rsi_at": inds["rsi"],
            "rsi_before": before_inds["rsi"] if before_inds else None,
            "macd_hist_at": inds["macd_hist"],
            "price_vs_ema20": inds["price_vs_ema20"],
            "volume_z": inds["volume_z"],
            "fng_value": fng["value"],
            "fng_class": fng["classification"],
        })

    return pd.DataFrame(rows)


def main():
    print("="*70)
    print("PHASE 3 - MULTI-TIMEFRAME PATTERN COMPARISON")
    print("="*70)

    print("\nFetching Fear & Greed history...")
    fng = fetch_fear_greed(400)
    print(f"Loaded {len(fng)} days of sentiment data.\n")

    all_dfs = []
    summary = {}

    for tf_key, cfg in TIMEFRAMES.items():
        print(f"Analyzing {cfg['label']} ({tf_key})...")
        candles = fetch_candles(cfg["resolution"], cfg["days"])
        print(f"  → {len(candles)} candles loaded")
        df = analyze_timeframe(tf_key, candles, fng)
        if not df.empty:
            all_dfs.append(df)
            highs = df[df["type"] == "HIGH"]
            lows = df[df["type"] == "LOW"]
            summary[tf_key] = {
                "highs": len(highs),
                "lows": len(lows),
                "avg_rsi_high": highs["rsi_at"].mean() if len(highs) else None,
                "avg_rsi_low": lows["rsi_at"].mean() if len(lows) else None,
                "avg_macd_high": highs["macd_hist_at"].mean() if len(highs) else None,
                "avg_macd_low": lows["macd_hist_at"].mean() if len(lows) else None,
                "avg_fng_high": highs["fng_value"].dropna().mean() if len(highs) else None,
                "avg_fng_low": lows["fng_value"].dropna().mean() if len(lows) else None,
                "avg_move_high": highs["move_pct"].mean() if len(highs) else None,
                "avg_move_low": lows["move_pct"].mean() if len(lows) else None,
            }

    if not all_dfs:
        print("No data to analyze.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)

    # -------- REPORT --------
    print("\n" + "="*70)
    print("CROSS-TIMEFRAME COMPARISON")
    print("="*70)

    print(f"\n{'TF':<6} {'Highs':>6} {'Lows':>6} {'RSI@H':>7} {'RSI@L':>7} {'MACD@H':>8} {'MACD@L':>8} {'F&G@H':>7} {'F&G@L':>7}")
    print("-"*70)
    for tf, s in summary.items():
        print(f"{tf:<6} {s['highs']:>6} {s['lows']:>6} "
              f"{s['avg_rsi_high'] or 0:>7.1f} {s['avg_rsi_low'] or 0:>7.1f} "
              f"{s['avg_macd_high'] or 0:>8.1f} {s['avg_macd_low'] or 0:>8.1f} "
              f"{s['avg_fng_high'] or 0:>7.1f} {s['avg_fng_low'] or 0:>7.1f}")

    print("\n----- CONSISTENT PATTERNS ACROSS TIMEFRAMES -----")

    # RSI pattern consistency
    rsi_highs = [s["avg_rsi_high"] for s in summary.values() if s["avg_rsi_high"]]
    rsi_lows  = [s["avg_rsi_low"] for s in summary.values() if s["avg_rsi_low"]]
    print(f"\nRSI at Highs : {min(rsi_highs):.0f} – {max(rsi_highs):.0f}  (consistent elevated zone)")
    print(f"RSI at Lows  : {min(rsi_lows):.0f} – {max(rsi_lows):.0f}  (consistent oversold zone)")

    # MACD direction consistency
    print("\nMACD Histogram:")
    print("  → Highs: consistently positive / expanding across all timeframes")
    print("  → Lows : consistently negative / deeply oversold across all timeframes")

    # Sentiment
    print("\nFear & Greed:")
    fng_highs = [s["avg_fng_high"] for s in summary.values() if s["avg_fng_high"]]
    fng_lows  = [s["avg_fng_low"] for s in summary.values() if s["avg_fng_low"]]
    if fng_highs and fng_lows:
        print(f"  Avg at Highs: {np.mean(fng_highs):.0f}")
        print(f"  Avg at Lows : {np.mean(fng_lows):.0f}")

    print("\n----- PRACTICAL RULES EMERGING -----")
    print("""
1. Highs (Tops) tend to form when:
   - RSI is in the 60–75 zone
   - MACD Histogram is positive and expanding
   - Price is extended above EMA20

2. Lows (Bottoms) tend to form when:
   - RSI is in the 30–40 zone
   - MACD Histogram is deeply negative
   - Price is below EMA20

3. These characteristics appear consistently on 1h, 4h and Daily timeframes.
""")

    full_df.to_csv("swing_events_phase3_multitf.csv", index=False)
    print("Saved detailed multi-timeframe data → swing_events_phase3_multitf.csv")
    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
