"""
candle_patterns.py
Detect simple last-bar / last-2-bar candlestick patterns.
Returns pattern name and BUY / SELL / HOLD bias.
"""

import numpy as np


def _bar(open_, high, low, close, i):
    o, h, l, c = float(open_[i]), float(high[i]), float(low[i]), float(close[i])
    body = abs(c - o)
    rng = max(h - l, 1e-12)
    upper = h - max(o, c)
    lower = min(o, c) - l
    bull = c > o
    return o, h, l, c, body, rng, upper, lower, bull


def detect_candle_pattern(open_, high, low, close):
    """
    Look at the last 1-2 completed bars.
    Returns (pattern_name, bias) where bias is BUY / SELL / HOLD.
    """
    n = len(close)
    if n < 3:
        return "None", "HOLD"

    o1, h1, l1, c1, body1, rng1, up1, lo1, bull1 = _bar(open_, high, low, close, -1)
    o0, h0, l0, c0, body0, rng0, up0, lo0, bull0 = _bar(open_, high, low, close, -2)

    # Doji
    if body1 / rng1 < 0.12:
        return "Doji", "HOLD"

    # Hammer (small body, long lower wick, near highs)
    if lo1 >= 2.0 * body1 and up1 <= body1 * 0.6 and body1 / rng1 < 0.40:
        return "Hammer", "BUY"

    # Hanging man looks like hammer but after an up bar - still flag as caution sell if prior was bullish
    if lo1 >= 2.0 * body1 and up1 <= body1 * 0.6 and body1 / rng1 < 0.40 and bull0:
        return "Hanging Man", "SELL"

    # Shooting star (long upper wick)
    if up1 >= 2.0 * body1 and lo1 <= body1 * 0.6 and body1 / rng1 < 0.40:
        return "Shooting Star", "SELL"

    # Inverted hammer (long upper wick after decline)
    if up1 >= 2.0 * body1 and lo1 <= body1 * 0.6 and body1 / rng1 < 0.40 and not bull0:
        return "Inverted Hammer", "BUY"

    # Bullish engulfing
    if (not bull0) and bull1 and c1 >= o0 and o1 <= c0 and body1 > body0 * 1.05:
        return "Bull Engulf", "BUY"

    # Bearish engulfing
    if bull0 and (not bull1) and o1 >= c0 and c1 <= o0 and body1 > body0 * 1.05:
        return "Bear Engulf", "SELL"

    # Marubozu-ish strong close
    if bull1 and body1 / rng1 > 0.75 and up1 / rng1 < 0.12 and lo1 / rng1 < 0.12:
        return "Bull Marubozu", "BUY"
    if (not bull1) and body1 / rng1 > 0.75 and up1 / rng1 < 0.12 and lo1 / rng1 < 0.12:
        return "Bear Marubozu", "SELL"

    return "No clear pattern", "HOLD"
