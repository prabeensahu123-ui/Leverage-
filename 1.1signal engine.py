"""
signal_engine.py - Weighted technical signal scoring.

Weights are exposed as constants so the backtester can be used to tune
them empirically, instead of guessing which indicator matters most.
"""

import numpy as np
from forecasting import ensemble_forecast

# Default weights - START here, then adjust based on backtest.py results.
WEIGHTS = {
    "trend_sma": 1.0,
    "trend_ema": 1.0,
    "rsi": 2.0,
    "macd": 1.5,
    "bollinger": 1.5,
    "forecast": 1.5,
}

BUY_THRESHOLD = 3.0
SELL_THRESHOLD = -3.0


def sma(prices, period):
    return np.mean(prices[-period:])


def ema_series(prices, period):
    weights = np.exp(np.linspace(-1.0, 0.0, period))
    weights /= weights.sum()
    return np.convolve(prices, weights, mode="valid")


def ema(prices, period):
    return ema_series(prices, period)[-1]


def rsi(prices, period=14):
    deltas = np.diff(prices[-(period + 1):])
    gains = deltas[deltas > 0].sum() / period
    losses = -deltas[deltas < 0].sum() / period
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - (100 / (1 + rs))


def macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal:
        return 0.0, 0.0
    ema_fast = ema_series(prices, fast)
    ema_slow = ema_series(prices, slow)
    min_len = min(len(ema_fast), len(ema_slow))
    macd_line_series = ema_fast[-min_len:] - ema_slow[-min_len:]
    signal_line = np.mean(macd_line_series[-signal:])
    return macd_line_series[-1], signal_line


def bollinger_position(prices, period=20, num_std=2):
    """Returns where price sits relative to the bands: <0 below lower, >0 above upper, 0 = middle."""
    window = prices[-period:]
    mid = np.mean(window)
    std = np.std(window)
    upper, lower = mid + num_std * std, mid - num_std * std
    price = prices[-1]
    if price <= lower:
        return -1.0
    if price >= upper:
        return 1.0
    return (price - mid) / (upper - mid) if upper != mid else 0.0


def generate_signal(prices: np.ndarray, weights: dict = None):
    """
    Returns a dict with the weighted score, decision, and every
    component so you can see WHY a signal fired - critical for
    trusting (or debugging) the system.
    """
    w = weights or WEIGHTS
    price = prices[-1]

    sma20, ema20 = sma(prices, 20), ema(prices, 20)
    rsi14 = rsi(prices, 14)
    macd_line, macd_signal = macd(prices)
    boll_pos = bollinger_position(prices)
    fc = ensemble_forecast(prices, steps=1)
    forecast_price = fc["forecast"][0]

    components = {}
    components["trend_sma"] = w["trend_sma"] * (1 if price > sma20 else -1)
    components["trend_ema"] = w["trend_ema"] * (1 if price > ema20 else -1)

    if rsi14 < 30:
        components["rsi"] = w["rsi"] * 1
    elif rsi14 > 70:
        components["rsi"] = w["rsi"] * -1
    else:
        components["rsi"] = 0.0

    components["macd"] = w["macd"] * (1 if macd_line > macd_signal else -1)
    components["bollinger"] = w["bollinger"] * (-boll_pos)  # near lower band = bullish
    components["forecast"] = w["forecast"] * (1 if forecast_price > price else -1)

    score = sum(components.values())

    if score >= BUY_THRESHOLD:
        decision = "BUY"
    elif score <= SELL_THRESHOLD:
        decision = "SELL"
    else:
        decision = "HOLD"

    return {
        "price": price, "score": round(score, 2), "decision": decision,
        "components": {k: round(v, 2) for k, v in components.items()},
        "rsi": round(rsi14, 1), "macd": round(macd_line, 2),
        "forecast": round(forecast_price, 2),
        "forecast_band": (round(fc["lower_band"][0], 2), round(fc["upper_band"][0], 2)),
    }
