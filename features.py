"""
features.py - Feature engineering using full OHLCV data.

Adds volume and volatility-based features on top of the original
price-only indicators - these tend to carry more real signal than
close-price indicators alone.
"""

import numpy as np


def atr(high, low, close, period=14):
    """Average True Range - a standard volatility measure."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return np.mean(tr[-period:])


def volume_zscore(volume, period=20):
    """How unusual is today's volume vs recent history? High z-score = unusual spike."""
    window = volume[-period:]
    mean, std = np.mean(window), np.std(window)
    if std == 0:
        return 0.0
    return (volume[-1] - mean) / std


def volatility_regime(close, period=20, lookback=100):
    """
    Is current volatility high or low relative to its own recent history?
    Returns >1 if volatility is elevated, <1 if unusually calm.
    """
    returns = np.diff(np.log(close))
    current_vol = np.std(returns[-period:])
    historical_vol = np.std(returns[-lookback:])
    if historical_vol == 0:
        return 1.0
    return current_vol / historical_vol


def price_volume_trend(close, volume, period=14):
    """
    Does volume confirm the price move? Positive = rising price on rising
    volume (healthy trend); negative = rising price on falling volume
    (weak, possibly unsustainable move).
    """
    returns = np.diff(close[-period - 1:]) / close[-period - 1:-1]
    vol_changes = np.diff(volume[-period - 1:])
    if np.std(vol_changes) == 0 or np.std(returns) == 0:
        return 0.0
    correlation = np.corrcoef(returns, vol_changes)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0
