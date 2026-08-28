"""
ml_features.py - Builds a feature matrix + labels for machine learning,
using the same indicators/features already validated in the pipeline,
plus a few additional lagged features ML models can exploit that a
simple weighted score can't.
"""

import numpy as np
from signal_engine import sma, ema, rsi, macd, bollinger_position
from features import atr, volume_zscore, volatility_regime, price_volume_trend
from forecasting import ensemble_forecast

FEATURE_NAMES = [
    "price_vs_sma20", "price_vs_ema20", "rsi14", "macd_hist",
    "bollinger_pos", "forecast_gap", "volume_z", "vol_regime",
    "price_vol_trend", "return_1d", "return_3d", "return_7d", "atr_pct",
]


def build_features(ohlcv: dict, index: int):
    """Builds one feature row using data available up to `index` (inclusive)."""
    close = ohlcv["close"][: index + 1]
    high = ohlcv["high"][: index + 1]
    low = ohlcv["low"][: index + 1]
    volume = ohlcv["volume"][: index + 1]
    price = close[-1]

    sma20, ema20 = sma(close, 20), ema(close, 20)
    rsi14 = rsi(close, 14)
    macd_line, macd_signal = macd(close)
    boll_pos = bollinger_position(close)
    fc = ensemble_forecast(close, steps=1)
    forecast_gap = (fc["forecast"][0] - price) / price

    vol_z = volume_zscore(volume)
    vol_regime = volatility_regime(close)
    pvt = price_volume_trend(close, volume)
    atr_val = atr(high, low, close)

    return [
        (price - sma20) / sma20,
        (price - ema20) / ema20,
        rsi14 / 100,
        macd_line - macd_signal,
        boll_pos,
        forecast_gap,
        vol_z,
        vol_regime,
        pvt,
        (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0,
        (close[-1] - close[-4]) / close[-4] if len(close) > 3 else 0,
        (close[-1] - close[-8]) / close[-8] if len(close) > 7 else 0,
        atr_val / price,
    ]


def build_dataset(ohlcv: dict, min_history=60, label_horizon=1):
    """
    Builds X (features) and y (labels: 1 if price rises over next
    `label_horizon` bars, else 0) for every valid index in the data.
    """
    close = ohlcv["close"]
    n = len(close)
    X, y, idxs = [], [], []

    for i in range(min_history, n - label_horizon):
        X.append(build_features(ohlcv, i))
        future_return = (close[i + label_horizon] - close[i]) / close[i]
        y.append(1 if future_return > 0 else 0)
        idxs.append(i)

    return np.array(X), np.array(y), idxs
