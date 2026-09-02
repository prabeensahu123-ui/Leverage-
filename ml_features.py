"""
ml_features.py - Builds a feature matrix + labels for machine learning.

Updated to include high-value technical features:
  - price_vs_sma200 (major trend regime)
  - macd_histogram (momentum)
  - improved RSI handling
"""

import numpy as np
from signal_engine import sma, ema, rsi, macd, bollinger_position
from features import atr, volume_zscore, volatility_regime, price_volume_trend
from forecasting import ensemble_forecast

FEATURE_NAMES = [
    "price_vs_sma20",
    "price_vs_ema20",
    "price_vs_sma200",      # NEW - major trend filter
    "rsi14",
    "macd_hist",            # already present, now more prominent
    "bollinger_pos",
    "forecast_gap",
    "volume_z",
    "vol_regime",
    "price_vol_trend",
    "return_1d",
    "return_3d",
    "return_7d",
    "atr_pct",
]

# Previously validated subset + new strong candidates
# Re-validate these on out-of-sample data before trusting fully
WINNING_FEATURES = [
    "atr_pct",
    "price_vol_trend",
    "return_3d",
    "price_vs_ema20",
    "price_vs_sma200",      # NEW
    "macd_hist",            # NEW emphasis
    "rsi14",
]


def build_features(ohlcv: dict, index: int):
    """Builds one feature row using data available up to `index` (inclusive)."""
    close = ohlcv["close"][: index + 1]
    high = ohlcv["high"][: index + 1]
    low = ohlcv["low"][: index + 1]
    volume = ohlcv["volume"][: index + 1]
    price = close[-1]

    # Moving averages
    sma20 = sma(close, 20) if len(close) >= 20 else price
    ema20 = ema(close, 20) if len(close) >= 20 else price
    sma200 = sma(close, 200) if len(close) >= 200 else sma(close, min(len(close), 100))

    # RSI
    rsi14 = rsi(close, 14)

    # MACD
    macd_line, macd_signal = macd(close)
    macd_hist = macd_line - macd_signal

    # Bollinger
    boll_pos = bollinger_position(close)

    # Forecast gap
    fc = ensemble_forecast(close, steps=1)
    forecast_gap = (fc["forecast"][0] - price) / price if price != 0 else 0.0

    # Volume & volatility features
    vol_z = volume_zscore(volume)
    vol_regime = volatility_regime(close)
    pvt = price_volume_trend(close, volume)
    atr_val = atr(high, low, close)

    return [
        (price - sma20) / sma20 if sma20 != 0 else 0.0,          # price_vs_sma20
        (price - ema20) / ema20 if ema20 != 0 else 0.0,          # price_vs_ema20
        (price - sma200) / sma200 if sma200 != 0 else 0.0,       # price_vs_sma200 (NEW)
        rsi14 / 100.0,                                          # rsi14
        macd_hist,                                              # macd_hist
        boll_pos,                                               # bollinger_pos
        forecast_gap,                                           # forecast_gap
        vol_z,                                                  # volume_z
        vol_regime,                                             # vol_regime
        pvt,                                                    # price_vol_trend
        (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0.0,   # return_1d
        (close[-1] - close[-4]) / close[-4] if len(close) > 3 else 0.0,   # return_3d
        (close[-1] - close[-8]) / close[-8] if len(close) > 7 else 0.0,   # return_7d
        atr_val / price if price != 0 else 0.0,                 # atr_pct
    ]


def build_dataset(ohlcv: dict, min_history=60, label_horizon=1):
    """
    Builds X (features) and y (labels) for supervised learning.
    Note: For triple-barrier labels use the dedicated function in paper_trade / app.
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
