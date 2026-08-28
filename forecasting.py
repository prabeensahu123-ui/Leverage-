"""
forecasting.py - Statistical forecasting for price series.

Improves on plain Holt's linear trend with:
  - Damped trend (prevents runaway extrapolation on volatile assets)
  - Confidence bands derived from actual historical residual error,
    not an arbitrary guess
  - A simple ensemble blend with a log-return linear regression,
    so the forecast isn't relying on one model's assumptions
"""

import numpy as np


class DampedHoltForecaster:
    """
    Holt's double exponential smoothing with a damping parameter (phi).
    phi < 1 pulls the trend toward flat over longer horizons - important
    for crypto, where an aggressive short-term trend rarely continues
    linearly for days.
    """

    def __init__(self, alpha=0.3, beta=0.1, phi=0.90):
        self.alpha = alpha
        self.beta = beta
        self.phi = phi

    def fit(self, prices: np.ndarray):
        level, trend = prices[0], prices[1] - prices[0]
        fitted = [level]
        for price in prices[1:]:
            last_level = level
            level = self.alpha * price + (1 - self.alpha) * (level + self.phi * trend)
            trend = self.beta * (level - last_level) + (1 - self.beta) * self.phi * trend
            fitted.append(level)
        self.level, self.trend = level, trend
        self.residuals = prices - np.array(fitted)
        self.resid_std = np.std(self.residuals)
        return self

    def forecast(self, steps=1):
        """Returns (point_forecasts, lower_band, upper_band) for `steps` ahead."""
        damp_sum = sum(self.phi ** i for i in range(1, steps + 1))
        points, lowers, uppers = [], [], []
        cumulative_damp = 0
        for h in range(1, steps + 1):
            cumulative_damp += self.phi ** h
            point = self.level + cumulative_damp * self.trend
            # error grows with sqrt(horizon) - standard for random-walk-type uncertainty
            band = 1.96 * self.resid_std * np.sqrt(h)
            points.append(point)
            lowers.append(point - band)
            uppers.append(point + band)
        return np.array(points), np.array(lowers), np.array(uppers)


def log_return_trend_forecast(prices: np.ndarray, steps=1, lookback=30):
    """
    Secondary model: fits a linear trend to recent log returns and
    projects forward. Captures momentum differently than Holt's -
    useful as an ensemble check rather than sole source of truth.
    """
    window = prices[-lookback:]
    log_prices = np.log(window)
    x = np.arange(len(log_prices))
    slope, intercept = np.polyfit(x, log_prices, 1)
    last_x = len(log_prices) - 1
    forecasts = [np.exp(intercept + slope * (last_x + h)) for h in range(1, steps + 1)]
    return np.array(forecasts)


def ensemble_forecast(prices: np.ndarray, steps=1, holt_weight=0.65):
    """
    Blends damped Holt's and log-return trend. holt_weight controls how
    much to trust the smoothing model vs. the momentum model - tune this
    based on what the backtester shows for your specific asset/timeframe.
    """
    holt = DampedHoltForecaster().fit(prices)
    holt_points, lower, upper = holt.forecast(steps)
    trend_points = log_return_trend_forecast(prices, steps)

    blended = holt_weight * holt_points + (1 - holt_weight) * trend_points
    # widen bands slightly to reflect model disagreement
    disagreement = np.abs(holt_points - trend_points)
    lower = blended - (upper - holt_points) - disagreement * 0.5
    upper = blended + (upper - holt_points) + disagreement * 0.5

    return {
        "forecast": blended,
        "lower_band": lower,
        "upper_band": upper,
        "holt_only": holt_points,
        "trend_only": trend_points,
    }


if __name__ == "__main__":
    # quick sanity check with synthetic data
    np.random.seed(0)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5 + 0.05)
    result = ensemble_forecast(prices, steps=5)
    print("5-step forecast:", np.round(result["forecast"], 2))
    print("Lower band:     ", np.round(result["lower_band"], 2))
    print("Upper band:     ", np.round(result["upper_band"], 2))
