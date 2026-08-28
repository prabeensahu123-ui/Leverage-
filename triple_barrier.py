"""
triple_barrier.py - Implements Marcos Lopez de Prado's triple-barrier
labeling method (Advances in Financial Machine Learning, Ch. 3).

Instead of labeling "did price go up next bar" (naive, and blind to
whether the move was big enough to matter), this labels each point by
which of three barriers gets touched first:
  - Upper barrier (profit-take): price rises by k * volatility
  - Lower barrier (stop-loss):   price falls by k * volatility
  - Time barrier: neither hit within max_holding_bars -> label 0 (neutral)

Barriers scale with each point's LOCAL volatility (ATR), so the
labeling automatically adjusts to calm vs. turbulent periods - a
fixed-percent barrier would be too tight in calm markets and too loose
in volatile ones.
"""

import numpy as np


def compute_local_volatility(close, period=20):
    """Rolling volatility of log returns, used to scale barriers."""
    log_returns = np.diff(np.log(close))
    vol = np.zeros(len(close))
    for i in range(period, len(close)):
        vol[i] = np.std(log_returns[i - period:i])
    vol[:period] = vol[period]  # backfill early bars with first computable value
    return vol


def triple_barrier_labels(close, k_profit=2.0, k_stop=2.0, max_holding_bars=24, vol_period=20):
    """
    Returns labels: 1 (profit-take hit first), -1 (stop-loss hit first),
    0 (neither - timed out). Also returns the bar index where each
    label was determined, for diagnostic purposes.
    """
    n = len(close)
    vol = compute_local_volatility(close, vol_period)
    labels = np.zeros(n, dtype=int)
    valid = np.zeros(n, dtype=bool)  # False for bars too close to the end to have a full window
    exit_prices = np.zeros(n)   # the ACTUAL price at which the outcome was determined
    exit_indices = np.zeros(n, dtype=int)  # which future bar triggered the outcome

    for i in range(vol_period, n - 1):
        entry_price = close[i]
        upper_barrier = entry_price * (1 + k_profit * vol[i])
        lower_barrier = entry_price * (1 - k_stop * vol[i])

        end_idx = min(i + max_holding_bars, n - 1)
        window = close[i + 1: end_idx + 1]

        if len(window) == 0:
            continue

        hit_upper = np.where(window >= upper_barrier)[0]
        hit_lower = np.where(window <= lower_barrier)[0]

        first_upper = hit_upper[0] if len(hit_upper) > 0 else np.inf
        first_lower = hit_lower[0] if len(hit_lower) > 0 else np.inf

        if first_upper == np.inf and first_lower == np.inf:
            labels[i] = 0  # timed out - use the actual price at the time barrier as exit
            exit_prices[i] = window[-1]
            exit_indices[i] = i + len(window)
        elif first_upper < first_lower:
            labels[i] = 1
            exit_prices[i] = window[first_upper]  # actual real price when profit-take was hit
            exit_indices[i] = i + 1 + first_upper
        else:
            labels[i] = -1
            exit_prices[i] = window[first_lower]  # actual real price when stop-loss was hit
            exit_indices[i] = i + 1 + first_lower

        valid[i] = True

    return labels, valid, vol, exit_prices, exit_indices
