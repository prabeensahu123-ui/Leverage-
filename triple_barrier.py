"""
triple_barrier.py - Lopez de Prado triple-barrier labels + cost filter.
A move that cannot beat fees is labeled HOLD (0), not a win/loss.
"""

import numpy as np


def compute_local_volatility(close, period=20):
    log_returns = np.diff(np.log(close))
    vol = np.zeros(len(close))
    for i in range(period, len(close)):
        vol[i] = np.std(log_returns[i - period:i])
    vol[:period] = vol[period] if period < len(close) else 0.0
    return vol


def triple_barrier_labels(close, k_profit=2.0, k_stop=2.0, max_holding_bars=24, vol_period=20):
    n = len(close)
    vol = compute_local_volatility(close, vol_period)
    labels = np.zeros(n, dtype=int)
    valid = np.zeros(n, dtype=bool)
    exit_prices = np.zeros(n)
    exit_indices = np.zeros(n, dtype=int)

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
            labels[i] = 0
            exit_prices[i] = window[-1]
            exit_indices[i] = i + len(window)
        elif first_upper < first_lower:
            labels[i] = 1
            exit_prices[i] = window[first_upper]
            exit_indices[i] = i + 1 + first_upper
        else:
            labels[i] = -1
            exit_prices[i] = window[first_lower]
            exit_indices[i] = i + 1 + first_lower
        valid[i] = True

    return labels, valid, vol, exit_prices, exit_indices


def apply_cost_filter(labels, close, exit_prices, valid, min_move_pct=0.0015):
    """
    Relabel outcomes whose |exit-entry|/entry is below the fee buffer as 0.
    min_move_pct default 0.15% = ~0.118% taker round-trip + slack.
    """
    out = labels.copy()
    for i in range(len(close)):
        if not valid[i] or close[i] == 0:
            continue
        move = abs(exit_prices[i] - close[i]) / close[i]
        if move < min_move_pct:
            out[i] = 0
    return out
