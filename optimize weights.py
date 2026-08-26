"""
optimize_weights.py - Tries many weight combinations against your
historical data and reports which ones actually performed best.

This replaces guessing at WEIGHTS in signal_engine.py with real evidence.

Usage:
    python optimize_weights.py BTCUSD_history.csv
"""

import sys
import itertools
import numpy as np
from backtest import run_backtest, compute_metrics
import signal_engine

def optimize(prices, starting_equity=10_000.0):
    # A range of values to try for each weight - kept modest to avoid
    # overfitting to one year of data (more values = more overfitting risk)
    option_values = [0.5, 1.0, 1.5, 2.0]
    keys = ["trend_sma", "trend_ema", "rsi", "macd", "bollinger", "forecast"]

    best = None
    results = []

    # Full grid search would be 4^6 = 4096 combos - a bit much, so we
    # sample a reasonable number randomly instead of testing every one.
    rng = np.random.default_rng(42)
    num_trials = 150

    for _ in range(num_trials):
        weights = {k: rng.choice(option_values) for k in keys}
        result = run_backtest(prices, starting_equity, weights)
        metrics = compute_metrics(result, starting_equity)

        if metrics["num_trades"] < 10:
            continue  # skip combos that barely trade - not enough signal to judge

        results.append((weights, metrics))

        # Rank by Sharpe ratio primarily - risk-adjusted return matters
        # more than raw return for something you'll trust with money
        if best is None or metrics["sharpe_ratio"] > best[1]["sharpe_ratio"]:
            best = (weights, metrics)

    return best, sorted(results, key=lambda r: r[1]["sharpe_ratio"], reverse=True)[:5]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python optimize_weights.py your_history.csv")
        sys.exit(1)

    prices = np.loadtxt(sys.argv[1], delimiter=",")
    best, top5 = optimize(prices)

    print("=" * 50)
    print("TOP 5 WEIGHT COMBINATIONS (ranked by Sharpe ratio)")
    print("=" * 50)
    for i, (weights, metrics) in enumerate(top5, 1):
        print(f"\n#{i}")
        print(f"  Weights: {weights}")
        print(f"  Return: {metrics['total_return_pct']}% | "
              f"Trades: {metrics['num_trades']} | "
              f"Win rate: {metrics['win_rate_pct']}% | "
              f"Sharpe: {metrics['sharpe_ratio']} | "
              f"Max DD: {metrics['max_drawdown_pct']}%")

    print("\n" + "=" * 50)
    print("BEST WEIGHTS FOUND - copy this into signal_engine.py's WEIGHTS:")
    print("=" * 50)
    print(best[0])
    print("\nIMPORTANT: this is fit to ONE dataset. Before trusting it,")
    print("re-test these weights on a DIFFERENT time period or asset (like")
    print("PAXG) to check it's not just overfit to this specific year.")
