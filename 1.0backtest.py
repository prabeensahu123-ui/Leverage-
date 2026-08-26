"""
backtest.py - Walk-forward backtesting engine.

The single most important file in this project. It tells you whether
signal_engine.py actually produces a trading edge, or whether it just
looks reasonable. Every decision at step `i` uses ONLY prices[0:i] -
no lookahead into the future, which is the #1 way backtests lie to you.

Usage:
    python backtest.py path_to_price_history.csv
    (CSV should have one column of closing prices, oldest first)

Or import and call run_backtest(prices) directly with a numpy array.
"""

import sys
import numpy as np
from signal_engine import generate_signal, WEIGHTS

MIN_HISTORY = 60          # bars needed before generating any signal
POSITION_SIZE_PCT = 0.10  # fraction of equity risked per trade
FEE_PCT = 0.0005           # per-side trading fee assumption (0.05%) - set to your exchange's real rate


def run_backtest(prices: np.ndarray, starting_equity=10_000.0, weights=None):
    equity = starting_equity
    equity_curve = [equity]
    position = None  # {"entry": price, "size": qty}
    trades = []

    for i in range(MIN_HISTORY, len(prices)):
        history = prices[: i + 1]          # only past + current bar - no future leakage
        signal = generate_signal(history, weights)
        price = signal["price"]

        if signal["decision"] == "BUY" and position is None:
            size = (equity * POSITION_SIZE_PCT) / price
            cost = size * price * FEE_PCT
            equity -= cost
            position = {"entry": price, "size": size}

        elif signal["decision"] == "SELL" and position is not None:
            pnl = (price - position["entry"]) * position["size"]
            fee = position["size"] * price * FEE_PCT
            equity += pnl - fee
            trades.append(pnl - fee)
            position = None

        # mark-to-market unrealized pnl for the equity curve
        unrealized = (price - position["entry"]) * position["size"] if position else 0
        equity_curve.append(equity + unrealized)

    # close any open position at the final price
    if position is not None:
        pnl = (prices[-1] - position["entry"]) * position["size"]
        equity += pnl
        trades.append(pnl)

    return {
        "final_equity": equity,
        "equity_curve": np.array(equity_curve),
        "trades": trades,
    }


def compute_metrics(result, starting_equity=10_000.0):
    curve = result["equity_curve"]
    trades = result["trades"]

    total_return_pct = (result["final_equity"] / starting_equity - 1) * 100

    wins = [t for t in trades if t > 0]
    win_rate = (len(wins) / len(trades) * 100) if trades else 0.0

    # max drawdown
    running_max = np.maximum.accumulate(curve)
    drawdowns = (curve - running_max) / running_max
    max_drawdown_pct = drawdowns.min() * 100

    # Sharpe ratio on the equity curve's step returns (not annualized - depends on your bar interval)
    step_returns = np.diff(curve) / curve[:-1]
    sharpe = (np.mean(step_returns) / np.std(step_returns) * np.sqrt(len(step_returns))
              if np.std(step_returns) > 0 else 0.0)

    return {
        "total_return_pct": round(total_return_pct, 2),
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate, 1),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
    }


def print_report(prices, weights=None, starting_equity=10_000.0):
    result = run_backtest(prices, starting_equity, weights)
    metrics = compute_metrics(result, starting_equity)

    print("=" * 40)
    print("BACKTEST REPORT")
    print("=" * 40)
    print(f"Bars tested:       {len(prices) - MIN_HISTORY}")
    print(f"Total return:      {metrics['total_return_pct']}%")
    print(f"Number of trades:  {metrics['num_trades']}")
    print(f"Win rate:          {metrics['win_rate_pct']}%")
    print(f"Max drawdown:      {metrics['max_drawdown_pct']}%")
    print(f"Sharpe ratio:      {metrics['sharpe_ratio']}")
    print("=" * 40)
    if metrics["num_trades"] < 20:
        print("NOTE: fewer than 20 trades - not enough data to trust these stats yet.")
    return metrics


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prices = np.loadtxt(sys.argv[1], delimiter=",")
    else:
        # synthetic fallback so the script is runnable standalone for a smoke test
        np.random.seed(1)
        prices = 100 + np.cumsum(np.random.randn(500) * 0.5 + 0.02)
        print("No CSV provided - running on synthetic data as a smoke test.\n")

    print_report(prices)
