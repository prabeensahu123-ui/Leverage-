"""
paper_trade_step.py - Paper trading runner for the triple-barrier model.

Updated to use the expanded feature set (SMA200 + MACD Histogram + RSI).

Run this periodically (e.g. once per hour). Each run:
  1. Fetches the latest hourly BTC data from Delta Exchange
  2. Checks any OPEN paper position against new prices
  3. If no position is open, retrains the model and checks for a confident signal
  4. Opens a new paper position if confidence is high enough

NO REAL ORDERS ARE EVER PLACED.
"""

import json
import time
import requests
import numpy as np
import csv
import os
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier

from ml_features import build_features, FEATURE_NAMES, WINNING_FEATURES
from triple_barrier import triple_barrier_labels

DELTA_BASE_URL = "https://api.india.delta.exchange"
SYMBOL = "BTCUSD"
K_PROFIT = 2.0
K_STOP = 2.0
MAX_HOLDING_BARS = 24
FEE_PCT = 0.0005
POSITION_SIZE_PCT = 0.10
STARTING_EQUITY = 10_000.0

WINNING_FEATURE_INDICES = [FEATURE_NAMES.index(f) for f in WINNING_FEATURES]

STATE_FILE = "paper_trade_state.json"
LOG_FILE = "paper_trade_log.csv"


def fetch_latest_candles(days=120):
    end = int(time.time())
    start = end - days * 86400
    url = f"{DELTA_BASE_URL}/v2/history/candles"
    params = {"resolution": "1h", "symbol": SYMBOL, "start": start, "end": end}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("result", [])
    return list(reversed(data))


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"equity": STARTING_EQUITY, "open_position": None, "last_processed_time": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def log_trade(row):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["closed_at", "direction", "entry_price", "exit_price",
                              "entry_time", "exit_reason", "pnl", "equity_after"])
        writer.writerow(row)


def main():
    print(f"[{datetime.now(timezone.utc).isoformat()}] Paper trading step starting...")
    print(f"Using features: {WINNING_FEATURES}")

    candles = fetch_latest_candles()
    close = np.array([c["close"] for c in candles])
    high = np.array([c["high"] for c in candles])
    low = np.array([c["low"] for c in candles])
    volume = np.array([c["volume"] for c in candles])
    times = np.array([c["time"] for c in candles])
    ohlcv = {"open": np.array([c["open"] for c in candles]), "high": high,
             "low": low, "close": close, "volume": volume}

    state = load_state()

    last_processed = state.get("last_processed_time")
    if last_processed is None:
        bars_to_process = [len(times) - 1]
    else:
        bars_to_process = [i for i in range(len(times)) if times[i] > last_processed]
        if not bars_to_process:
            print("No new bars since last run - nothing to do yet.")
            return

    print(f"Processing {len(bars_to_process)} new hourly bar(s)...")

    for bar_idx in bars_to_process:
        current_price = close[bar_idx]
        current_time = int(times[bar_idx])
        _process_bar(state, ohlcv, close, bar_idx, current_price, current_time)

    state["last_processed_time"] = int(times[-1])
    save_state(state)
    print(f"\nCurrent paper equity: {state['equity']:.2f}")
    print("Done.\n")


def _process_bar(state, ohlcv, close, bar_idx, current_price, current_time):
    print(f"\n--- Bar: {datetime.fromtimestamp(current_time, tz=timezone.utc)} | Price: {current_price:.2f} ---")

    if state["open_position"] is not None:
        pos = state["open_position"]
        entry_price = pos["entry_price"]
        direction = pos["direction"]
        upper_barrier = pos["upper_barrier"]
        lower_barrier = pos["lower_barrier"]
        opened_time = pos["opened_time"]
        bars_open = pos["bars_open"] + 1

        hit_upper = current_price >= upper_barrier
        hit_lower = current_price <= lower_barrier
        timed_out = bars_open >= MAX_HOLDING_BARS

        if hit_upper or hit_lower or timed_out:
            exit_price = current_price
            exit_reason = "upper_barrier" if hit_upper else ("lower_barrier" if hit_lower else "timeout")

            size = (state["equity"] * POSITION_SIZE_PCT) / entry_price
            fees = size * entry_price * FEE_PCT * 2
            gross_pnl = direction * (exit_price - entry_price) * size
            net_pnl = gross_pnl - fees
            state["equity"] += net_pnl

            log_trade([
                datetime.now(timezone.utc).isoformat(), direction, entry_price, exit_price,
                opened_time, exit_reason, round(net_pnl, 2), round(state["equity"], 2)
            ])
            print(f"POSITION CLOSED: direction={direction} entry={entry_price:.2f} exit={exit_price:.2f} "
                  f"reason={exit_reason} pnl={net_pnl:.2f} equity={state['equity']:.2f}")
            state["open_position"] = None
        else:
            state["open_position"]["bars_open"] = bars_open
            print(f"Position still open: direction={direction} entry={entry_price:.2f} "
                  f"bars_open={bars_open}/{MAX_HOLDING_BARS}")
        return

    # No open position → look for new signal
    ohlcv_so_far = {k: v[:bar_idx + 1] for k, v in ohlcv.items()}
    close_so_far = close[:bar_idx + 1]

    labels, valid, vol, _, _ = triple_barrier_labels(
        close_so_far, K_PROFIT, K_STOP, MAX_HOLDING_BARS
    )
    X, y = [], []
    min_history = 60
    for i in range(min_history, len(close_so_far) - 1):
        if valid[i] and labels[i] != 0:
            X.append(build_features(ohlcv_so_far, i))
            y.append(1 if labels[i] == 1 else 0)

    if len(X) < 100:
        print("Not enough training data yet - skipping signal generation.")
        return

    X, y = np.array(X), np.array(y)
    X = X[:, WINNING_FEATURE_INDICES]
    model = RandomForestClassifier(
        n_estimators=150, max_depth=5, random_state=42, min_samples_leaf=10
    )
    model.fit(X, y)

    current_features = np.array([build_features(ohlcv_so_far, bar_idx)])[:, WINNING_FEATURE_INDICES]
    prob = model.predict_proba(current_features)[0, 1]

    print(f"Model confidence (prob of upward barrier hit first): {prob:.3f}")

    if prob > 0.60 or prob < 0.40:
        direction = 1 if prob > 0.60 else -1
        current_vol = vol[bar_idx] if vol[bar_idx] > 0 else np.std(np.diff(np.log(close_so_far[-20:])))
        upper_barrier = current_price * (1 + K_PROFIT * current_vol)
        lower_barrier = current_price * (1 - K_STOP * current_vol)

        state["open_position"] = {
            "direction": direction,
            "entry_price": current_price,
            "upper_barrier": upper_barrier,
            "lower_barrier": lower_barrier,
            "opened_time": datetime.now(timezone.utc).isoformat(),
            "bars_open": 0,
            "confidence": float(prob),
        }
        print(f"NEW PAPER POSITION OPENED: direction={direction} entry={current_price:.2f} "
              f"upper={upper_barrier:.2f} lower={lower_barrier:.2f} confidence={prob:.3f}")
    else:
        print("No confident signal this bar - staying flat.")


if __name__ == "__main__":
    main()
