"""
paper_trading.py
Background Paper Trading System

- Monitors selected timeframes
- Opens virtual trades when model confidence is high
- Manages TP / SL / Max Holding
- Logs every trade for performance analysis
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier

from ml_features import build_features, FEATURE_NAMES, WINNING_FEATURES
from triple_barrier import triple_barrier_labels

# -------------------- CONFIG --------------------
DELTA_BASE = "https://api.india.delta.exchange"
SYMBOL = "BTCUSD"
WINNING_FEATURE_INDICES = [FEATURE_NAMES.index(f) for f in WINNING_FEATURES]

K_PROFIT = 2.0
K_STOP = 2.0
MIN_CONFIDENCE = 0.62
TRADE_LOG_FILE = "paper_trades.csv"
STATE_FILE = "paper_state.json"

PAPER_TIMEFRAMES = {
    "1h": {"resolution": "1h", "days": 90,  "max_holding_bars": 24, "label": "1 Hour"},
    "4h": {"resolution": "4h", "days": 180, "max_holding_bars": 18, "label": "4 Hour"},
}

# -------------------- HELPERS --------------------
def fetch_candles(symbol, resolution, days):
    end = int(time.time())
    start = end - days * 86400
    try:
        r = requests.get(
            f"{DELTA_BASE}/v2/history/candles",
            params={"resolution": resolution, "symbol": symbol, "start": start, "end": end},
            timeout=12
        ).json()
        return list(reversed(r.get("result", [])))
    except Exception as e:
        print(f"Candle fetch error ({resolution}): {e}")
        return []

def fetch_ticker(symbol):
    try:
        return requests.get(f"{DELTA_BASE}/v2/tickers/{symbol}", timeout=5).json().get("result", {})
    except Exception:
        return {}

def get_model_proba(candles, max_holding):
    if len(candles) < 80:
        return None, None

    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)
    volume = np.array([c["volume"] for c in candles], dtype=float)
    ohlcv = {"open": close, "high": high, "low": low, "close": close, "volume": volume}

    labels, valid, vol, _, _ = triple_barrier_labels(close, K_PROFIT, K_STOP, max_holding)

    X, y = [], []
    for i in range(60, len(close) - 1):
        if valid[i] and labels[i] != 0:
            X.append(build_features(ohlcv, i))
            y.append(1 if labels[i] == 1 else 0)

    if len(X) < 80:
        return None, None

    X = np.array(X)[:, WINNING_FEATURE_INDICES]
    y = np.array(y)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    model.fit(X, y)

    proba = float(model.predict_proba(
        np.array([build_features(ohlcv, len(close) - 1)])[:, WINNING_FEATURE_INDICES]
    )[0, 1])

    cur_vol = float(vol[-1]) if len(vol) > 0 and vol[-1] > 0 else float(np.std(np.diff(np.log(close[-20:]))))
    return proba, cur_vol

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"open_trades": {}}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def log_trade(trade):
    df = pd.DataFrame([trade])
    if os.path.exists(TRADE_LOG_FILE):
        df.to_csv(TRADE_LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(TRADE_LOG_FILE, index=False)

def close_trade(trade, exit_price, reason):
    entry = trade["entry"]
    side = trade["side"]
    if side == "BUY":
        pnl_pct = (exit_price - entry) / entry * 100
    else:
        pnl_pct = (entry - exit_price) / entry * 100

    result = {
        **trade,
        "exit_price": round(exit_price, 2),
        "exit_time": datetime.now(timezone.utc).isoformat(),
        "exit_reason": reason,
        "pnl_pct": round(pnl_pct, 3),
        "win": pnl_pct > 0
    }
    log_trade(result)
    return result

# -------------------- CORE LOGIC --------------------
def check_and_trade():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running paper trading check...")
    state = load_state()
    open_trades = state.get("open_trades", {})
    ticker = fetch_ticker(SYMBOL)
    live_price = float(ticker.get("close", ticker.get("mark_price", 0)))

    if live_price <= 0:
        print("Could not fetch live price.")
        return

    print(f"Live price: ${live_price:,.2f}")

    for tf_key, cfg in PAPER_TIMEFRAMES.items():
        # Manage existing open trade
        if tf_key in open_trades:
            trade = open_trades[tf_key]
            bars_held = trade.get("bars_held", 0) + 1
            trade["bars_held"] = bars_held

            if trade["side"] == "BUY":
                if live_price >= trade["take_profit"]:
                    result = close_trade(trade, live_price, "Take Profit")
                    print(f"  ✅ {tf_key} BUY closed at TP | PnL: {result['pnl_pct']:+.2f}%")
                    del open_trades[tf_key]
                elif live_price <= trade["stop_loss"]:
                    result = close_trade(trade, live_price, "Stop Loss")
                    print(f"  ❌ {tf_key} BUY closed at SL | PnL: {result['pnl_pct']:+.2f}%")
                    del open_trades[tf_key]
                elif bars_held >= cfg["max_holding_bars"]:
                    result = close_trade(trade, live_price, "Max Holding")
                    print(f"  ⏰ {tf_key} BUY closed (time) | PnL: {result['pnl_pct']:+.2f}%")
                    del open_trades[tf_key]
                else:
                    open_trades[tf_key] = trade
            else:
                if live_price <= trade["take_profit"]:
                    result = close_trade(trade, live_price, "Take Profit")
                    print(f"  ✅ {tf_key} SELL closed at TP | PnL: {result['pnl_pct']:+.2f}%")
                    del open_trades[tf_key]
                elif live_price >= trade["stop_loss"]:
                    result = close_trade(trade, live_price, "Stop Loss")
                    print(f"  ❌ {tf_key} SELL closed at SL | PnL: {result['pnl_pct']:+.2f}%")
                    del open_trades[tf_key]
                elif bars_held >= cfg["max_holding_bars"]:
                    result = close_trade(trade, live_price, "Max Holding")
                    print(f"  ⏰ {tf_key} SELL closed (time) | PnL: {result['pnl_pct']:+.2f}%")
                    del open_trades[tf_key]
                else:
                    open_trades[tf_key] = trade
            continue

        # Check for new signal
        candles = fetch_candles(SYMBOL, cfg["resolution"], cfg["days"])
        proba, cur_vol = get_model_proba(candles, cfg["max_holding_bars"])

        if proba is None:
            print(f"  {tf_key}: Not enough data")
            continue

        print(f"  {tf_key}: Model proba = {proba:.3f}")

        if proba >= MIN_CONFIDENCE:
            side = "BUY"
            tp = live_price * (1 + K_PROFIT * cur_vol)
            sl = live_price * (1 - K_STOP * cur_vol)
        elif proba <= (1 - MIN_CONFIDENCE):
            side = "SELL"
            tp = live_price * (1 - K_PROFIT * cur_vol)
            sl = live_price * (1 + K_STOP * cur_vol)
        else:
            print(f"  {tf_key}: No clear signal")
            continue

        trade = {
            "id": f"{tf_key}_{int(time.time())}",
            "symbol": SYMBOL,
            "timeframe": tf_key,
            "side": side,
            "entry": round(live_price, 2),
            "take_profit": round(tp, 2),
            "stop_loss": round(sl, 2),
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "bars_held": 0,
            "confidence": round(proba if side == "BUY" else (1 - proba), 3),
            "max_holding": cfg["max_holding_bars"]
        }
        open_trades[tf_key] = trade
        print(f"  📈 NEW {side} paper trade opened on {tf_key} @ ${live_price:,.2f}")
        print(f"     TP: ${tp:,.2f} | SL: ${sl:,.2f}")

    state["open_trades"] = open_trades
    save_state(state)
    print("State saved.")

def show_performance():
    if not os.path.exists(TRADE_LOG_FILE):
        print("No closed trades yet.")
        return

    df = pd.read_csv(TRADE_LOG_FILE)
    if df.empty:
        print("No closed trades yet.")
        return

    print("\n========== PAPER TRADING PERFORMANCE ==========")
    print(f"Total closed trades : {len(df)}")
    print(f"Win rate            : {(df['win'].mean()*100):.1f}%")
    print(f"Average PnL         : {df['pnl_pct'].mean():+.2f}%")
    print(f"Total PnL (sum)     : {df['pnl_pct'].sum():+.2f}%")
    print(f"Best trade          : {df['pnl_pct'].max():+.2f}%")
    print(f"Worst trade         : {df['pnl_pct'].min():+.2f}%")

    print("\nBy Timeframe:")
    print(df.groupby("timeframe").agg(
        trades=("pnl_pct", "count"),
        winrate=("win", "mean"),
        avg_pnl=("pnl_pct", "mean")
    ).round(3))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one check and exit")
    parser.add_argument("--performance", action="store_true", help="Show performance report")
    parser.add_argument("--loop", action="store_true", help="Run continuously every 15 minutes")
    args = parser.parse_args()

    if args.performance:
        show_performance()
    elif args.loop:
        print("Starting paper trading loop (every 15 min)...")
        while True:
            check_and_trade()
            show_performance()
            print("Sleeping 15 minutes...\n")
            time.sleep(15 * 60)
    else:
        check_and_trade()
        show_performance()
