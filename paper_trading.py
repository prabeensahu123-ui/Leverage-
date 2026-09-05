"""
paper_trading.py
4-Timeframe Paper Trading System (Max 4 trades/day)

Timeframes: 15m | 1h | 4h | Daily
Rule: Only 1 open trade per timeframe
Uses balanced 3-class model (BUY / SELL / HOLD)
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

DELTA_BASE = "https://api.india.delta.exchange"
SYMBOL = "BTCUSD"
WINNING_FEATURE_INDICES = [FEATURE_NAMES.index(f) for f in WINNING_FEATURES]

K_PROFIT = 2.0
K_STOP = 2.0
MIN_BUY_PROBA = 0.45
MIN_SELL_PROBA = 0.45
MIN_EDGE = 0.10

TRADE_LOG_FILE = "paper_trades.csv"
STATE_FILE = "paper_state.json"

# 4 Timeframes = Max 4 paper trades per day
PAPER_TIMEFRAMES = {
    "15m": {"resolution": "15m", "days": 30,  "max_holding_bars": 32, "label": "15m"},
    "1h":  {"resolution": "1h",  "days": 90,  "max_holding_bars": 24, "label": "1h"},
    "4h":  {"resolution": "4h",  "days": 180, "max_holding_bars": 18, "label": "4h"},
    "1D":  {"resolution": "1d",  "days": 300, "max_holding_bars": 15, "label": "Daily"},
}

def fetch_candles(symbol, resolution, days):
    end = int(time.time())
    start = end - days * 86400
    try:
        r = requests.get(
            f"{DELTA_BASE}/v2/history/candles",
            params={"resolution": resolution, "symbol": symbol, "start": start, "end": end},
            timeout=14
        ).json()
        return list(reversed(r.get("result", [])))
    except Exception as e:
        print(f"Candle error ({resolution}): {e}")
        return []

def fetch_ticker(symbol):
    try:
        return requests.get(f"{DELTA_BASE}/v2/tickers/{symbol}", timeout=5).json().get("result", {})
    except Exception:
        return {}

def get_balanced_proba(candles, max_holding):
    if len(candles) < 80:
        return None, None, None, None

    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)
    volume = np.array([c["volume"] for c in candles], dtype=float)
    ohlcv = {"open": close, "high": high, "low": low, "close": close, "volume": volume}

    labels, valid, vol, _, _ = triple_barrier_labels(close, K_PROFIT, K_STOP, max_holding)

    X, y = [], []
    for i in range(55, len(close) - 1):
        if valid[i]:
            X.append(build_features(ohlcv, i))
            y.append(labels[i])

    if len(X) < 90:
        return None, None, None, None

    X = np.array(X)[:, WINNING_FEATURE_INDICES]
    y = np.array(y)

    model = RandomForestClassifier(
        n_estimators=120, max_depth=6, min_samples_leaf=8,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X, y)

    current_feats = np.array([build_features(ohlcv, len(close)-1)])[:, WINNING_FEATURE_INDICES]
    proba = model.predict_proba(current_feats)[0]
    classes = list(model.classes_)

    buy_p = proba[classes.index(1)] if 1 in classes else 0.0
    sell_p = proba[classes.index(-1)] if -1 in classes else 0.0
    hold_p = proba[classes.index(0)] if 0 in classes else 0.0

    cur_vol = float(vol[-1]) if len(vol) and vol[-1] > 0 else float(np.std(np.diff(np.log(close[-15:]))))
    return buy_p, sell_p, hold_p, cur_vol

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
    pnl_pct = (exit_price - entry) / entry * 100 if side == "BUY" else (entry - exit_price) / entry * 100

    result = {
        **trade,
        "exit_price": round(exit_price, 2),
        "exit_time": datetime.now(timezone.utc).isoformat(),
        "exit_reason": reason,
        "pnl_pct": round(pnl_pct, 3),
        "win": bool(pnl_pct > 0)
    }
    log_trade(result)
    return result

def check_and_trade():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Paper Trading Check (4 TF)")
    state = load_state()
    open_trades = state.get("open_trades", {})
    ticker = fetch_ticker(SYMBOL)
    live_price = float(ticker.get("close", ticker.get("mark_price", 0)))

    if live_price <= 0:
        print("Failed to get live price.")
        return

    print(f"BTC: ${live_price:,.2f} | Open: {list(open_trades.keys())}")

    for tf_key, cfg in PAPER_TIMEFRAMES.items():
        if tf_key in open_trades:
            trade = open_trades[tf_key]
            bars_held = trade.get("bars_held", 0) + 1
            trade["bars_held"] = bars_held
            closed = False

            if trade["side"] == "BUY":
                if live_price >= trade["take_profit"]:
                    res = close_trade(trade, live_price, "Take Profit")
                    print(f"  ✅ {tf_key} BUY TP | {res['pnl_pct']:+.2f}%")
                    closed = True
                elif live_price <= trade["stop_loss"]:
                    res = close_trade(trade, live_price, "Stop Loss")
                    print(f"  ❌ {tf_key} BUY SL | {res['pnl_pct']:+.2f}%")
                    closed = True
                elif bars_held >= cfg["max_holding_bars"]:
                    res = close_trade(trade, live_price, "Max Holding")
                    print(f"  ⏰ {tf_key} BUY Time | {res['pnl_pct']:+.2f}%")
                    closed = True
            else:
                if live_price <= trade["take_profit"]:
                    res = close_trade(trade, live_price, "Take Profit")
                    print(f"  ✅ {tf_key} SELL TP | {res['pnl_pct']:+.2f}%")
                    closed = True
                elif live_price >= trade["stop_loss"]:
                    res = close_trade(trade, live_price, "Stop Loss")
                    print(f"  ❌ {tf_key} SELL SL | {res['pnl_pct']:+.2f}%")
                    closed = True
                elif bars_held >= cfg["max_holding_bars"]:
                    res = close_trade(trade, live_price, "Max Holding")
                    print(f"  ⏰ {tf_key} SELL Time | {res['pnl_pct']:+.2f}%")
                    closed = True

            if closed:
                del open_trades[tf_key]
            else:
                open_trades[tf_key] = trade
            continue

        # Look for new strong signal
        candles = fetch_candles(SYMBOL, cfg["resolution"], cfg["days"])
        buy_p, sell_p, hold_p, cur_vol = get_balanced_proba(candles, cfg["max_holding_bars"])

        if buy_p is None:
            print(f"  {tf_key}: No data")
            continue

        print(f"  {tf_key}: BUY {buy_p:.2f} | SELL {sell_p:.2f} | HOLD {hold_p:.2f}")

        side = None
        if buy_p >= MIN_BUY_PROBA and buy_p > sell_p + MIN_EDGE:
            side = "BUY"
            tp = live_price * (1 + K_PROFIT * cur_vol)
            sl = live_price * (1 - K_STOP * cur_vol)
            conf = buy_p
        elif sell_p >= MIN_SELL_PROBA and sell_p > buy_p + MIN_EDGE:
            side = "SELL"
            tp = live_price * (1 - K_PROFIT * cur_vol)
            sl = live_price * (1 + K_STOP * cur_vol)
            conf = sell_p

        if side is None:
            print(f"  {tf_key}: No strong signal")
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
            "confidence": round(conf, 3),
            "max_holding": cfg["max_holding_bars"]
        }
        open_trades[tf_key] = trade
        print(f"  📈 NEW {side} on {tf_key} @ ${live_price:,.2f} | TP ${tp:,.2f} | SL ${sl:,.2f}")

    state["open_trades"] = open_trades
    save_state(state)
    print(f"Saved. Open trades: {list(open_trades.keys())}")

def show_performance():
    print("\n" + "="*55)
    print("PAPER TRADING PERFORMANCE (Max 4 trades/day)")
    print("="*55)

    if not os.path.exists(TRADE_LOG_FILE):
        print("No closed trades yet.")
        return

    df = pd.read_csv(TRADE_LOG_FILE)
    if df.empty:
        print("No closed trades yet.")
        return

    print(f"Total closed trades : {len(df)}")
    print(f"Win rate            : {df['win'].mean()*100:.1f}%")
    print(f"Average PnL         : {df['pnl_pct'].mean():+.2f}%")
    print(f"Total PnL           : {df['pnl_pct'].sum():+.2f}%")
    print(f"Best / Worst        : {df['pnl_pct'].max():+.2f}% / {df['pnl_pct'].min():+.2f}%")

    print("\nBy Timeframe:")
    print(df.groupby("timeframe").agg(
        trades=("pnl_pct", "count"),
        winrate=("win", lambda x: f"{x.mean()*100:.1f}%"),
        avg_pnl=("pnl_pct", "mean"),
        total_pnl=("pnl_pct", "sum")
    ).round(2))

    state = load_state()
    open_trades = state.get("open_trades", {})
    if open_trades:
        print("\nCurrently Open:")
        for tf, t in open_trades.items():
            print(f"  {tf}: {t['side']} @ ${t['entry']} → TP ${t['take_profit']} | SL ${t['stop_loss']}")
    else:
        print("\nNo open trades.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    if args.performance:
        show_performance()
    elif args.loop:
        print("Continuous mode (every 15 min) — Max 4 paper trades/day")
        while True:
            check_and_trade()
            show_performance()
            time.sleep(15 * 60)
    else:
        check_and_trade()
        show_performance()
