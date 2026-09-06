"""
algo_bot/bot.py
Standalone Algo Trading Bot
"""

import os
import sys
import json
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo_bot.config import (
    SYMBOL, TIMEFRAMES, TRADE_LOG_FILE, STATE_FILE,
    CHECK_INTERVAL_MINUTES, MODE, ROUND_TRIP_COST_PCT
)
from algo_bot.model_engine import get_signal
from algo_bot.notifier import (
    is_configured, notify_trade_opened, notify_trade_closed
)


def ensure_dirs():
    os.makedirs("algo_bot", exist_ok=True)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"open_trades": {}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def log_trade(trade: dict):
    df = pd.DataFrame([trade])
    if os.path.exists(TRADE_LOG_FILE):
        df.to_csv(TRADE_LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(TRADE_LOG_FILE, index=False)


def close_trade(trade, exit_price, reason):
    entry = trade["entry"]
    side = trade["side"]
    gross = (exit_price - entry) / entry * 100 if side == "BUY" else (entry - exit_price) / entry * 100
    net = gross - (ROUND_TRIP_COST_PCT * 100)
    result = {
        **trade,
        "exit_price": round(exit_price, 2),
        "exit_time": datetime.now(timezone.utc).isoformat(),
        "exit_reason": reason,
        "pnl_pct": round(gross, 3),
        "pnl_pct_net": round(net, 3),
        "win": bool(net > 0)
    }
    log_trade(result)
    notify_trade_closed(result)
    return result


def run_once():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Algo Bot Check | Mode: {MODE}")
    state = load_state()
    open_trades = state.get("open_trades", {})

    for tf_key, cfg in TIMEFRAMES.items():
        print(f"\n> {cfg['label']} ({tf_key})")

        if tf_key in open_trades:
            trade = open_trades[tf_key]
            signal_data = get_signal(SYMBOL, cfg["resolution"], cfg["days"], cfg["max_holding_bars"])
            if signal_data is None:
                print("  Could not fetch price")
                continue

            live_price = signal_data["price"]
            bars_held = trade.get("bars_held", 0) + 1
            trade["bars_held"] = bars_held
            closed = False

            if trade["side"] == "BUY":
                if live_price >= trade["take_profit"]:
                    res = close_trade(trade, live_price, "Take Profit")
                    print(f"  BUY TP net {res.get('pnl_pct_net', res['pnl_pct']):+.2f}%")
                    closed = True
                elif live_price <= trade["stop_loss"]:
                    res = close_trade(trade, live_price, "Stop Loss")
                    print(f"  BUY SL net {res.get('pnl_pct_net', res['pnl_pct']):+.2f}%")
                    closed = True
                elif bars_held >= cfg["max_holding_bars"]:
                    res = close_trade(trade, live_price, "Max Holding")
                    print(f"  BUY Time net {res.get('pnl_pct_net', res['pnl_pct']):+.2f}%")
                    closed = True
            else:
                if live_price <= trade["take_profit"]:
                    res = close_trade(trade, live_price, "Take Profit")
                    print(f"  SELL TP net {res.get('pnl_pct_net', res['pnl_pct']):+.2f}%")
                    closed = True
                elif live_price >= trade["stop_loss"]:
                    res = close_trade(trade, live_price, "Stop Loss")
                    print(f"  SELL SL net {res.get('pnl_pct_net', res['pnl_pct']):+.2f}%")
                    closed = True
                elif bars_held >= cfg["max_holding_bars"]:
                    res = close_trade(trade, live_price, "Max Holding")
                    print(f"  SELL Time net {res.get('pnl_pct_net', res['pnl_pct']):+.2f}%")
                    closed = True

            if closed:
                del open_trades[tf_key]
            else:
                open_trades[tf_key] = trade
                print(f"  Open: {trade['side']} @ ${trade['entry']} | Bars {bars_held}")
            continue

        if not cfg.get("trade_enabled", True):
            print("  Watch only — no new paper trades")
            continue

        signal = get_signal(SYMBOL, cfg["resolution"], cfg["days"], cfg["max_holding_bars"])
        if signal is None:
            print("  Insufficient data")
            continue

        print(
            f"  BUY {signal['buy_proba']:.2f} | SELL {signal['sell_proba']:.2f} | "
            f"HOLD {signal['hold_proba']:.2f} | regime {signal.get('regime')}"
        )

        if signal["side"] is None:
            print("  No strong cost-aware + regime signal")
            continue

        trade = {
            "id": f"{tf_key}_{int(time.time())}",
            "symbol": SYMBOL,
            "timeframe": tf_key,
            "side": signal["side"],
            "entry": round(signal["price"], 2),
            "take_profit": round(signal["take_profit"], 2),
            "stop_loss": round(signal["stop_loss"], 2),
            "qty_btc": round(signal.get("qty_btc", 0), 6),
            "regime": signal.get("regime"),
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "bars_held": 0,
            "confidence": round(signal["confidence"], 3),
            "max_holding": cfg["max_holding_bars"],
            "mode": MODE
        }
        open_trades[tf_key] = trade
        notify_trade_opened(trade)
        print(f"  NEW {signal['side']} @ ${signal['price']:,.2f} qty {trade['qty_btc']}")

    state["open_trades"] = open_trades
    save_state(state)
    print(f"\nOpen trades: {list(open_trades.keys())}")


def show_status():
    state = load_state()
    open_trades = state.get("open_trades", {})
    print(f"Mode {MODE} | Open {len(open_trades)}")
    for tf, t in open_trades.items():
        print(f"  {tf} {t['side']} @ ${t['entry']}")
    if os.path.exists(TRADE_LOG_FILE):
        df = pd.read_csv(TRADE_LOG_FILE)
        if not df.empty:
            col = "pnl_pct_net" if "pnl_pct_net" in df.columns else "pnl_pct"
            print(f"Closed {len(df)} | win {df['win'].mean()*100:.1f}% | {col} sum {df[col].sum():+.3f}")


def show_performance():
    if not os.path.exists(TRADE_LOG_FILE):
        print("No closed trades yet.")
        return
    df = pd.read_csv(TRADE_LOG_FILE)
    if df.empty:
        print("No closed trades yet.")
        return
    col = "pnl_pct_net" if "pnl_pct_net" in df.columns else "pnl_pct"
    print(df.groupby("timeframe").agg(
        trades=("pnl_pct", "count"),
        winrate=("win", lambda x: f"{x.mean()*100:.1f}%"),
        total=(col, "sum")
    ))


def main():
    ensure_dirs()
    if len(sys.argv) < 2:
        print("python -m algo_bot.bot once|start|status|performance")
        return
    cmd = sys.argv[1].lower()
    if cmd == "start":
        while True:
            try:
                run_once()
                show_status()
                time.sleep(CHECK_INTERVAL_MINUTES * 60)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(60)
    elif cmd == "once":
        run_once()
        show_status()
    elif cmd == "status":
        show_status()
    elif cmd == "performance":
        show_performance()
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
