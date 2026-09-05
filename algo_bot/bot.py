"""
algo_bot/bot.py
Standalone Algo Trading Bot

Commands:
  python -m algo_bot.bot start         → Run continuous loop
  python -m algo_bot.bot once          → Run one check
  python -m algo_bot.bot status        → Show open trades + performance
  python -m algo_bot.bot performance   → Detailed performance report
"""

import os
import sys
import json
import time
from datetime import datetime, timezone

import pandas as pd

# Allow running both as module and as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo_bot.config import (
    SYMBOL, TIMEFRAMES, TRADE_LOG_FILE, STATE_FILE,
    CHECK_INTERVAL_MINUTES, MODE
)
from algo_bot.model_engine import get_signal


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


def run_once():
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Algo Bot Check | Mode: {MODE}")
    print(f"{'='*60}")

    state = load_state()
    open_trades = state.get("open_trades", {})

    for tf_key, cfg in TIMEFRAMES.items():
        print(f"\n▶ {cfg['label']} ({tf_key})")

        # ----- Manage open trade -----
        if tf_key in open_trades:
            trade = open_trades[tf_key]
            signal_data = get_signal(SYMBOL, cfg["resolution"], cfg["days"], cfg["max_holding_bars"])
            if signal_data is None:
                print("  Could not fetch price, skipping management")
                continue

            live_price = signal_data["price"]
            bars_held = trade.get("bars_held", 0) + 1
            trade["bars_held"] = bars_held

            closed = False
            if trade["side"] == "BUY":
                if live_price >= trade["take_profit"]:
                    res = close_trade(trade, live_price, "Take Profit")
                    print(f"  ✅ BUY closed at TP | PnL: {res['pnl_pct']:+.2f}%")
                    closed = True
                elif live_price <= trade["stop_loss"]:
                    res = close_trade(trade, live_price, "Stop Loss")
                    print(f"  ❌ BUY closed at SL | PnL: {res['pnl_pct']:+.2f}%")
                    closed = True
                elif bars_held >= cfg["max_holding_bars"]:
                    res = close_trade(trade, live_price, "Max Holding")
                    print(f"  ⏰ BUY time exit | PnL: {res['pnl_pct']:+.2f}%")
                    closed = True
            else:
                if live_price <= trade["take_profit"]:
                    res = close_trade(trade, live_price, "Take Profit")
                    print(f"  ✅ SELL closed at TP | PnL: {res['pnl_pct']:+.2f}%")
                    closed = True
                elif live_price >= trade["stop_loss"]:
                    res = close_trade(trade, live_price, "Stop Loss")
                    print(f"  ❌ SELL closed at SL | PnL: {res['pnl_pct']:+.2f}%")
                    closed = True
                elif bars_held >= cfg["max_holding_bars"]:
                    res = close_trade(trade, live_price, "Max Holding")
                    print(f"  ⏰ SELL time exit | PnL: {res['pnl_pct']:+.2f}%")
                    closed = True

            if closed:
                del open_trades[tf_key]
            else:
                open_trades[tf_key] = trade
                print(f"  Still open: {trade['side']} @ ${trade['entry']} | Bars: {bars_held}")
            continue

        # ----- Look for new signal -----
        signal = get_signal(SYMBOL, cfg["resolution"], cfg["days"], cfg["max_holding_bars"])

        if signal is None:
            print("  Insufficient data")
            continue

        print(f"  BUY: {signal['buy_proba']:.2f} | SELL: {signal['sell_proba']:.2f} | HOLD: {signal['hold_proba']:.2f}")

        if signal["side"] is None:
            print("  No strong signal")
            continue

        # Open new paper trade
        trade = {
            "id": f"{tf_key}_{int(time.time())}",
            "symbol": SYMBOL,
            "timeframe": tf_key,
            "side": signal["side"],
            "entry": round(signal["price"], 2),
            "take_profit": round(signal["take_profit"], 2),
            "stop_loss": round(signal["stop_loss"], 2),
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "bars_held": 0,
            "confidence": round(signal["confidence"], 3),
            "max_holding": cfg["max_holding_bars"],
            "mode": MODE
        }
        open_trades[tf_key] = trade
        print(f"  📈 NEW {signal['side']} @ ${signal['price']:,.2f}")
        print(f"     TP: ${signal['take_profit']:,.2f} | SL: ${signal['stop_loss']:,.2f} | Conf: {signal['confidence']:.1%}")

    state["open_trades"] = open_trades
    save_state(state)
    print(f"\nOpen trades now: {list(open_trades.keys())}")


def show_status():
    state = load_state()
    open_trades = state.get("open_trades", {})

    print(f"\n{'='*50}")
    print("ALGO BOT STATUS")
    print(f"{'='*50}")
    print(f"Mode          : {MODE}")
    print(f"Symbol        : {SYMBOL}")
    print(f"Open Trades   : {len(open_trades)}")

    if open_trades:
        print("\nCurrently Open:")
        for tf, t in open_trades.items():
            print(f"  {tf:5} | {t['side']:4} | Entry ${t['entry']:,.0f} | TP ${t['take_profit']:,.0f} | SL ${t['stop_loss']:,.0f} | Conf {t['confidence']:.0%}")
    else:
        print("\nNo open trades.")

    if os.path.exists(TRADE_LOG_FILE):
        df = pd.read_csv(TRADE_LOG_FILE)
        if not df.empty:
            print(f"\nClosed Trades : {len(df)}")
            print(f"Win Rate      : {df['win'].mean()*100:.1f}%")
            print(f"Avg PnL       : {df['pnl_pct'].mean():+.2f}%")
            print(f"Total PnL     : {df['pnl_pct'].sum():+.2f}%")


def show_performance():
    print(f"\n{'='*55}")
    print("PERFORMANCE REPORT")
    print(f"{'='*55}")

    if not os.path.exists(TRADE_LOG_FILE):
        print("No closed trades yet.")
        return

    df = pd.read_csv(TRADE_LOG_FILE)
    if df.empty:
        print("No closed trades yet.")
        return

    print(f"Total Trades  : {len(df)}")
    print(f"Win Rate      : {df['win'].mean()*100:.1f}%")
    print(f"Average PnL   : {df['pnl_pct'].mean():+.2f}%")
    print(f"Total PnL     : {df['pnl_pct'].sum():+.2f}%")
    print(f"Best Trade    : {df['pnl_pct'].max():+.2f}%")
    print(f"Worst Trade   : {df['pnl_pct'].min():+.2f}%")

    print("\nBy Timeframe:")
    summary = df.groupby("timeframe").agg(
        trades=("pnl_pct", "count"),
        winrate=("win", lambda x: f"{x.mean()*100:.1f}%"),
        avg_pnl=("pnl_pct", "mean"),
        total_pnl=("pnl_pct", "sum")
    ).round(2)
    print(summary)


def main():
    ensure_dirs()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m algo_bot.bot start         # Continuous mode")
        print("  python -m algo_bot.bot once          # Single check")
        print("  python -m algo_bot.bot status        # Current status")
        print("  python -m algo_bot.bot performance   # Full report")
        return

    cmd = sys.argv[1].lower()

    if cmd == "start":
        print(f"Starting Algo Bot in continuous mode (every {CHECK_INTERVAL_MINUTES} min)")
        print("Press Ctrl+C to stop.\n")
        while True:
            try:
                run_once()
                show_status()
                print(f"\nSleeping {CHECK_INTERVAL_MINUTES} minutes...\n")
                time.sleep(CHECK_INTERVAL_MINUTES * 60)
            except KeyboardInterrupt:
                print("\nBot stopped by user.")
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
