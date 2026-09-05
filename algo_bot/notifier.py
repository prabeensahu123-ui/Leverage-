"""
algo_bot/notifier.py
Telegram notification support

How to enable:
1. Create a bot with @BotFather on Telegram → get TOKEN
2. Get your CHAT_ID (message the bot, then visit https://api.telegram.org/bot<TOKEN>/getUpdates)
3. Set environment variables:
   export TELEGRAM_TOKEN="your_token"
   export TELEGRAM_CHAT_ID="your_chat_id"
"""

import os
import requests
from datetime import datetime

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def is_configured() -> bool:
    return bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)


def send_message(text: str) -> bool:
    """Send a Telegram message. Returns True if successful."""
    if not is_configured():
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"[Notifier] Telegram error: {e}")
        return False


def notify_trade_opened(trade: dict):
    msg = (
        f"📈 <b>NEW PAPER TRADE</b>\n"
        f"Timeframe: <b>{trade.get('timeframe')}</b>\n"
        f"Side: <b>{trade.get('side')}</b>\n"
        f"Entry: ${trade.get('entry'):,.2f}\n"
        f"TP: ${trade.get('take_profit'):,.2f}\n"
        f"SL: ${trade.get('stop_loss'):,.2f}\n"
        f"Confidence: {trade.get('confidence', 0):.0%}\n"
        f"Time: {datetime.now().strftime('%H:%M:%S')}"
    )
    send_message(msg)


def notify_trade_closed(result: dict):
    emoji = "✅" if result.get("win") else "❌"
    msg = (
        f"{emoji} <b>TRADE CLOSED</b>\n"
        f"Timeframe: <b>{result.get('timeframe')}</b>\n"
        f"Side: {result.get('side')}\n"
        f"Entry: ${result.get('entry'):,.2f} → Exit: ${result.get('exit_price'):,.2f}\n"
        f"PnL: <b>{result.get('pnl_pct'):+.2f}%</b>\n"
        f"Reason: {result.get('exit_reason')}\n"
        f"Time: {datetime.now().strftime('%H:%M:%S')}"
    )
    send_message(msg)


def notify_status(open_count: int, message: str = ""):
    msg = f"🤖 Algo Bot Status\nOpen trades: {open_count}\n{message}"
    send_message(msg)
