"""
algo_bot/config.py
Central configuration for the Algo Trading Bot
"""

# -------------------- GENERAL --------------------
SYMBOL = "BTCUSD"
MODE = "PAPER"          # PAPER or LIVE (LIVE not enabled yet)

# -------------------- TIMEFRAMES --------------------
# Max 1 open trade per timeframe = Max 4 trades
TIMEFRAMES = {
    "15m": {
        "resolution": "15m",
        "days": 30,
        "max_holding_bars": 32,
        "label": "15 Minutes"
    },
    "1h": {
        "resolution": "1h",
        "days": 90,
        "max_holding_bars": 24,
        "label": "1 Hour"
    },
    "4h": {
        "resolution": "4h",
        "days": 180,
        "max_holding_bars": 18,
        "label": "4 Hours"
    },
    "1D": {
        "resolution": "1d",
        "days": 300,
        "max_holding_bars": 15,
        "label": "Daily"
    },
}

# -------------------- MODEL & SIGNAL --------------------
K_PROFIT = 2.0
K_STOP = 2.0
MIN_BUY_PROBA = 0.45
MIN_SELL_PROBA = 0.45
MIN_EDGE = 0.10          # BUY must be higher than SELL by this margin

# -------------------- RISK --------------------
# Currently fixed size for paper trading
POSITION_SIZE = 1        # 1 unit per trade

# -------------------- FILES --------------------
TRADE_LOG_FILE = "algo_bot/trades.csv"
STATE_FILE = "algo_bot/state.json"
LOG_FILE = "algo_bot/bot.log"

# -------------------- LOOP --------------------
CHECK_INTERVAL_MINUTES = 15   # How often the bot checks for signals

# -------------------- API --------------------
DELTA_BASE = "https://api.india.delta.exchange"
