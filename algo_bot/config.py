"""
algo_bot/config.py
Deeper settings: cost-aware signals, all TFs paper-traded for validation.
"""

SYMBOL = "BTCUSD"
MODE = "PAPER"

# All predicted timeframes can open paper trades (validation of the call).
# Score 15m on direction AND on net-after-fees separately.
TIMEFRAMES = {
    "15m": {
        "resolution": "15m",
        "days": 30,
        "max_holding_bars": 32,
        "label": "15 Minutes",
        "trade_enabled": True,
    },
    "1h": {
        "resolution": "1h",
        "days": 90,
        "max_holding_bars": 24,
        "label": "1 Hour",
        "trade_enabled": True,
    },
    "4h": {
        "resolution": "4h",
        "days": 180,
        "max_holding_bars": 18,
        "label": "4 Hours",
        "trade_enabled": True,
    },
    "1D": {
        "resolution": "1d",
        "days": 300,
        "max_holding_bars": 15,
        "label": "Daily",
        "trade_enabled": True,
    },
}

K_PROFIT = 2.0
K_STOP = 2.0
MIN_BUY_PROBA = 0.52
MIN_SELL_PROBA = 0.52
MIN_EDGE = 0.12

ROUND_TRIP_COST_PCT = 0.00118
MIN_LABEL_MOVE_PCT = 0.00150

ACCOUNT_INR = 15000.0
LEVERAGE = 10.0
USD_INR = 85.0
RISK_FRACTION = 0.01

TRADE_LOG_FILE = "algo_bot/trades.csv"
STATE_FILE = "algo_bot/state.json"
LOG_FILE = "algo_bot/bot.log"

CHECK_INTERVAL_MINUTES = 15
DELTA_BASE = "https://api.india.delta.exchange"
