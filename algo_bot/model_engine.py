"""
algo_bot/model_engine.py
Balanced 3-class engine with cost-aware labels and trend regime filter.
"""

import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier

from ml_features import build_features, FEATURE_NAMES, WINNING_FEATURES
from triple_barrier import triple_barrier_labels, apply_cost_filter
from algo_bot.config import (
    DELTA_BASE, K_PROFIT, K_STOP,
    MIN_BUY_PROBA, MIN_SELL_PROBA, MIN_EDGE,
    MIN_LABEL_MOVE_PCT, ACCOUNT_INR, LEVERAGE, USD_INR, RISK_FRACTION
)

WINNING_FEATURE_INDICES = [FEATURE_NAMES.index(f) for f in WINNING_FEATURES]


def fetch_candles(symbol: str, resolution: str, days: int):
    end = int(__import__("time").time())
    start = end - days * 86400
    try:
        r = requests.get(
            f"{DELTA_BASE}/v2/history/candles",
            params={"resolution": resolution, "symbol": symbol, "start": start, "end": end},
            timeout=14
        ).json()
        return list(reversed(r.get("result", [])))
    except Exception as e:
        print(f"[Model] Candle fetch error: {e}")
        return []


def fetch_ticker(symbol: str):
    try:
        return requests.get(f"{DELTA_BASE}/v2/tickers/{symbol}", timeout=5).json().get("result", {})
    except Exception:
        return {}


def _sma(arr, n):
    if len(arr) < n:
        return float(arr[-1]) if len(arr) else 0.0
    return float(np.mean(arr[-n:]))


def position_qty_btc(entry, stop):
    """Size so that a stop hit loses about RISK_FRACTION of ACCOUNT_INR."""
    risk_inr = ACCOUNT_INR * RISK_FRACTION
    stop_dist = abs(entry - stop)
    if stop_dist <= 0 or entry <= 0:
        return 0.0
    qty = (risk_inr / USD_INR) / stop_dist
    max_qty = (ACCOUNT_INR * LEVERAGE / USD_INR) / entry
    return float(min(qty, max_qty))


def get_signal(symbol: str, resolution: str, days: int, max_holding: int):
    candles = fetch_candles(symbol, resolution, days)
    if len(candles) < 80:
        return None

    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)
    volume = np.array([c["volume"] for c in candles], dtype=float)
    ohlcv = {"open": close, "high": high, "low": low, "close": close, "volume": volume}

    labels, valid, vol, exit_prices, _ = triple_barrier_labels(close, K_PROFIT, K_STOP, max_holding)
    labels = apply_cost_filter(labels, close, exit_prices, valid, MIN_LABEL_MOVE_PCT)

    X, y = [], []
    for i in range(55, len(close) - 1):
        if valid[i]:
            X.append(build_features(ohlcv, i))
            y.append(labels[i])

    if len(X) < 90:
        return None

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

    buy_p = float(proba[classes.index(1)]) if 1 in classes else 0.0
    sell_p = float(proba[classes.index(-1)]) if -1 in classes else 0.0
    hold_p = float(proba[classes.index(0)]) if 0 in classes else 0.0

    cur_vol = float(vol[-1]) if len(vol) and vol[-1] > 0 else float(np.std(np.diff(np.log(close[-15:]))))
    sma200 = _sma(close, min(200, len(close)))
    price_now = float(close[-1])
    regime = "UP" if price_now >= sma200 else "DOWN"

    ticker = fetch_ticker(symbol)
    live_price = float(ticker.get("close", ticker.get("mark_price", close[-1])))

    side = None
    confidence = 0.0
    if buy_p >= MIN_BUY_PROBA and buy_p > sell_p + MIN_EDGE and regime == "UP":
        side = "BUY"
        confidence = buy_p
    elif sell_p >= MIN_SELL_PROBA and sell_p > buy_p + MIN_EDGE and regime == "DOWN":
        side = "SELL"
        confidence = sell_p

    base = {
        "side": side,
        "buy_proba": buy_p,
        "sell_proba": sell_p,
        "hold_proba": hold_p,
        "price": live_price,
        "volatility": cur_vol,
        "regime": regime,
        "sma200": sma200,
    }
    if side is None:
        return base

    if side == "BUY":
        tp = live_price * (1 + K_PROFIT * cur_vol)
        sl = live_price * (1 - K_STOP * cur_vol)
    else:
        tp = live_price * (1 - K_PROFIT * cur_vol)
        sl = live_price * (1 + K_STOP * cur_vol)

    qty = position_qty_btc(live_price, sl)
    base.update({
        "confidence": confidence,
        "take_profit": tp,
        "stop_loss": sl,
        "qty_btc": qty,
    })
    return base
