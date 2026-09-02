"""
app.py - Leverage Signal Engine
Clear Buy/Sell prediction + Entry / Take Profit / Stop Loss + Holding time
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

from ml_features import build_features, FEATURE_NAMES, WINNING_FEATURES
from triple_barrier import triple_barrier_labels
from signal_engine import sma, ema, rsi, macd
from features import atr
from forecasting import ensemble_forecast

st.set_page_config(page_title="Leverage Signal", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stButton > button { border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

DELTA_BASE = "https://api.india.delta.exchange"
WINNING_FEATURE_INDICES = [FEATURE_NAMES.index(f) for f in WINNING_FEATURES]
K_PROFIT, K_STOP = 2.0, 2.0

TOP_PROFILE = {"rsi_center": 64.0, "rsi_range": 8.0, "macd_hist_min": 10.0, "price_vs_ema20_min": 0.3}
BOTTOM_PROFILE = {"rsi_center": 38.0, "rsi_range": 8.0, "macd_hist_max": -10.0, "price_vs_ema20_max": -0.3}

TIMEFRAMES = {
    "15m": {"resolution": "15m", "days": 30,  "max_holding": 32, "label": "15m", "hold_text": "~8 hours"},
    "30m": {"resolution": "30m", "days": 45,  "max_holding": 28, "label": "30m", "hold_text": "~14 hours"},
    "1h":  {"resolution": "1h",  "days": 90,  "max_holding": 24, "label": "1h",  "hold_text": "~24 hours"},
    "4h":  {"resolution": "4h",  "days": 180, "max_holding": 18, "label": "4h",  "hold_text": "~3 days"},
    "1D":  {"resolution": "1d",  "days": 300, "max_holding": 15, "label": "Daily", "hold_text": "~15 days"},
}

ASSETS = ["BTCUSD", "ETHUSD", "SOLUSD"]

if "tf" not in st.session_state:
    st.session_state.tf = "1h"
if "symbol" not in st.session_state:
    st.session_state.symbol = "BTCUSD"

# -------------------- DATA --------------------
@st.cache_data(ttl=5, show_spinner=False)
def fetch_ticker(symbol):
    try:
        return requests.get(f"{DELTA_BASE}/v2/tickers/{symbol}", timeout=4).json().get("result", {})
    except Exception:
        return {}

@st.cache_data(ttl=40, show_spinner=False)
def fetch_candles(symbol, resolution, days):
    end = int(time.time())
    start = end - days * 86400
    try:
        r = requests.get(f"{DELTA_BASE}/v2/history/candles",
                         params={"resolution": resolution, "symbol": symbol, "start": start, "end": end},
                         timeout=12).json()
        return list(reversed(r.get("result", [])))
    except Exception:
        return []

@st.cache_data(ttl=90, show_spinner=False)
def run_model(close_t, high_t, low_t, vol_t, max_holding):
    close = np.array(close_t, dtype=float)
    high = np.array(high_t, dtype=float)
    low = np.array(low_t, dtype=float)
    volume = np.array(vol_t, dtype=float)
    ohlcv = {"open": close, "high": high, "low": low, "close": close, "volume": volume}

    labels, valid, vol, _, _ = triple_barrier_labels(close, K_PROFIT, K_STOP, max_holding)
    X, y = [], []
    for i in range(60, len(close)-1):
        if valid[i] and labels[i] != 0:
            X.append(build_features(ohlcv, i))
            y.append(1 if labels[i] == 1 else 0)

    if len(X) < 80:
        return None, None

    X = np.array(X)[:, WINNING_FEATURE_INDICES]
    y = np.array(y)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    proba = float(model.predict_proba(np.array([build_features(ohlcv, len(close)-1)])[:, WINNING_FEATURE_INDICES])[0, 1])
    cur_vol = float(vol[-1]) if len(vol) and vol[-1] > 0 else float(np.std(np.diff(np.log(close[-20:]))))
    return proba, cur_vol

def score_pattern(inds, profile, is_top=True):
    score = 0.0
    rsi_diff = abs(inds["rsi"] - profile["rsi_center"])
    score += max(0, 100 - (rsi_diff / profile["rsi_range"]) * 50) * 0.45
    macd_h = inds["macd_hist"]
    if is_top:
        macd_score = min(100, 50 + macd_h/5) if macd_h >= profile["macd_hist_min"] else max(0, 50 + macd_h)
    else:
        macd_score = min(100, 50 + abs(macd_h)/5) if macd_h <= profile["macd_hist_max"] else max(0, 50 - macd_h)
    score += macd_score * 0.35
    pve = inds["price_vs_ema20"]
    if is_top:
        ema_score = min(100, 60 + pve*10) if pve >= profile.get("price_vs_ema20_min", 0) else max(0, 50 + pve*20)
    else:
        ema_score = min(100, 60 + abs(pve)*10) if pve <= profile.get("price_vs_ema20_max", 0) else max(0, 50 - pve*20)
    score += ema_score * 0.20
    return round(score, 1)

# -------------------- UI HEADER --------------------
st.markdown("### Leverage Signal")
st.caption(f"live · {datetime.now().strftime('%H:%M:%S')}")

# Asset selector
cols = st.columns(len(ASSETS))
for col, sym in zip(cols, ASSETS):
    with col:
        if st.button(sym.replace("USD", ""), key=f"a_{sym}", use_container_width=True):
            st.session_state.symbol = sym
            st.rerun()

symbol = st.session_state.symbol

# Timeframe pills
st.write("")
tf_cols = st.columns(len(TIMEFRAMES))
for col, (key, cfg) in zip(tf_cols, TIMEFRAMES.items()):
    with col:
        if st.button(cfg["label"], key=f"t_{key}", use_container_width=True):
            st.session_state.tf = key
            st.rerun()

tf = st.session_state.tf
tf_cfg = TIMEFRAMES[tf]

# -------------------- LOAD DATA --------------------
ticker = fetch_ticker(symbol)
candles = fetch_candles(symbol, tf_cfg["resolution"], tf_cfg["days"])

if not candles or len(candles) < 60:
    st.warning("Not enough data for this timeframe.")
    st.stop()

close = np.array([c["close"] for c in candles], dtype=float)
high = np.array([c["high"] for c in candles], dtype=float)
low = np.array([c["low"] for c in candles], dtype=float)
volume = np.array([c["volume"] for c in candles], dtype=float)

live_price = float(ticker.get("close", ticker.get("mark_price", close[-1])))
chg = float(ticker.get("price_change_24h", 0.0)) * 100

# Indicators
ema20 = ema(close, 20) if len(close) >= 20 else close[-1]
sma200 = sma(close, 200) if len(close) >= 200 else sma(close, min(len(close), 80))
rsi_val = rsi(close, 14)
macd_line, macd_sig = macd(close)
macd_hist = macd_line - macd_sig
pve = (close[-1] - ema20) / ema20 * 100
atr_val = atr(high, low, close)

inds = {"rsi": rsi_val, "macd_hist": macd_hist, "price_vs_ema20": pve}
top_score = score_pattern(inds, TOP_PROFILE, True)
bottom_score = score_pattern(inds, BOTTOM_PROFILE, False)

proba, cur_vol = run_model(tuple(close), tuple(high), tuple(low), tuple(volume), tf_cfg["max_holding"])

# -------------------- PREDICTION LOGIC --------------------
if proba is None:
    signal = "NO DATA"
    signal_color = "gray"
elif proba >= 0.62:
    signal = "BUY"
    signal_color = "green"
elif proba <= 0.38:
    signal = "SELL"
    signal_color = "red"
else:
    signal = "HOLD / NEUTRAL"
    signal_color = "orange"

# Entry / Exit / Stop levels
entry = live_price
if signal == "BUY":
    take_profit = live_price * (1 + K_PROFIT * (cur_vol or 0.01))
    stop_loss = live_price * (1 - K_STOP * (cur_vol or 0.01))
elif signal == "SELL":
    take_profit = live_price * (1 - K_PROFIT * (cur_vol or 0.01))
    stop_loss = live_price * (1 + K_STOP * (cur_vol or 0.01))
else:
    take_profit = live_price * (1 + K_PROFIT * (cur_vol or 0.01))
    stop_loss = live_price * (1 - K_STOP * (cur_vol or 0.01))

holding_time = tf_cfg["hold_text"]

# -------------------- DISPLAY --------------------
st.markdown(f"## ${live_price:,.2f}")
st.markdown(f"{'+' if chg >= 0 else ''}{chg:.2f}% (24h)")

st.line_chart(pd.DataFrame({"Price": close[-90:]}), height=200)

st.markdown("---")

# ========== MAIN PREDICTION BLOCK ==========
st.subheader("Prediction")

# Big signal
if signal == "BUY":
    st.success(f"### {signal}")
elif signal == "SELL":
    st.error(f"### {signal}")
else:
    st.warning(f"### {signal}")

if proba is not None:
    st.write(f"Model confidence: **{proba*100:.1f}%** chance of upward barrier being hit first")

# Key trade levels
st.markdown("#### Trade Setup")
col1, col2, col3 = st.columns(3)
col1.metric("Entry", f"${entry:,.2f}")
col2.metric("Take Profit", f"${take_profit:,.2f}")
col3.metric("Stop Loss", f"${stop_loss:,.2f}")

col4, col5 = st.columns(2)
col4.metric("Holding Time", holding_time)
col5.metric("Risk : Reward", f"1 : {K_PROFIT/K_STOP:.1f}")

st.caption(f"Based on Triple Barrier (K={K_PROFIT}) + RandomForest on {tf_cfg['label']} timeframe")

st.markdown("---")

# Pattern detector
st.subheader("Pattern Match")
p1, p2 = st.columns(2)
p1.metric("Top Match", f"{top_score:.0f}%")
p2.metric("Bottom Match", f"{bottom_score:.0f}%")

st.markdown("---")

# Indicators
st.subheader("Indicators")
i1, i2, i3, i4 = st.columns(4)
i1.metric("RSI (14)", f"{rsi_val:.1f}")
i2.metric("MACD Hist", f"{macd_hist:.1f}")
i3.metric("EMA 20", f"${ema20:,.0f}")
i4.metric("SMA 200", f"${sma200:,.0f}")

st.caption(f"Price vs EMA20: {pve:+.2f}% · Regime: {'Bullish' if live_price > sma200 else 'Bearish'}")

# Auto refresh
time.sleep(12)
st.rerun()
