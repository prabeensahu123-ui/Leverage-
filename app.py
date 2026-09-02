"""
app.py - Leverage Signal Engine
Multi-timeframe prediction table with color coding
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

from ml_features import build_features, FEATURE_NAMES, WINNING_FEATURES
from triple_barrier import triple_barrier_labels
from signal_engine import sma, ema, rsi, macd
from features import atr

st.set_page_config(page_title="Leverage Signal", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stButton > button { border-radius: 18px; font-weight: 600; font-size: 0.85rem; }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

DELTA_BASE = "https://api.india.delta.exchange"
WINNING_FEATURE_INDICES = [FEATURE_NAMES.index(f) for f in WINNING_FEATURES]
K_PROFIT, K_STOP = 2.0, 2.0

TIMEFRAMES = {
    "15m": {"resolution": "15m", "days": 25,  "max_holding": 32, "label": "15m",  "hold": "~8h"},
    "30m": {"resolution": "30m", "days": 40,  "max_holding": 28, "label": "30m",  "hold": "~14h"},
    "1h":  {"resolution": "1h",  "days": 80,  "max_holding": 24, "label": "1h",   "hold": "~24h"},
    "4h":  {"resolution": "4h",  "days": 160, "max_holding": 18, "label": "4h",   "hold": "~3d"},
    "1D":  {"resolution": "1d",  "days": 300, "max_holding": 15, "label": "Daily","hold": "~15d"},
}

ASSETS = ["BTCUSD", "ETHUSD", "SOLUSD"]

if "symbol" not in st.session_state:
    st.session_state.symbol = "BTCUSD"

@st.cache_data(ttl=6, show_spinner=False)
def fetch_ticker(symbol):
    try:
        return requests.get(f"{DELTA_BASE}/v2/tickers/{symbol}", timeout=4).json().get("result", {})
    except Exception:
        return {}

@st.cache_data(ttl=50, show_spinner=False)
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
    except Exception:
        return []

@st.cache_data(ttl=100, show_spinner=False)
def run_model(close_t, high_t, low_t, vol_t, max_holding):
    close = np.array(close_t, dtype=float)
    high = np.array(high_t, dtype=float)
    low = np.array(low_t, dtype=float)
    volume = np.array(vol_t, dtype=float)
    ohlcv = {"open": close, "high": high, "low": low, "close": close, "volume": volume}

    labels, valid, vol, _, _ = triple_barrier_labels(close, K_PROFIT, K_STOP, max_holding)
    X, y = [], []
    for i in range(55, len(close) - 1):
        if valid[i] and labels[i] != 0:
            X.append(build_features(ohlcv, i))
            y.append(1 if labels[i] == 1 else 0)

    if len(X) < 70:
        return None, None

    X = np.array(X)[:, WINNING_FEATURE_INDICES]
    y = np.array(y)
    model = RandomForestClassifier(n_estimators=80, max_depth=5, min_samples_leaf=8, random_state=42, n_jobs=-1)
    model.fit(X, y)
    proba = float(model.predict_proba(np.array([build_features(ohlcv, len(close)-1)])[:, WINNING_FEATURE_INDICES])[0, 1])
    cur_vol = float(vol[-1]) if len(vol) and vol[-1] > 0 else float(np.std(np.diff(np.log(close[-15:]))))
    return proba, cur_vol

def get_signal(proba):
    if proba is None:
        return "N/A", 0.0
    if proba >= 0.62:
        return "BUY", proba * 100
    elif proba <= 0.38:
        return "SELL", (1 - proba) * 100
    else:
        return "HOLD", max(proba, 1 - proba) * 100

def style_table(df):
    def color_signal(val):
        if val == "BUY":
            return "background-color: #16a34a; color: white; font-weight: bold; text-align: center"
        if val == "SELL":
            return "background-color: #dc2626; color: white; font-weight: bold; text-align: center"
        if val == "HOLD":
            return "background-color: #ea580c; color: white; font-weight: bold; text-align: center"
        return "text-align: center"

    def color_conf(val):
        try:
            v = float(val)
        except Exception:
            return "text-align: center"
        if v >= 70:
            return "background-color: #16a34a; color: white; text-align: center"
        if v >= 60:
            return "background-color: #4ade80; color: black; text-align: center"
        if v >= 50:
            return "background-color: #fdba74; color: black; text-align: center"
        return "background-color: #fca5a5; color: black; text-align: center"

    return (df.style
            .applymap(color_signal, subset=["Signal"])
            .applymap(color_conf, subset=["Confidence"])
            .set_properties(**{"text-align": "center"}))

# -------------------- HEADER --------------------
st.markdown("### Leverage Signal")
st.caption(f"Multi-Timeframe View · {datetime.now().strftime('%H:%M:%S')}")

cols = st.columns(len(ASSETS))
for col, sym in zip(cols, ASSETS):
    with col:
        if st.button(sym.replace("USD", ""), key=f"a_{sym}", use_container_width=True):
            st.session_state.symbol = sym
            st.rerun()

symbol = st.session_state.symbol
ticker = fetch_ticker(symbol)
live_price = float(ticker.get("close", ticker.get("mark_price", 0)) or 0)
chg = float(ticker.get("price_change_24h", 0.0)) * 100

st.markdown(f"## {symbol.replace('USD','')}  ${live_price:,.2f}")
st.markdown(f"{'+' if chg >= 0 else ''}{chg:.2f}% (24h)")

st.markdown("---")
st.subheader("Multi-Timeframe Predictions")

rows = []
progress = st.progress(0)
status = st.empty()

for i, (tf_key, cfg) in enumerate(TIMEFRAMES.items()):
    status.caption(f"Analyzing {cfg['label']}...")
    candles = fetch_candles(symbol, cfg["resolution"], cfg["days"])

    if not candles or len(candles) < 60:
        rows.append({
            "Timeframe": cfg["label"],
            "Signal": "N/A",
            "Confidence": "-",
            "Entry": "-",
            "Take Profit": "-",
            "Stop Loss": "-",
            "Hold": cfg["hold"]
        })
        progress.progress((i + 1) / len(TIMEFRAMES))
        continue

    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)
    volume = np.array([c["volume"] for c in candles], dtype=float)

    proba, cur_vol = run_model(tuple(close), tuple(high), tuple(low), tuple(volume), cfg["max_holding"])
    signal, conf = get_signal(proba)

    entry = live_price
    vol = cur_vol or 0.01
    if signal == "BUY":
        tp = live_price * (1 + K_PROFIT * vol)
        sl = live_price * (1 - K_STOP * vol)
    elif signal == "SELL":
        tp = live_price * (1 - K_PROFIT * vol)
        sl = live_price * (1 + K_STOP * vol)
    else:
        tp = live_price * (1 + K_PROFIT * vol)
        sl = live_price * (1 - K_STOP * vol)

    rows.append({
        "Timeframe": cfg["label"],
        "Signal": signal,
        "Confidence": f"{conf:.0f}" if proba is not None else "-",
        "Entry": f"${entry:,.0f}",
        "Take Profit": f"${tp:,.0f}",
        "Stop Loss": f"${sl:,.0f}",
        "Hold": cfg["hold"]
    })
    progress.progress((i + 1) / len(TIMEFRAMES))

status.empty()
progress.empty()

df = pd.DataFrame(rows)
st.dataframe(style_table(df), use_container_width=True, hide_index=True)

st.caption("🟢 Strong Buy (≥70) · Light Green (60-69) · 🟠 Hold (50-59) · 🔴 Low / Sell side")

st.markdown("---")
st.caption("Delta Exchange data · RandomForest + Triple Barrier · Auto refresh ~18s")

time.sleep(18)
st.rerun()
