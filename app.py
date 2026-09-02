"""
app.py - Leverage Bitcoin Signal Engine

Faster refresh version:
- Live price updates quickly (short cache)
- Heavy model training is cached longer
- Manual + auto refresh support
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import hmac
import hashlib
import json
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

from ml_features import build_features, FEATURE_NAMES, WINNING_FEATURES
from triple_barrier import triple_barrier_labels
from signal_engine import sma, ema, rsi, macd
from features import atr, volume_zscore

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Leverage - BTC Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stButton button { width: 100%; border-radius: 8px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

DELTA_BASE = "https://api.india.delta.exchange"
WINNING_FEATURE_INDICES = [FEATURE_NAMES.index(f) for f in WINNING_FEATURES]
K_PROFIT = 2.0
K_STOP = 2.0

TOP_PROFILE = {
    "rsi_center": 64.0, "rsi_range": 8.0,
    "macd_hist_min": 10.0, "price_vs_ema20_min": 0.3,
}
BOTTOM_PROFILE = {
    "rsi_center": 38.0, "rsi_range": 8.0,
    "macd_hist_max": -10.0, "price_vs_ema20_max": -0.3,
}

TIMEFRAME_CONFIG = {
    "15m": {"resolution": "15m", "days": 45,  "max_holding": 32, "label": "15 Minutes (Scalp)"},
    "1h":  {"resolution": "1h",  "days": 90,  "max_holding": 24, "label": "1 Hour (Intraday)"},
    "4h":  {"resolution": "4h",  "days": 180, "max_holding": 18, "label": "4 Hours (Swing)"},
    "1D":  {"resolution": "1d",  "days": 300, "max_holding": 15, "label": "1 Day (Position)"},
}

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("🔑 API Credentials")
    api_key = st.text_input("API Key", type="password")
    api_secret = st.text_input("API Secret", type="password")

    st.header("⚙️ Settings")
    symbol = st.selectbox("Symbol", ["BTCUSD", "ETHUSD", "SOLUSD"])
    timeframe = st.selectbox(
        "Timeframe",
        options=list(TIMEFRAME_CONFIG.keys()),
        format_func=lambda x: TIMEFRAME_CONFIG[x]["label"],
        index=1
    )
    leverage = st.selectbox("Leverage", ["1x", "5x", "10x", "20x", "50x"], index=1)
    paper_mode = st.checkbox("Paper Mode (Recommended)", value=True)

    st.header("🔄 Refresh")
    auto_refresh = st.checkbox("Auto-refresh every 15s", value=False)
    if st.button("⚡ Refresh Now"):
        st.cache_data.clear()
        st.rerun()

tf_cfg = TIMEFRAME_CONFIG[timeframe]
MAX_HOLDING = tf_cfg["max_holding"]

# -------------------- FAST DATA FETCHING --------------------
# Live ticker - very short cache for accurate price
@st.cache_data(ttl=5, show_spinner=False)
def fetch_ticker(symbol: str):
    try:
        url = f"{DELTA_BASE}/v2/tickers/{symbol}"
        res = requests.get(url, timeout=4).json()
        if "result" in res:
            return res["result"]
    except Exception:
        pass
    return {}

# Candles - medium cache (model doesn't need tick-by-tick)
@st.cache_data(ttl=45, show_spinner=False)
def fetch_candles(symbol: str, resolution: str, days: int):
    end = int(time.time())
    start = end - days * 86400
    url = f"{DELTA_BASE}/v2/history/candles"
    params = {"resolution": resolution, "symbol": symbol, "start": start, "end": end}
    try:
        resp = requests.get(url, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json().get("result", [])
        return list(reversed(data))
    except Exception as e:
        return []

# Heavy model - longer cache so it doesn't retrain every few seconds
@st.cache_data(ttl=120, show_spinner="Training model...")
def train_and_predict_cached(close_tuple, high_tuple, low_tuple, volume_tuple, max_holding):
    close = np.array(close_tuple, dtype=float)
    high = np.array(high_tuple, dtype=float)
    low = np.array(low_tuple, dtype=float)
    volume = np.array(volume_tuple, dtype=float)

    ohlcv = {"open": close, "high": high, "low": low, "close": close, "volume": volume}

    labels, valid, vol, _, _ = triple_barrier_labels(close, K_PROFIT, K_STOP, max_holding)

    X, y = [], []
    for i in range(60, len(close) - 1):
        if valid[i] and labels[i] != 0:
            X.append(build_features(ohlcv, i))
            y.append(1 if labels[i] == 1 else 0)

    if len(X) < 100:
        return None, None, None

    X = np.array(X)[:, WINNING_FEATURE_INDICES]
    y = np.array(y)

    model = RandomForestClassifier(
        n_estimators=120, max_depth=5, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    model.fit(X, y)

    current_feats = np.array([build_features(ohlcv, len(close) - 1)])[:, WINNING_FEATURE_INDICES]
    proba = float(model.predict_proba(current_feats)[0, 1])
    current_vol = float(vol[-1]) if len(vol) > 0 and vol[-1] > 0 else float(np.std(np.diff(np.log(close[-20:]))))

    importance = model.feature_importances_.tolist()
    return proba, current_vol, importance

# -------------------- LOAD DATA --------------------
ticker = fetch_ticker(symbol)
candles = fetch_candles(symbol, tf_cfg["resolution"], tf_cfg["days"])

if not candles or len(candles) < 80:
    st.error("Not enough market data. Try another timeframe.")
    st.stop()

close = np.array([c["close"] for c in candles], dtype=float)
high = np.array([c["high"] for c in candles], dtype=float)
low = np.array([c["low"] for c in candles], dtype=float)
volume = np.array([c["volume"] for c in candles], dtype=float)

live_price = float(ticker.get("close", ticker.get("mark_price", close[-1])))
price_change_24h = float(ticker.get("price_change_24h", 0.0)) * 100

# Indicators (fast)
price = close[-1]
ema20_val = ema(close, 20) if len(close) >= 20 else price
sma200_val = sma(close, 200) if len(close) >= 200 else sma(close, min(len(close), 100))
rsi_val = rsi(close, 14)
macd_line, macd_sig = macd(close)
macd_hist = macd_line - macd_sig
price_vs_ema20 = (price - ema20_val) / ema20_val * 100 if ema20_val else 0.0

indicators = {
    "ema20": ema20_val,
    "sma200": sma200_val,
    "rsi": rsi_val,
    "macd_hist": macd_hist,
    "price_vs_ema20": price_vs_ema20,
}

# Pattern scores (fast)
def score_pattern(inds, profile, is_top=True):
    score = 0.0
    rsi_diff = abs(inds["rsi"] - profile["rsi_center"])
    score += max(0, 100 - (rsi_diff / profile["rsi_range"]) * 50) * 0.45

    macd_h = inds["macd_hist"]
    if is_top:
        macd_score = min(100, 50 + macd_h / 5) if macd_h >= profile["macd_hist_min"] else max(0, 50 + macd_h)
    else:
        macd_score = min(100, 50 + abs(macd_h) / 5) if macd_h <= profile["macd_hist_max"] else max(0, 50 - macd_h)
    score += macd_score * 0.35

    pve = inds["price_vs_ema20"]
    if is_top:
        ema_score = min(100, 60 + pve * 10) if pve >= profile.get("price_vs_ema20_min", 0) else max(0, 50 + pve * 20)
    else:
        ema_score = min(100, 60 + abs(pve) * 10) if pve <= profile.get("price_vs_ema20_max", 0) else max(0, 50 - pve * 20)
    score += ema_score * 0.20
    return round(score, 1)

def interpret_match(score):
    if score >= 75: return "STRONG"
    if score >= 60: return "Moderate"
    if score >= 45: return "Weak"
    return "None"

top_score = score_pattern(indicators, TOP_PROFILE, is_top=True)
bottom_score = score_pattern(indicators, BOTTOM_PROFILE, is_top=False)

# Model (cached heavier)
proba, current_vol, importance_list = train_and_predict_cached(
    tuple(close), tuple(high), tuple(low), tuple(volume), MAX_HOLDING
)

importance_df = None
if importance_list is not None:
    importance_df = pd.DataFrame({
        "Feature": WINNING_FEATURES,
        "Importance": importance_list
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

# -------------------- UI --------------------
st.title("⚡ Leverage - Bitcoin Signal Engine")

# Live status line
last_update = datetime.now().strftime("%H:%M:%S")
st.caption(f"Timeframe: **{tf_cfg['label']}**  |  Updated: {last_update}  |  Bars: {len(close)}")

m1, m2, m3 = st.columns(3)
m1.metric("Live Price", f"${live_price:,.2f}", f"{price_change_24h:.2f}%")
m2.metric("Model Confidence (Up)", f"{proba*100:.1f}%" if proba is not None else "N/A")
m3.metric("Volatility", f"{current_vol*100:.3f}%" if current_vol else "N/A")

st.markdown("---")

# Pattern Detector
st.subheader("🎯 Live Pattern Detector")
p1, p2 = st.columns(2)
p1.metric("Top (High) Match", f"{top_score:.1f}%", interpret_match(top_score))
p2.metric("Bottom (Low) Match", f"{bottom_score:.1f}%", interpret_match(bottom_score))

if top_score >= 65 and top_score > bottom_score + 10:
    st.warning("⚠️ Current setup **resembles historical TOPS**. Caution on new longs.")
elif bottom_score >= 65 and bottom_score > top_score + 10:
    st.success("✅ Current setup **resembles historical BOTTOMS**. Potential long zone.")
elif max(top_score, bottom_score) < 50:
    st.info("No strong Top or Bottom pattern active. Market appears neutral / transitional.")
else:
    st.info("Mixed signals — no clear dominant pattern.")

st.markdown("---")

# Technicals
st.subheader("📊 Key Technicals")
t1, t2, t3, t4 = st.columns(4)
t1.metric("RSI (14)", f"{indicators['rsi']:.1f}")
t2.metric("MACD Hist", f"{indicators['macd_hist']:.1f}")
t3.metric("EMA 20", f"${indicators['ema20']:,.0f}")
t4.metric("SMA 200", f"${indicators['sma200']:,.0f}")

trend_text = "Above SMA200 (Bullish)" if live_price > indicators["sma200"] else "Below SMA200 (Bearish)"
st.caption(f"{trend_text}  |  Price vs EMA20: {indicators['price_vs_ema20']:+.2f}%")

st.markdown("---")

if proba is None:
    st.warning("Not enough data to train a reliable model on this timeframe.")
else:
    if proba > 0.60:
        bias, advice = "LONG 🟢", f"Model assigns **{proba*100:.1f}%** probability that the upper barrier will be hit first."
    elif proba < 0.40:
        bias, advice = "SHORT 🔴", f"Model assigns **{(1-proba)*100:.1f}%** probability that the lower barrier will be hit first."
    else:
        bias, advice = "NEUTRAL / NO TRADE ⚪", "Confidence is too low. Better to stay flat."

    st.subheader(f"Signal: {bias}")
    st.write(advice)

    upper = live_price * (1 + K_PROFIT * current_vol)
    lower = live_price * (1 - K_STOP * current_vol)

    c1, c2, c3 = st.columns(3)
    c1.metric("Entry", f"${live_price:,.2f}")
    c2.metric("Take Profit", f"${upper:,.2f}")
    c3.metric("Stop Loss", f"${lower:,.2f}")

    if importance_df is not None:
        st.markdown("---")
        st.subheader("🔍 Feature Importance")
        st.bar_chart(importance_df.set_index("Feature"), height=240)

    st.markdown("---")
    st.subheader(f"Recent Price Action ({timeframe})")
    st.line_chart(pd.DataFrame({"Close": close[-100:]}), height=230)

# -------------------- ORDERS --------------------
st.markdown("---")
st.subheader("🔒 Order Execution")

if paper_mode:
    st.info("Paper Mode is ON. No real orders will be sent.")
else:
    st.warning("Live Mode is active.")

col1, col2 = st.columns(2)
with col1:
    if st.button("🟢 Open LONG", disabled=(proba is None or proba < 0.55)):
        st.session_state["pending"] = "buy"
with col2:
    if st.button("🔴 Open SHORT", disabled=(proba is None or proba > 0.45)):
        st.session_state["pending"] = "sell"

if "pending" in st.session_state and st.session_state["pending"]:
    side = st.session_state["pending"]
    st.warning(f"Confirm **{side.upper()}** on **{symbol}** ({timeframe}) with **{leverage}**?")
    cy, cn = st.columns(2)
    if cy.button("✅ Confirm"):
        if paper_mode:
            st.success(f"Paper {side.upper()} recorded at ${live_price:,.2f}")
        else:
            st.error("Live order placement requires valid API keys and is disabled in this view for safety." if not (api_key and api_secret) else "Implement live order carefully.")
        st.session_state["pending"] = None
    if cn.button("❌ Cancel"):
        st.session_state["pending"] = None

# Auto refresh
if auto_refresh:
    time.sleep(15)
    st.rerun()

st.caption("Faster refresh enabled • Live price ~5s • Model cache ~2min • Pattern Detector active")
