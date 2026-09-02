"""
app.py - Delta Exchange Companion (Clean Version)

Uses real OHLCV data + RandomForest trained with triple-barrier labels.
Now includes SMA200 + MACD Histogram + RSI in the feature set.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import hmac
import hashlib
import json
import time
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier

from ml_features import build_features, FEATURE_NAMES, WINNING_FEATURES
from triple_barrier import triple_barrier_labels
from features import atr
from signal_engine import sma, ema, rsi, macd

# -------------------- CONFIG --------------------
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
MAX_HOLDING = 24

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("🔑 API Credentials")
    api_key = st.text_input("API Key", type="password")
    api_secret = st.text_input("API Secret", type="password")
    st.info("Credentials are only kept in memory for this session.")

    st.header("⚙️ Settings")
    symbol = st.selectbox("Symbol", ["BTCUSD", "ETHUSD", "SOLUSD"])
    leverage = st.selectbox("Leverage", ["1x", "5x", "10x", "20x", "50x"], index=1)
    paper_mode = st.checkbox("Paper Mode (Recommended)", value=True)

# -------------------- DATA FETCHING --------------------
@st.cache_data(ttl=60)
def fetch_candles(symbol: str, days: int = 120):
    """Fetch more history so SMA200 is meaningful."""
    end = int(time.time())
    start = end - days * 86400
    url = f"{DELTA_BASE}/v2/history/candles"
    params = {"resolution": "1h", "symbol": symbol, "start": start, "end": end}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("result", [])
        return list(reversed(data))
    except Exception as e:
        st.error(f"Failed to fetch candles: {e}")
        return []

@st.cache_data(ttl=30)
def fetch_ticker(symbol: str):
    try:
        url = f"{DELTA_BASE}/v2/tickers/{symbol}"
        res = requests.get(url, timeout=5).json()
        if "result" in res:
            return res["result"]
    except Exception:
        pass
    return {}

candles = fetch_candles(symbol)
ticker = fetch_ticker(symbol)

if not candles:
    st.error("Could not load market data. Please try again later.")
    st.stop()

close = np.array([c["close"] for c in candles], dtype=float)
high = np.array([c["high"] for c in candles], dtype=float)
low = np.array([c["low"] for c in candles], dtype=float)
volume = np.array([c["volume"] for c in candles], dtype=float)
ohlcv = {
    "open": np.array([c["open"] for c in candles], dtype=float),
    "high": high,
    "low": low,
    "close": close,
    "volume": volume,
}

live_price = float(ticker.get("close", ticker.get("mark_price", close[-1])))
price_change_24h = float(ticker.get("price_change_24h", 0.0)) * 100

# -------------------- HELPER INDICATORS FOR DISPLAY --------------------
def get_key_indicators(close):
    price = close[-1]
    sma20_val = sma(close, 20) if len(close) >= 20 else price
    ema20_val = ema(close, 20) if len(close) >= 20 else price
    sma200_val = sma(close, 200) if len(close) >= 200 else sma(close, min(len(close), 100))
    rsi_val = rsi(close, 14)
    macd_line, macd_sig = macd(close)
    macd_hist = macd_line - macd_sig
    return {
        "sma20": sma20_val,
        "ema20": ema20_val,
        "sma200": sma200_val,
        "rsi": rsi_val,
        "macd_hist": macd_hist,
        "price_vs_sma200": (price - sma200_val) / sma200_val if sma200_val else 0,
    }

indicators = get_key_indicators(close)

# -------------------- MODEL --------------------
def train_and_predict(ohlcv, close):
    labels, valid, vol, _, _ = triple_barrier_labels(
        close, K_PROFIT, K_STOP, MAX_HOLDING
    )

    X, y = [], []
    min_history = 60
    for i in range(min_history, len(close) - 1):
        if valid[i] and labels[i] != 0:
            feats = build_features(ohlcv, i)
            X.append(feats)
            y.append(1 if labels[i] == 1 else 0)

    if len(X) < 120:
        return None, None, None

    X = np.array(X)[:, WINNING_FEATURE_INDICES]
    y = np.array(y)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    current_feats = np.array([build_features(ohlcv, len(close) - 1)])[:, WINNING_FEATURE_INDICES]
    proba = model.predict_proba(current_feats)[0, 1]

    current_vol = vol[-1] if vol[-1] > 0 else np.std(np.diff(np.log(close[-20:])))
    return proba, current_vol, model

proba, current_vol, model = train_and_predict(ohlcv, close)

# -------------------- UI --------------------
st.title("⚡ Leverage - Bitcoin Signal Engine")

m1, m2, m3 = st.columns(3)
m1.metric("Live Price", f"${live_price:,.2f}", f"{price_change_24h:.2f}%")
m2.metric("Model Confidence (Up)", f"{proba*100:.1f}%" if proba is not None else "N/A")
m3.metric("Current Volatility", f"{current_vol*100:.3f}%" if current_vol else "N/A")

st.markdown("---")

# Technical Snapshot
st.subheader("📊 Key Technicals")
t1, t2, t3, t4 = st.columns(4)
t1.metric("RSI (14)", f"{indicators['rsi']:.1f}")
t2.metric("MACD Hist", f"{indicators['macd_hist']:.1f}")
t3.metric("EMA 20", f"${indicators['ema20']:,.0f}")
t4.metric("SMA 200", f"${indicators['sma200']:,.0f}")

# Trend context
trend_text = "Above SMA200 (Bullish Regime)" if live_price > indicators["sma200"] else "Below SMA200 (Bearish Regime)"
st.caption(f"Trend Context: {trend_text}")

st.markdown("---")

if proba is None:
    st.warning("Not enough data yet to train a reliable model.")
else:
    if proba > 0.60:
        bias = "LONG 🟢"
        advice = f"Model assigns {proba*100:.1f}% probability that the upper barrier will be hit first."
        direction = 1
    elif proba < 0.40:
        bias = "SHORT 🔴"
        advice = f"Model assigns {(1-proba)*100:.1f}% probability that the lower barrier will be hit first."
        direction = -1
    else:
        bias = "NEUTRAL / NO TRADE ⚪"
        advice = "Confidence is too low. Better to stay flat."
        direction = 0

    st.subheader(f"Signal: {bias}")
    st.write(advice)

    # Dynamic barriers
    upper = live_price * (1 + K_PROFIT * current_vol)
    lower = live_price * (1 - K_STOP * current_vol)

    c1, c2, c3 = st.columns(3)
    c1.metric("Entry", f"${live_price:,.2f}")
    c2.metric("Take Profit (Upper)", f"${upper:,.2f}")
    c3.metric("Stop Loss (Lower)", f"${lower:,.2f}")

    st.markdown("---")

    # Simple chart of recent closes
    st.subheader("Recent Price Action (1h)")
    chart_df = pd.DataFrame({"Close": close[-100:]})
    st.line_chart(chart_df, height=250)

# -------------------- ORDER SECTION --------------------
st.markdown("---")
st.subheader("🔒 Order Execution")

if paper_mode:
    st.info("Paper Mode is ON. No real orders will be sent.")
else:
    st.warning("Live Mode is active. Real money can be used.")

col1, col2 = st.columns(2)

with col1:
    if st.button("🟢 Open LONG", disabled=(proba is None or proba < 0.55)):
        st.session_state["pending"] = "buy"

with col2:
    if st.button("🔴 Open SHORT", disabled=(proba is None or proba > 0.45)):
        st.session_state["pending"] = "sell"

if "pending" in st.session_state and st.session_state["pending"]:
    side = st.session_state["pending"]
    st.warning(f"Confirm **{side.upper()}** market order on **{symbol}** with **{leverage}**?")

    cy, cn = st.columns(2)
    if cy.button("✅ Confirm"):
        if paper_mode:
            st.success(f"Paper {side.upper()} order recorded at ${live_price:,.2f}")
        else:
            if not api_key or not api_secret:
                st.error("Please enter API credentials in the sidebar.")
            else:
                try:
                    prod_res = requests.get(f"{DELTA_BASE}/v2/products", timeout=5).json()
                    product_id = None
                    for p in prod_res.get("result", []):
                        if p.get("symbol") == symbol and p.get("contract_type") == "perpetual_futures":
                            product_id = p["id"]
                            break

                    if not product_id:
                        st.error("Could not find product_id")
                    else:
                        path = "/v2/orders"
                        method = "POST"
                        timestamp = str(int(time.time() * 1000))
                        payload = {
                            "product_id": int(product_id),
                            "size": 1,
                            "side": side,
                            "order_type": "market_order",
                        }
                        payload_str = json.dumps(payload, separators=(',', ':'))
                        signature_data = method + timestamp + path + payload_str
                        signature = hmac.new(
                            api_secret.encode(),
                            signature_data.encode(),
                            hashlib.sha256
                        ).hexdigest()

                        headers = {
                            "api-key": api_key,
                            "timestamp": timestamp,
                            "signature": signature,
                            "Content-Type": "application/json",
                            "User-Agent": "LeverageBot/2.1",
                        }

                        resp = requests.post(DELTA_BASE + path, data=payload_str, headers=headers, timeout=8)
                        res = resp.json()
                        if res.get("success"):
                            st.success(f"Order placed! ID: {res.get('result', {}).get('id')}")
                        else:
                            st.error(f"API Error: {res.get('error')}")
                except Exception as e:
                    st.error(f"Order failed: {e}")

        st.session_state["pending"] = None

    if cn.button("❌ Cancel"):
        st.session_state["pending"] = None
        st.info("Cancelled.")

st.caption("Features: ATR% • Price-Vol Trend • Return 3d • EMA20 • SMA200 • MACD Hist • RSI | Triple Barrier Labels")
