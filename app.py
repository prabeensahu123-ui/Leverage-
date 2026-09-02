"""
app.py - Leverage Bitcoin Signal Engine

Multi-timeframe support + RandomForest + Triple Barrier
+ Feature Importance + Live Top/Bottom Pattern Detector
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import hmac
import hashlib
import json
import time
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

# Pattern profiles learned from Phases 1-3
TOP_PROFILE = {
    "rsi_center": 64.0,
    "rsi_range": 8.0,
    "macd_hist_min": 10.0,
    "price_vs_ema20_min": 0.3,
}

BOTTOM_PROFILE = {
    "rsi_center": 38.0,
    "rsi_range": 8.0,
    "macd_hist_max": -10.0,
    "price_vs_ema20_max": -0.3,
}

TIMEFRAME_CONFIG = {
    "15m": {"resolution": "15m", "days": 45,  "max_holding": 32, "label": "15 Minutes (Scalp)"},
    "1h":  {"resolution": "1h",  "days": 120, "max_holding": 24, "label": "1 Hour (Intraday)"},
    "4h":  {"resolution": "4h",  "days": 240, "max_holding": 18, "label": "4 Hours (Swing)"},
    "1D":  {"resolution": "1d",  "days": 400, "max_holding": 15, "label": "1 Day (Position)"},
}

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("🔑 API Credentials")
    api_key = st.text_input("API Key", type="password")
    api_secret = st.text_input("API Secret", type="password")
    st.info("Credentials stay in memory only for this session.")

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

tf_cfg = TIMEFRAME_CONFIG[timeframe]
MAX_HOLDING = tf_cfg["max_holding"]

# -------------------- DATA FETCHING --------------------
@st.cache_data(ttl=60)
def fetch_candles(symbol: str, resolution: str, days: int):
    end = int(time.time())
    start = end - days * 86400
    url = f"{DELTA_BASE}/v2/history/candles"
    params = {"resolution": resolution, "symbol": symbol, "start": start, "end": end}
    try:
        resp = requests.get(url, params=params, timeout=18)
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

candles = fetch_candles(symbol, tf_cfg["resolution"], tf_cfg["days"])
ticker = fetch_ticker(symbol)

if not candles or len(candles) < 80:
    st.error("Not enough market data for this timeframe. Try another timeframe or try again later.")
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

# -------------------- KEY INDICATORS --------------------
def get_key_indicators(close, high, low, volume):
    price = close[-1]
    ema20_val = ema(close, 20) if len(close) >= 20 else price
    sma200_val = sma(close, 200) if len(close) >= 200 else sma(close, min(len(close), 100))
    rsi_val = rsi(close, 14)
    macd_line, macd_sig = macd(close)
    macd_hist = macd_line - macd_sig
    price_vs_ema20 = (price - ema20_val) / ema20_val * 100 if ema20_val else 0
    return {
        "ema20": ema20_val,
        "sma200": sma200_val,
        "rsi": rsi_val,
        "macd_hist": macd_hist,
        "price_vs_ema20": price_vs_ema20,
        "volume_z": volume_zscore(volume),
        "atr_pct": atr(high, low, close) / price * 100 if price else 0,
    }

indicators = get_key_indicators(close, high, low, volume)

# -------------------- PATTERN MATCHING --------------------
def score_pattern(inds, profile, is_top=True):
    if inds is None:
        return 0.0

    score = 0.0

    # RSI (45% weight)
    rsi_diff = abs(inds["rsi"] - profile["rsi_center"])
    rsi_score = max(0, 100 - (rsi_diff / profile["rsi_range"]) * 50)
    score += rsi_score * 0.45

    # MACD Hist (35% weight)
    macd_h = inds["macd_hist"]
    if is_top:
        macd_score = min(100, 50 + macd_h / 5) if macd_h >= profile["macd_hist_min"] else max(0, 50 + macd_h)
    else:
        macd_score = min(100, 50 + abs(macd_h) / 5) if macd_h <= profile["macd_hist_max"] else max(0, 50 - macd_h)
    score += macd_score * 0.35

    # Price vs EMA20 (20% weight)
    pve = inds["price_vs_ema20"]
    if is_top:
        ema_score = min(100, 60 + pve * 10) if pve >= profile.get("price_vs_ema20_min", 0) else max(0, 50 + pve * 20)
    else:
        ema_score = min(100, 60 + abs(pve) * 10) if pve <= profile.get("price_vs_ema20_max", 0) else max(0, 50 - pve * 20)
    score += ema_score * 0.20

    return round(score, 1)

def interpret_match(score):
    if score >= 75:
        return "STRONG"
    elif score >= 60:
        return "Moderate"
    elif score >= 45:
        return "Weak"
    return "None"

top_score = score_pattern(indicators, TOP_PROFILE, is_top=True)
bottom_score = score_pattern(indicators, BOTTOM_PROFILE, is_top=False)

# -------------------- MODEL --------------------
def train_and_predict(ohlcv, close, max_holding):
    labels, valid, vol, _, _ = triple_barrier_labels(close, K_PROFIT, K_STOP, max_holding)

    X, y = [], []
    min_history = 60
    for i in range(min_history, len(close) - 1):
        if valid[i] and labels[i] != 0:
            X.append(build_features(ohlcv, i))
            y.append(1 if labels[i] == 1 else 0)

    if len(X) < 100:
        return None, None, None, None

    X = np.array(X)[:, WINNING_FEATURE_INDICES]
    y = np.array(y)

    model = RandomForestClassifier(
        n_estimators=150, max_depth=5, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    model.fit(X, y)

    current_feats = np.array([build_features(ohlcv, len(close) - 1)])[:, WINNING_FEATURE_INDICES]
    proba = model.predict_proba(current_feats)[0, 1]
    current_vol = vol[-1] if len(vol) > 0 and vol[-1] > 0 else np.std(np.diff(np.log(close[-20:])))

    importance_df = pd.DataFrame({
        "Feature": WINNING_FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return proba, current_vol, model, importance_df

proba, current_vol, model, importance_df = train_and_predict(ohlcv, close, MAX_HOLDING)

# -------------------- UI --------------------
st.title("⚡ Leverage - Bitcoin Signal Engine")
st.caption(f"Timeframe: **{tf_cfg['label']}**  |  Bars: {len(close)}  |  Max hold: {MAX_HOLDING} bars")

m1, m2, m3 = st.columns(3)
m1.metric("Live Price", f"${live_price:,.2f}", f"{price_change_24h:.2f}%")
m2.metric("Model Confidence (Up)", f"{proba*100:.1f}%" if proba is not None else "N/A")
m3.metric("Current Volatility", f"{current_vol*100:.3f}%" if current_vol else "N/A")

st.markdown("---")

# ---------- LIVE PATTERN DETECTOR ----------
st.subheader("🎯 Live Pattern Detector")
st.caption("How closely current conditions match historical Tops & Bottoms (from Phases 1-3)")

p1, p2 = st.columns(2)
with p1:
    st.metric("Top (High) Match", f"{top_score:.1f}%", interpret_match(top_score))
with p2:
    st.metric("Bottom (Low) Match", f"{bottom_score:.1f}%", interpret_match(bottom_score))

# Combined verdict
if top_score >= 65 and top_score > bottom_score + 10:
    st.warning("⚠️ Current setup **resembles historical TOPS**. Caution on new longs.")
elif bottom_score >= 65 and bottom_score > top_score + 10:
    st.success("✅ Current setup **resembles historical BOTTOMS**. Potential long zone.")
elif max(top_score, bottom_score) < 50:
    st.info("No strong Top or Bottom pattern active. Market appears neutral / transitional.")
else:
    st.info("Mixed signals — no clear dominant pattern.")

st.markdown("---")

# Key Technicals
st.subheader("📊 Key Technicals")
t1, t2, t3, t4 = st.columns(4)
t1.metric("RSI (14)", f"{indicators['rsi']:.1f}")
t2.metric("MACD Hist", f"{indicators['macd_hist']:.1f}")
t3.metric("EMA 20", f"${indicators['ema20']:,.0f}")
t4.metric("SMA 200", f"${indicators['sma200']:,.0f}")

trend_text = "Above SMA200 (Bullish Regime)" if live_price > indicators["sma200"] else "Below SMA200 (Bearish Regime)"
st.caption(f"Trend Context: {trend_text}  |  Price vs EMA20: {indicators['price_vs_ema20']:+.2f}%")

st.markdown("---")

if proba is None:
    st.warning("Not enough clean labeled data to train a reliable model on this timeframe.")
else:
    if proba > 0.60:
        bias = "LONG 🟢"
        advice = f"Model assigns **{proba*100:.1f}%** probability that the upper barrier will be hit first."
    elif proba < 0.40:
        bias = "SHORT 🔴"
        advice = f"Model assigns **{(1-proba)*100:.1f}%** probability that the lower barrier will be hit first."
    else:
        bias = "NEUTRAL / NO TRADE ⚪"
        advice = "Confidence is too low. Better to stay flat."

    st.subheader(f"Signal: {bias}")
    st.write(advice)

    upper = live_price * (1 + K_PROFIT * current_vol)
    lower = live_price * (1 - K_STOP * current_vol)

    c1, c2, c3 = st.columns(3)
    c1.metric("Entry", f"${live_price:,.2f}")
    c2.metric("Take Profit (Upper)", f"${upper:,.2f}")
    c3.metric("Stop Loss (Lower)", f"${lower:,.2f}")

    st.markdown("---")

    # Feature Importance
    st.subheader("🔍 Feature Importance")
    st.caption(f"Feature contribution on the **{timeframe}** model")
    st.bar_chart(importance_df.set_index("Feature"), height=260)

    with st.expander("View detailed importance values"):
        st.dataframe(importance_df.style.format({"Importance": "{:.3f}"}), use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader(f"Recent Price Action ({timeframe})")
    lookback = min(120, len(close))
    st.line_chart(pd.DataFrame({"Close": close[-lookback:]}), height=240)

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
    st.warning(f"Confirm **{side.upper()}** market order on **{symbol}** ({timeframe}) with **{leverage}**?")

    cy, cn = st.columns(2)
    if cy.button("✅ Confirm"):
        if paper_mode:
            st.success(f"Paper {side.upper()} order recorded at ${live_price:,.2f} on {timeframe}")
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
                        timestamp = str(int(time.time() * 1000))
                        payload = {"product_id": int(product_id), "size": 1, "side": side, "order_type": "market_order"}
                        payload_str = json.dumps(payload, separators=(',', ':'))
                        signature = hmac.new(api_secret.encode(), ("POST" + timestamp + path + payload_str).encode(), hashlib.sha256).hexdigest()
                        headers = {"api-key": api_key, "timestamp": timestamp, "signature": signature,
                                   "Content-Type": "application/json", "User-Agent": "LeverageBot/2.4"}
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

st.caption("Pattern Detector + RandomForest + Triple Barrier | Phases 1-3 Integrated")
