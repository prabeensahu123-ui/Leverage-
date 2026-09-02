"""
app.py - Leverage Bitcoin Signal Engine
Clean UI + horizontal timeframe pills + auto update + indicators
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
from features import atr, volume_zscore
from forecasting import ensemble_forecast

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Leverage Signal",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stButton > button {
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-weight: 600;
        font-size: 0.85rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    .block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

DELTA_BASE = "https://api.india.delta.exchange"
WINNING_FEATURE_INDICES = [FEATURE_NAMES.index(f) for f in WINNING_FEATURES]
K_PROFIT, K_STOP = 2.0, 2.0

TOP_PROFILE = {"rsi_center": 64.0, "rsi_range": 8.0, "macd_hist_min": 10.0, "price_vs_ema20_min": 0.3}
BOTTOM_PROFILE = {"rsi_center": 38.0, "rsi_range": 8.0, "macd_hist_max": -10.0, "price_vs_ema20_max": -0.3}

# Expanded timeframe options closer to reference image
TIMEFRAMES = {
    "15m": {"resolution": "15m", "days": 30,  "max_holding": 32, "label": "15m"},
    "30m": {"resolution": "30m", "days": 45,  "max_holding": 28, "label": "30m"},
    "1h":  {"resolution": "1h",  "days": 90,  "max_holding": 24, "label": "1h"},
    "4h":  {"resolution": "4h",  "days": 180, "max_holding": 18, "label": "4h"},
    "1D":  {"resolution": "1d",  "days": 300, "max_holding": 15, "label": "Daily"},
}

ASSETS = ["BTCUSD", "ETHUSD", "SOLUSD"]

# -------------------- SESSION STATE --------------------
if "tf" not in st.session_state:
    st.session_state.tf = "1h"
if "symbol" not in st.session_state:
    st.session_state.symbol = "BTCUSD"

# -------------------- DATA FUNCTIONS --------------------
@st.cache_data(ttl=4, show_spinner=False)
def fetch_ticker(symbol: str):
    try:
        r = requests.get(f"{DELTA_BASE}/v2/tickers/{symbol}", timeout=4).json()
        return r.get("result", {})
    except Exception:
        return {}

@st.cache_data(ttl=40, show_spinner=False)
def fetch_candles(symbol: str, resolution: str, days: int):
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
        return None, None, None

    X = np.array(X)[:, WINNING_FEATURE_INDICES]
    y = np.array(y)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    proba = float(model.predict_proba(np.array([build_features(ohlcv, len(close)-1)])[:, WINNING_FEATURE_INDICES])[0, 1])
    cur_vol = float(vol[-1]) if len(vol) and vol[-1] > 0 else float(np.std(np.diff(np.log(close[-20:]))))
    return proba, cur_vol, model.feature_importances_.tolist()

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

# -------------------- HEADER / ASSET SELECTOR --------------------
st.markdown("### Signal.")
st.caption(f"live · updated {datetime.now().strftime('%H:%M:%S')}")

# Asset pills
a1, a2, a3 = st.columns(3)
for col, sym in zip([a1, a2, a3], ASSETS):
    with col:
        if st.button(sym.replace("USD", ""), key=f"asset_{sym}", use_container_width=True):
            st.session_state.symbol = sym
            st.rerun()

symbol = st.session_state.symbol
tf = st.session_state.tf
tf_cfg = TIMEFRAMES[tf]

# -------------------- TIMEFRAME PILLS (like reference) --------------------
st.write("")
tf_cols = st.columns(len(TIMEFRAMES))
for col, (key, cfg) in zip(tf_cols, TIMEFRAMES.items()):
    with col:
        if st.button(cfg["label"], key=f"tf_{key}", use_container_width=True):
            st.session_state.tf = key
            st.rerun()

# -------------------- FETCH DATA --------------------
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

inds = {"rsi": rsi_val, "macd_hist": macd_hist, "price_vs_ema20": pve}

top_score = score_pattern(inds, TOP_PROFILE, True)
bottom_score = score_pattern(inds, BOTTOM_PROFILE, False)

# Model
proba, cur_vol, imp = run_model(tuple(close), tuple(high), tuple(low), tuple(volume), tf_cfg["max_holding"])

# Holt projection (like reference app)
try:
    fc = ensemble_forecast(close, steps=7)
    proj_low = fc["lower_band"][-1]
    proj_high = fc["upper_band"][-1]
    proj_dir = "upward" if fc["forecast"][-1] > close[-1] else "downward"
except Exception:
    proj_low = proj_high = close[-1]
    proj_dir = "sideways"

# -------------------- MAIN DISPLAY --------------------
st.markdown(f"## ${live_price:,.2f}")
color = "green" if chg >= 0 else "red"
st.markdown(f"<span style='color:{color}; font-size:1.1rem'>{chg:+.2f}% (24h)</span>", unsafe_allow_html=True)

# Mini chart
st.line_chart(pd.DataFrame({"Price": close[-80:]}), height=220)

# Projection text (similar to reference)
st.info(
    f"7-bar statistical projection (Holt’s model): **{proj_dir}** trend, "
    f"est. range ${proj_low:,.2f} – ${proj_high:,.2f}. "
    f"This is a trend extrapolation, not a guarantee."
)

st.markdown("---")

# Pattern + Model
c1, c2, c3 = st.columns(3)
c1.metric("Top Match", f"{top_score:.0f}%")
c2.metric("Bottom Match", f"{bottom_score:.0f}%")
c3.metric("Model Up Prob", f"{proba*100:.0f}%" if proba is not None else "N/A")

if top_score >= 65 and top_score > bottom_score + 10:
    st.warning("Setup resembles historical **TOPS**")
elif bottom_score >= 65 and bottom_score > top_score + 10:
    st.success("Setup resembles historical **BOTTOMS**")
else:
    st.caption("No strong top/bottom pattern active")

st.markdown("---")

# Technical Indicators section (like reference)
st.subheader("TECHNICAL INDICATORS")
i1, i2, i3, i4 = st.columns(4)
i1.metric("RSI (14)", f"{rsi_val:.1f}")
i2.metric("MACD Hist", f"{macd_hist:.1f}")
i3.metric("EMA 20", f"${ema20:,.0f}")
i4.metric("SMA 200", f"${sma200:,.0f}")

st.caption(f"Price vs EMA20: {pve:+.2f}%  |  Trend: {'Bullish' if live_price > sma200 else 'Bearish'} regime")

# Auto update every 12 seconds without full manual refresh
time.sleep(0.1)  # small yield
st_autorefresh = st.empty()
# Simple auto-rerun
if True:
    time.sleep(12)
    st.rerun()
