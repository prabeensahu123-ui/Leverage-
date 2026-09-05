"""
app.py - Leverage Signal Engine
Model table + Candlestick comparison table
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import os
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

from ml_features import build_features, FEATURE_NAMES, WINNING_FEATURES
from triple_barrier import triple_barrier_labels
from candle_patterns import detect_candle_pattern

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

STATE_FILES = ["algo_bot/state.json", "paper_state.json"]
TRADE_FILES = ["algo_bot/trades.csv", "paper_trades.csv"]

TIMEFRAMES = {
    "15m": {"resolution": "15m", "days": 30,  "max_holding": 32, "label": "15m",  "hold": "~8h"},
    "30m": {"resolution": "30m", "days": 45,  "max_holding": 28, "label": "30m",  "hold": "~14h"},
    "1h":  {"resolution": "1h",  "days": 90,  "max_holding": 24, "label": "1h",   "hold": "~24h"},
    "4h":  {"resolution": "4h",  "days": 180, "max_holding": 18, "label": "4h",   "hold": "~3d"},
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

@st.cache_data(ttl=120, show_spinner=False)
def run_balanced_model(close_t, high_t, low_t, vol_t, max_holding):
    close = np.array(close_t, dtype=float)
    high = np.array(high_t, dtype=float)
    low = np.array(low_t, dtype=float)
    volume = np.array(vol_t, dtype=float)
    ohlcv = {"open": close, "high": high, "low": low, "close": close, "volume": volume}

    labels, valid, vol, _, _ = triple_barrier_labels(close, K_PROFIT, K_STOP, max_holding)

    X, y = [], []
    for i in range(55, len(close) - 1):
        if valid[i]:
            X.append(build_features(ohlcv, i))
            y.append(labels[i])

    if len(X) < 90:
        return None, None, None, None

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

    buy_proba = proba[classes.index(1)] if 1 in classes else 0.0
    sell_proba = proba[classes.index(-1)] if -1 in classes else 0.0
    hold_proba = proba[classes.index(0)] if 0 in classes else 0.0
    cur_vol = float(vol[-1]) if len(vol) and vol[-1] > 0 else float(np.std(np.diff(np.log(close[-15:]))))
    return buy_proba, sell_proba, hold_proba, cur_vol

def decide_signal(buy_p, sell_p, hold_p):
    if buy_p is None:
        return "N/A", 0.0
    if buy_p >= 0.45 and buy_p > sell_p + 0.10:
        return "BUY", buy_p * 100
    if sell_p >= 0.45 and sell_p > buy_p + 0.10:
        return "SELL", sell_p * 100
    if buy_p >= 0.38 and buy_p > sell_p:
        return "BUY", buy_p * 100
    if sell_p >= 0.38 and sell_p > buy_p:
        return "SELL", sell_p * 100
    return "HOLD", max(buy_p, sell_p, hold_p) * 100

def load_bot_state():
    for path in STATE_FILES:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
    return {"open_trades": {}}

def load_bot_trades():
    for path in TRADE_FILES:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                pass
    return pd.DataFrame()

def agree_label(model_sig, candle_sig):
    if model_sig in ("N/A", None) or candle_sig in (None,):
        return "-"
    if model_sig == candle_sig:
        return "AGREE"
    if "HOLD" in (model_sig, candle_sig):
        return "MIXED"
    return "CONFLICT"

def make_colored_table(df, extra_signal_cols=None):
    extra_signal_cols = extra_signal_cols or []
    html = "<table style='width:100%; border-collapse: collapse; font-size: 0.82rem;'>"
    html += "<thead><tr style='background:#1e293b; color:white;'>"
    for col in df.columns:
        html += f"<th style='padding:7px; text-align:center; border:1px solid #334155;'>{col}</th>"
    html += "</tr></thead><tbody>"
    signal_cols = ["Signal", "Model", "Candle"] + extra_signal_cols
    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            val = row[col]
            style = "padding:7px; text-align:center; border:1px solid #334155;"
            if col in signal_cols:
                if val == "BUY": style += "background-color:#16a34a; color:white; font-weight:bold;"
                elif val == "SELL": style += "background-color:#dc2626; color:white; font-weight:bold;"
                elif val == "HOLD": style += "background-color:#ea580c; color:white; font-weight:bold;"
            elif col == "Agree":
                if val == "AGREE": style += "background-color:#16a34a; color:white; font-weight:bold;"
                elif val == "CONFLICT": style += "background-color:#dc2626; color:white; font-weight:bold;"
                elif val == "MIXED": style += "background-color:#ca8a04; color:white; font-weight:bold;"
            elif col == "Confidence":
                try:
                    v = float(val)
                    if v >= 55: style += "background-color:#16a34a; color:white;"
                    elif v >= 45: style += "background-color:#4ade80; color:black;"
                    elif v >= 35: style += "background-color:#fdba74; color:black;"
                    else: style += "background-color:#fca5a5; color:black;"
                except: pass
            html += f"<td style='{style}'>{val}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html

# -------------------- HEADER --------------------
st.markdown("### Leverage Signal")
st.caption(f"Model + Candles · {datetime.now().strftime('%H:%M:%S')}")

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

model_rows = []
candle_rows = []
compare_rows = []
progress = st.progress(0)
status = st.empty()

for i, (tf_key, cfg) in enumerate(TIMEFRAMES.items()):
    status.caption(f"Analyzing {cfg['label']}...")
    candles = fetch_candles(symbol, cfg["resolution"], cfg["days"])

    if not candles or len(candles) < 70:
        model_rows.append({
            "Timeframe": cfg["label"], "Signal": "N/A", "Confidence": "-",
            "BUY %": "-", "SELL %": "-", "Entry": "-",
            "Take Profit": "-", "Stop Loss": "-", "Hold": cfg["hold"]
        })
        candle_rows.append({
            "Timeframe": cfg["label"], "Pattern": "-", "Candle": "N/A"
        })
        compare_rows.append({
            "Timeframe": cfg["label"], "Model": "N/A", "Candle": "N/A", "Agree": "-"
        })
        progress.progress((i + 1) / len(TIMEFRAMES))
        continue

    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)
    volume = np.array([c["volume"] for c in candles], dtype=float)
    open_ = np.array([c.get("open", c["close"]) for c in candles], dtype=float)

    buy_p, sell_p, hold_p, cur_vol = run_balanced_model(
        tuple(close), tuple(high), tuple(low), tuple(volume), cfg["max_holding"]
    )
    signal, conf = decide_signal(buy_p, sell_p, hold_p)
    pattern, candle_bias = detect_candle_pattern(open_, high, low, close)

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

    model_rows.append({
        "Timeframe": cfg["label"],
        "Signal": signal,
        "Confidence": f"{conf:.0f}" if buy_p is not None else "-",
        "BUY %": f"{buy_p*100:.0f}" if buy_p is not None else "-",
        "SELL %": f"{sell_p*100:.0f}" if sell_p is not None else "-",
        "Entry": f"${live_price:,.0f}",
        "Take Profit": f"${tp:,.0f}",
        "Stop Loss": f"${sl:,.0f}",
        "Hold": cfg["hold"]
    })
    candle_rows.append({
        "Timeframe": cfg["label"],
        "Pattern": pattern,
        "Candle": candle_bias
    })
    compare_rows.append({
        "Timeframe": cfg["label"],
        "Model": signal,
        "Candle": candle_bias,
        "Agree": agree_label(signal, candle_bias)
    })
    progress.progress((i + 1) / len(TIMEFRAMES))

status.empty()
progress.empty()

st.markdown("---")
st.subheader("1) Model Predictions")
st.caption("RandomForest + Triple Barrier + indicators")
st.markdown(make_colored_table(pd.DataFrame(model_rows)), unsafe_allow_html=True)

st.markdown("---")
st.subheader("2) Candlestick Patterns")
st.caption("Last candle / last 2 candles only: Hammer, Hanging Man, Shooting Star, Doji, Engulfing")
st.markdown(make_colored_table(pd.DataFrame(candle_rows)), unsafe_allow_html=True)

st.markdown("---")
st.subheader("3) Do they agree?")
st.caption("AGREE = same side · MIXED = one is HOLD · CONFLICT = BUY vs SELL")
st.markdown(make_colored_table(pd.DataFrame(compare_rows)), unsafe_allow_html=True)

st.caption("If they conflict, treat as HOLD unless model confidence is very high.")

# ==================== ALGO BOT DASHBOARD ====================
st.markdown("---")
st.subheader("Algo Bot Dashboard")
state = load_bot_state()
open_trades = state.get("open_trades", {})
trades_df = load_bot_trades()

if open_trades:
    open_rows = []
    for tf, t in open_trades.items():
        open_rows.append({
            "TF": tf, "Side": t.get("side", "-"),
            "Entry": f"${t.get('entry', 0):,.0f}",
            "TP": f"${t.get('take_profit', 0):,.0f}",
            "SL": f"${t.get('stop_loss', 0):,.0f}"
        })
    st.dataframe(pd.DataFrame(open_rows), use_container_width=True, hide_index=True)
else:
    st.info("No open algo-bot trades.")

time.sleep(25)
st.rerun()
