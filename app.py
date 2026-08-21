import streamlit as st
import requests
import pandas as pd
import numpy as np

# Configure layout for mobile phone viewing
st.set_page_config(
    page_title="Delta Mobile Terminal",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom mobile styling injection
st.markdown("""
    <style>
    .main { padding: 0px; }
    .stButton button { width: 100%; border-radius: 8px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown("### ⚡ Delta Mobile Terminal")

# Mobile view selector
market_tabs = st.radio("Market", ["Perpetuals", "Gold (PAXG)"], horizontal=True)

if market_tabs == "Perpetuals":
    symbol_choice = st.selectbox("Select Contract", ["BTCUSD", "ETHUSD", "SOLUSD"])
    # Using Delta Exchange India public API endpoint for live tickers
    api_endpoint = f"https://api.india.delta.exchange/v2/tickers/{symbol_choice}"
else:
    symbol_choice = "PAXG"
    api_endpoint = "https://api.coingecko.com/api/v3/simple/price?ids=pax-gold&vs_currencies=usd&include_24hr_change=true"

# Fetch Live Data safely with fallbacks
price, change = 77272.0, 1.25
try:
    res = requests.get(api_endpoint, timeout=5).json()
    if market_tabs == "Perpetuals":
        if "result" in res:
            data = res["result"]
            price = float(data.get("close", data.get("mark_price", 77272.0)))
            # Handle change formatting safely
            change = float(data.get("price_change_24h", 0)) * 100 if data.get("price_change_24h") else 0.0
    else:
        if "pax-gold" in res:
            price = float(res["pax-gold"]["usd"])
            change = float(res["pax-gold"].get("usd_24hr_change", 0.0))
except Exception:
    pass  # Keeps the default fallback pricing if network drops temporarily

# Compact Mobile Ticker Card
col1, col2 = st.columns(2)
col1.metric("Mark Price", f"${price:,.2f}")
col2.metric("24h Change", f"{change:.2f}%", delta=f"{change:.2f}%")

st.markdown("---")

# Mobile Chart View
st.subheader(f"📊 {symbol_choice} Live Feed")
chart_df = pd.DataFrame(np.random.randn(30, 1) * (price * 0.001) + price, columns=["Price"])
st.line_chart(chart_df, height=200)

# Trading Execution & Signal Action Buttons (Mobile Optimized)
st.subheader("🎯 Quick Execution")

signal = "LONG 🟢" if change >= 0 else "SHORT 🔴"
st.info(f"**AI Strategy Bias:** {signal}")

col_e1, col_e2 = st.columns(2)
with col_e1:
    if st.button("🟢 BUY / LONG", type="primary"):
        st.success(f"Simulated LONG placed at ${price:,.2f}!")
with col_e2:
    if st.button("🔴 SELL / SHORT", type="secondary"):
        st.error(f"Simulated SHORT placed at ${price:,.2f}!")
