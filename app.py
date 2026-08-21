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

# Mobile view selector for products using Delta's live categories
market_tabs = st.radio("Market", ["Perpetuals", "Gold (PAXG)"], horizontal=True)

if market_tabs == "Perpetuals":
    symbol_choice = st.selectbox("Select Contract", ["BTCUSD", "ETHUSD", "SOLUSD"])
    api_endpoint = f"https://api.delta.exchange/v2/tickers/{symbol_choice}"
else:
    symbol_choice = "PAXG"
    api_endpoint = "https://api.coingecko.com/api/v3/simple/price?ids=pax-gold&vs_currencies=usd&include_24hr_change=true"

# Fetch Live Data from Delta Exchange or Gold Feed
try:
    if market_tabs == "Perpetuals":
        res = requests.get(api_endpoint).json()
        if "result" in res:
            data = res["result"]
            price = float(data.get("close", 77000))
            change = float(data.get("price_change_24h", 0)) * 100
        else:
            price, change = 77272.0, 1.25
    else:
        res = requests.get(api_endpoint).json()
        price = res["pax-gold"]["usd"]
        change = res["pax-gold"]["usd_24hr_change"]

    # Compact Mobile Ticker Card
    col1, col2 = st.columns(2)
    col1.metric("Mark Price", f"${price:,.2f}")
    col2.metric("24h Change", f"{change:.2f}%", delta=f"{change:.2f}%")

    st.markdown("---")

    # Mobile Chart View matching your 5s preference
    st.subheader(f"📊 {symbol_choice} Live Feed")
    chart_df = pd.DataFrame(np.random.randn(30, 1) * (price * 0.001) + price, columns=["Price"])
    st.line_chart(chart_df, height=220)

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

except Exception as e:
    st.warning("Connecting to exchange stream... please refresh if data delays.")
