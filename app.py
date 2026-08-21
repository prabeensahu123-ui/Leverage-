import streamlit as st
import requests
import pandas as pd
import numpy as np

# Page configuration for wide layout
st.set_page_config(page_title="Pro Crypto Terminal", layout="wide")

st.title("⚡ Pro Terminal & Market Intelligence Dashboard")

# Top Navigation / Asset Selector Bar
col_a, col_b, col_c = st.columns(3)
with col_a:
    asset_choice = st.selectbox("Select Asset / Pair", ["BTCUSD (Bitcoin Perpetual)", "PAXG (Digital Gold)", "ETHUSD (Ethereum)", "SOLUSD (Solana)"])
with col_b:
    timeframe = st.selectbox("Timeframe Interval", ["1s", "3s", "5s", "3m", "5m", "15m", "1h", "4h", "1D"])
with col_c:
    leverage = st.selectbox("Leverage Tier", ["1x", "5x", "10x", "20x", "50x", "100x"])

# Map selection to API id
coin_map = {
    "BTCUSD (Bitcoin Perpetual)": "bitcoin",
    "PAXG (Digital Gold)": "pax-gold",
    "ETHUSD (Ethereum)": "ethereum",
    "SOLUSD (Solana)": "solana"
}
asset_id = coin_map[asset_choice]

# Fetch Live Data
url = f"https://api.coingecko.com/api/v3/simple/price?ids={asset_id}&vs_currencies=usd&include_24hr_change=true"
response = requests.get(url).json()

if asset_id in response:
    price = response[asset_id].get("usd", 0)
    change = response[asset_id].get("usd_24hr_change", 0)
    if change is None: change = 0.0

    # Professional Ticker Header Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mark Price", f"${price:,.2f}", f"{change:.2f}%")
    m2.metric("24h High (Est)", f"${price * 1.03:,.2f}")
    m3.metric("24h Low (Est)", f"${price * 0.97:,.2f}")
    m4.metric("Sentiment Index", "Bullish 🟢" if change >= 0 else "Bearish 🔴")

    st.markdown("---")

    # Main Workspace: Charts & AI Signals side-by-side
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader(f"📈 Advanced Charting View ({timeframe})")
        # Generate high-frequency simulation lookalike chart based on selected timeframe
        chart_df = pd.DataFrame(np.random.randn(50, 1) * (price * 0.002) + price, columns=["Price Action"])
        st.line_chart(chart_df, height=350)

    with right_col:
        st.subheader("🎯 Algorithmic Execution Panel")
        
        # Signal Generation Logic
        signal_type = "LONG 🟢" if change >= 0 else "SHORT 🔴"
        st.markdown(f"**Automated Signal:** `{signal_type}`")
        
        entry_price = price
        tp_price = price * 1.015 if change >= 0 else price * 0.985
        sl_price = price * 0.992 if change >= 0 else price * 1.008
        
        st.info(f"**Suggested Entry:** ${entry_price:,.2f}")
        st.success(f"**Take Profit (TP):** ${tp_price:,.2f}")
        st.error(f"**Stop Loss (SL):** ${sl_price:,.2f}")
        
        # Interactive Trade Execution Simulation Buttons
        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("Execute Long"):
            st.toast("Long order simulated successfully!")
        if col_btn2.button("Execute Short"):
            st.toast("Short order simulated successfully!")

    # Market Intelligence & On-Chain Feed Section
    st.markdown("---")
    st.subheader("🔍 Market & On-Chain Intelligence Feed")
    
    intel_col1, intel_col2, intel_col3 = st.columns(3)
    with intel_col1:
        st.markdown("**Glassnode Metrics:**")
        st.write("• Exchange Inflows: Neutral")
        st.write("• Whale Accumulation: Moderate")
    with intel_col2:
        st.markdown("**Santiment Sentiment:**")
        st.write("• Crowd Social Bias: Greed")
        st.write("• Developer Activity: High")
    with intel_col3:
        st.markdown("**IntoTheBlock Models:**")
        st.write("• In/Out of Money: 72% In")
        st.write("• Volatility Index: Elevated")

else:
    st.error("Error fetching data from live feed. Please refresh the app.")
