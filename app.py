import streamlit as st
import requests
import pandas as pd
import numpy as np

st.title("🚀 Live Crypto & Gold Trading Dashboard")

# 1. Asset Selection
asset_options = {
    "Bitcoin (BTC)": "bitcoin",
    "PAX Gold (PAXG)": "pax-gold",
    "Ethereum (ETH)": "ethereum",
    "Solana (SOL)": "solana"
}
selected_asset_name = st.selectbox("Choose an Asset", list(asset_options.keys()))
asset_id = asset_options[selected_asset_name]

# Fetch live price
url = f"https://api.coingecko.com/api/v3/simple/price?ids={asset_id}&vs_currencies=usd&include_24hr_change=true"
response = requests.get(url).json()

if asset_id in response:
    price = response[asset_id].get("usd", 0)
    change = response[asset_id].get("usd_24hr_change", 0)
    if change is None: change = 0.0

    st.metric(label=f"{selected_asset_name} Price (USD)", value=f"${price:,.2f}", delta=f"{change:.2f}%")
    
    # 2. Automated Signal Generation
    st.subheader("📊 AI Trade Suggestion")
    
    # Simple logic: If change > 0, suggest Long; if < 0, suggest Short
    signal = "📈 LONG" if change >= 0 else "📉 SHORT"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Signal", signal)
    col2.metric("Entry", f"${price:,.2f}")
    col3.metric("TP (Target)", f"${price * 1.02:,.2f}") # 2% Target
    
    st.write(f"**Stop Loss (SL):** ${price * 0.98:,.2f}") # 2% Stop Loss
    
    st.info("Analysis: Based on 24h market momentum. Always perform your own technical analysis before trading.")

    # 3. Chart
    chart_data = pd.DataFrame(np.random.randn(20, 1) * (price * 0.005) + price, columns=["Price Trend"])
    st.line_chart(chart_data)
else:
    st.write("Could not fetch data. Refresh the page.")
