import streamlit as st
import requests
import pandas as pd
import numpy as np

st.title("🚀 Live Crypto & Gold Trading Dashboard")

# 1. Asset Selection
asset_options = {
    "Bitcoin (BTC)": "bitcoin",
    "PAX Gold (PAXG - Digital Gold)": "pax-gold",
    "Ethereum (ETH)": "ethereum",
    "Solana (SOL)": "solana"
}
selected_asset_name = st.selectbox("Choose an Asset to Track", list(asset_options.keys()))
asset_id = asset_options[selected_asset_name]

# 2. Timeframe Selection
timeframe_options = [
    "1 sec", "3 sec", "5 sec", "3 min", "5 min", "10 min", 
    "15 min", "30 min", "45 min", "90 min", "1 day", 
    "1 week", "6 weeks", "1 month", "3 months"
]
selected_timeframe = st.selectbox("Select Timeframe Interval", timeframe_options)

# Fetch live price data
url = f"https://api.coingecko.com/api/v3/simple/price?ids={asset_id}&vs_currencies=usd&include_24hr_change=true"
response = requests.get(url).json()

if asset_id in response:
    price = response[asset_id].get("usd", 0)
    change = response[asset_id].get("usd_24hr_change", 0)
    if change is None:
        change = 0.0

    st.metric(label=f"{selected_asset_name} Price (USD)", value=f"${price:,.2f}", delta=f"{change:.2f}%")
    
    st.info(f"Timeframe selected: **{selected_timeframe}**. Loading performance charts for this window...")

    # Generating a mock historical trend chart for the selected timeframe view
    chart_data = pd.DataFrame(
        np.random.randn(20, 1) * (price * 0.001) + price,
        columns=["Price Trend"]
    )
    st.line_chart(chart_data)
else:
    st.write("Could not fetch live data at the moment. Please refresh!")

