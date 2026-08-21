import streamlit as st
import requests

st.title("🚀 Live Crypto & Gold Tracker")

# Dropdown to select coin / asset (using correct CoinGecko IDs)
asset_options = {
    "Bitcoin (BTC)": "bitcoin",
    "PAX Gold (PAXG - Digital Gold)": "pax-gold",
    "Ethereum (ETH)": "ethereum",
    "Solana (SOL)": "solana"
}

selected_asset_name = st.selectbox("Choose an Asset to Track", list(asset_options.keys()))
asset_id = asset_options[selected_asset_name]

# Fetch live price from CoinGecko API
url = f"https://api.coingecko.com/api/v3/simple/price?ids={asset_id}&vs_currencies=usd&include_24hr_change=true"
response = requests.get(url).json()

if asset_id in response:
    price = response[asset_id].get("usd", 0)
    change = response[asset_id].get("usd_24hr_change", 0)
    
    if change is None:
        change = 0.0

    st.metric(label=f"{selected_asset_name} Price (USD)", value=f"${price:,.2f}", delta=f"{change:.2f}%")
else:
    st.write("Could not fetch live data at the moment. Please refresh!")
