import streamlit as st
import requests

st.title("🚀 Live Crypto Tracker")

# Fetch live Bitcoin price from CoinGecko API
url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"
response = requests.get(url).json()

if "bitcoin" in response:
    btc_price = response["bitcoin"].get("usd", 0)
    btc_change = response["bitcoin"].get("usd_24hr_change", 0)
    
    # Fallback if change is None
    if btc_change is None:
        btc_change = 0.0

    st.metric(label="Bitcoin Price (USD)", value=f"${btc_price:,.2f}", delta=f"{btc_change:.2f}%")
else:
    st.write("Could not fetch live data at the moment. Please refresh!")
