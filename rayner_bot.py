import streamlit as st
from datetime import datetime
import base64

# =========================
# 1️⃣ BACKGROUND IMAGE
# =========================
def set_background(base64_image):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load your bull/bear background image in base64
with open("bull_bear.png", "rb") as image_file:
    base64_img = base64.b64encode(image_file.read()).decode()
set_background(base64_img)

# =========================
# 2️⃣ MARKET SYMBOLS
# =========================
MARKET_SYMBOLS = {
    # Forex
    "EUR/USD": ("OANDA", "EURUSD"),
    "GBP/JPY": ("OANDA", "GBPJPY"),
    "USD/JPY": ("OANDA", "USDJPY"),
    "AUD/USD": ("OANDA", "AUDUSD"),
    "GBP/USD": ("OANDA", "GBPUSD"),
    "USD/CAD": ("OANDA", "USDCAD"),
    "USD/CHF": ("OANDA", "USDCHF"),
    "NZD/USD": ("OANDA", "NZDUSD"),

    # Commodities
    "XAU/USD": ("OANDA", "XAUUSD"),
    "XAG/USD": ("OANDA", "XAGUSD"),
    "WTI Crude": ("OANDA", "WTICOUSD"),
    "Brent Crude": ("OANDA", "BCOUSD"),

    # Crypto
    "BTC/USD": ("BINANCE", "BTCUSDT"),
    "ETH/USD": ("BINANCE", "ETHUSDT"),

    # Indian Indices
    "NIFTY 50": ("NSE", "NIFTY"),
    "BANKNIFTY": ("NSE", "BANKNIFTY"),
    "SENSEX": ("BSE", "SENSEX")
}

# =========================
# 3️⃣ APP TITLE
# =========================
st.markdown("<h1 style='text-align: center;'>📊 Live Trading Bot</h1>", unsafe_allow_html=True)

# =========================
# 4️⃣ MARKET SELECTION
# =========================
selected_market = st.selectbox("Select Market", list(MARKET_SYMBOLS.keys()))

# =========================
# 5️⃣ TRADINGVIEW CHART
# =========================
market_info = MARKET_SYMBOLS.get(selected_market)

if not market_info:
    st.error(f"⚠ Market '{selected_market}' not found in dictionary.")
else:
    exchange, ticker = market_info

    # Embed TradingView chart
    tradingview_embed = f"""
        <iframe src="https://s.tradingview.com/widgetembed/?symbol={exchange}:{ticker}&interval=1&hidesidetoolbar=1&theme=dark&style=1&locale=en"
        width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
    """
    st.markdown(tradingview_embed, unsafe_allow_html=True)

# =========================
# 6️⃣ SIGNAL GENERATION (Placeholder)
# =========================
st.markdown("### 📈 Signal Analysis")
st.info("This is where your RSI + EMA + MACD + Candlestick + Risk Management strategy will be applied.")

# Example signal output (Dummy)
st.success(f"✅ Example Signal: {selected_market} — BUY at {datetime.now().strftime('%H:%M:%S')} with SL 50 pips, TP 100 pips, R:R = 1:2")

# =========================
# 7️⃣ NO CRASH ON MISSING MARKET
# =========================
# This is handled via `.get()` above
