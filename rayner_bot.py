import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Fenil Pro Trading Bot", layout="wide")

# --------------------------
# Market List (Forex, Crypto, Commodities, Indian Indices)
# --------------------------
MARKET_SYMBOLS = {
    "Forex": {
        "EUR/USD": "EURUSD=X",
        "GBP/JPY": "GBPJPY=X",
        "USD/JPY": "JPY=X",
        "AUD/USD": "AUDUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/CHF": "CHF=X",
        "USD/CAD": "CAD=X"
    },
    "Crypto": {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Litecoin": "LTC-USD",
        "BNB": "BNB-USD"
    },
    "Commodities": {
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Crude Oil": "CL=F",
        "Natural Gas": "NG=F"
    },
    "Indian Indices": {
        "NIFTY 50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN"
    }
}

# --------------------------
# Data Fetch Function (Fixed)
# --------------------------
def get_data(symbol, interval):
    # Automatically pick correct period
    interval_map = {
        "1m": "1d",
        "5m": "5d",
        "15m": "1mo",
        "30m": "1mo",
        "1h": "3mo",
        "1d": "6mo"
    }
    period = interval_map.get(interval, "1mo")

    try:
        df = yf.download(symbol, period=period, interval=interval)

        # Flatten columns if MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # Lowercase column names
        df.columns = [col.lower() for col in df.columns]
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# --------------------------
# Strategy Logic (Unchanged)
# --------------------------
def generate_signal(df):
    if df.empty:
        return "No Data", 0

    # Example: RSI-based strategy
    df["change"] = df["close"].pct_change()
    df["gain"] = np.where(df["change"] > 0, df["change"], 0)
    df["loss"] = np.where(df["change"] < 0, abs(df["change"]), 0)
    avg_gain = df["gain"].rolling(window=14).mean()
    avg_loss = df["loss"].rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    last_rsi = df["rsi"].iloc[-1]
    if last_rsi > 70:
        return "SELL", last_rsi
    elif last_rsi < 30:
        return "BUY", last_rsi
    else:
        return "HOLD", last_rsi

# --------------------------
# Sidebar
# --------------------------
st.sidebar.header("Market Selection")
market_type = st.sidebar.selectbox("Market Category", list(MARKET_SYMBOLS.keys()))
symbol_name = st.sidebar.selectbox("Select Market", list(MARKET_SYMBOLS[market_type].keys()))
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d"])

symbol = MARKET_SYMBOLS[market_type][symbol_name]

# --------------------------
# Fetch Data & Generate Signal
# --------------------------
df = get_data(symbol, interval)
signal, rsi_value = generate_signal(df)

# --------------------------
# Show Live TradingView Chart
# --------------------------
st.subheader(f"📈 Live Chart - {symbol_name}")
st.markdown(f"""
<iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_widget&symbol={symbol}&interval={interval.upper()}&hidesidetoolbar=1&symboledit=1&saveimage=0&toolbarbg=f1f3f6&studies=[]&theme=dark&style=1&timezone=Etc%2FUTC&hideideas=1" 
width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
""", unsafe_allow_html=True)

# --------------------------
# Signal Output
# --------------------------
st.subheader("Trading Signal")
st.write(f"**Signal:** {signal}")
st.write(f"**RSI:** {rsi_value:.2f}")

# --------------------------
# Data Table
# --------------------------
if not df.empty:
    st.subheader("Market Data")
    st.dataframe(df.tail(20))
