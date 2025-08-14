import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ==============================
# Streamlit Page Config
# ==============================
st.set_page_config(page_title="Rayner Bot - Trading Assistant", layout="wide")

# ==============================
# Functions
# ==============================
def get_data(symbol, interval, period):
    try:
        data = yf.download(symbol, period=period, interval=interval)
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_indicators(df):
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    return df

def generate_signal(df, rr_ratio=2):
    if df.empty:
        return "No data", None, None, None

    latest = df.iloc[-1]
    entry = latest['Close']
    atr = (df['High'] - df['Low']).rolling(window=14).mean().iloc[-1]  # ATR for volatility

    if latest['EMA20'] > latest['EMA50'] and latest['RSI'] < 70:
        direction = "BUY"
        sl = entry - atr
        tp = entry + atr * rr_ratio
    elif latest['EMA20'] < latest['EMA50'] and latest['RSI'] > 30:
        direction = "SELL"
        sl = entry + atr
        tp = entry - atr * rr_ratio
    else:
        direction = "HOLD"
        sl = None
        tp = None

    return direction, entry, sl, tp

# ==============================
# UI Layout
# ==============================
st.title("📈 Rayner Bot - Trading Assistant")
st.markdown("Get real-time trading signals using **EMA** and **RSI** strategy with **TP/SL** levels and TradingView live chart.")

col1, col2, col3 = st.columns(3)

with col1:
    symbol = st.text_input("Enter Market Symbol", "EURUSD=X")  # Forex example
with col2:
    interval = st.selectbox("Time Interval", ["1m", "5m", "15m", "30m", "1h", "1d"])
with col3:
    period = st.selectbox("Data Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])

# ==============================
# Signal Generation
# ==============================
if st.button("Generate Signal"):
    df = get_data(symbol, interval, period)

    if not df.empty:
        df = calculate_indicators(df)
        signal, entry, sl, tp = generate_signal(df)

        # Display Signal & TP/SL
        st.subheader(f"📊 Signal: **{signal}**")
        if signal != "HOLD":
            st.write(f"**Entry Price:** {entry:.5f}")
            st.write(f"**Stop Loss (SL):** {sl:.5f}")
            st.write(f"**Take Profit (TP):** {tp:.5f}")
            st.write(f"**Risk-Reward Ratio:** 1:2")

        # Show Last 5 Data Points
        st.dataframe(df.tail(5))

        # Plot Chart
        st.line_chart(df[['Close', 'EMA20', 'EMA50']])

        # ==============================
        # TradingView Live Chart Embed
        # ==============================
        tradingview_symbol = symbol.replace("=X", "")  # Adjust for forex
        st.markdown(f"""
            <iframe src="https://s.tradingview.com/widgetembed/?symbol={tradingview_symbol}&interval=1&hidesidetoolbar=1&symboledit=1&saveimage=1&toolbarbg=f1f3f6&studies=[]&theme=dark&style=1&timezone=Etc/UTC&studies_overrides={{}}&overrides={{}}&enabled_features=[]&disabled_features=[]"
            width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
        """, unsafe_allow_html=True)
    else:
        st.error("No data available. Please check the symbol or parameters.")
