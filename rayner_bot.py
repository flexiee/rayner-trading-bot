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

def generate_signal_with_tp_sl(df, rr_ratio=2, sl_pips=20):
    """
    rr_ratio = Risk:Reward ratio (e.g., 2 means TP is twice the SL distance)
    sl_pips = Stop Loss in pips
    """
    if df.empty:
        return "No data", None, None

    latest = df.iloc[-1]
    entry_price = latest['Close']

    if latest['EMA20'] > latest['EMA50'] and latest['RSI'] < 70:
        direction = "BUY"
        sl = entry_price - (sl_pips * pip_value(entry_price))
        tp = entry_price + (sl_pips * rr_ratio * pip_value(entry_price))
    elif latest['EMA20'] < latest['EMA50'] and latest['RSI'] > 30:
        direction = "SELL"
        sl = entry_price + (sl_pips * pip_value(entry_price))
        tp = entry_price - (sl_pips * rr_ratio * pip_value(entry_price))
    else:
        direction = "HOLD"
        sl, tp = None, None

    return direction, sl, tp

def pip_value(price):
    """Rough pip value calc for Forex pairs"""
    if price < 10:  # JPY pairs
        return 0.01
    return 0.0001

# ==============================
# UI Layout
# ==============================
st.title("📈 Rayner Bot - Trading Assistant (with TP/SL)")
st.markdown("Get real-time trading signals with **EMA + RSI** strategy and automatic TP/SL calculation.")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    symbol = st.text_input("Enter Market Symbol", "EURUSD=X")
with col2:
    interval = st.selectbox("Time Interval", ["1m", "5m", "15m", "30m", "1h", "1d"])
with col3:
    period = st.selectbox("Data Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])
with col4:
    rr_ratio = st.number_input("Risk-Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
with col5:
    sl_pips = st.number_input("Stop Loss (pips)", min_value=5, max_value=200, value=20, step=5)

# ==============================
# Signal Generation
# ==============================
if st.button("Generate Signal"):
    df = get_data(symbol, interval, period)
    
    if not df.empty:
        df = calculate_indicators(df)
        signal, sl, tp = generate_signal_with_tp_sl(df, rr_ratio, sl_pips)
        
        # Display Signal & TP/SL
        st.subheader(f"📊 Signal: **{signal}**")
        if signal != "HOLD":
            st.write(f"**Entry Price:** {df.iloc[-1]['Close']:.5f}")
            st.write(f"**Stop Loss:** {sl:.5f}")
            st.write(f"**Take Profit:** {tp:.5f}")
            st.write(f"**Risk-Reward Ratio:** {rr_ratio}:1")
        
        # Show Last 5 Data Points
        st.dataframe(df.tail(5))
        
        # Plot Chart
        st.line_chart(df[['Close', 'EMA20', 'EMA50']])
    else:
        st.error("No data available. Please check the symbol or parameters.")
