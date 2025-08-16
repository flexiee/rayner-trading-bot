# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import ta

st.set_page_config(page_title="Next Level Trading Bot", layout="wide")

# ==============================
# UTILITY FUNCTIONS
# ==============================

def fetch_data(symbol="EURUSD=X", interval="5m", lookback="5d"):
    try:
        data = yf.download(tickers=symbol, interval=interval, period=lookback)
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    df["EMA20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["MACD"] = ta.trend.MACD(df["Close"]).macd()
    df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    return df

def generate_signal(df, balance=1000, risk_pct=0.01):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    signal = None
    reason = ""
    strength = 0

    # Strategy 1: EMA Cross
    if latest["EMA20"] > latest["EMA50"] and prev["EMA20"] <= prev["EMA50"]:
        signal = "BUY"
        reason += "EMA20 crossed above EMA50. "
        strength += 1
    elif latest["EMA20"] < latest["EMA50"] and prev["EMA20"] >= prev["EMA50"]:
        signal = "SELL"
        reason += "EMA20 crossed below EMA50. "
        strength += 1

    # Strategy 2: RSI
    if latest["RSI"] < 30:
        signal = "BUY"
        reason += "RSI oversold. "
        strength += 1
    elif latest["RSI"] > 70:
        signal = "SELL"
        reason += "RSI overbought. "
        strength += 1

    # Strategy 3: MACD
    if latest["MACD"] > 0:
        reason += "MACD bullish. "
        if signal == "BUY": strength += 1
    elif latest["MACD"] < 0:
        reason += "MACD bearish. "
        if signal == "SELL": strength += 1

    # Risk Management
    atr = latest["ATR"]
    entry = latest["Close"]
    if signal == "BUY":
        sl = entry - atr
        tp = entry + 2 * atr
    elif signal == "SELL":
        sl = entry + atr
        tp = entry - 2 * atr
    else:
        return None

    # Lot size calculation
    risk_amount = balance * risk_pct
    lot_size = risk_amount / atr if atr > 0 else 0

    return {
        "signal": signal,
        "entry": round(entry, 5),
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "strength": strength,
        "reason": reason.strip(),
        "lot_size": round(lot_size, 2),
        "risk_amount": round(risk_amount, 2),
        "rr": round(abs(tp - entry) / abs(entry - sl), 2)
    }

# ==============================
# STREAMLIT UI
# ==============================

st.title("📈 Next Level Trading Bot")

markets = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "XAU/USD (Gold)": "GC=F",
    "BTC/USD": "BTC-USD"
}

col1, col2 = st.columns([2, 1])

with col1:
    market = st.selectbox("Choose Market", list(markets.keys()))
    interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h"])
    balance = st.number_input("Account Balance ($)", value=1000, step=100)

    if st.button("🔮 Generate Signal"):
        df = fetch_data(markets[market], interval=interval)
        if not df.empty:
            df = calculate_indicators(df)
            signal_data = generate_signal(df, balance=balance)
            if signal_data:
                st.subheader(f"Signal for {market}")
                st.write(signal_data)

                # Show TradingView chart
                st.markdown(f"""
                <iframe src="https://s.tradingview.com/widgetembed/?symbol={markets[market]}&interval={interval}&hidesidetoolbar=1&theme=dark"
                width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
                """, unsafe_allow_html=True)
            else:
                st.warning("No signal generated.")
        else:
            st.error("Failed to load market data.")

with col2:
    st.markdown("### ℹ️ How this bot works")
    st.write("""
    - Combines **EMA Cross, RSI, MACD, ATR** strategies  
    - Uses **risk management (1% default)**  
    - Calculates **lot size, SL/TP, R:R**  
    - Shows **signal strength & reasoning**  
    - Live TradingView chart embedded
    """)
