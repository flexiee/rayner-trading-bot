# =============================
# UNIVERSAL TRADING BOT (REAL-TIME, MULTI-MARKET)
# =============================

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import yfinance as yf

# =============================
# TECHNICAL INDICATORS
# =============================
def calculate_ema(prices, period):
    prices = np.asarray(prices).flatten()
    return pd.Series(prices).ewm(span=period, adjust=False).mean().values

def calculate_rsi(prices, period=14):
    prices = np.asarray(prices).flatten()
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period=14):
    high = np.asarray(high).flatten()
    low = np.asarray(low).flatten()
    close = np.asarray(close).flatten()
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.zeros_like(close)
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, len(close)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
    return atr

def calculate_macd(prices, fast=12, slow=26, signal=9):
    prices = np.asarray(prices).flatten()
    ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean()
    ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

# =============================
# STRATEGY ENGINE
# =============================
def enhanced_rr_strategy(df, risk_percent=1.0, min_vol=1.5):
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)

    df['ema_20'] = calculate_ema(df['close'], 20)
    df['ema_50'] = calculate_ema(df['close'], 50)
    df['rsi'] = calculate_rsi(df['close'])
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])

    # Jesse Livermore-style breakout: price makes new high
    df['rolling_high'] = df['high'].rolling(20).max()
    breakout_condition = df['close'] > df['rolling_high'].shift(1)

    df['volume_ma'] = df['volume'].rolling(20).mean().fillna(0)
    volume_condition = df['volume'] > df['volume_ma'] * 1.2
    trend_condition = df['ema_20'] > df['ema_50']
    rsi_condition = (df['rsi'] > 45) & (df['rsi'] < 65) & (df['rsi'].diff() > 0)
    macd_condition = df['macd'] > df['macd_signal']
    movement_condition = (df['atr'] / df['close']) * 100 > min_vol

    df['entry_signal'] = trend_condition & rsi_condition & macd_condition & volume_condition & movement_condition & breakout_condition
    df['recent_low'] = df['low'].rolling(5).min()
    df['stop_loss'] = df['recent_low'] - (df['atr'] * 1.5)
    df['risk_distance'] = df['close'] - df['stop_loss']
    df['tp_1'] = df['close'] + df['risk_distance'] * 1.618
    df['tp_2'] = df['close'] + df['risk_distance'] * 2.618
    df['tp_3'] = df['close'] + df['risk_distance'] * 4.236
    df['position_size'] = (risk_percent / 100) / (df['risk_distance'] / df['close'])

    df['entry_price'] = df['close'].where(df['entry_signal'])
    df['trade_result'] = np.nan
    for i in range(len(df) - 5):
        if df['entry_signal'].iloc[i]:
            future = df['close'].iloc[i + 1:i + 6]
            if (future >= df['tp_1'].iloc[i]).any():
                df.at[df.index[i], 'trade_result'] = 'TP Hit ✅'
            elif (future <= df['stop_loss'].iloc[i]).any():
                df.at[df.index[i], 'trade_result'] = 'SL Hit ❌'
            else:
                df.at[df.index[i], 'trade_result'] = 'Open 🟡'
    return df

# =============================
# STREAMLIT UI + DATA FETCHING
# =============================
st.set_page_config(layout="wide")
st.title("📈 Universal Trading Bot (Live Market Signals)")

markets = {
    'BTC/USD': 'BTC-USD',
    'ETH/USD': 'ETH-USD',
    'Gold (XAU)': 'GC=F',
    'Apple': 'AAPL',
    'Nifty 50': '^NSEI',
    'Tesla': 'TSLA',
    'Crude Oil': 'CL=F'
}

selected_market = st.selectbox("Select Market", list(markets.keys()))
ticker = markets[selected_market]

st.components.v1.iframe(
    f"https://www.tradingview.com/widgetembed/?symbol={ticker}&interval=1&theme=dark",
    height=400
)

if st.button("🚀 Generate Signal"):
    end = datetime.now()
    start = end - timedelta(days=90)
    df = yf.download(ticker, start=start, end=end, interval="1h")
    df = enhanced_rr_strategy(df)

    if df['entry_signal'].iloc[-1]:
        st.success(f"✅ Signal Generated for {selected_market}")
        st.metric("Entry Price", round(df['entry_price'].iloc[-1], 2))
        st.metric("Stop Loss", round(df['stop_loss'].iloc[-1], 2))
        st.metric("Take Profit (TP1)", round(df['tp_1'].iloc[-1], 2))
        st.metric("Position Size", round(df['position_size'].iloc[-1], 2))
    else:
        st.warning(f"No valid entry signal for {selected_market}")

    st.subheader("📜 Trade History")
    trade_df = df.dropna(subset=['entry_price', 'trade_result'])
    st.dataframe(trade_df[['entry_price', 'stop_loss', 'tp_1', 'tp_2', 'tp_3', 'trade_result']].tail(10))
