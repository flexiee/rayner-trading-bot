import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tradingview_ta import TA_Handler, Interval, Exchange
from datetime import datetime, timedelta

# --- Utility Functions ---

def fetch_price_yahoo(symbol, tf="1m", period="5d"):
    df = yf.download(tickers=symbol, period=period, interval=tf)
    return df

def get_tradingview_signal(symbol, screener, exchange, tf):
    handler = TA_Handler(
        symbol=symbol,
        screener=screener,
        exchange=exchange,
        interval=getattr(Interval, tf)
    )
    analysis = handler.get_analysis()
    return analysis

def calculate_indicators(df):
    # EMA 50/200
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
    # ATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    return df

def generate_signal(df):
    # Basic: EMA, RSI, MACD, Bollinger, ATR confluence for BUY/SELL
    # EMA trend
    last = df.iloc[-1]
    ema_bull = last['EMA50'] > last['EMA200']
    ema_bear = last['EMA50'] < last['EMA200']
    # RSI
    rsi_bull = last['RSI'] > 55
    rsi_bear = last['RSI'] < 45
    # MACD
    macd_bull = last['MACD'] > last['MACD_signal']
    macd_bear = last['MACD'] < last['MACD_signal']
    # Bollinger Band
    bb_buy = last['Close'] < last['BB_Lower']
    bb_sell = last['Close'] > last['BB_Upper']
    # Signal voting
    buy_votes = sum([ema_bull, rsi_bull, macd_bull, bb_buy])
    sell_votes = sum([ema_bear, rsi_bear, macd_bear, bb_sell])
    if buy_votes >= 3:
        return "BUY", buy_votes
    elif sell_votes >= 3:
        return "SELL", sell_votes
    else:
        return "WAIT", max(buy_votes, sell_votes)

def calc_position(account_balance, risk_pct, sl_pips, pip_value):
    # Risk = account_balance * risk_pct
    # position_size = risk_amt / (sl_pips * pip_value)
    risk_amt = account_balance * (risk_pct/100)
    lots = risk_amt / (sl_pips * pip_value)
    return round(lots, 2)

# --- Streamlit UI ---

st.title("Pro Trading Signals Bot")

markets = {
    "EURUSD": {"symbol":"EURUSD", "exchange":"FX_IDC", "screener":"forex", "pip":0.0001},
    "XAUUSD": {"symbol":"XAUUSD", "exchange":"OANDA", "screener":"forex", "pip":0.1},
    "BTCUSD": {"symbol":"BTCUSD", "exchange":"BINANCE", "screener":"crypto", "pip":1}
}
selected = st.selectbox("Select Market", list(markets.keys()))
tf = st.selectbox("Select Timeframe", ["INTERVAL_1_MINUTE", "INTERVAL_5_MINUTES", "INTERVAL_15_MINUTES", "INTERVAL_1_HOUR"])
acc_bal = st.number_input("Account Balance ($)", value=1000)
risk_pct = st.slider("Risk per trade (%)", 0.5, 2.0, 1.0, 0.1)

if st.button("Generate Signal"):
    m = markets[selected]
    df = fetch_price_yahoo(m["symbol"]+"=X" if "forex" in m["screener"] else m["symbol"], period="10d", tf="1m")
    df = calculate_indicators(df)
    signal, votes = generate_signal(df)
    entry = float(df.iloc[-1]['Close'])
    atr = float(df.iloc[-1]['ATR']) if not np.isnan(df.iloc[-1]['ATR']) else m["pip"]*10
    sl = round(entry - atr*0.3, 5) if signal=="BUY" else round(entry + atr*0.3, 5)
    tp = round(entry + (entry-sl)*3, 5) if signal=="BUY" else round(entry - (sl-entry)*3, 5)
    pip_value = m["pip"]*10000 if "forex" in m["screener"] else m["pip"]
    sl_pips = abs(entry-sl)/m["pip"]
    lots = calc_position(acc_bal, risk_pct, sl_pips, pip_value)
    conf = int((votes/4)*100)
    st.metric(label="Signal", value=signal)
    st.metric(label="Entry Price", value=entry)
    st.metric(label="Stop Loss", value=sl)
    st.metric(label="Take Profit", value=tp)
    st.metric(label="R:R", value="1:3")
    st.metric(label="Confidence %", value=conf)
    st.metric(label="Recommended Lot Size", value=lots)
    # Save to history - Could write to a .csv or SQL db for review/journal

st.info("This is a professional trading assistant, not a guarantee of profits. Apply smart risk management always.")

# -- Optionally add Telegram alerts and full backtest module per requirements (expand as desired)
