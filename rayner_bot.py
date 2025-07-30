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
    prices = pd.Series(prices).squeeze()
    return prices.ewm(span=period, adjust=False).mean().values

def calculate_rsi(prices, period=14):
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
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.zeros_like(close)
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, len(close)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
    return atr

# =============================
# STRATEGY ENGINE
# =============================
def enhanced_rr_strategy(df, risk_percent=1.0, min_vol=1.5):
    df['ema_20'] = calculate_ema(df['close'].values, 20)
    df['ema_50'] = calculate_ema(df['close'].values, 50)
    df['rsi'] = calculate_rsi(df['close'].values)
    df['atr'] = calculate_atr(df['high'].values, df['low'].values, df['close'].values)

    df['volume_ma'] = df['volume'].rolling(20).mean().fillna(0)
    volume_condition = df['volume'] > df['volume_ma'] * 1.2
    trend_condition = df['ema_20'] > df['ema_50']
    rsi_condition = (df['rsi'] > 45) & (df['rsi'] < 65) & (df['rsi'].diff() > 0)
    movement_condition = (df['atr'] / df['close']) * 100 > min_vol

    df['entry_signal'] = trend_condition & rsi_condition & volume_condition & movement_condition
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
# DATA FETCHING
# =============================
def fetch_market_data(market_type, symbol, interval='1h'):
    end = datetime.now()
    start = end - timedelta(days=15)
    if market_type == "Crypto":
        symbol = symbol.replace("/", "-")
    elif market_type == "Forex":
        symbol = symbol.replace("/", "") + "=X"
    elif market_type == "Commodities":
        mapping = {"Gold": "GC=F", "Silver": "SI=F", "Oil": "CL=F", "Natural Gas": "NG=F", "Copper": "HG=F"}
        symbol = mapping.get(symbol, symbol)
    elif market_type == "Indices":
        mapping = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI", "FTSE 100": "^FTSE", "DAX": "^GDAXI"}
        symbol = mapping.get(symbol, symbol)
    df = yf.download(symbol, start=start, end=end, interval=interval)
    df = df.rename(columns=str.lower)
    return df

# =============================
# TOP MOVERS
# =============================
def get_top_movers(market_type, symbols, interval="1h"):
    movers = []
    for sym in symbols:
        try:
            df = fetch_market_data(market_type, sym, interval)
            if df is not None and not df.empty:
                open_price = df['open'].iloc[0]
                close_price = df['close'].iloc[-1]
                if isinstance(open_price, (float, int)) and isinstance(close_price, (float, int)):
                    pct = (close_price - open_price) / open_price * 100
                    movers.append((sym, round(pct, 2)))
        except Exception as e:
            print(f"Error processing {sym}: {e}")
    movers.sort(key=lambda x: abs(x[1]), reverse=True)
    return movers[:5]

# =============================
# STREAMLIT APP
# =============================
def main():
    st.set_page_config("Universal Trading Bot", layout="wide")
    st.title("📊 Universal Trading Bot (Live)")

    market_options = {
        "Crypto": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "Stocks": ["AAPL", "MSFT", "TSLA"],
        "Forex": ["EUR/USD", "USD/JPY", "GBP/USD"],
        "Commodities": ["Gold", "Silver", "Oil"],
        "Indices": ["S&P 500", "NASDAQ", "DAX"]
    }

    with st.sidebar:
        market_type = st.selectbox("Market Type", list(market_options.keys()))
        symbol = st.selectbox("Symbol", market_options[market_type])
        interval = st.selectbox("Interval", ["15m", "1h", "4h", "1d"], index=1)
        risk_percent = st.slider("Risk per Trade (%)", 0.1, 10.0, 1.0, 0.1)
        min_volatility = st.slider("Min ATR%", 0.5, 5.0, 1.5, 0.1)
        generate_signal = st.button("🔍 Generate Signal")

    st.markdown("### 🚀 Top Movers")
    movers = get_top_movers(market_type, market_options[market_type], interval)
    for m in movers:
        st.write(f"**{m[0]}**: {m[1]}%")

    if generate_signal:
        df = fetch_market_data(market_type, symbol, interval)
        if df is None or df.empty:
            st.error("No data available.")
            return

        results = enhanced_rr_strategy(df.copy(), risk_percent, min_volatility)
        last = results.iloc[-1]

        st.header(f"📈 {symbol} | {interval} | Price: ${last['close']:.2f}")
        st.metric("ATR %", f"{(last['atr']/last['close'])*100:.2f}%")
        st.metric("Entry Signal", "✅" if last['entry_signal'] else "❌")

        st.subheader("📜 Trade History")
        trade_log = results[results['entry_signal']][['close', 'entry_price', 'stop_loss', 'tp_1', 'trade_result']].dropna()
        st.dataframe(trade_log.tail(10).style.format({'close': '${:.2f}', 'entry_price': '${:.2f}'}))

        st.subheader("📊 Live Chart")
        tv_symbol = symbol.replace("/", "") if market_type == "Forex" else symbol.replace("/", "") if market_type == "Crypto" else symbol
        st.components.v1.iframe(f"https://s.tradingview.com/widgetembed/?frameElementId=tradingview_{tv_symbol}&symbol={tv_symbol}&interval=60&hidesidetoolbar=1&symboledit=1&saveimage=1&toolbarbg=f1f3f6&studies=[]&theme=dark&style=1&timezone=Etc%2FUTC", height=500)

if __name__ == '__main__':
    main()
