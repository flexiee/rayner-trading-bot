# =============================
# UNIVERSAL TRADING BOT (REAL-TIME, MULTI-MARKET) with Full Trader Strategies + Signal History & Live Signal Monitoring
# =============================

import sys
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import time

try:
    import streamlit as st
    from streamlit.components.v1 import iframe
    import plotly.graph_objs as go
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    sys.exit("Please install tvDatafeed: pip install git+https://github.com/rongardF/tvdatafeed.git")

# Initialize
if STREAMLIT_AVAILABLE:
    st.set_page_config(layout="wide", page_title="Universal Trading Bot")

# Login to TradingView
tv = TvDatafeed()

# Market Mapping
MARKET_SYMBOLS = {
    "EUR/USD": ("OANDA", "EURUSD"),
    "GBP/JPY": ("OANDA", "GBPJPY"),
    "USD/JPY": ("OANDA", "USDJPY"),
    "AUD/USD": ("OANDA", "AUDUSD"),
    "XAU/USD": ("OANDA", "XAUUSD"),
    "BTC/USD": ("BINANCE", "BTCUSDT"),
    "ETH/USD": ("BINANCE", "ETHUSDT"),
    "Gold": ("OANDA", "XAUUSD"),
    "Silver": ("OANDA", "XAGUSD"),
    "Oil WTI": ("OANDA", "WTICOUSD"),
    "NIFTY 50": ("NSE", "NIFTY"),
    "BANKNIFTY": ("NSE", "BANKNIFTY"),
}

CATEGORIES = {
    "Forex": ["EUR/USD", "GBP/JPY", "USD/JPY", "AUD/USD", "XAU/USD"],
    "Crypto": ["BTC/USD", "ETH/USD"],
    "Commodities": ["Gold", "Silver", "Oil WTI"],
    "Indices": ["NIFTY 50", "BANKNIFTY"]
}

SESSIONS = {
    "London (Forex)": {"start": 8, "end": 17, "timezone": "Europe/London"},
    "New York (US)": {"start": 8, "end": 17, "timezone": "America/New_York"},
    "Tokyo (Asia)": {"start": 9, "end": 16, "timezone": "Asia/Tokyo"},
    "Sydney (Pacific)": {"start": 9, "end": 17, "timezone": "Australia/Sydney"},
    "Crypto Peak (UTC)": {"start": 12, "end": 21, "timezone": "UTC"},
}

SIGNAL_HISTORY = []

# Session detection

def get_active_sessions():
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    active = []
    for name, info in SESSIONS.items():
        tz = pytz.timezone(info["timezone"])
        local = now_utc.astimezone(tz)
        if info["start"] <= local.hour < info["end"]:
            active.append(name)
    return active

# Volatility ranking

def get_top_moving_markets():
    market_vols = []
    for name, (exchange, symbol) in MARKET_SYMBOLS.items():
        try:
            df = tv.get_hist(symbol, exchange, interval=Interval.in_1_minute, n_bars=20)
            if df is not None and not df.empty:
                std_dev = df['high'].std()
                volatility = round(std_dev * 10000, 2)
                market_vols.append((name, volatility))
        except Exception:
            continue
    sorted_markets = sorted(market_vols, key=lambda x: x[1], reverse=True)
    return sorted_markets[:3]

# Indicators

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

def calculate_macd(prices, fast=12, slow=26, signal=9):
    prices = np.asarray(prices).flatten()
    ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean()
    ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_atr(high, low, close, period=14):
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.zeros_like(close)
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, len(close)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
    return atr

# Live Data & Strategy

def get_live_data(symbol_info):
    exchange, symbol = symbol_info
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_minute, n_bars=100)
    if df is None or df.empty:
        return None
    df['ema_20'] = calculate_ema(df['close'], 20)
    df['ema_50'] = calculate_ema(df['close'], 50)
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    df['atr'] = calculate_atr(df['high'].values, df['low'].values, df['close'].values)
    df['breakout'] = df['close'] > df['close'].rolling(window=20).max().shift(1)

    last = df.iloc[-1]
    trend = "uptrend" if last['ema_20'] > last['ema_50'] else "downtrend"
    momentum = "strong" if last['macd'] > last['macd_signal'] else "weak"
    volatility = round(df['high'].std() * 10000)
    signal_strength = min(100, max(10, volatility))

    return {
        "price": round(last['close'], 5),
        "trend": trend,
        "momentum": momentum,
        "support": round(df['low'].min(), 5),
        "resistance": round(df['high'].max(), 5),
        "volatility": volatility,
        "signal_strength": signal_strength,
        "breakout": bool(last['breakout']),
        "atr": last['atr'],
        "df": df
    }

# Enhanced Signal Logic

def generate_signal(data, account_balance):
    entry = data["price"]
    risk_amount = account_balance * 0.01
    atr = data['atr']
    sl = entry - atr if data["trend"] == "uptrend" else entry + atr
    tp = entry + (atr * 3) if data["trend"] == "uptrend" else entry - (atr * 3)
    signal = "WAIT"
    reasons = []

    if data['breakout'] and data['momentum'] == 'strong':
        signal = "BUY" if data["trend"] == "uptrend" else "SELL"
        reasons.append("Breakout with trend and momentum alignment")

    if signal in ["BUY", "SELL"]:
        df = data['df']
        df_tail = df.tail(10)
        hit_tp = (df_tail['high'] >= tp).any() if signal == "BUY" else (df_tail['low'] <= tp).any()
        hit_sl = (df_tail['low'] <= sl).any() if signal == "BUY" else (df_tail['high'] >= sl).any()
        result = {
            "signal": signal,
            "entry": round(entry, 5),
            "stop_loss": round(sl, 5),
            "take_profit": round(tp, 5),
            "confidence": data["signal_strength"],
            "trend": data['trend'],
            "momentum": data['momentum'],
            "breakout": data['breakout'],
            "reasons": reasons,
            "risk_amount": round(risk_amount, 2),
            "reward_amount": round(risk_amount * 3, 2),
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "result": "WIN" if hit_tp and not hit_sl else "LOSS" if hit_sl else "UNKNOWN"
        }
        SIGNAL_HISTORY.append(result)
        return result, df
    return None, data['df']

# Metrics

def calculate_win_rate():
    if not SIGNAL_HISTORY:
        return 0.0
    wins = sum(1 for s in SIGNAL_HISTORY if s['result'] == 'WIN')
    return round((wins / len(SIGNAL_HISTORY)) * 100, 2)

def calculate_avg_rr():
    if not SIGNAL_HISTORY:
        return 0.0
    rrs = [s['reward_amount'] / s['risk_amount'] for s in SIGNAL_HISTORY if s['risk_amount'] > 0]
    return round(np.mean(rrs), 2) if rrs else 0.0

# === STREAMLIT UI ===
if STREAMLIT_AVAILABLE:
    st.title("📈 Universal Trading Bot")
    account_balance = st.number_input("Account Balance ($)", value=1000.0)

    category = st.selectbox("Select Category", list(CATEGORIES.keys()))
    market = st.selectbox("Select Market", CATEGORIES[category])
    active_sessions = get_active_sessions()
    st.markdown(f"**🕒 Active Sessions:** {', '.join(active_sessions)}")
    top_markets = get_top_moving_markets()
    st.markdown("**🔥 Top Moving Markets:**")
    for m, v in top_markets:
        st.write(f"{m}: {v}")

    st.markdown("---")

    with st.spinner("Fetching real-time data..."):
        info = get_live_data(MARKET_SYMBOLS[market])
        if info:
            signal, df = generate_signal(info, account_balance)

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Candles"
            ))
            if signal:
                fig.add_hline(y=signal["entry"], line_color="blue", annotation_text="Entry", line_dash="dash")
                fig.add_hline(y=signal["stop_loss"], line_color="red", annotation_text="SL", line_dash="dot")
                fig.add_hline(y=signal["take_profit"], line_color="green", annotation_text="TP", line_dash="dot")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            if signal:
                st.subheader(f"📢 Signal: {signal['signal']}")
                st.write(f"**Entry:** {signal['entry']}")
                st.write(f"**Stop Loss:** {signal['stop_loss']}")
                st.write(f"**Take Profit:** {signal['take_profit']}")
                st.write(f"**Trend:** {signal['trend']}")
                st.write(f"**Momentum:** {signal['momentum']}")
                st.write(f"**Confidence:** {signal['confidence']}%")
                st.write(f"**Result (Simulated):** {signal['result']}")
                st.write(f"**Reasons:** {', '.join(signal['reasons'])}")
            else:
                st.info("No trade signal at the moment. Waiting for valid breakout and trend alignment.")

            st.markdown("---")
            st.subheader("📊 Performance Metrics")
            st.write(f"**Win Rate:** {calculate_win_rate()}%")
            st.write(f"**Avg Risk/Reward:** {calculate_avg_rr()}")

            if SIGNAL_HISTORY:
                st.markdown("---")
                st.subheader("📜 Signal History")
                st.dataframe(pd.DataFrame(SIGNAL_HISTORY[::-1]))

        else:
            st.error("⚠️ Could not fetch live data. Please try again later.")
