import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from urllib.parse import quote

st.set_page_config(page_title="Fenil Pro Trading Bot", layout="wide")

MARKET_SYMBOLS = {
    "Forex": {
        "EUR/USD": "EURUSD=X",
        "GBP/JPY": "GBPJPY=X",
        "USD/JPY": "JPY=X",
        "AUD/USD": "AUDUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/CHF": "CHF=X",
        "USD/CAD": "CAD=X"
    }
}

# TradingView-compatible symbols
TRADINGVIEW_SYMBOLS = {
    "EUR/USD": "OANDA:EURUSD",
    "GBP/JPY": "OANDA:GBPJPY",
    "USD/JPY": "OANDA:USDJPY",
    "AUD/USD": "OANDA:AUDUSD",
    "GBP/USD": "OANDA:GBPUSD",
    "USD/CHF": "OANDA:USDCHF",
    "USD/CAD": "OANDA:USDCAD"
}

def map_interval_to_period(interval):
    return {
        "1m": "1d", "5m": "5d", "15m": "1mo",
        "30m": "1mo", "1h": "3mo", "1d": "6mo",
    }.get(interval, "1mo")

def map_interval_to_tv(interval):
    return {
        "1m": "1", "5m": "5", "15m": "15",
        "30m": "30", "1h": "60", "1d": "D",
    }.get(interval, "15")

def normalize_ohlc(df):
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c != ""]).lower() for col in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    def pick(starts_with):
        for c in df.columns:
            if c.startswith(starts_with):
                return c
        return None

    o_col, h_col, l_col, c_col = pick("open"), pick("high"), pick("low"), pick("close")
    v_col = pick("volume")

    out = pd.DataFrame({
        "date": df.index,
        "open": df[o_col], "high": df[h_col], "low": df[l_col], "close": df[c_col]
    })
    if v_col: out["volume"] = df[v_col]
    return out.reset_index(drop=True).dropna()

def get_data(symbol, interval):
    period = map_interval_to_period(interval)
    raw = yf.download(symbol, period=period, interval=interval, progress=False)
    if raw.empty: return pd.DataFrame()
    return normalize_ohlc(raw)

def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def generate_signal(df):
    if df.empty or len(df) < 20:
        return None
    df["change"] = df["close"].pct_change()
    df["gain"] = np.where(df["change"] > 0, df["change"], 0.0)
    df["loss"] = np.where(df["change"] < 0, -df["change"], 0.0)
    avg_gain = df["gain"].rolling(14).mean()
    avg_loss = df["loss"].rolling(14).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    last_rsi = df["rsi"].iloc[-1]

    atr = calculate_atr(df).iloc[-1]
    last_price = df["close"].iloc[-1]

    if last_rsi >= 70:
        return {
            "signal": "SELL",
            "rsi": last_rsi,
            "tp": last_price - atr * 2,
            "sl": last_price + atr * 1.5
        }
    elif last_rsi <= 30:
        return {
            "signal": "BUY",
            "rsi": last_rsi,
            "tp": last_price + atr * 2,
            "sl": last_price - atr * 1.5
        }
    else:
        return {
            "signal": "HOLD",
            "rsi": last_rsi,
            "tp": None,
            "sl": None
        }

# Sidebar
st.sidebar.header("Market Selection")
market_type = st.sidebar.selectbox("Market Category", list(MARKET_SYMBOLS.keys()))
symbol_name = st.sidebar.selectbox("Select Market", list(MARKET_SYMBOLS[market_type].keys()))
interval = st.sidebar.selectbox("Interval", ["1m","5m","15m","30m","1h","1d"])
symbol = MARKET_SYMBOLS[market_type][symbol_name]

# Chart
st.subheader(f"📈 Live Chart — {symbol_name}")
tv_symbol = TRADINGVIEW_SYMBOLS.get(symbol_name, symbol_name)
tv_int = map_interval_to_tv(interval)
st.markdown(
    f"""
<iframe src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval={tv_int}&theme=dark"
width="100%" height="520" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
""",
    unsafe_allow_html=True
)

# Generate signal button
if st.button("Generate Signal"):
    df = get_data(symbol, interval)
    result = generate_signal(df)
    if result is None:
        st.warning("Not enough data to generate signal.")
    else:
        st.success(f"Signal: {result['signal']} | RSI: {result['rsi']:.2f}")
        if result["tp"] and result["sl"]:
            st.write(f"**Take Profit:** {result['tp']:.5f}")
            st.write(f"**Stop Loss:** {result['sl']:.5f}")
    if not df.empty:
        st.subheader("Recent Data")
        st.dataframe(df.tail(20))
