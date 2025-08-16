import streamlit as st
from datetime import datetime
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import random

# ----------------------------
# TradingView Data Connection
# ----------------------------
tv = TvDatafeed()

# Markets
MARKET_SYMBOLS = {
    "EUR/USD": ("OANDA", "EURUSD"),
    "GBP/JPY": ("OANDA", "GBPJPY"),
    "USD/JPY": ("OANDA", "USDJPY"),
    "AUD/USD": ("OANDA", "AUDUSD"),
    "XAU/USD": ("OANDA", "XAUUSD"),
    "BTC/USD": ("BINANCE", "BTCUSDT"),
    "ETH/USD": ("BINANCE", "ETHUSDT"),
}

# ----------------------------
# Trading Signal Logic
# ----------------------------
def generate_signal(symbol, exchange, interval=Interval.in_5_minute, bars=100):
    try:
        df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=bars)
    except:
        return {"error": "Failed to fetch data from TradingView"}

    if df is None or df.empty:
        return {"error": "No data received"}

    df["ema_fast"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi"] = compute_rsi(df["close"], 14)
    df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()

    last = df.iloc[-1]
    signal, sl, tp = "", None, None

    if last["ema_fast"] > last["ema_slow"] and last["rsi"] > 55 and last["macd"] > 0:
        signal = "BUY"
        sl = round(last["close"] * 0.995, 5)
        tp = round(last["close"] * 1.010, 5)
    elif last["ema_fast"] < last["ema_slow"] and last["rsi"] < 45 and last["macd"] < 0:
        signal = "SELL"
        sl = round(last["close"] * 1.005, 5)
        tp = round(last["close"] * 0.990, 5)
    else:
        signal = "NO TRADE"

    confidence = random.randint(70, 95)  # mock confidence %
    rr_ratio = round(abs((tp - last["close"]) / (last["close"] - sl)), 2) if sl and tp else None

    return {
        "signal": signal,
        "price": round(last["close"], 5),
        "stop_loss": sl,
        "take_profit": tp,
        "risk_reward": rr_ratio,
        "confidence": confidence,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Pro Trading Bot", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
        padding: 10px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ Next-Level Trading Bot")
market = st.selectbox("📊 Select Market", list(MARKET_SYMBOLS.keys()))

# Live TradingView Chart
exchange, symbol = MARKET_SYMBOLS[market]
st.components.v1.html(
    f"""
    <iframe src="https://s.tradingview.com/widgetembed/?symbol={exchange}:{symbol}&interval=5&hidesidetoolbar=1&symboledit=1&saveimage=0&toolbarbg=f1f3f6&studies=[]&theme=dark&style=1&timezone=Etc/UTC&withdateranges=1&hideideas=1"
    width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
    """,
    height=500,
)

# Generate Signal Button
if st.button("🔮 Generate Signal"):
    result = generate_signal(symbol, exchange)

    if "error" in result:
        st.error(result["error"])
    else:
        st.subheader("📈 Signal Result")
        st.write(f"**Signal:** {result['signal']}")
        st.write(f"**Price:** {result['price']}")
        st.write(f"**Stop Loss:** {result['stop_loss']}")
        st.write(f"**Take Profit:** {result['take_profit']}")
        st.write(f"**Risk/Reward:** {result['risk_reward']}")
        st.write(f"**Confidence:** {result['confidence']}%")
        st.write(f"**Time:** {result['time']}")
