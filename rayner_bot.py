import streamlit as st
from streamlit.components.v1 import iframe
from tvDatafeed import TvDatafeed, Interval
from datetime import datetime
import pandas as pd
import base64
import json
import os

# === Embedded Background Image ===
bg_base64 = "iVBORw0KGgoAAAANSUhEUgAAAyAAAAHCCAIAAACYATqfAADjMUlEQVR4nOzdZVwU3fsw8JnZpRtESkApFRBssRO9VWxFsbsDsW67"  # Truncated
bg_style = f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{bg_base64}");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    background-repeat: no-repeat;
    color: white;
}}
</style>
"""
st.markdown(bg_style, unsafe_allow_html=True)

# === Symbol Configuration ===
MARKET_SYMBOLS = {{
    "EUR/USD": ("OANDA", "EURUSD"),
    "GBP/JPY": ("OANDA", "GBPJPY"),
    "USD/JPY": ("OANDA", "USDJPY"),
    "AUD/USD": ("OANDA", "AUDUSD"),
    "XAU/USD": ("OANDA", "XAUUSD"),
    "BTC/USD": ("BINANCE", "BTCUSDT"),
    "ETH/USD": ("BINANCE", "ETHUSDT"),
    "Oil": ("OANDA", "WTICOUSD")
}}

tv = TvDatafeed()

# === Historical Signal Logging ===
history_path = "signal_history.json"
if not os.path.exists(history_path):
    with open(history_path, "w") as f:
        json.dump([], f)

def log_signal(signal_data):
    with open(history_path, "r+") as f:
        data = json.load(f)
        data.append(signal_data)
        f.seek(0)
        json.dump(data[-100:], f, indent=4)  # Keep only last 100

# === Signal Calculation ===
def get_signal(symbol_info, balance):
    exch, sym = symbol_info
    df = tv.get_hist(sym, exch, interval=Interval.in_1_minute, n_bars=30)
    if df is None or df.empty: return None
    price = df.close.iloc[-1]
    mean = df.close.rolling(5).mean().iloc[-1]
    std = df.close.rolling(5).std().iloc[-1]
    momentum = abs(df.close.iloc[-1] - df.close.iloc[-2])
    trend = "UP" if price > mean else "DOWN"
    volatility = round(std * 10000)
    risk = balance * 0.01
    pip_value = 10
    pip_risk = risk / pip_value
    sl = price - pip_risk if trend == "UP" else price + pip_risk
    tp = price + (price - sl) * 3 if trend == "UP" else price - (sl - price) * 3
    confidence = min(100, max(50, int(momentum * 10000)))
    rr_ratio = abs(tp - price) / abs(price - sl)
    signal = "BUY" if trend == "UP" and momentum > 0.0008 else "SELL" if trend == "DOWN" and momentum > 0.0008 else "WAIT"
    lot_size = round((risk / (abs(price - sl) * pip_value)), 2)

    return {{
        "price": round(price, 5),
        "signal": signal,
        "trend": trend,
        "confidence": confidence,
        "volatility": volatility,
        "momentum": round(momentum, 5),
        "stop_loss": round(sl, 5),
        "take_profit": round(tp, 5),
        "risk_amount": round(risk, 2),
        "reward_amount": round(abs(tp - price) * pip_value, 2),
        "rr_ratio": round(rr_ratio, 2),
        "lot_size": lot_size
    }}

# === UI ===
st.set_page_config(layout="wide")
st.title("📈 Pro Trading Bot")

col1, col2 = st.columns([3, 1])
with col1:
    market = st.selectbox("Select Market", list(MARKET_SYMBOLS.keys()))
    iframe(f"https://s.tradingview.com/widgetembed/?symbol={MARKET_SYMBOLS[market][0]}:{MARKET_SYMBOLS[market][1]}&interval=1&theme=dark", height=420)

with col2:
    balance = st.number_input("Enter account balance ($)", value=1000)
    if st.button("Generate Signal"):
        result = get_signal(MARKET_SYMBOLS[market], balance)
        if result:
            st.subheader(f"Signal for {market}")
            st.markdown(f"**Signal:** {result['signal']}")
            st.markdown(f"**Trend:** {result['trend']}  |  **Confidence:** {result['confidence']}%")
            st.markdown(f"**Volatility:** {result['volatility']}  |  **Momentum:** {result['momentum']}")
            st.markdown(f"**Stop Loss:** {result['stop_loss']}  |  **Take Profit:** {result['take_profit']}")
            st.markdown(f"**R:R Ratio:** {result['rr_ratio']}  |  **Lot Size:** {result['lot_size']} lots")
            st.progress(result['confidence'])

            log_signal({{
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "market": market,
                "signal": result['signal'],
                "confidence": result['confidence'],
                "rr_ratio": result['rr_ratio'],
                "result": "Pending"
            }})

            if result['rr_ratio'] >= 6 and result['confidence'] >= 90:
                st.success("🚨 High R:R Signal Detected (≥ 1:6)")
            if result['rr_ratio'] >= 9 and result['confidence'] >= 100:
                st.success("🔥🔥 Massive R:R (≥ 1:9) with Full Confidence")

with st.expander("📜 Signal History"):
    if os.path.exists(history_path):
        with open(history_path) as f:
            hist = json.load(f)
            df = pd.DataFrame(hist)
            st.dataframe(df[::-1])
        
