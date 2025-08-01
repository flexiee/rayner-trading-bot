import streamlit as st
import pandas as pd
import random
from datetime import datetime

st.set_page_config(page_title="📈 Signal Bot", layout="wide")
st.title("📈 Live Market Signal Bot (Pro Strategy)")

# Initialize session state
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []

# Market list
markets = ["EUR/USD", "GBP/USD", "USD/JPY", "BTC/USD", "ETH/USD", "GOLD", "OIL", "NIFTY 50", "BANKNIFTY"]
confidence_levels = [88, 95, 98, 101, 105, 110]
rr_ratios = ["1:2", "1:3", "1:6", "1:9"]

# Inputs
selected_market = st.selectbox("Select Market", markets)
account_balance = st.number_input("💰 Account Balance ($)", min_value=10, value=1000, step=10)

# Generate fake signal logic (use real data source in future)
def generate_signal(market, balance):
    confidence = random.choice(confidence_levels)
    rr = random.choice(rr_ratios)
    signal_type = random.choice(["BUY", "SELL"])
    result = random.choice(["✅ Success", "❌ Failed"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    risk_amount = round(balance * 0.01, 2)
    reward_amount = round(risk_amount * int(rr.split(":")[1]), 2)
    pip_value = 10  # Simplified assumption
    lot_size = round(risk_amount / pip_value, 2)

    signal = {
        "time": timestamp,
        "market": market,
        "signal": signal_type,
        "rr_ratio": rr,
        "confidence": f"{confidence}%",
        "result": result,
        "risk": f"${risk_amount}",
        "reward": f"${reward_amount}",
        "lot_size": lot_size
    }

    st.session_state.signal_history.append(signal)
    return signal

# Generate button
if st.button("🔄 Generate Signal"):
    signal = generate_signal(selected_market, account_balance)

    st.subheader(f"📍 Signal for {selected_market}")
    st.markdown(f"- Action: `{signal['signal']}`")
    st.markdown(f"- R:R: `{signal['rr_ratio']}` | Confidence: `{signal['confidence']}`")
    st.markdown(f"- Lot Size: `{signal['lot_size']} lot`")
    st.markdown(f"- Risk: `{signal['risk']}` | Reward: `{signal['reward']}`")
    st.markdown(f"- Result: `{signal['result']}`")
    st.markdown(f"- Time: {signal['time']}")
    st.progress(int(signal["confidence"].strip('%')))

    if signal["rr_ratio"] in ["1:6", "1:9"] and signal["confidence"] == "110%":
        st.success("🚨 High R:R with 110% Confidence! Perfect Opportunity!")
        st.balloons()

# Live chart
ex = "OANDA" if "/" in selected_market else "BINANCE"
symbol = selected_market.replace("/", "")
symbol = "XAUUSD" if selected_market == "GOLD" else symbol
symbol = "XAGUSD" if selected_market == "SILVER" else symbol
symbol = "BTCUSDT" if selected_market == "BTC/USD" else symbol
symbol = "ETHUSDT" if selected_market == "ETH/USD" else symbol
st.subheader("📺 Live Market Chart")
st.components.v1.iframe(
    f"https://s.tradingview.com/widgetembed/?symbol={ex}:{symbol}&interval=1&theme=dark",
    height=400
)

# Signal history
if st.checkbox("📜 Show Signal History"):
    df = pd.DataFrame(st.session_state.signal_history)
    st.dataframe(df[::-1], use_container_width=True)
