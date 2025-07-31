# rayner_bot.py

import streamlit as st
from tvDatafeed import TvDatafeed, Interval
from datetime import datetime
import base64
import pandas as pd

tv = TvDatafeed()

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

def get_encoded_bg():
    with open("13812.png", "rb") as img:
        return base64.b64encode(img.read()).decode()

def get_live_data(exchange, symbol):
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_minute, n_bars=20)
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = round(last['close'], 5)
    prev_price = round(prev['close'], 5)
    support = round(df['low'].min(), 5)
    resistance = round(df['high'].max(), 5)
    momentum = "strong" if abs(price - prev_price) > 0.0008 else "weak"
    volatility = round(df['high'].std() * 10000)
    trend = "uptrend" if price > df['close'].rolling(5).mean().iloc[-1] else "downtrend"
    return {
        "price": price,
        "trend": trend,
        "support": support,
        "resistance": resistance,
        "momentum": momentum,
        "volatility": volatility,
        "signal_strength": min(100, max(10, volatility))
    }

def generate_signal(data, account_balance):
    entry = data["price"]
    risk_amount = account_balance * 0.01
    pip_value = 10 if data["volatility"] < 100 else 20
    pip_risk = risk_amount / pip_value
    sl, tp = None, None
    signal = "WAIT"
    reasons = []

    if data["trend"] == "uptrend" and data["momentum"] == "strong":
        sl = entry - pip_risk
        tp = entry + (entry - sl) * 3
        signal = "BUY"
        reasons.append("Strong uptrend confirmed")
    elif data["trend"] == "downtrend" and data["momentum"] == "strong":
        sl = entry + pip_risk
        tp = entry - (sl - entry) * 3
        signal = "SELL"
        reasons.append("Strong downtrend confirmed")

    rr_ratio = round(abs((tp - entry) / (entry - sl)), 2) if sl and tp else None
    return {
        "signal": signal,
        "entry": round(entry, 5),
        "stop_loss": round(sl, 5) if sl else None,
        "take_profit": round(tp, 5) if tp else None,
        "confidence": data["signal_strength"],
        "risk_amount": round(risk_amount, 2),
        "reward_amount": round(risk_amount * 3, 2),
        "lot_size": round(risk_amount / 10, 2),
        "rr_ratio": rr_ratio,
        "reasons": reasons
    }

def save_to_history(market, signal_info, result="PENDING"):
    filename = "signal_history.csv"
    record = {
        "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "market": market,
        "signal": signal_info["signal"],
        "entry": signal_info["entry"],
        "sl": signal_info["stop_loss"],
        "tp": signal_info["take_profit"],
        "r:r": signal_info["rr_ratio"],
        "lot": signal_info["lot_size"],
        "confidence": signal_info["confidence"],
        "result": result
    }
    try:
        df = pd.read_csv(filename)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    except:
        df = pd.DataFrame([record])
    df.to_csv(filename, index=False)

def main():
    st.set_page_config(layout="wide", page_title="📊 Rayner Trading Bot")
    bg = get_encoded_bg()
    st.markdown(f"""<style>.stApp {{
        background-image: url("data:image/png;base64,{bg}");
        background-size: cover; background-position: center; }}</style>""", unsafe_allow_html=True)

    st.title("⚡ Pro Trading Bot with Alerts & Risk Management")

    account_balance = st.number_input("💰 Enter Account Balance ($)", min_value=10, value=1000)
    market = st.selectbox("📈 Select Market", list(MARKET_SYMBOLS.keys()))
    if st.button("🔄 Generate Signal"):
        exch, sym = MARKET_SYMBOLS[market]
        data = get_live_data(exch, sym)
        if data:
            signal = generate_signal(data, account_balance)
            save_to_history(market, signal)
            st.subheader("✅ Signal")
            st.write(signal)
            if signal["rr_ratio"] in [6, 9] and signal["confidence"] >= 100:
                st.success(f"🚨 High R:R Signal ({signal['rr_ratio']}:1) with 110% confirmation!")
        else:
            st.error("⚠ Failed to fetch live data")

    if st.checkbox("📜 Show Signal History"):
        try:
            df = pd.read_csv("signal_history.csv")
            st.dataframe(df)
        except:
            st.info("No signal history found.")

if __name__ == "__main__":
    main()
