import sys
import time
from datetime import datetime
import streamlit as st
from streamlit.components.v1 import iframe
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from tvDatafeed import TvDatafeed, Interval

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

CATEGORIES = {
    "Forex": ["EUR/USD", "GBP/JPY", "USD/JPY", "AUD/USD", "XAU/USD"],
    "Crypto": ["BTC/USD", "ETH/USD"],
    "Commodities": ["Gold", "Silver", "Oil WTI"],
    "Indices": ["NIFTY 50", "BANKNIFTY"]
}

def get_live_data(symbol_info):
    exchange, symbol = symbol_info
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
    pip_value = 10
    pip_risk = risk_amount / pip_value
    sl, tp = None, None
    signal = "WAIT"
    reasons = []

    if data["trend"] == "uptrend" and entry > data["support"]:
        if data["momentum"] == "strong" and data["volatility"] > 50:
            sl = entry - pip_risk
            tp = entry + (entry - sl) * 3
            signal = "BUY"
            reasons.append("Strong uptrend breakout")
    elif data["trend"] == "downtrend" and entry < data["resistance"]:
        if data["momentum"] == "strong" and data["volatility"] > 50:
            sl = entry + pip_risk
            tp = entry - (sl - entry) * 3
            signal = "SELL"
            reasons.append("Strong downtrend breakout")

    return {
        "signal": signal,
        "entry": round(entry, 5),
        "stop_loss": round(sl, 5) if sl else None,
        "take_profit": round(tp, 5) if tp else None,
        "confidence": data["signal_strength"],
        "reasons": reasons,
        "risk_amount": round(risk_amount, 2),
        "reward_amount": round(risk_amount * 3, 2)
    }

def execute_trade_exness(signal_type, symbol, email, password):
    try:
        options = Options()
        options.add_argument("--start-maximized")
        driver = webdriver.Chrome(options=options)
        driver.get("https://www.exness.com")
        time.sleep(3)

        driver.find_element(By.LINK_TEXT, "Sign in").click()
        time.sleep(3)
        driver.find_element(By.NAME, "email").send_keys(email)
        driver.find_element(By.NAME, "password").send_keys(password)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        time.sleep(5)

        driver.get("https://trade.exness.com/")
        time.sleep(8)

        search = driver.find_element(By.XPATH, "//input[@placeholder='Search']")
        search.send_keys(symbol)
        time.sleep(3)

        driver.find_element(By.XPATH, f"//div[contains(text(), '{symbol}')]").click()
        time.sleep(3)

        if signal_type == "BUY":
            driver.find_element(By.XPATH, "//button[contains(text(),'Buy')]").click()
        elif signal_type == "SELL":
            driver.find_element(By.XPATH, "//button[contains(text(),'Sell')]").click()

        time.sleep(5)
        print(f"✅ {signal_type} trade executed.")
    except Exception as e:
        print("❌ Trade failed:", e)
    finally:
        driver.quit()

def run_ui():
    st.set_page_config(layout="wide", page_title="Exness AutoBot")
    st.title("📈 Exness Trading Bot with Auto-Trade")

    if "favorites" not in st.session_state:
        st.session_state.favorites = []
    if "selected_market" not in st.session_state:
        st.session_state.selected_market = "EUR/USD"
    if "auto_trade" not in st.session_state:
        st.session_state.auto_trade = False

    category = st.sidebar.selectbox("Category", list(CATEGORIES.keys()))
    for market in CATEGORIES[category]:
        col1, col2 = st.columns([8, 1])
        if col1.button(market):
            st.session_state.selected_market = market
        if col2.button("⭐" if market in st.session_state.favorites else "☆", key=market):
            if market in st.session_state.favorites:
                st.session_state.favorites.remove(market)
            else:
                st.session_state.favorites.append(market)

    st.sidebar.markdown("---")
    st.sidebar.checkbox("✅ Enable Auto-Trade", key="auto_trade")
    email = st.sidebar.text_input("📧 Exness Email", type="default")
    password = st.sidebar.text_input("🔐 Exness Password", type="password")
    account_balance = st.sidebar.number_input("💰 Account Balance ($)", min_value=10, value=1000)

    st.subheader(f"📉 Chart - {st.session_state.selected_market}")
    exch, sym = MARKET_SYMBOLS[st.session_state.selected_market]
    iframe(f"https://s.tradingview.com/widgetembed/?symbol={exch}:{sym}&interval=1&theme=dark", height=400)

    if st.button("🔄 Refresh Signal"):
        data = get_live_data((exch, sym))
        if data:
            signal = generate_signal(data, account_balance)

            st.subheader("📊 Market Info")
            st.markdown(f"- Trend: **{data['trend']}**, Momentum: **{data['momentum']}**")
            st.markdown(f"- Volatility: **{data['volatility']}**, Support: **{data['support']}**, Resistance: **{data['resistance']}**")

            st.subheader("✅ Signal")
            st.markdown(f"- Signal: `{signal['signal']}` | Confidence: **{signal['confidence']}%**")
            st.progress(signal['confidence'])
            st.markdown(f"- Entry: `{signal['entry']}`, SL: `{signal['stop_loss']}`, TP: `{signal['take_profit']}`")
            st.markdown(f"- 💸 Risk: `${signal['risk_amount']}`, Reward: `${signal['reward_amount']}`")
            if signal['reasons']:
                st.markdown(f"**Reasons:** {', '.join(signal['reasons'])}")
            st.caption(f"Updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            if st.session_state.auto_trade and signal['signal'] in ["BUY", "SELL"]:
                execute_trade_exness(signal['signal'], sym, email, password)
        else:
            st.error("Failed to fetch data.")

if __name__ == "__main__":
    run_ui()
