import sys
from datetime import datetime
from base64 import b64encode

try:
    import streamlit as st
    from streamlit.components.v1 import iframe
except ImportError:
    raise SystemExit("Please install Streamlit: pip install streamlit")

try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    raise SystemExit("Install tvDatafeed: pip install git+https://github.com/rongardF/tvdatafeed.git")

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
    "BANKNIFTY": ("NSE", "BANKNIFTY")
}

CATEGORIES = {
    "Forex": ["EUR/USD", "GBP/JPY", "USD/JPY", "AUD/USD", "XAU/USD"],
    "Crypto": ["BTC/USD", "ETH/USD"],
    "Commodities": ["Gold", "Silver", "Oil WTI"],
    "Indices": ["NIFTY 50", "BANKNIFTY"]
}

def get_encoded_image():
    with open("13812.png", "rb") as img:
        return b64encode(img.read()).decode()

def get_live_data(symbol_info):
    exchange, symbol = symbol_info
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_minute, n_bars=20)
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = round(last["close"], 5)
    prev_price = round(prev["close"], 5)
    support = round(df["low"].min(), 5)
    resistance = round(df["high"].max(), 5)
    momentum = "strong" if abs(price - prev_price) > 0.0008 else "weak"
    volatility = round(df["high"].std() * 10000)
    trend = "uptrend" if price > df["close"].rolling(5).mean().iloc[-1] else "downtrend"
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

    if data["trend"] == "uptrend" and data["momentum"] == "strong":
        sl = entry - pip_risk
        tp = entry + (entry - sl) * 3
        signal = "BUY"
        reasons.append("Uptrend + strong momentum")
    elif data["trend"] == "downtrend" and data["momentum"] == "strong":
        sl = entry + pip_risk
        tp = entry - (sl - entry) * 3
        signal = "SELL"
        reasons.append("Downtrend + strong momentum")

    lot_size = round(risk_amount / abs(entry - sl), 2) if sl else 0

    return {
        "signal": signal,
        "entry": entry,
        "stop_loss": round(sl, 5) if sl else None,
        "take_profit": round(tp, 5) if tp else None,
        "confidence": data["signal_strength"],
        "risk_amount": round(risk_amount, 2),
        "reward_amount": round(risk_amount * 3, 2),
        "lot_size": lot_size,
        "reasons": reasons
    }

def get_top_movers():
    movers = []
    for name, info in MARKET_SYMBOLS.items():
        data = get_live_data(info)
        if data:
            movers.append((name, data["volatility"]))
    movers.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in movers[:3]]

def main():
    st.set_page_config(layout="wide", page_title="Pro Trading Bot")
    bg = get_encoded_image()
    st.markdown(f"""<style>.stApp {{
        background-image: url("data:image/png;base64,{bg}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}</style>""", unsafe_allow_html=True)

    if "favorites" not in st.session_state:
        st.session_state.favorites = []
    if "selected_market" not in st.session_state:
        st.session_state.selected_market = "EUR/USD"

    top_movers = get_top_movers()
    st.sidebar.header("📊 Favorites Watchlist")
    for fav in st.session_state.favorites:
        exch, sym = MARKET_SYMBOLS[fav]
        df = tv.get_hist(sym, exch, Interval.in_1_minute, n_bars=1)
        if df is not None and not df.empty:
            price = df.iloc[-1]["close"]
            st.sidebar.markdown(f"✅ {fav}: {round(price, 5)}")

    st.sidebar.markdown("---")
    category = st.sidebar.selectbox("Select Category", list(CATEGORIES.keys()))
    for market in CATEGORIES[category]:
        col1, col2 = st.sidebar.columns([8, 1])
        if col1.button(market):
            st.session_state.selected_market = market
        if col2.button("⭐" if market in st.session_state.favorites else "☆", key=market):
            if market in st.session_state.favorites:
                st.session_state.favorites.remove(market)
            else:
                st.session_state.favorites.append(market)

    st.markdown("---")
    st.subheader(f"📈 Chart: {st.session_state.selected_market}")
    exch, sym = MARKET_SYMBOLS[st.session_state.selected_market]
    iframe(f"https://s.tradingview.com/widgetembed/?symbol={exch}:{sym}&interval=1&theme=dark", height=400)

    account_balance = st.number_input("💰 Your Account Balance ($)", min_value=10, value=1000)

    st.markdown("### 🔄 Click to refresh signal")
    if st.button("Refresh"):
        data = get_live_data((exch, sym))
        if not data:
            st.error("Could not fetch live data.")
            return
        signal = generate_signal(data, account_balance)

        st.success(f"Market: {st.session_state.selected_market}")
        st.markdown(f"**Signal**: {signal['signal']} | Confidence: {signal['confidence']}%")
        st.progress(signal['confidence'])

        st.markdown(f"**Entry**: {signal['entry']} | SL: {signal['stop_loss']} | TP: {signal['take_profit']}")
        st.markdown(f"**Risk**: ${signal['risk_amount']} | Reward: ${signal['reward_amount']}")
        st.markdown(f"📌 Recommended lot size: `{signal['lot_size']} units`")
        if signal['reasons']:
            st.info("Reason: " + " | ".join(signal['reasons']))

        if data["volatility"] > 85:
            st.warning("🚨 Huge move detected in market!")

        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    st.markdown("---")
    st.subheader("🔥 Top High-Movement Markets Now")
    st.markdown(", ".join(top_movers))

if __name__ == "__main__":
    main()
