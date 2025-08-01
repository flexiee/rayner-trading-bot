import sys
from datetime import datetime

try:
    import streamlit as st
    from streamlit.components.v1 import iframe
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    sys.exit("Please install tvDatafeed: pip install git+https://github.com/rongardF/tvdatafeed.git")

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

# Base64-encoded image (13812.png) as background
BACKGROUND_BASE64 = 'iVBORw0KGgoAAAANSUhEUgAAAyAAAAHCCAIAAACYATqfAADjMUlEQVR4nOzdZVwU3fsw8JnZpRtESkApFRBssRO9VWxFsbsDsW67'  # truncated for brevity

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
        "signal_strength": min(120, max(10, volatility))
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

def scan_high_rr_markets(account_balance):
    high_rr_alerts = []
    for market, symbol_info in MARKET_SYMBOLS.items():
        data = get_live_data(symbol_info)
        if data and data["signal_strength"] >= 110:
            signal = generate_signal(data, account_balance)
            if signal["stop_loss"] and signal["take_profit"]:
                rr_ratio = abs((signal["take_profit"] - signal["entry"]) / (signal["entry"] - signal["stop_loss"]))
                if rr_ratio >= 6:
                    high_rr_alerts.append({
                        "market": market,
                        "signal": signal["signal"],
                        "rr": round(rr_ratio, 2),
                        "confidence": signal["confidence"]
                    })
    return high_rr_alerts

if STREAMLIT_AVAILABLE:
    def run_ui():
        st.set_page_config(layout="wide", page_title="📊 Rayner Pro Bot")

        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{BACKGROUND_BASE64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white;
        }}
        </style>
        """, unsafe_allow_html=True)

        st.title("📊 Rayner Pro Trading Bot")

        if "favorites" not in st.session_state:
            st.session_state.favorites = []
        if "selected_market" not in st.session_state:
            st.session_state.selected_market = "EUR/USD"

        account_balance = st.sidebar.number_input("💰 Account Balance ($)", min_value=10, value=1000)

        st.sidebar.subheader("⭐ Favorite Markets")
        for fav in st.session_state.favorites:
            exch, sym = MARKET_SYMBOLS[fav]
            df = tv.get_hist(sym, exch, Interval.in_1_minute, n_bars=1)
            if df is not None and not df.empty:
                price = df.iloc[-1]["close"]
                st.sidebar.markdown(f"✔️ {fav}: {round(price, 5)}")

        st.sidebar.markdown("---")
        category = st.sidebar.selectbox("Market Category", list(CATEGORIES.keys()))
        for market in CATEGORIES[category]:
            col1, col2 = st.columns([8, 1])
            if col1.button(market):
                st.session_state.selected_market = market
            if col2.button("⭐" if market in st.session_state.favorites else "☆", key=f"fav_{market}"):
                if market in st.session_state.favorites:
                    st.session_state.favorites.remove(market)
                else:
                    st.session_state.favorites.append(market)

        st.markdown("---")
        st.subheader(f"📈 {st.session_state.selected_market} Chart")
        exch, sym = MARKET_SYMBOLS[st.session_state.selected_market]
        iframe(f"https://s.tradingview.com/widgetembed/?symbol={exch}:{sym}&interval=1&theme=dark", height=400)

        if st.button("🔄 Refresh Signal"):
            data = get_live_data((exch, sym))
            if data:
                signal = generate_signal(data, account_balance)

                st ​:contentReference[oaicite:0]{index=0}​
