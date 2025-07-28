import sys
from datetime import datetime
import pytz

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
    "BANKNIFTY": ("NSE", "BANKNIFTY"),
}

CATEGORIES = {
    "Forex": ["EUR/USD", "GBP/JPY", "USD/JPY", "AUD/USD", "XAU/USD"],
    "Crypto": ["BTC/USD", "ETH/USD"],
    "Commodities": ["Gold", "Silver", "Oil WTI"],
    "Indices": ["NIFTY 50", "BANKNIFTY"]
}

# World Market Sessions
SESSIONS = {
    "London (Forex)": {"start": 8, "end": 17, "timezone": "Europe/London"},
    "New York (US)": {"start": 8, "end": 17, "timezone": "America/New_York"},
    "Tokyo (Asia)": {"start": 9, "end": 16, "timezone": "Asia/Tokyo"},
    "Sydney (Pacific)": {"start": 9, "end": 17, "timezone": "Australia/Sydney"},
    "Crypto Peak (UTC)": {"start": 12, "end": 21, "timezone": "UTC"},
}

def get_active_sessions():
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    active = []
    for name, info in SESSIONS.items():
        tz = pytz.timezone(info["timezone"])
        local = now_utc.astimezone(tz)
        if info["start"] <= local.hour < info["end"]:
            active.append(name)
    return active

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

if STREAMLIT_AVAILABLE:
    def run_ui():
        st.set_page_config(layout="wide", page_title="TradingView Risk Bot")
        st.title("📊 TradingView-Style Risk Bot")

        # High movement sessions
        active_sessions = get_active_sessions()
        if active_sessions:
            st.markdown("### 🌍 Active Market Sessions Now")
            for s in active_sessions:
                st.markdown(f"- ✅ **{s}**")

        # Top market pairs with highest volatility
        movers = get_top_moving_markets()
        if movers:
            st.markdown("### 🔥 Top Moving Markets Now")
            for pair, vol in movers:
                st.markdown(f"- 🔹 {pair}: {vol} volatility")

        st.markdown("""
            <style>
            .stApp { background-color: #0e1117; color: white; font-family: 'Segoe UI', sans-serif; }
            .stSidebar { background-color: #1c1f26; }
            </style>
        """, unsafe_allow_html=True)

        if "favorites" not in st.session_state:
            st.session_state.favorites = []
        if "selected_market" not in st.session_state:
            st.session_state.selected_market = "EUR/USD"

        st.sidebar.header("⭐ Favorites")
        for fav in st.session_state.favorites:
            exch, sym = MARKET_SYMBOLS[fav]
            df = tv.get_hist(sym, exch, Interval.in_1_minute, n_bars=1)
            if df is not None and not df.empty:
                price = df.iloc[-1]["close"]
                st.sidebar.markdown(f"{fav}: {round(price, 5)}")

        st.sidebar.markdown("---")
        st.sidebar.caption("Select category and star your favorites")

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

        st.markdown("---")
        account_balance = st.number_input("💰 Enter your account balance ($)", min_value=10, value=1000)

        if st.button("🔄 Refresh Signal"):
            data = get_live_data((exch, sym))
            if data:
                signal = generate_signal(data, account_balance)

                st.subheader("📌 Market Snapshot")
                st.markdown(f"- Trend: **{data['trend']}**")
                st.markdown(f"- Momentum: **{data['momentum']}**")
                st.markdown(f"- Volatility: **{data['volatility']}**")
                st.markdown(f"- Support: **{data['support']}**")
                st.markdown(f"- Resistance: **{data['resistance']}**")

                st.subheader("✅ Signal Result")
                st.markdown(f"- Signal: `{signal['signal']}`")
                st.markdown(f"- Confidence: **{signal['confidence']}%**")
                st.progress(signal['confidence'])

                st.markdown(f"- Entry Price: **{signal['entry']}**")
                st.markdown(f"- Stop Loss: **{signal['stop_loss']}**  |  Take Profit: **{signal['take_profit']}**")
                st.markdown(f"- 💸 Risk: `${signal['risk_amount']}` | 🟢 Reward: `${signal['reward_amount']}`")

                if signal['reasons']:
                    st.markdown(f"**Reason:** {' | '.join(signal['reasons'])}")
                st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.error("❌ Failed to fetch live data.")

    if __name__ == "__main__":
        run_ui()
else:
    print("Streamlit not installed.")
