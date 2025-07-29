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

    lot_size = round(risk_amount / abs(entry - sl), 2) if sl else 0

    return {
        "signal": signal,
        "entry": round(entry, 5),
        "stop_loss": round(sl, 5) if sl else None,
        "take_profit": round(tp, 5) if tp else None,
        "confidence": data["signal_strength"],
        "reasons": reasons,
        "risk_amount": round(risk_amount, 2),
        "reward_amount": round(risk_amount * 3, 2),
        "lot_size": lot_size
    }

def get_high_movement_markets():
    movement_scores = []
    for market, (exch, sym) in MARKET_SYMBOLS.items():
        try:
            data = get_live_data((exch, sym))
            if data:
                movement_scores.append((market, data["volatility"]))
        except:
            continue
    movement_scores.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in movement_scores[:3]]

if STREAMLIT_AVAILABLE:
    def run_ui():
        st.set_page_config(layout="wide", page_title="Pro Trading Bot")

        st.markdown("""
            <style>
                .stApp {
                    background-image: url('https://images.unsplash.com/photo-1611971260625-2c8d9b6de636?auto=format&fit=crop&w=1470&q=80');
                    background-size: cover;
                    background-position: center;
                    color: white;
                }
            </style>
        """, unsafe_allow_html=True)

        if "favorites" not in st.session_state:
            st.session_state.favorites = []
        if "selected_market" not in st.session_state:
            st.session_state.selected_market = "EUR/USD"

        st.sidebar.title("⭐ Favorite Watchlist")
        for fav in st.session_state.favorites:
            exch, sym = MARKET_SYMBOLS[fav]
            try:
                df = tv.get_hist(sym, exch, Interval.in_1_minute, n_bars=1)
                if df is not None and not df.empty:
                    price = df.iloc[-1]["close"]
                    st.sidebar.markdown(f"**{fav}**: {round(price, 5)}")
            except:
                pass

        st.sidebar.markdown("---")
        st.sidebar.caption("📈 Select market")

        high_moves = get_high_movement_markets()
        st.sidebar.markdown("🔥 High Movement:")
        for mover in high_moves:
            st.sidebar.markdown(f"- {mover}")

        category = st.sidebar.selectbox("Category", list(CATEGORIES.keys()))
        for market in CATEGORIES[category]:
            col1, col2 = st.columns([8, 1])
            if col1.button(market):
                st.session_state.selected_market = market
            if col2.button("⭐" if market in st.session_state.favorites else "☆", key=f"fav_{market}"):
                if market in st.session_state.favorites:
                    st.session_state.favorites.remove(market)
                else:
                    st.session_state.favorites.append(market)

        market = st.session_state.selected_market
        st.subheader(f"📊 {market} Live Chart")
        exch, sym = MARKET_SYMBOLS[market]
        iframe(f"https://s.tradingview.com/widgetembed/?symbol={exch}:{sym}&interval=1&theme=dark", height=400)

        st.markdown("---")
        account_balance = st.number_input("💰 Account Balance", min_value=10, value=1000)

        if st.button("🔄 Refresh Signal"):
            data = get_live_data((exch, sym))
            if data:
                signal = generate_signal(data, account_balance)

                st.subheader("Market Conditions")
                st.markdown(f"- Trend: **{data['trend']}**")
                st.markdown(f"- Momentum: **{data['momentum']}**")
                st.markdown(f"- Volatility: **{data['volatility']}**")
                st.markdown(f"- Support: **{data['support']}**, Resistance: **{data['resistance']}**")

                st.subheader("📍 Signal")
                st.markdown(f"- Signal: `{signal['signal']}`")
                st.markdown(f"- Entry: `{signal['entry']}`")
                st.markdown(f"- Stop Loss: `{signal['stop_loss']}`, Take Profit: `{signal['take_profit']}`")
                st.markdown(f"- Confidence: **{signal['confidence']}%**")
                st.progress(signal['confidence'])

                st.markdown(f"- 💸 Risk: `${signal['risk_amount']}` | 🟢 Reward: `${signal['reward_amount']}`")
                st.markdown(f"- 📦 Suggested Lot Size: **{signal['lot_size']} lots**")
                if signal["reasons"]:
                    st.markdown(f"Reason: {' | '.join(signal['reasons'])}")
                st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.error("❌ Could not load data. Try again later.")

    if __name__ == "__main__":
        run_ui()
else:
    print("Streamlit not installed.")
