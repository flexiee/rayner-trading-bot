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
    sys.exit("Install tvDatafeed: pip install git+https://github.com/rongardF/tvdatafeed.git")

tv = TvDatafeed()

MARKET_SYMBOLS = {
    "NIFTY 50": ("NSE", "NIFTY"),
    "BANKNIFTY": ("NSE", "BANKNIFTY"),
    "SENSEX": ("BSE", "SENSEX"),
    "EUR/USD": ("OANDA", "EURUSD"),
    "GBP/JPY": ("OANDA", "GBPJPY"),
    "USD/JPY": ("OANDA", "USDJPY"),
    "AUD/USD": ("OANDA", "AUDUSD"),
    "XAU/USD": ("OANDA", "XAUUSD"),
    "Gold": ("OANDA", "XAUUSD"),
    "Silver": ("OANDA", "XAGUSD"),
    "Oil WTI": ("OANDA", "WTICOUSD"),
    "Oil Brent": ("OANDA", "BCOUSD"),
    "Natural Gas": ("OANDA", "NATGASUSD"),
    "Bitcoin": ("BINANCE", "BTCUSDT"),
    "Ethereum": ("BINANCE", "ETHUSDT"),
    "Solana": ("BINANCE", "SOLUSDT"),
    "XRP": ("BINANCE", "XRPUSDT"),
    "Dogecoin": ("BINANCE", "DOGEUSDT"),
}

CATEGORIES = {
    "Indices": ["NIFTY 50", "BANKNIFTY", "SENSEX"],
    "Forex": ["EUR/USD", "GBP/JPY", "USD/JPY", "AUD/USD", "XAU/USD"],
    "Commodities": ["Gold", "Silver", "Oil WTI", "Oil Brent", "Natural Gas"],
    "Crypto": ["Bitcoin", "Ethereum", "Solana", "XRP", "Dogecoin"],
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
        "previous_price": prev_price,
        "trend": trend,
        "support": support,
        "resistance": resistance,
        "volatility": volatility,
        "momentum": momentum,
        "signal_strength": min(100, max(10, volatility)),
    }

def generate_signal(data):
    entry = data["price"]
    risk = 0.0015
    sl, tp = None, None
    reasons = []
    if data["trend"] == "uptrend" and entry > data["support"]:
        if data["momentum"] == "strong" and data["volatility"] > 50:
            sl = entry - risk
            tp = entry + (risk * 3)
            signal = "BUY"
            reasons.append("Breakout confirmation in uptrend")
        else:
            signal = "WAIT"
    elif data["trend"] == "downtrend" and entry < data["resistance"]:
        if data["momentum"] == "strong" and data["volatility"] > 50:
            sl = entry + risk
            tp = entry - (risk * 3)
            signal = "SELL"
            reasons.append("Breakout confirmation in downtrend")
        else:
            signal = "WAIT"
    else:
        signal = "WAIT"
    return {
        "signal": signal,
        "entry": round(entry, 5),
        "stop_loss": round(sl, 5) if sl else None,
        "take_profit": round(tp, 5) if tp else None,
        "confidence": data["signal_strength"],
        "reasons": reasons,
    }

if STREAMLIT_AVAILABLE:
    def run_ui():
        st.set_page_config(layout="wide", page_title="TradingView Pro Bot")

        st.markdown("""
            <style>
            .stApp {
                background-color: #0e1117;
                font-family: 'Segoe UI', sans-serif;
            }
            .sidebar .sidebar-content {
                background-color: #1c1f26;
            }
            .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
                background-color: rgba(20, 20, 25, 0.85);
                border-radius: 10px;
            }
            </style>
        """, unsafe_allow_html=True)

        if "favorites" not in st.session_state:
            st.session_state.favorites = []
        if "selected_market" not in st.session_state:
            st.session_state.selected_market = "EUR/USD"

        st.sidebar.title("📌 Favorites")
        for fav in st.session_state.favorites:
            exch, sym = MARKET_SYMBOLS[fav]
            df = tv.get_hist(sym, exch, Interval.in_1_minute, n_bars=1)
            if df is not None and not df.empty:
                price = df.iloc[-1]["close"]
                st.sidebar.markdown(f"<b>{fav}</b>: {round(price, 5)}", unsafe_allow_html=True)

        st.sidebar.markdown("---")
        st.sidebar.caption("Select category and star your favorites")

        st.markdown("### Market Categories")
        category = st.radio("Choose", list(CATEGORIES.keys()), horizontal=True)

        cols = st.columns(len(CATEGORIES[category]))
        for i, market in enumerate(CATEGORIES[category]):
            star = "⭐" if market in st.session_state.favorites else "☆"
            if cols[i].button(market):
                st.session_state.selected_market = market
            if cols[i].button(star, key=f"star_{market}"):
                if market in st.session_state.favorites:
                    st.session_state.favorites.remove(market)
                else:
                    st.session_state.favorites.append(market)

        selected = st.session_state.selected_market
        st.markdown(f"## 📈 {selected} Chart and Signal")

        exch, sym = MARKET_SYMBOLS[selected]
        iframe(f"https://s.tradingview.com/widgetembed/?symbol={exch}:{sym}&interval=1&theme=dark", height=400)

        if st.button("🔄 Refresh Signal"):
            data = get_live_data((exch, sym))
            if data:
                signal = generate_signal(data)

                st.subheader("📊 Snapshot")
                st.markdown(f"- **Trend:** {data['trend']}")
                st.markdown(f"- **Momentum:** {data['momentum']}")
                st.markdown(f"- **Volatility:** {data['volatility']}")
                st.markdown(f"- **Support:** {data['support']}")
                st.markdown(f"- **Resistance:** {data['resistance']}")

                st.subheader("✅ Signal")
                st.markdown(f"**Signal:** `{signal['signal']}`")
                st.markdown(f"**Confidence:** {signal['confidence']}%")
                st.progress(signal['confidence'])
                st.markdown(f"**Entry:** {signal['entry']}")
                if signal['stop_loss'] and signal['take_profit']:
                    st.markdown(f"**SL:** {signal['stop_loss']}  |  **TP (1:3):** {signal['take_profit']}")
                if signal['reasons']:
                    st.markdown(f"**Reason:** {' | '.join(signal['reasons'])}")
                st.caption(f"Updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.error("Could not fetch data.")

    if __name__ == "__main__":
        run_ui()
else:
    print("Streamlit not available.")
