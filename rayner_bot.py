import sys
from datetime import datetime

# Streamlit support (optional)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    sys.exit("tvDatafeed is not installed. Run: pip install git+https://github.com/rongardF/tvdatafeed.git")

tv = TvDatafeed()

# Market List
MARKET_SYMBOLS = {
    # Forex
    "EUR/USD": ("OANDA", "EURUSD"),
    "GBP/JPY": ("OANDA", "GBPJPY"),
    "USD/JPY": ("OANDA", "USDJPY"),
    "AUD/USD": ("OANDA", "AUDUSD"),
    "USD/CAD": ("OANDA", "USDCAD"),
    "NZD/USD": ("OANDA", "NZDUSD"),
    "USD/CHF": ("OANDA", "USDCHF"),
    "XAU/USD": ("OANDA", "XAUUSD"),

    # Commodities
    "Gold": ("OANDA", "XAUUSD"),
    "Silver": ("OANDA", "XAGUSD"),
    "Crude Oil WTI": ("OANDA", "WTICOUSD"),
    "Crude Oil Brent": ("OANDA", "BCOUSD"),
    "Natural Gas": ("OANDA", "NATGASUSD"),
    "Platinum": ("OANDA", "XPTUSD"),

    # Crypto
    "Bitcoin (BTC/USD)": ("BINANCE", "BTCUSDT"),
    "Ethereum (ETH/USD)": ("BINANCE", "ETHUSDT"),
    "BNB (BNB/USD)": ("BINANCE", "BNBUSDT"),
    "Solana (SOL/USD)": ("BINANCE", "SOLUSDT"),
    "XRP (XRP/USD)": ("BINANCE", "XRPUSDT"),
    "Dogecoin (DOGE/USD)": ("BINANCE", "DOGEUSDT"),
    "Cardano (ADA/USD)": ("BINANCE", "ADAUSDT"),
    "Polkadot (DOT/USD)": ("BINANCE", "DOTUSDT"),
    "Litecoin (LTC/USD)": ("BINANCE", "LTCUSDT"),
    "Avalanche (AVAX/USD)": ("BINANCE", "AVAXUSDT"),
}

def get_live_data(symbol_info):
    exchange, symbol = symbol_info
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_minute, n_bars=20)
    if df is None or df.empty:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    price = round(last['close'], 5)
    previous_price = round(prev['close'], 5)
    support = round(df['low'].min(), 5)
    resistance = round(df['high'].max(), 5)

    momentum = "strong" if abs(price - previous_price) > 0.0008 else "weak"
    volatility = round(df['high'].std() * 10000)
    trend = "uptrend" if price > df['close'].rolling(5).mean().iloc[-1] else "downtrend"

    return {
        "price": price,
        "previous_price": previous_price,
        "trend": trend,
        "support": support,
        "resistance": resistance,
        "volatility": volatility,
        "momentum": momentum,
        "signal_strength": min(100, max(10, volatility)),
    }

def generate_signal(data):
    entry = data["price"]
    sl, tp = None, None
    reasons = []
    risk = 0.0015

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
    def run_streamlit_ui():
        st.set_page_config(page_title="Rayner Bot", layout="centered")
        st.title("📈 Rayner Price Action Bot")

        # Initialize favorites
        if "favorites" not in st.session_state:
            st.session_state.favorites = ["EUR/USD", "Gold", "Bitcoin (BTC/USD)"]

        st.subheader("⭐ Your Favorite Markets")
        fav_cols = st.columns(len(st.session_state.favorites))
        for i, fav in enumerate(st.session_state.favorites):
            if fav_cols[i].button(fav):
                st.session_state["selected_market"] = fav

        st.subheader("📝 Manage Favorites")
        new_favs = st.multiselect("Select favorite markets:", list(MARKET_SYMBOLS.keys()), default=st.session_state.favorites)
        st.session_state.favorites = new_favs

        st.subheader("🌍 Select Market")
        selected_market = st.selectbox("Choose Market", list(MARKET_SYMBOLS.keys()),
                                       index=list(MARKET_SYMBOLS).index(st.session_state.get("selected_market", "EUR/USD")))
        st.session_state["selected_market"] = selected_market

        if st.button("📤 Generate Signal"):
            with st.spinner("Loading data..."):
                data = get_live_data(MARKET_SYMBOLS[selected_market])
                if data:
                    signal = generate_signal(data)

                    exch, sym = MARKET_SYMBOLS[selected_market]
                    chart_url = f"https://s.tradingview.com/widgetembed/?symbol={exch}:{sym}&interval=1&theme=light"
                    st.subheader("📺 Live Market Chart")
                    st.components.v1.iframe(chart_url, height=400)

                    st.subheader("📊 Market Snapshot")
                    st.markdown(f"**Trend:** {data['trend']}")
                    st.markdown(f"**Momentum:** {data['momentum']}")
                    st.markdown(f"**Volatility:** {data['volatility']}%")
                    st.markdown(f"**Support:** {data['support']}")
                    st.markdown(f"**Resistance:** {data['resistance']}")

                    st.subheader("✅ Trade Signal")
                    st.markdown(f"**Signal:** `{signal['signal']}`")
                    st.markdown(f"**Confidence:** {signal['confidence']}%")
                    st.progress(signal['confidence'])
                    st.markdown(f"**Entry Price:** {signal['entry']}")
                    if signal['stop_loss'] and signal['take_profit']:
                        st.markdown(f"**Stop Loss:** {signal['stop_loss']}")
                        st.markdown(f"**Take Profit (1:3):** {signal['take_profit']}")

                    st.markdown("**Reasons:**")
                    for r in signal['reasons']:
                        st.markdown(f"✔️ {r}")

                    st.markdown(f"🕒 Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    st.error("❌ Failed to fetch data. Please try again later.")

    if __name__ == "__main__":
        run_streamlit_ui()
else:
    print("Streamlit is not installed. Install using: pip install streamlit")
    print("You can still use the backend signal logic.")
