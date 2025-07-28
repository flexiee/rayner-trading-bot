# rayner_bot.py

import sys
import datetime

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    sys.exit("tvDatafeed is not installed. Please install it using 'pip install tvDatafeed'")

tv = TvDatafeed()

# Market List (Includes Forex, Crypto, Commodities, Indices)
MARKET_SYMBOLS = {
    "EUR/USD": ("OANDA", "EURUSD"),
    "GBP/JPY": ("OANDA", "GBPJPY"),
    "USD/JPY": ("OANDA", "USDJPY"),
    "AUD/USD": ("OANDA", "AUDUSD"),
    "XAU/USD (Gold)": ("OANDA", "XAUUSD"),
    "BTC/USD": ("BINANCE", "BTCUSDT"),
    "ETH/USD": ("BINANCE", "ETHUSDT"),
    "USOIL": ("TVC", "USOIL"),
    "NATGAS": ("TVC", "NATGASUSD"),
    "NIFTY 50": ("NSE", "NIFTY"),
    "SENSEX": ("BSE", "SENSEX"),
    "S&P 500": ("SP", "SPX"),
}

FAVORITES = ["NIFTY 50", "EUR/USD", "XAU/USD (Gold)"]

def get_live_data(symbol_info):
    exchange, symbol = symbol_info
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_minute, n_bars=20)

    if df is None or df.empty:
        return None

    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]

    price = round(last_candle['close'], 5)
    previous_price = round(prev_candle['close'], 5)
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
    reasons = []
    entry_price = data["price"]
    sl = None
    tp = None

    if data["trend"] == "uptrend" and entry_price > data["support"]:
        if data["momentum"] == "strong" and data["volatility"] > 50:
            sl = entry_price - 0.0015
            tp = entry_price + (entry_price - sl) * 3
            reasons.append("Breakout confirmation in uptrend")
            signal = "BUY"
        else:
            signal = "WAIT"
    elif data["trend"] == "downtrend" and entry_price < data["resistance"]:
        if data["momentum"] == "strong" and data["volatility"] > 50:
            sl = entry_price + 0.0015
            tp = entry_price - (sl - entry_price) * 3
            reasons.append("Breakout confirmation in downtrend")
            signal = "SELL"
        else:
            signal = "WAIT"
    else:
        signal = "WAIT"

    return {
        "signal": signal,
        "entry": entry_price,
        "stop_loss": round(sl, 5) if sl else None,
        "take_profit": round(tp, 5) if tp else None,
        "confidence": data["signal_strength"],
        "reasons": reasons,
    }

if STREAMLIT_AVAILABLE:
    def run_ui():
        st.set_page_config(layout="wide", page_title="TradingView Style Bot")
        st.markdown("""
            <style>
                .block-container {
                    padding: 1rem 2rem;
                    background-image: url("https://images.unsplash.com/photo-1616857593881-03c2d3ad88f4?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80");
                    background-size: cover;
                    color: white;
                }
                .stButton>button {
                    width: 100%;
                }
            </style>
        """, unsafe_allow_html=True)

        st.title("📈 Rayner Teo Style Bot (TradingView Look)")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("🌟 Favorite Markets")
            for market in FAVORITES:
                data = get_live_data(MARKET_SYMBOLS[market])
                if data:
                    signal = generate_signal(data)
                    with st.expander(f"🔸 {market} ({signal['signal']})"):
                        st.metric("Price", data["price"])
                        st.metric("Trend", data["trend"])
                        st.metric("Momentum", data["momentum"])
                        st.metric("Volatility", f"{data['volatility']}%")

        with col2:
            st.subheader("📊 Select Market to Analyze")
            market = st.selectbox("Choose market", list(MARKET_SYMBOLS.keys()))
            data = get_live_data(MARKET_SYMBOLS[market])
            if data:
                signal = generate_signal(data)
                st.components.v1.iframe(f"https://s.tradingview.com/widgetembed/?frameElementId=tradingview_{market}&symbol={MARKET_SYMBOLS[market][0]}%3A{MARKET_SYMBOLS[market][1]}&interval=1&theme=dark&style=1", height=400)
                st.write("### Signal Details")
                st.success(f"📌 Signal: {signal['signal']}")
                st.info(f"Confidence: {signal['confidence']}%")
                st.markdown(f"**Entry:** {signal['entry']}")
                if signal['stop_loss'] and signal['take_profit']:
                    st.markdown(f"**SL:** {signal['stop_loss']} | **TP:** {signal['take_profit']}")

    if __name__ == "__main__":
        run_ui()
else:
    print("Streamlit not available. Install it using pip install streamlit")
