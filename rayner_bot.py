# --- Rayner Teo Style Trading Bot Web App (Streamlit Optional) with TradingView Data ---
import sys
import datetime

# Streamlit is optional
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    sys.exit("tvDatafeed is not installed. Use: pip install git+https://github.com/rongardF/tvdatafeed.git")

# --- Connect to TradingView ---
tv = TvDatafeed()

# --- Markets: Forex, Crypto, Commodities ---
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

# --- Fetch real-time data from TradingView ---
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

# --- Signal Logic (1:3 Risk:Reward) ---
def generate_signal(data):
    reasons = []
    entry = data["price"]
    sl, tp = None, None
    risk = 0.0015

    if data["trend"] == "uptrend" and entry > data["support"]:
        if data["momentum"] == "strong" and data["volatility"] > 50:
            sl = entry - risk
            tp = entry + (risk * 3)
            reasons.append("Breakout confirmation in uptrend")
            signal = "BUY"
        else:
            signal = "WAIT"
    elif data["trend"] == "downtrend" and entry < data["resistance"]:
        if data["momentum"] == "strong" and data["volatility"] > 50:
            sl = entry + risk
            tp = entry - (risk * 3)
            reasons.append("Breakout confirmation in downtrend")
            signal = "SELL"
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

# --- Streamlit UI ---
if STREAMLIT_AVAILABLE:
    def run_streamlit_ui():
        st.set_page_config(page_title="Rayner Bot", layout="centered")
        st.title("📈 Rayner Price Action Bot")

        market = st.selectbox("Select Market", list(MARKET_SYMBOLS.keys()))

        if st.button("📤 Generate Signal"):
            with st.spinner("Fetching data..."):
                data = get_live_data(MARKET_SYMBOLS[market])
                if data:
                    signal = generate_signal(data)

                    # Live TradingView Chart
                    st.subheader("📺 Live Market Chart")
                    symbol = MARKET_SYMBOLS[market][1]
                    exchange = MARKET_SYMBOLS[market][0]
                    embed_url = f"https://s.tradingview.com/widgetembed/?symbol={exchange}:{symbol}&interval=1&theme=light&style=1&timezone=Etc%2FUTC"
                    st.components.v1.iframe(embed_url, height=400)

                    # Market Snapshot
                    st.subheader("📊 Market Snapshot")
                    st.markdown(f"**Trend:** {data['trend']}")
                    st.markdown(f"**Momentum:** {data['momentum']}")
                    st.markdown(f"**Volatility:** {data['volatility']}")
                    st.markdown(f"**Support:** {data['support']}")
                    st.markdown(f"**Resistance:** {data['resistance']}")

                    # Signal Section
                    st.subheader("✅ Signal")
                    st.markdown(f"**Signal:** `{signal['signal']}`")
                    st.markdown(f"**Confidence:** {signal['confidence']}%")
                    st.progress(signal['confidence'])
                    st.markdown(f"**Entry Price:** {signal['entry']}")

                    if signal['stop_loss'] and signal['take_profit']:
                        st.markdown(f"**Stop Loss:** {signal['stop_loss']}")
                        st.markdown(f"**Take Profit (1:3):** {signal['take_profit']}")

                    st.markdown("**Reasons:**")
                    for r in signal['reasons']:
                        st.markdown(f"✅ {r}")

                    st.markdown(f"📅 Generated At: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    st.error("❌ Could not load live data. Try again later.")
        else:
            st.info("Click 'Generate Signal' to start.")

    if __name__ == "__main__":
        try:
            run_streamlit_ui()
        except Exception as e:
            print("Streamlit UI failed. Backend logic still works.", str(e))
else:
    print("Streamlit not installed. Run: pip install streamlit")
    print("Backend signal engine is still functional.")
