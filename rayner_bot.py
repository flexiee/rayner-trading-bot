# trading_bot.py
import streamlit as st
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import datetime

# Initialize TradingView connection
tv = TvDatafeed()

# Global risk config
RISK_PERCENT = 1.0  # risk 1% per trade

# Supported markets
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
    "S&P 500": ("SP", "SPX")
}

# Get real-time data from TradingView
def get_live_data(symbol_info):
    exchange, symbol = symbol_info
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_minute, n_bars=20)

    if df is None or df.empty:
        return None

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    price = round(latest['close'], 5)
    previous_price = round(previous['close'], 5)
    support = round(df['low'].min(), 5)
    resistance = round(df['high'].max(), 5)
    trend = "uptrend" if price > df['close'].rolling(5).mean().iloc[-1] else "downtrend"
    momentum = "strong" if abs(price - previous_price) > 0.0008 else "weak"
    volatility = round(df['high'].std() * 10000)
    signal_strength = min(100, max(10, volatility))

    return {
        "price": price,
        "previous_price": previous_price,
        "trend": trend,
        "support": support,
        "resistance": resistance,
        "momentum": momentum,
        "volatility": volatility,
        "signal_strength": signal_strength
    }

# Signal logic (unchanged core logic)
def generate_signal(data, balance):
    reasons = []
    signal = "WAIT"
    entry = data["price"]
    sl = tp = None

    if data["trend"] == "uptrend" and entry > data["support"]:
        if data["momentum"] == "strong" and data["volatility"] > 50:
            sl = entry - 0.0015
            tp = entry + (entry - sl) * 3
            signal = "BUY"
            reasons.append("Breakout confirmation in uptrend")
    elif data["trend"] == "downtrend" and entry < data["resistance"]:
        if data["momentum"] == "strong" and data["volatility"] > 50:
            sl = entry + 0.0015
            tp = entry - (sl - entry) * 3
            signal = "SELL"
            reasons.append("Breakout confirmation in downtrend")

    lot_size, risk_amt = calculate_lot_size(balance, sl, entry) if sl else (0, 0)

    return {
        "signal": signal,
        "entry": entry,
        "stop_loss": round(sl, 5) if sl else None,
        "take_profit": round(tp, 5) if tp else None,
        "confidence": data["signal_strength"],
        "lot_size": lot_size,
        "risk_amount": round(risk_amt, 2),
        "reasons": reasons
    }

# Lot size calculator based on account and SL
def calculate_lot_size(balance, stop_loss, entry_price):
    risk_amt = balance * (RISK_PERCENT / 100)
    pip_distance = abs(entry_price - stop_loss)
    if pip_distance == 0:
        return 0, 0
    lot_size = round(risk_amt / pip_distance, 2)
    return lot_size, risk_amt

# Streamlit styling
def apply_styling():
    st.set_page_config(page_title="Trading Bot", layout="wide")
    st.markdown("""
        <style>
        .main {
            background: url('https://images.unsplash.com/photo-1605902711622-cfb43c44367d') no-repeat center center fixed;
            background-size: cover;
        }
        </style>
    """, unsafe_allow_html=True)

# UI components
def show_chart(exchange, symbol):
    url = f"https://s.tradingview.com/widgetembed/?frameElementId=tv&symbol={exchange}%3A{symbol}&interval=1&theme=dark&style=1"
    st.components.v1.iframe(url, height=400)

def show_signal_result(signal, market, account):
    symbol = MARKET_SYMBOLS[market][1]
    exchange = MARKET_SYMBOLS[market][0]
    show_chart(exchange, symbol)

    st.subheader(f"📢 Signal: {signal['signal']}")
    st.markdown(f"**Confidence:** {signal['confidence']}%")
    st.markdown(f"**Entry Price:** {signal['entry']}")
    if signal['stop_loss']:
        st.markdown(f"**Stop Loss:** {signal['stop_loss']}")
        st.markdown(f"**Take Profit (1:3):** {signal['take_profit']}")
        st.markdown(f"**Lot Size:** {signal['lot_size']} lot")
        st.markdown(f"**Risk Amount:** ${signal['risk_amount']}")

    st.markdown("### 🔍 Reason(s):")
    if signal["reasons"]:
        for r in signal["reasons"]:
            st.markdown(f"- {r}")
    else:
        st.markdown("No strong signal at the moment.")

    st.caption(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Main App
def main():
    apply_styling()
    st.title("📈 Trading Bot (Rayner Strategy) with Risk Management")

    col1, col2 = st.columns([1, 3])
    with col1:
        balance = st.number_input("💰 Account Balance ($)", value=1000.0)
        market = st.selectbox("🌐 Select Market", list(MARKET_SYMBOLS.keys()))
        run = st.button("🔁 Refresh Signal")

    with col2:
        if run:
            st.subheader(f"🔎 Analyzing Market: {market}")
            market_data = get_live_data(MARKET_SYMBOLS[market])
            if market_data:
                signal = generate_signal(market_data, balance)
                show_signal_result(signal, market, balance)
            else:
                st.error("❌ Failed to fetch market data. Try again later.")
        else:
            st.info("Click 'Refresh Signal' to analyze the selected market.")

if __name__ == "__main__":
    main()
