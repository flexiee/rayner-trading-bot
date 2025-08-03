import pytz
import streamlit as st
from datetime import datetime
from streamlit.components.v1 import iframe
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
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_minute, n_bars=30)
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
    atr = round((df['high'] - df['low']).rolling(5).mean().iloc[-1], 5)
    trend = "uptrend" if price > df['close'].rolling(5).mean().iloc[-1] else "downtrend"
    return {
        "price": price,
        "trend": trend,
        "support": support,
        "resistance": resistance,
        "momentum": momentum,
        "volatility": volatility,
        "atr": atr,
        "signal_strength": min(100, max(10, volatility)),
        "change": price - prev_price
    }

def generate_signal(data, account_balance, market_name):
    entry = data["price"]
    risk_amount = account_balance * 0.01
    atr_buffer = data["atr"] if data["atr"] > 0 else 0.002
    sl, tp, signal = None, None, "WAIT"
    reasons = []

    if data["trend"] == "uptrend" and data["momentum"] == "strong":
        sl = entry - atr_buffer
        tp = entry + (entry - sl) * 3
        signal = "BUY"
        reasons.append("Confirmed uptrend breakout")
    elif data["trend"] == "downtrend" and data["momentum"] == "strong":
        sl = entry + atr_buffer
        tp = entry - (sl - entry) * 3
        signal = "SELL"
        reasons.append("Confirmed downtrend breakout")

    sl_distance = abs(entry - sl) if sl else 0.001
    base_lot = risk_amount / sl_distance if sl_distance > 0 else 0

    # Risk-limiting factor by market
    scale_factor = 1.0
    if market_name in ["BTC/USD", "ETH/USD", "XAU/USD"]:
        scale_factor = 0.3  # reduce lot size for high volatility
    lot_size = round(base_lot * scale_factor, 3)

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

def run_ui():
    st.set_page_config(layout="wide", page_title="🔥 Real-Time Trading Bot")
    st.title("🔥 Most Accurate Trading Signal Bot")

    india_time = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
    st.sidebar.write(f"🕒 India Time: {india_time}")

    if "favorites" not in st.session_state:
        st.session_state.favorites = []
    if "selected_market" not in st.session_state:
        st.session_state.selected_market = "EUR/USD"

    account_balance = st.sidebar.number_input("💰 Account Balance ($)", value=1000, min_value=10)

    st.sidebar.subheader("⭐ Favorites")
    for fav in st.session_state.favorites:
        exch, sym = MARKET_SYMBOLS[fav]
        df = tv.get_hist(sym, exch, Interval.in_1_minute, n_bars=1)
        if df is not None and not df.empty:
            price = df.iloc[-1]["close"]
            st.sidebar.write(f"{fav}: {round(price, 5)}")

    st.sidebar.subheader("📂 Categories")
    category = st.sidebar.selectbox("Select Category", list(CATEGORIES.keys()))
    for market in CATEGORIES[category]:
        col1, col2 = st.columns([8, 1])
        if col1.button(market):
            st.session_state.selected_market = market
        if col2.button("⭐" if market in st.session_state.favorites else "☆", key=market):
            if market in st.session_state.favorites:
                st.session_state.favorites.remove(market)
            else:
                st.session_state.favorites.append(market)

    selected = st.session_state.selected_market
    exch, sym = MARKET_SYMBOLS[selected]
    st.subheader(f"📊 {selected} Live Chart")
    iframe(f"https://s.tradingview.com/widgetembed/?symbol={exch}:{sym}&interval=1&theme=dark", height=400)

    if st.button("🔁 Refresh Signal"):
        data = get_live_data((exch, sym))
        if data:
            signal = generate_signal(data, account_balance, selected)

            st.subheader("📌 Market Snapshot")
            st.write(f"Trend: {data['trend']}")
            st.write(f"Momentum: {data['momentum']}")
            st.write(f"Volatility: {data['volatility']}")
            st.write(f"ATR (buffer): {data['atr']}")
            st.write(f"Support: {data['support']} | Resistance: {data['resistance']}")

            st.subheader("✅ Signal Result")
            st.write(f"Signal: {signal['signal']}")
            st.progress(signal['confidence'])
            st.write(f"Entry: {signal['entry']} | SL: {signal['stop_loss']} | TP: {signal['take_profit']}")
            st.write(f"Risk: ${signal['risk_amount']} | Reward: ${signal['reward_amount']}")
            st.write(f"Recommended Lot Size: {signal['lot_size']}")

            if signal['signal'] in ["BUY", "SELL"] and signal['confidence'] >= 90:
                st.success(f"📢 High Accuracy {signal['signal']} Opportunity Detected!")
        else:
            st.error("❌ Failed to fetch data.")

if __name__ == "__main__":
    run_ui()
