import base64
from datetime import datetime
import streamlit as st
from tvDatafeed import TvDatafeed, Interval
import pandas as pd

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
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_minute, n_bars=50)
    if df is None or df.empty:
        return None
    df['EMA5'] = df['close'].ewm(span=5).mean()
    df['EMA20'] = df['close'].ewm(span=20).mean()
    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = round(last['close'], 5)
    support = round(df['low'].min(), 5)
    resistance = round(df['high'].max(), 5)
    momentum = "strong" if abs(price - prev['close']) > 0.0008 else "weak"
    volatility = round(df['high'].std() * 10000)
    trend = "uptrend" if last['EMA5'] > last['EMA20'] else "downtrend"
    return {
        "price": price,
        "trend": trend,
        "support": support,
        "resistance": resistance,
        "momentum": momentum,
        "volatility": volatility,
        "ema_gap": round(last['EMA5'] - last['EMA20'], 5),
        "change": price - prev['close']
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
        if data["momentum"] == "strong" and data["volatility"] > 50 and data["ema_gap"] > 0:
            sl = entry - pip_risk
            tp = entry + (entry - sl) * 3
            signal = "BUY"
            reasons.append("EMA crossover + strong uptrend")

    elif data["trend"] == "downtrend" and entry < data["resistance"]:
        if data["momentum"] == "strong" and data["volatility"] > 50 and data["ema_gap"] < 0:
            sl = entry + pip_risk
            tp = entry - (sl - entry) * 3
            signal = "SELL"
            reasons.append("EMA crossover + strong downtrend")

    lot_size = round(risk_amount / abs(entry - sl), 2) if sl else 0

    return {
        "signal": signal,
        "entry": round(entry, 5),
        "stop_loss": round(sl, 5) if sl else None,
        "take_profit": round(tp, 5) if tp else None,
        "confidence": min(100, data["volatility"]),
        "reasons": reasons,
        "risk_amount": round(risk_amount, 2),
        "reward_amount": round(risk_amount * 3, 2),
        "lot_size": lot_size
    }

def run_ui():
    st.set_page_config(layout="wide", page_title="📈 Advanced Trading Bot")
    st.title("📊 Pro Strategy Signal Bot")

    if "favorites" not in st.session_state:
        st.session_state.favorites = []
    if "selected_market" not in st.session_state:
        st.session_state.selected_market = "EUR/USD"

    account_balance = st.sidebar.number_input("💰 Account Balance ($)", value=1000, min_value=10)

    # Movement Tracker
    st.sidebar.subheader("🔥 Top Movers")
    movement = {}
    for market, info in MARKET_SYMBOLS.items():
        data = get_live_data(info)
        if data:
            movement[market] = abs(data["change"])
    movers = sorted(movement.items(), key=lambda x: x[1], reverse=True)[:3]
    for m in movers:
        st.sidebar.write(f"{m[0]}: {round(m[1], 5)}")

    # Watchlist
    st.sidebar.subheader("⭐ Watchlist")
    for fav in st.session_state.favorites:
        exch, sym = MARKET_SYMBOLS[fav]
        df = tv.get_hist(sym, exch, Interval.in_1_minute, n_bars=1)
        if df is not None and not df.empty:
            price = df.iloc[-1]['close']
            st.sidebar.write(f"**{fav}**: {round(price, 5)}")

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
    st.components.v1.iframe(f"https://s.tradingview.com/widgetembed/?symbol={exch}:{sym}&interval=1&theme=dark", height=400)

    if st.button("🔁 Refresh Signal"):
        data = get_live_data((exch, sym))
        if data:
            signal = generate_signal(data, account_balance)
            st.subheader("📌 Market Snapshot")
            st.write(f"Trend: **{data['trend']}**")
            st.write(f"Momentum: **{data['momentum']}**")
            st.write(f"Volatility: **{data['volatility']}**")
            st.write(f"EMA GAP: {data['ema_gap']}")

            st.subheader("✅ Signal Output")
            st.write(f"Signal: `{signal['signal']}`")
            st.progress(signal["confidence"])
            st.write(f"Entry: {signal['entry']} | SL: {signal['stop_loss']} | TP: {signal['take_profit']}")
            st.write(f"Risk: ${signal['risk_amount']} | Reward: ${signal['reward_amount']}")
            st.write(f"Lot Size: {signal['lot_size']}")
            st.markdown(f"**Reason:** {' | '.join(signal['reasons'])}")
            st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.error("❌ Could not fetch data. Try again later.")

if __name__ == "__main__":
    run_ui()
