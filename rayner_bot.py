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
    rsi = calculate_rsi(df['close'], 14).iloc[-1]
    pattern = detect_candlestick_pattern(df)

    return {
        "price": price,
        "trend": trend,
        "support": support,
        "resistance": resistance,
        "momentum": momentum,
        "volatility": volatility,
        "signal_strength": min(100, max(10, volatility)),
        "change": price - prev_price,
        "rsi": round(rsi, 2),
        "pattern": pattern
    }

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_candlestick_pattern(df):
    if df['close'].iloc[-1] > df['open'].iloc[-1] and df['open'].iloc[-1] < df['low'].iloc[-1]:
        return "Hammer"
    if df['close'].iloc[-1] < df['open'].iloc[-1] and df['open'].iloc[-1] > df['high'].iloc[-1]:
        return "Shooting Star"
    return "None"

def generate_signal(data, account_balance):
    entry = data["price"]
    risk_amount = account_balance * 0.01
    pip_value = 10
    pip_risk = risk_amount / pip_value
    sl, tp = None, None
    signal = "WAIT"
    reasons = []

    # ✅ Condition 1: Pro Trader Trend
    if data["trend"] == "uptrend" and entry > data["support"]:
        if data["momentum"] == "strong" and data["volatility"] > 50 and data["rsi"] < 70:
            sl = entry - pip_risk
            tp = entry + (entry - sl) * 3
            signal = "BUY"
            reasons.append("Livermore trend confirmation")
    elif data["trend"] == "downtrend" and entry < data["resistance"]:
        if data["momentum"] == "strong" and data["volatility"] > 50 and data["rsi"] > 30:
            sl = entry + pip_risk
            tp = entry - (sl - entry) * 3
            signal = "SELL"
            reasons.append("Livermore trend confirmation")

    # ✅ Condition 2: RSI extreme levels
    if data["rsi"] > 80:
        signal = "SELL"
        reasons.append("RSI overbought level")
    elif data["rsi"] < 20:
        signal = "BUY"
        reasons.append("RSI oversold level")

    # ✅ Condition 3: Candlestick pattern confirmation
    if data["pattern"] == "Hammer":
        signal = "BUY"
        reasons.append("Hammer pattern detected")
    elif data["pattern"] == "Shooting Star":
        signal = "SELL"
        reasons.append("Shooting Star pattern detected")

    # ✅ Force BUY/SELL on 100% confidence
    if data["signal_strength"] >= 100 and signal == "WAIT":
        signal = "BUY" if data["trend"] == "uptrend" else "SELL"
        reasons.append("High confidence forced signal")

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
        "lot_size": lot_size,
        "r_r_ratio": round((abs(tp - entry) / abs(entry - sl)), 2) if sl and tp else None
    }

def run_ui():
    st.set_page_config(layout="wide", page_title="🔥 Pro Signal Bot")

    st.title("🔥 Pro Signal Trading Bot")
    if "favorites" not in st.session_state:
        st.session_state.favorites = []
    if "selected_market" not in st.session_state:
        st.session_state.selected_market = "EUR/USD"
    if "history" not in st.session_state:
        st.session_state.history = []

    account_balance = st.sidebar.number_input("Account Balance ($)", value=1000, min_value=10)

    # 🔥 High Movement Markets
    movement_scores = {}
    for market, info in MARKET_SYMBOLS.items():
        data = get_live_data(info)
        if data:
            movement_scores[market] = abs(data["change"])

    high_movement = sorted(movement_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    st.sidebar.subheader("🔥 High Movement")
    for market, delta in high_movement:
        st.sidebar.write(f"{market}: {round(delta, 5)}")

    # ⭐ Favorites Watchlist
    st.sidebar.subheader("⭐ Favorites")
    for fav in st.session_state.favorites:
        exch, sym = MARKET_SYMBOLS[fav]
        df = tv.get_hist(sym, exch, Interval.in_1_minute, n_bars=1)
        if df is not None and not df.empty:
            price = df.iloc[-1]["close"]
            st.sidebar.markdown(f"**{fav}**: {round(price, 5)}")

    # 📂 Market Category Picker
    st.sidebar.subheader("📂 Market Category")
    category = st.sidebar.selectbox("Market Category", list(CATEGORIES.keys()))
    for market in CATEGORIES[category]:
        col1, col2 = st.columns([8, 1])
        if col1.button(market):
            st.session_state.selected_market = market
        if col2.button("⭐" if market in st.session_state.favorites else "☆", key=market):
            if market in st.session_state.favorites:
                st.session_state.favorites.remove(market)
            else:
                st.session_state.favorites.append(market)

    # 📊 Display Market + Chart
    selected = st.session_state.selected_market
    exch, sym = MARKET_SYMBOLS[selected]
    st.subheader(f"📊 {selected} Live Chart")
    st.components.v1.iframe(f"https://s.tradingview.com/widgetembed/?symbol={exch}:{sym}&interval=1&theme=dark", height=400)

    # 🔎 Analyze Button
    if st.button("🔍 Analyze"):
        data = get_live_data((exch, sym))
        if data:
            signal = generate_signal(data, account_balance)
            st.subheader("📌 Market Data")
            st.write(f"Trend: {data['trend']}")
            st.write(f"Momentum: {data['momentum']} | Volatility: {data['volatility']}")
            st.write(f"Support: {data['support']} | Resistance: {data['resistance']}")
            st.write(f"RSI: {data['rsi']} | Pattern: {data['pattern']}")

            st.subheader("✅ Signal")
            st.write(f"Signal: {signal['signal']} | Confidence: {signal['confidence']}%")
            st.progress(signal["confidence"])
            st.write(f"Entry: {signal['entry']} | SL: {signal['stop_loss']} | TP: {signal['take_profit']}")
            st.write(f"Risk: {signal['risk_amount']} | Reward: {signal['reward_amount']}")
            st.write(f"Lot Size: {signal['lot_size']} | R:R Ratio: {signal['r_r_ratio']}")
            st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            for reason in signal["reasons"]:
                st.info(reason)

            # 💾 Save to History
            result = {
                "Time": datetime.now().strftime('%H:%M:%S'),
                "Market": selected,
                "Signal": signal["signal"],
                "Result": "Pending"
            }
            st.session_state.history.append(result)
        else:
            st.error("❌ Unable to fetch live data.")

    # 🕘 Signal History
    if st.session_state.history:
        st.subheader("📋 Signal History (Live)")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)

if __name__ == "__main__":
    run_ui()
