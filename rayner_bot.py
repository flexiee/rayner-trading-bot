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

def get_encoded_image():
    return ""  # Background image removed as per request

def get_live_data(symbol_info):
    exchange, symbol = symbol_info
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_minute, n_bars=30)
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = round(last['close'], 5)
    support = round(df['low'].min(), 5)
    resistance = round(df['high'].max(), 5)
    change = price - prev['close']
    trend = "uptrend" if price > df['close'].rolling(5).mean().iloc[-1] else "downtrend"
    momentum = "strong" if abs(change) > 0.001 else "weak"
    volatility = round(df['high'].std() * 10000)
    return {
        "df": df,
        "price": price,
        "trend": trend,
        "support": support,
        "resistance": resistance,
        "momentum": momentum,
        "volatility": volatility,
        "change": change,
        "signal_strength": min(100, max(20, volatility))
    }

def apply_pro_strategies(df, price, trend):
    strategy_score = 0
    summary = []

    # 1. Turtle breakout logic
    breakout_20 = df['high'].rolling(20).max().iloc[-2]
    breakdown_20 = df['low'].rolling(20).min().iloc[-2]
    if trend == 'uptrend' and price > breakout_20:
        strategy_score += 1
        summary.append("Turtle breakout confirmed")
    elif trend == 'downtrend' and price < breakdown_20:
        strategy_score += 1
        summary.append("Turtle breakdown confirmed")

    # 2. Livermore strength confirmation
    if df['close'].iloc[-1] > df['close'].rolling(10).mean().iloc[-1]:
        strategy_score += 1
        summary.append("Livermore trend confirmation")

    # 3. Soros-style volatility push
    if df['volume'].mean() > 0 and df['volume'].iloc[-1] > df['volume'].mean() * 1.5:
        strategy_score += 1
        summary.append("Soros-style volume spike")

    return strategy_score, summary

def generate_signal(data, account_balance):
    entry = data["price"]
    risk_amount = account_balance * 0.01
    pip_value = 10
    pip_risk = risk_amount / pip_value
    sl, tp = None, None
    signal = "WAIT"
    reasons = []

    strat_score, strat_signals = apply_pro_strategies(data["df"], entry, data["trend"])
    confirm = strat_score >= 2  # Require 2 out of 3

    if data["trend"] == "uptrend" and entry > data["support"] and confirm:
        sl = entry - pip_risk
        tp = entry + (entry - sl) * 3
        signal = "BUY"
        reasons.append("Uptrend breakout with pro confirmation")
    elif data["trend"] == "downtrend" and entry < data["resistance"] and confirm:
        sl = entry + pip_risk
        tp = entry - (sl - entry) * 3
        signal = "SELL"
        reasons.append("Downtrend breakout with pro confirmation")

    lot_size = round(risk_amount / abs(entry - sl), 2) if sl else 0

    return {
        "signal": signal,
        "entry": round(entry, 5),
        "stop_loss": round(sl, 5) if sl else None,
        "take_profit": round(tp, 5) if tp else None,
        "confidence": data["signal_strength"] + strat_score * 10,
        "risk_amount": round(risk_amount, 2),
        "reward_amount": round(risk_amount * 3, 2),
        "lot_size": lot_size,
        "reasons": reasons + strat_signals
    }

def run_ui():
    st.set_page_config(layout="wide", page_title="🔥 Pro Strategy Trading Bot")
    st.title("📈 Pro Strategy Trading Bot")

    if "favorites" not in st.session_state:
        st.session_state.favorites = []
    if "selected_market" not in st.session_state:
        st.session_state.selected_market = "EUR/USD"

    st.sidebar.header("⚙️ Settings")
    balance = st.sidebar.number_input("Account Balance ($)", 100, step=10)

    st.sidebar.subheader("🔥 High Movement")
    movers = {}
    for m, info in MARKET_SYMBOLS.items():
        data = get_live_data(info)
        if data:
            movers[m] = abs(data["change"])
    top_movers = sorted(movers.items(), key=lambda x: x[1], reverse=True)[:3]
    for market, ch in top_movers:
        st.sidebar.write(f"{market}: {round(ch, 5)}")

    st.sidebar.subheader("⭐ Watchlist")
    for fav in st.session_state.favorites:
        exch, sym = MARKET_SYMBOLS[fav]
        df = tv.get_hist(sym, exch, Interval.in_1_minute, n_bars=1)
        if df is not None and not df.empty:
            st.sidebar.write(f"{fav}: {round(df.iloc[-1]['close'], 5)}")

    st.sidebar.subheader("📂 Markets")
    cat = st.sidebar.selectbox("Category", list(CATEGORIES.keys()))
    for m in CATEGORIES[cat]:
        c1, c2 = st.columns([8, 1])
        if c1.button(m):
            st.session_state.selected_market = m
        if c2.button("⭐" if m in st.session_state.favorites else "☆", key=m):
            if m in st.session_state.favorites:
                st.session_state.favorites.remove(m)
            else:
                st.session_state.favorites.append(m)

    market = st.session_state.selected_market
    exch, sym = MARKET_SYMBOLS[market]
    st.subheader(f"📊 {market} Chart")
    st.components.v1.iframe(f"https://s.tradingview.com/widgetembed/?symbol={exch}:{sym}&interval=1&theme=dark", height=400)

    if st.button("🔍 Analyze"):
        data = get_live_data((exch, sym))
        if data:
            signal = generate_signal(data, balance)
            st.subheader("🧠 Market Snapshot")
            st.write(f"Trend: {data['trend']}, Momentum: {data['momentum']}, Volatility: {data['volatility']}")
            st.write(f"Support: {data['support']} | Resistance: {data['resistance']}")

            st.subheader("📣 Signal")
            st.write(f"Signal: `{signal['signal']}` with {signal['confidence']}% Confidence")
            st.progress(min(signal["confidence"], 100))
            st.write(f"Entry: {signal['entry']} | SL: {signal['stop_loss']} | TP: {signal['take_profit']}")
            st.write(f"Lot size: {signal['lot_size']} | Risk: ${signal['risk_amount']} | Reward: ${signal['reward_amount']}")
            st.markdown("**Reasons:**")
            for r in signal['reasons']:
                st.markdown(f"- {r}")
            if signal['confidence'] >= 110 and signal['reward_amount'] / signal['risk_amount'] >= 6:
                st.success("⚡ High Confidence 1:6 R:R Setup Detected!")

if __name__ == "__main__":
    run_ui()
