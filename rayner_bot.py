import streamlit as st
from datetime import datetime
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import os

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

HISTORY_FILE = "signal_history.csv"
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=[
        "Datetime", "Market", "Signal", "Entry", "Stop Loss", "Take Profit",
        "Confidence", "Lot Size", "Status"
    ]).to_csv(HISTORY_FILE, index=False)

def log_signal_history(market, signal_data):
    row = {
        "Datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "Market": market,
        "Signal": signal_data["signal"],
        "Entry": signal_data["entry"],
        "Stop Loss": signal_data["stop_loss"],
        "Take Profit": signal_data["take_profit"],
        "Confidence": signal_data["confidence"],
        "Lot Size": signal_data["lot_size"],
        "Status": "pending"
    }
    df = pd.read_csv(HISTORY_FILE)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

def update_signal_status(latest_price, market):
    df = pd.read_csv(HISTORY_FILE)
    updated = False
    for i in df.index:
        if df.loc[i, "Market"] == market and df.loc[i, "Status"] == "pending":
            signal = df.loc[i, "Signal"]
            entry = float(df.loc[i, "Entry"])
            sl = float(df.loc[i, "Stop Loss"])
            tp = float(df.loc[i, "Take Profit"])
            if signal == "BUY":
                if latest_price >= tp:
                    df.loc[i, "Status"] = "success"
                    updated = True
                elif latest_price <= sl:
                    df.loc[i, "Status"] = "fail"
                    updated = True
            elif signal == "SELL":
                if latest_price <= tp:
                    df.loc[i, "Status"] = "success"
                    updated = True
                elif latest_price >= sl:
                    df.loc[i, "Status"] = "fail"
                    updated = True
    if updated:
        df.to_csv(HISTORY_FILE, index=False)

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
        "signal_strength": min(100, max(10, volatility)),
        "change": price - prev_price
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

def run_ui():
    st.set_page_config(layout="wide", page_title="📈 Pro Trading Bot")
    st.title("📈 Pro Trading Bot")

    if "favorites" not in st.session_state:
        st.session_state.favorites = []
    if "selected_market" not in st.session_state:
        st.session_state.selected_market = "EUR/USD"

    account_balance = st.sidebar.number_input("💰 Account Balance", min_value=10, value=1000)

    movement_scores = {}
    for market, info in MARKET_SYMBOLS.items():
        data = get_live_data(info)
        if data:
            movement_scores[market] = abs(data["change"])

    high_movement = sorted(movement_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    st.sidebar.subheader("🔥 High Movement Markets")
    for market, delta in high_movement:
        st.sidebar.write(f"{market}: {round(delta, 5)}")

    st.sidebar.subheader("⭐ Watchlist")
    for fav in st.session_state.favorites:
        exch, sym = MARKET_SYMBOLS[fav]
        df = tv.get_hist(sym, exch, Interval.in_1_minute, n_bars=1)
        if df is not None and not df.empty:
            price = df.iloc[-1]["close"]
            st.sidebar.markdown(f"{fav}: {round(price, 5)}")

    st.sidebar.subheader("📂 Market Categories")
    category = st.sidebar.selectbox("Choose Category", list(CATEGORIES.keys()))
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
    st.subheader(f"📊 {selected} Chart")
    st.components.v1.iframe(f"https://s.tradingview.com/widgetembed/?symbol={exch}:{sym}&interval=1&theme=dark", height=400)

    if st.button("🔄 Refresh Signal"):
        data = get_live_data((exch, sym))
        if data:
            signal = generate_signal(data, account_balance)
            log_signal_history(selected, signal)
            update_signal_status(data["price"], selected)

            st.subheader("📌 Market Snapshot")
            st.write(f"Trend: {data['trend']}")
            st.write(f"Momentum: {data['momentum']}")
            st.write(f"Volatility: {data['volatility']}")
            st.write(f"Support: {data['support']} | Resistance: {data['resistance']}")

            st.subheader("✅ Signal")
            st.write(f"Signal: {signal['signal']}")
            st.progress(signal["confidence"])
            st.write(f"Entry: {signal['entry']} | SL: {signal['stop_loss']} | TP: {signal['take_profit']}")
            st.write(f"Risk: ${signal['risk_amount']} | Reward: ${signal['reward_amount']}")
            st.write(f"Recommended Lot Size: {signal['lot_size']}")
            st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.error("❌ Data fetch failed.")

    st.markdown("---")
    st.subheader("📜 Signal History")
    try:
        df = pd.read_csv(HISTORY_FILE)
        df_sorted = df.sort_values(by="Datetime", ascending=False)
        st.dataframe(df_sorted, use_container_width=True)
    except:
        st.info("No signal history yet.")

if __name__ == "__main__":
    run_ui()
