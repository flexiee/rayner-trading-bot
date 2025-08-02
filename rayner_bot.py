from datetime import datetime
import streamlit as st
import pandas as pd
from tvDatafeed import TvDatafeed, Interval

tv = TvDatafeed()
st.set_page_config(layout="wide", page_title="📈 Pro Signal Bot")

# ---- MARKET SETUP ----
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

# ---- DATA FUNCTION ----
def get_live_data(symbol_info):
    exchange, symbol = symbol_info
    df = tv.get_hist(symbol, exchange, interval=Interval.in_1_minute, n_bars=50)
    if df is None or df.empty or len(df) < 20:
        return None

    close = df['close']
    price = round(close.iloc[-1], 5)
    prev_price = round(close.iloc[-2], 5)
    change = price - prev_price
    support = round(df['low'].min(), 5)
    resistance = round(df['high'].max(), 5)

    rsi = 100 - (100 / (1 + df['close'].pct_change().dropna().rolling(14).mean().iloc[-1]))
    ema_fast = close.ewm(span=9).mean().iloc[-1]
    ema_slow = close.ewm(span=21).mean().iloc[-1]
    macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    signal_line = macd.ewm(span=9).mean()
    macd_cross = macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]

    candle_pattern = "Bullish Engulfing" if close.iloc[-2] > close.iloc[-3] and close.iloc[-1] > close.iloc[-2] else "None"

    volatility = round(df['high'].std() * 10000)
    trend = "uptrend" if ema_fast > ema_slow else "downtrend"
    momentum = "strong" if abs(change) > 0.001 else "weak"
    confidence = min(100, max(10, volatility + (20 if macd_cross else 0)))

    return {
        "price": price, "trend": trend, "support": support, "resistance": resistance,
        "momentum": momentum, "volatility": volatility, "change": change,
        "rsi": rsi, "macd_cross": macd_cross, "ema_fast": ema_fast, "ema_slow": ema_slow,
        "pattern": candle_pattern, "signal_strength": confidence
    }

# ---- SIGNAL FUNCTION ----
def generate_signal(data, balance):
    entry = data["price"]
    risk_amount = balance * 0.01
    pip_value = 10
    pip_risk = risk_amount / pip_value
    sl, tp = None, None
    signal = "WAIT"
    reasons = []

    if data["trend"] == "uptrend" and data["momentum"] == "strong" and data["macd_cross"] and data["rsi"] < 70:
        sl = entry - pip_risk
        tp = entry + (entry - sl) * 3
        signal = "BUY"
        reasons.append("Trend Up + MACD + RSI")
    elif data["trend"] == "downtrend" and data["momentum"] == "strong" and not data["macd_cross"] and data["rsi"] > 30:
        sl = entry + pip_risk
        tp = entry - (sl - entry) * 3
        signal = "SELL"
        reasons.append("Trend Down + MACD + RSI")

    lot_size = round(risk_amount / abs(entry - sl), 2) if sl else 0
    rr_ratio = round(abs(tp - entry) / abs(entry - sl), 2) if sl and tp else None

    return {
        "signal": signal, "entry": round(entry, 5),
        "stop_loss": round(sl, 5) if sl else None,
        "take_profit": round(tp, 5) if tp else None,
        "confidence": data["signal_strength"],
        "reasons": reasons,
        "risk_amount": round(risk_amount, 2),
        "reward_amount": round(risk_amount * 3, 2),
        "lot_size": lot_size,
        "rr_ratio": rr_ratio
    }

# ---- HISTORY TRACKING ----
if "history" not in st.session_state:
    st.session_state.history = []

# ---- UI START ----
st.title("📊 Pro Trading Signal Bot")

if "selected_market" not in st.session_state:
    st.session_state.selected_market = "EUR/USD"
if "favorites" not in st.session_state:
    st.session_state.favorites = []

balance = st.sidebar.number_input("Account Balance ($)", value=1000)
category = st.sidebar.selectbox("Market Category", list(CATEGORIES.keys()))
for m in CATEGORIES[category]:
    col1, col2 = st.columns([8,1])
    if col1.button(m):
        st.session_state.selected_market = m
    if col2.button("⭐" if m in st.session_state.favorites else "☆", key=f"fav_{m}"):
        if m in st.session_state.favorites:
            st.session_state.favorites.remove(m)
        else:
            st.session_state.favorites.append(m)

# ---- HIGH MOVEMENT ----
movement_scores = {}
for m, info in MARKET_SYMBOLS.items():
    d = get_live_data(info)
    if d:
        movement_scores[m] = abs(d["change"])
high_move = sorted(movement_scores.items(), key=lambda x: x[1], reverse=True)[:3]
st.sidebar.subheader("🔥 High Movement")
for m, v in high_move:
    st.sidebar.write(f"{m}: {round(v, 5)}")

# ---- WATCHLIST ----
st.sidebar.markdown("---")
st.sidebar.subheader("⭐ Favorites")
for fav in st.session_state.favorites:
    exch, sym = MARKET_SYMBOLS[fav]
    df = tv.get_hist(sym, exch, interval=Interval.in_1_minute, n_bars=1)
    if df is not None and not df.empty:
        price = df.iloc[-1]["close"]
        st.sidebar.write(f"{fav}: {round(price, 5)}")

# ---- CHART ----
selected = st.session_state.selected_market
exch, sym = MARKET_SYMBOLS[selected]
st.subheader(f"📈 {selected} Live Chart")
st.components.v1.iframe(f"https://s.tradingview.com/widgetembed/?symbol={exch}:{sym}&interval=1&theme=dark", height=400)

# ---- REFRESH + SIGNAL ----
if st.button("🔁 Refresh Signal"):
    data = get_live_data((exch, sym))
    if data:
        signal = generate_signal(data, balance)

        st.subheader("📌 Market Data")
        st.write(f"Trend: {data['trend']}")
        st.write(f"Momentum: {data['momentum']} | Volatility: {data['volatility']}")
        st.write(f"Support: {data['support']} | Resistance: {data['resistance']}")
        st.write(f"RSI: {round(data['rsi'], 2)} | Pattern: {data['pattern']}")

        st.subheader("✅ Signal")
        st.write(f"Signal: {signal['signal']} | Confidence: {signal['confidence']}%")
        st.progress(signal['confidence'])
        st.write(f"Entry: {signal['entry']} | SL: {signal['stop_loss']} | TP: {signal['take_profit']}")
        st.write(f"Risk: ${signal['risk_amount']} | Reward: ${signal['reward_amount']}")
        st.write(f"Lot Size: {signal['lot_size']} | R:R Ratio: {signal['rr_ratio']}")
        if signal['reasons']:
            st.write("Reason:", ', '.join(signal['reasons']))

        # Save to history
        result = {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Market": selected,
            "Signal": signal['signal'],
            "Entry": signal['entry'],
            "SL": signal['stop_loss'],
            "TP": signal['take_profit'],
            "Result": "PENDING"
        }
        st.session_state.history.append(result)
    else:
        st.error("⚠️ Unable to fetch data")

# ---- HISTORY DISPLAY ----
st.markdown("---")
st.subheader("📊 Signal History")
if st.session_state.history:
    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist, use_container_width=True)
else:
    st.info("No signals yet.")
