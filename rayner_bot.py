import streamlit as st
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import datetime

tv = TvDatafeed()

# Predefined market list
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

# Risk settings (adjust as needed)
DEFAULT_RISK_PERCENT = 1  # 1% risk per trade

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
        "df": df
    }

def calculate_risk(account_balance, sl, entry_price):
    risk_amount = account_balance * (DEFAULT_RISK_PERCENT / 100)
    pip_risk = abs(entry_price - sl)
    if pip_risk == 0:
        return 0, 0
    lot_size = round(risk_amount / pip_risk, 2)
    return lot_size, round(risk_amount, 2)

def generate_signal(data, account_balance):
    reasons = []
    entry = data["price"]
    sl = None
    tp = None

    if data["trend"] == "uptrend" and entry > data["support"]:
        if data["momentum"] == "strong" and data["volatility"] > 50:
            sl = entry - 0.0015
            tp = entry + (entry - sl) * 3
            signal = "BUY"
            reasons.append("Breakout confirmation in uptrend")
        else:
            signal = "WAIT"
    elif data["trend"] == "downtrend" and entry < data["resistance"]:
        if data["momentum"] == "strong" and data["volatility"] > 50:
            sl = entry + 0.0015
            tp = entry - (sl - entry) * 3
            signal = "SELL"
            reasons.append("Breakout confirmation in downtrend")
        else:
            signal = "WAIT"
    else:
        signal = "WAIT"

    lot_size, risk_amount = calculate_risk(account_balance, sl, entry) if sl else (0, 0)

    return {
        "signal": signal,
        "entry": entry,
        "stop_loss": round(sl, 5) if sl else None,
        "take_profit": round(tp, 5) if tp else None,
        "confidence": data["signal_strength"],
        "lot_size": lot_size,
        "risk_amount": risk_amount,
        "reasons": reasons
    }

# UI
st.set_page_config(page_title="Trading Bot", layout="wide")
st.markdown("""
    <style>
    .main {
        background: url('https://images.unsplash.com/photo-1605902711622-cfb43c44367d') no-repeat center center fixed;
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Professional Trading Bot with Risk Management")

col1, col2 = st.columns([1, 3])

with col1:
    account_balance = st.number_input("💰 Enter Account Balance ($)", value=1000.0)
    selected_market = st.selectbox("📊 Select Market", list(MARKET_SYMBOLS.keys()))
    refresh = st.button("🔁 Refresh Signal")

with col2:
    if refresh:
        st.subheader(f"📍 Market Analysis: {selected_market}")
        data = get_live_data(MARKET_SYMBOLS[selected_market])
        if data:
            signal_data = generate_signal(data, account_balance)

            symbol_code = MARKET_SYMBOLS[selected_market][1]
            exchange = MARKET_SYMBOLS[selected_market][0]
            iframe_url = f"https://s.tradingview.com/widgetembed/?frameElementId=tradingview_{symbol_code}&symbol={exchange}%3A{symbol_code}&interval=1&theme=dark&style=1"
            st.components.v1.iframe(iframe_url, height=400)

            st.markdown(f"### 📢 Signal: {signal_data['signal']}")
            st.markdown(f"**Confidence:** {signal_data['confidence']}%")
            st.markdown(f"**Entry Price:** {signal_data['entry']}")
            if signal_data["stop_loss"]:
                st.markdown(f"**Stop Loss:** {signal_data['stop_loss']}")
                st.markdown(f"**Take Profit (1:3):** {signal_data['take_profit']}")
                st.markdown(f"**Lot Size:** {signal_data['lot_size']}")
                st.markdown(f"**Risk Amount:** ${signal_data['risk_amount']}")

            st.markdown("### Reason for Signal")
            if signal_data["reasons"]:
                for r in signal_data["reasons"]:
                    st.markdown(f"- {r}")
            else:
                st.info("⚠️ No strong signal identified.")
        else:
            st.error("⚠️ Unable to fetch live data.")
    else:
        st.info("🔄 Click 'Refresh Signal' to fetch and analyze live market data.")
