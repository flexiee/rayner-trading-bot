import sys
from datetime import datetime

try:
    import streamlit as st
    from streamlit.components.v1 import iframe
    import ta
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    sys.exit("Please install tvDatafeed: pip install git+https://github.com/rongardF/tvdatafeed.git")

tv = TvDatafeed()

# --- Market Symbols Map ---
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

# --- Data Fetch and Indicator Logic ---
def get_data(symbol_info, interval, bars=50):
    exchange, symbol = symbol_info
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=bars)
    return df

def add_indicators(df):
    df["ema9"] = ta.trend.ema_indicator(df["close"], window=9)
    df["ema21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    return df

def get_analysis(symbol_info):
    df_1m = get_data(symbol_info, Interval.in_1_minute)
    df_5m = get_data(symbol_info, Interval.in_5_minute)

    if df_1m is None or df_5m is None or df_1m.empty or df_5m.empty:
        return None

    df_1m = add_indicators(df_1m)
    df_5m = add_indicators(df_5m)

    last_1m = df_1m.iloc[-1]
    last_5m = df_5m.iloc[-1]

    trend_1m = "uptrend" if last_1m["ema9"] > last_1m["ema21"] else "downtrend"
    trend_5m = "uptrend" if last_5m["ema9"] > last_5m["ema21"] else "downtrend"

    rsi_ok = last_1m["rsi"] > 50 if trend_1m == "uptrend" else last_1m["rsi"] < 50
    ema_ok = (trend_1m == trend_5m)

    return {
        "price": round(last_1m["close"], 5),
        "trend": trend_1m,
        "trend_match": trend_1m == trend_5m,
        "rsi": round(last_1m["rsi"], 2),
        "ema_fast": round(last_1m["ema9"], 5),
        "ema_slow": round(last_1m["ema21"], 5),
        "support": round(df_1m["low"].min(), 5),
        "resistance": round(df_1m["high"].max(), 5),
        "rsi_confirm": rsi_ok,
        "ema_confirm": ema_ok
    }

# --- Signal Generator ---
def generate_signal(analysis, balance, risk_percent=1):
    entry = analysis["price"]
    sl = None
    tp = None
    signal = "WAIT"
    reasons = []

    if analysis["trend"] == "uptrend" and analysis["trend_match"] and analysis["rsi_confirm"] and analysis["ema_confirm"]:
        sl = entry - 0.0015
        tp = entry + (entry - sl) * 3
        signal = "BUY"
        reasons.append("Uptrend confirmed (EMA & RSI)")

    elif analysis["trend"] == "downtrend" and analysis["trend_match"] and analysis["rsi_confirm"] and analysis["ema_confirm"]:
        sl = entry + 0.0015
        tp = entry - (sl - entry) * 3
        signal = "SELL"
        reasons.append("Downtrend confirmed (EMA & RSI)")

    risk_amt = balance * (risk_percent / 100)
    pip_risk = abs(entry - sl) if sl else 0
    lot_size = round(risk_amt / (pip_risk * 10), 2) if pip_risk else 0

    return {
        "signal": signal,
        "entry": entry,
        "stop_loss": round(sl, 5) if sl else None,
        "take_profit": round(tp, 5) if tp else None,
        "risk": round(risk_amt, 2),
        "reward": round(risk_amt * 3, 2),
        "lot_size": lot_size,
        "reasons": reasons
    }

# --- Streamlit Interface ---
if STREAMLIT_AVAILABLE:
    def run_ui():
        st.set_page_config(layout="wide", page_title="Smart Trading Bot")
        st.title("📈 Smart Trading Bot (Multi-Timeframe, RSI, EMA, Risk Management)")

        market = st.selectbox("📌 Select Market", list(MARKET_SYMBOLS.keys()))
        exch, sym = MARKET_SYMBOLS[market]
        iframe(f"https://s.tradingview.com/widgetembed/?symbol={exch}:{sym}&interval=1&theme=dark", height=400)

        balance = st.number_input("💰 Account Balance ($)", min_value=100, value=1000)
        risk_percent = st.slider("📉 Risk per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

        st.markdown("----")

        with st.spinner("⏳ Analyzing market..."):
            analysis = get_analysis((exch, sym))
            if analysis:
                signal = generate_signal(analysis, balance, risk_percent)

                st.subheader("📊 Technical Snapshot")
                st.markdown(f"🔹 Trend (1m): **{analysis['trend']}**")
                st.markdown(f"🔹 RSI: **{analysis['rsi']}**")
                st.markdown(f"🔹 EMA 9: {analysis['ema_fast']} | EMA 21: {analysis['ema_slow']}")
                st.markdown(f"🔹 Support: {analysis['support']} | Resistance: {analysis['resistance']}")
                st.markdown(f"🔹 Timeframe Agreement (1m vs 5m): `{analysis['trend_match']}`")

                st.subheader("✅ Signal")
                st.markdown(f"📈 Signal: `{signal['signal']}`")
                st.markdown(f"💲 Entry: {signal['entry']} | SL: {signal['stop_loss']} | TP: {signal['take_profit']}")
                st.markdown(f"📉 Risk: ${signal['risk']} | 🟢 Reward: ${signal['reward']} | 📦 Lot Size: `{signal['lot_size']}`")
                if signal["reasons"]:
                    st.markdown("📝 Reason: " + " | ".join(signal["reasons"]))
                st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.error("⚠️ Failed to retrieve data.")

    if __name__ == "__main__":
        run_ui()
else:
    print("Streamlit not installed.")
