# rayner_bot.py
import os
import time
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from tradingview_ta import TA_Handler, Interval

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="Rayner Trading Bot", layout="wide")
st.title("📈 Rayner Trading Bot (Fixed Version)")

# ---------------------------
# Market Config
# ---------------------------
MARKETS = {
    "EUR/USD": {"tv_symbol": "EURUSD", "exchange": "OANDA", "screener": "FOREX", "yf": "EURUSD=X"},
    "USD/JPY": {"tv_symbol": "USDJPY", "exchange": "OANDA", "screener": "FOREX", "yf": "JPY=X"},
    "XAU/USD (Gold)": {"tv_symbol": "XAUUSD", "exchange": "OANDA", "screener": "FOREX", "yf": "GC=F"},
    "BTC/USD": {"tv_symbol": "BTCUSDT", "exchange": "BINANCE", "screener": "CRYPTO", "yf": "BTC-USD"},
    "ETH/USD": {"tv_symbol": "ETHUSDT", "exchange": "BINANCE", "screener": "CRYPTO", "yf": "ETH-USD"},
}

TV_INTERVALS = {
    "1m": Interval.INTERVAL_1_MINUTE,
    "5m": Interval.INTERVAL_5_MINUTES,
    "15m": Interval.INTERVAL_15_MINUTES,
    "1h": Interval.INTERVAL_1_HOUR,
    "4h": Interval.INTERVAL_4_HOURS,
}

# ---------------------------
# Functions
# ---------------------------
def get_tv_analysis(symbol, exchange, screener, interval):
    """Fetch TradingView summary for a symbol."""
    try:
        handler = TA_Handler(
            symbol=symbol,
            exchange=exchange,
            screener=screener,
            interval=interval
        )
        return handler.get_analysis()
    except Exception as e:
        st.error(f"TradingView error: {e}")
        return None


def generate_signal(market_key, interval_key):
    """Generate trading signal with TP/SL."""
    m = MARKETS[market_key]
    analysis = get_tv_analysis(m["tv_symbol"], m["exchange"], m["screener"], TV_INTERVALS[interval_key])

    if not analysis:
        return {"error": "No analysis available."}

    summary = analysis.summary
    indicators = analysis.indicators

    # Direction
    rec = summary.get("RECOMMENDATION", "NEUTRAL").upper()
    direction = "WAIT"
    if "BUY" in rec:
        direction = "BUY"
    elif "SELL" in rec:
        direction = "SELL"

    # Entry price
    entry = indicators.get("close") or indicators.get("Close")
    if not entry and m.get("yf"):
        try:
            df = yf.download(m["yf"], period="1d", interval="1m")
            entry = float(df["Close"].iloc[-1])
        except:
            entry = None

    # Risk/Reward
    if entry:
        sl = round(entry * (0.99 if direction == "BUY" else 1.01), 5)
        tp = round(entry * (1.03 if direction == "BUY" else 0.97), 5)
        rr = 3.0
    else:
        sl = tp = rr = None

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "summary": summary,
    }


def run_backtest(market_key, interval="15m"):
    """Simple backtest with EMA/RSI strategy."""
    yf_symbol = MARKETS[market_key]["yf"]
    df = yf.download(yf_symbol, period="60d", interval=interval)

    if df.empty:
        return {"error": "No data for backtest"}

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["RSI"] = ta_rsi(df["Close"])

    signals = []
    for i in range(50, len(df)):
        if df["EMA20"].iloc[i] > df["EMA50"].iloc[i] and df["RSI"].iloc[i] > 50:
            signals.append("BUY")
        elif df["EMA20"].iloc[i] < df["EMA50"].iloc[i] and df["RSI"].iloc[i] < 50:
            signals.append("SELL")
        else:
            signals.append("WAIT")

    win_rate = (signals.count("BUY") + signals.count("SELL")) / len(signals) * 100
    return {"trades": len(signals), "win_rate": round(win_rate, 2)}


def ta_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def run_scanner(interval_key):
    results = []
    for m in MARKETS:
        analysis = get_tv_analysis(
            MARKETS[m]["tv_symbol"], MARKETS[m]["exchange"], MARKETS[m]["screener"], TV_INTERVALS[interval_key]
        )
        if analysis:
            rec = analysis.summary.get("RECOMMENDATION", "NEUTRAL")
            results.append({"Market": m, "Signal": rec})
        time.sleep(0.2)
    return pd.DataFrame(results)

# ---------------------------
# UI
# ---------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Settings")
    market = st.selectbox("Select Market", list(MARKETS.keys()))
    tf = st.selectbox("Timeframe", list(TV_INTERVALS.keys()))

    if st.button("🔮 Generate Signal"):
        sig = generate_signal(market, tf)
        if "error" in sig:
            st.error(sig["error"])
        else:
            st.success(f"{market} | {tf} | {sig['direction']}")
            st.write(f"Entry: {sig['entry']} | SL: {sig['sl']} | TP: {sig['tp']} | R:R = {sig['rr']}")

    if st.button("📊 Run Backtest"):
        res = run_backtest(market)
        if "error" in res:
            st.error(res["error"])
        else:
            st.info(f"Trades: {res['trades']} | Win Rate: {res['win_rate']}%")

    if st.button("📡 Run Market Scanner"):
        scan = run_scanner(tf)
        st.dataframe(scan)

with col2:
    st.subheader("📺 Live Chart")
    sym = MARKETS[market]["tv_symbol"]
    exch = MARKETS[market]["exchange"]
    iframe = f"""<iframe src="https://s.tradingview.com/widgetembed/?symbol={exch}:{sym}&interval=5&theme=dark"
                 width="100%" height="500" frameborder="0"></iframe>"""
    st.markdown(iframe, unsafe_allow_html=True)
