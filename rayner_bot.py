# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os

st.set_page_config(page_title="Rayner Style Risk Bot", layout="wide")
st.title("📈 Rayner Style Risk Bot — ATR-based Risk Engine")

# ---------------------------
# MARKET SYMBOL CONFIG
# (yfinance tickers where possible)
# ---------------------------
MARKETS = {
    # Forex (yfinance uses pairs like EURUSD=X)
    "EUR/USD": {"ticker": "EURUSD=X", "type": "forex", "pip": 0.0001, "pip_value_per_lot": 10},
    "GBP/JPY": {"ticker": "GBPJPY=X", "type": "forex", "pip": 0.01, "pip_value_per_lot": 9.12},
    "USD/JPY": {"ticker": "JPY=X", "type": "forex", "pip": 0.01, "pip_value_per_lot": 9.12},
    "AUD/USD": {"ticker": "AUDUSD=X", "type": "forex", "pip": 0.0001, "pip_value_per_lot": 10},
    # Commodities
    "Gold (GC)": {"ticker": "GC=F", "type": "commodity", "pip": 0.1, "pip_value_per_lot": 100},
    "Oil WTI (CL)": {"ticker": "CL=F", "type": "commodity", "pip": 0.01, "pip_value_per_lot": 10},
    # Crypto
    "BTC/USD": {"ticker": "BTC-USD", "type": "crypto", "pip": 1.0, "pip_value_per_lot": 1},
    "ETH/USD": {"ticker": "ETH-USD", "type": "crypto", "pip": 0.1, "pip_value_per_lot": 1},
    # Indices (common tickers)
    "NIFTY 50": {"ticker": "^NSEI", "type": "index", "pip": 1.0, "pip_value_per_lot": 1},
    "BANKNIFTY": {"ticker": "^NSEBANK", "type": "index", "pip": 1.0, "pip_value_per_lot": 1},
}

# ---------------------------
# Helper functions
# ---------------------------
def fetch_ohlc(ticker, period="7d", interval="1m"):
    """Fetch OHLC data from yfinance."""
    try:
        df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return None
        df = df.dropna()
        return df
    except Exception as e:
        return None

def atr(df, period=14):
    """Calculate ATR (True Range rolling mean)"""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr

def ema(df, span=20):
    return df["Close"].ewm(span=span, adjust=False).mean()

def detect_trend(df):
    ma5 = df["Close"].rolling(5).mean().iloc[-1]
    ma20 = df["Close"].rolling(20).mean().iloc[-1] if len(df) >= 20 else ema(df).iloc[-1]
    return "uptrend" if ma5 > ma20 else "downtrend"

def calc_sl_tp(entry, side, atr_val, market_info, rr=3.0, atr_mult=1.5):
    """SL based on ATR × multiplier, TP = SL * RR"""
    # For crypto we use percent-based approach to avoid massive values
    mtype = market_info["type"]
    pip = market_info["pip"]
    if mtype == "crypto":
        # SL = entry * (ATR_percent * atr_mult)
        # approximate ATR percent
        sl_distance = atr_val * atr_mult
    else:
        sl_distance = atr_val * atr_mult
    if side == "BUY":
        sl = entry - sl_distance
        tp = entry + sl_distance * rr
    else:
        sl = entry + sl_distance
        tp = entry - sl_distance * rr
    # Round according to pip step
    try:
        step = pip
        sl = round(sl / step) * step
        tp = round(tp / step) * step
    except Exception:
        pass
    return sl, tp, abs(sl - entry)

def calculate_lot_size(account_balance, risk_percent, risk_amount_price_diff, market_info):
    """Estimate lot size / units using risk amount and pip value."""
    risk_amount_money = account_balance * (risk_percent / 100)
    pip_value_per_lot = market_info.get("pip_value_per_lot", 1)
    # avoid division by zero
    if risk_amount_price_diff == 0:
        return 0
    # convert price diff to pips
    pip = market_info.get("pip", 1)
    pips = risk_amount_price_diff / pip
    if pips == 0:
        return 0
    # number of standard lots (approx) = risk money / (pips * pip value per lot)
    lots = risk_amount_money / (abs(pips) * pip_value_per_lot)
    # For instruments where lot concept not meaningful (crypto/index), return units estimated
    return round(lots, 4)

def save_signal_history(row, filename="signal_history.csv"):
    cols = ["timestamp","market","signal","entry","sl","tp","confidence","risk_percent","lot_size"]
    df_row = pd.DataFrame([row], columns=cols)
    if not os.path.exists(filename):
        df_row.to_csv(filename, index=False)
    else:
        df_row.to_csv(filename, mode="a", header=False, index=False)

def load_history(filename="signal_history.csv"):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return pd.DataFrame(columns=["timestamp","market","signal","entry","sl","tp","confidence","risk_percent","lot_size"])

# ---------------------------
# UI: Sidebar (settings)
# ---------------------------
st.sidebar.header("⚙️ Account & Settings")
account_balance = st.sidebar.number_input("Account Balance ($)", value=1000.0, min_value=1.0, step=50.0)
risk_percent = st.sidebar.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
timeframe = st.sidebar.selectbox("Chart timeframe", ["1m","5m","15m","1h","4h"], index=0)
auto_refresh = st.sidebar.number_input("Auto-refresh every N seconds (0=off)", min_value=0, value=0, step=10)
st.sidebar.markdown("---")
st.sidebar.markdown("Made for manual execution. No auto-trading included.")

# Favorites (session)
if "favorites" not in st.session_state:
    st.session_state.favorites = []

# ---------------------------
# Left column: favorites, high movement
# ---------------------------
col1, col2 = st.columns([1,3])

with col1:
    st.subheader("🔥 High Movement Markets")
    # compute movement using past N candles' std dev for each market
    movement = {}
    for m, info in MARKETS.items():
        dfm = fetch_ohlc(info["ticker"], period="2d", interval=timeframe)
        if dfm is not None and not dfm.empty:
            movement[m] = float((dfm["Close"].pct_change().abs().sum())*100)
    sorted_m = sorted(movement.items(), key=lambda x: x[1], reverse=True)[:5]
    for m, score in sorted_m:
        st.write(f"{m}: {round(score,3)}")

    st.markdown("---")
    st.subheader("⭐ Watchlist / Favorites")
    for m in MARKETS.keys():
        fav = "⭐" if m in st.session_state.favorites else "☆"
        if st.button(f"{fav} {m}", key=f"fav_{m}"):
            if m in st.session_state.favorites:
                st.session_state.favorites.remove(m)
            else:
                st.session_state.favorites.append(m)
    st.write("Your favorites:", st.session_state.favorites)

# ---------------------------
# Main column: select market & show chart
# ---------------------------
with col2:
    st.subheader("Select Market")
    market_choice = st.selectbox("Market", list(MARKETS.keys()))
    market_info = MARKETS[market_choice]
    ticker = market_info["ticker"]

    # TradingView iframe embed (visual only)
    st.markdown("### Live chart")
    symbol_for_tv = ticker.replace("^","").replace("=X","").replace("-USD","USD").replace("BTC-USD","BINANCE:BTCUSDT")
    # Use a safe TradingView URL; for many tickers TradingView accepts "OANDA:EURUSD"
    iframe_url = f"https://s.tradingview.com/widgetembed/?symbol={ticker}&interval={timeframe}&theme=dark"
    st.components.v1.iframe(f"https://s.tradingview.com/widgetembed/?frameElementId=tradingview_{symbol_for_tv}&symbol={ticker}&interval=1&hidesidetoolbar=1&symboledit=1&saveimage=1&toolbarbg=f1f3f6&studies=[]&theme=dark&style=1&timezone=Etc%2FUTC&withdateranges=1&hidevolume=0", height=480)

    # Manual generate button
    if st.button("🔮 Generate Signal"):
        with st.spinner("Fetching data..."):
            # Fetch OHLC for analysis
            df = fetch_ohlc(ticker, period="7d", interval=timeframe)
            if df is None or df.empty:
                st.error("Unable to fetch live OHLC data for the selected market/timeframe. Try again or choose a different timeframe.")
            else:
                # compute indicators
                atr_series = atr(df, period=14)
                atr_val = float(atr_series.iloc[-1])
                trend = detect_trend(df)
                momentum = "strong" if abs(df["Close"].pct_change().iloc[-1]) > 0.001 else "weak"
                support = float(df["Low"].rolling(20, min_periods=1).min().iloc[-1])
                resistance = float(df["High"].rolling(20, min_periods=1).max().iloc[-1])
                entry = float(df["Close"].iloc[-1])

                # Decide side using simple Rayner-style breakout logic (preserve user's original technique)
                side = "WAIT"
                confidence = 50
                if trend == "uptrend" and entry > support and momentum == "strong" and atr_val > 0:
                    side = "BUY"
                    confidence = min(100, int(50 + (atr_val*100)))
                elif trend == "downtrend" and entry < resistance and momentum == "strong" and atr_val > 0:
                    side = "SELL"
                    confidence = min(100, int(50 + (atr_val*100)))
                else:
                    side = "WAIT"
                    confidence = 40

                # Calculate SL/TP using ATR-based engine and per market pip rounding
                sl, tp, sl_distance = calc_sl_tp(entry, side if side != "WAIT" else "BUY", atr_val, market_info, rr=3.0, atr_mult=1.5)
                if side == "WAIT":
                    st.warning("No clear signal based on current rules (WAIT). You can inspect snapshot below.")
                else:
                    # recommended lot size
                    lot_size = calculate_lot_size(account_balance, risk_percent, sl_distance, market_info)
                    st.success(f"Signal: {side}")
                    st.markdown(f"- **Trend:** {trend}")
                    st.markdown(f"- **Momentum:** {momentum}")
                    st.markdown(f"- **Volatility (ATR):** {round(atr_val,6)}")
                    st.markdown(f"- **Support:** {round(support,6)} | **Resistance:** {round(resistance,6)}")
                    st.markdown(f"**Entry:** {round(entry,6)}  |  **SL:** {sl}  |  **TP:** {tp}")
                    st.markdown(f"**Risk per trade:** {risk_percent}% (${round(account_balance * risk_percent/100,2)})")
                    st.markdown(f"**Recommended lots (approx):** {lot_size}")
                    st.progress(min(confidence,100))

                    # save to history
                    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    row = [timestamp, market_choice, side, entry, sl, tp, confidence, risk_percent, lot_size]
                    save_signal_history(row)
                    st.info("Signal saved to history.")

    # Show a small market snapshot if not generating
    st.markdown("---")
    st.subheader("Market Snapshot (quick)")
    dfq = fetch_ohlc(ticker, period="2d", interval=timeframe)
    if dfq is None or dfq.empty:
        st.write("No quick data available.")
    else:
        atr_q = float(atr(dfq).iloc[-1])
        trend_q = detect_trend(dfq)
        last_price = float(dfq["Close"].iloc[-1])
        st.write(f"Price: {round(last_price,6)}  |  Trend: {trend_q}  |  ATR: {round(atr_q,6)}")

# ---------------------------
# Right column: History table
# ---------------------------
col3 = st.columns([1])[0]
st.markdown("---")
st.subheader("📜 Signal History")
history_df = load_history()
if history_df.empty:
    st.write("No history yet.")
else:
    st.dataframe(history_df.sort_values("timestamp", ascending=False).reset_index(drop=True), height=300)

# ---------------------------
# Footer: small tips
# ---------------------------
st.markdown("---")
st.caption("This app provides signals for manual trading only. Always confirm with your own analysis. Built with yfinance (data may be delayed).")

# Auto-refresh (if enabled) - simple rerun
if auto_refresh and auto_refresh > 0:
    st.experimental_rerun()
