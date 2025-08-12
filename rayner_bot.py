# trading_bot.py
# Single-file Streamlit trading signal app (no background image).
# Works on Streamlit Cloud (uses yfinance + pandas + numpy + ta).
# Save alongside: requirements.txt (see below).

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os

st.set_page_config(page_title="Rayner Style Risk Bot", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# --- CONFIG / SYMBOLS ---
# -------------------------
MARKETS = {
    "EUR/USD": {"tv": "FX:EURUSD", "yf": "EURUSD=X", "type": "forex", "pip": 0.0001},
    "GBP/JPY": {"tv": "FX:GBPJPY", "yf": "GBPJPY=X", "type": "forex", "pip": 0.01},
    "USD/JPY": {"tv": "FX:USDJPY", "yf": "JPY=X", "type": "forex", "pip": 0.01},  # fallback
    "AUD/USD": {"tv": "FX:AUDUSD", "yf": "AUDUSD=X", "type": "forex", "pip": 0.0001},
    "XAU/USD": {"tv": "OANDA:XAUUSD", "yf": "XAUUSD=X", "type": "commodity", "pip": 0.01},
    "BTC/USD": {"tv": "BINANCE:BTCUSDT", "yf": "BTC-USD", "type": "crypto", "pip": 1},
    "ETH/USD": {"tv": "BINANCE:ETHUSDT", "yf": "ETH-USD", "type": "crypto", "pip": 0.1},
    "NIFTY 50": {"tv": "NSE:NIFTY", "yf": "^NSEI", "type": "index", "pip": 1},
    "BANKNIFTY": {"tv": "NSE:BANKNIFTY", "yf": "^NSEBANK", "type": "index", "pip": 1},
    "WTI": {"tv": "NYMEX:CL1!", "yf": "CL=F", "type": "commodity", "pip": 0.01}
}

# ensure signal history file exists
HISTORY_FILE = "signal_history.csv"
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["timestamp","market","timeframe","signal","entry","sl","tp","risk_usd","lot","result"]).to_csv(HISTORY_FILE, index=False)

# -------------------------
# --- Sidebar controls ---
# -------------------------
st.sidebar.header("Account & Settings")
account_balance = st.sidebar.number_input("Account Balance ($)", value=1000.0, min_value=1.0, step=100.0, format="%.2f")
risk_percent = st.sidebar.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
timeframe = st.sidebar.selectbox("Chart timeframe", ["1m","5m","15m","1h","1d"])
auto_refresh = st.sidebar.number_input("Auto-refresh (sec, 0=off)", min_value=0, value=0, step=1)
st.sidebar.markdown("Made for manual execution. No auto-trading included.")

# -------------------------
# --- Main layout ---
# -------------------------
col1, col2 = st.columns([1.6, 3])
with col1:
    st.subheader("Watchlist / Favorites")
    # favorites stored in session_state
    if "favorites" not in st.session_state:
        st.session_state.favorites = ["EUR/USD","BTC/USD","XAU/USD"]

    # list favorites with add/remove
    for m in MARKETS.keys():
        fav = m in st.session_state.favorites
        if st.button(("★ " if fav else "☆ ") + m, key=f"fav_{m}"):
            if fav:
                st.session_state.favorites.remove(m)
            else:
                st.session_state.favorites.append(m)

    st.markdown("---")
    st.subheader("High Movement (live scan)")
    # quick volatility ranking (small sample)
    movement_scores = {}
    for k,v in MARKETS.items():
        try:
            df0 = yf.download(tickers=v["yf"], period="2d", interval="1d", progress=False)
            if not df0.empty:
                movement_scores[k] = df0["Close"].pct_change().abs().mean()
        except Exception:
            movement_scores[k] = 0
    top = sorted(movement_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for name,score in top:
        st.markdown(f"- **{name}**: {round(score*100,3)}%")

with col2:
    st.title("Rayner Style Risk Bot")
    st.markdown("Select market from left favorites or pick below.")
    selected_market = st.selectbox("Select Market", list(MARKETS.keys()), index=0)
    st.markdown("---")

    # TradingView iframe (fill center)
    tv_symbol = MARKETS[selected_market]["tv"]
    iframe_url = f"https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval={timeframe}&hidesidetoolbar=1&theme=dark&style=1"
    st.components.v1.iframe(src=iframe_url, height=480)

    st.markdown("---")
    st.subheader("Signal Summary")

    # -------------------------
    # --- Fetch OHLC data ---
    # -------------------------
    yf_symbol = MARKETS[selected_market]["yf"]
    interval_map = {"1m":"1m","5m":"5m","15m":"15m","1h":"60m","1d":"1d"}
    yf_interval = interval_map.get(timeframe, "1m")
    # yfinance supports intraday for period <=7d for minute intervals
    period = "7d" if yf_interval.endswith("m") else "2y"
    try:
        df = yf.download(tickers=yf_symbol, period=period, interval=yf_interval, progress=False)
    except Exception as e:
        df = pd.DataFrame()

    if df.empty:
        st.error("Unable to fetch live OHLC data for the selected market/timeframe. Try a different timeframe.")
    else:
        df = df.dropna().tail(500)

        # indicators: EMA20, EMA50, RSI(14), ATR(14)
        df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
        delta = df["Close"].diff()
        up = delta.clip(lower=0)
        down = -1*delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up/ema_down
        df["rsi"] = 100 - (100/(1+rs))
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean().fillna(method="bfill")

        latest = df.iloc[-1]
        price = latest["Close"]
        ema20 = latest["ema20"]
        ema50 = latest["ema50"]
        rsi = latest["rsi"]
        atr = latest["atr"]

        st.markdown(f"**Price:** {price:.5f}  •  **EMA20:** {ema20:.5f}  •  **EMA50:** {ema50:.5f}")
        st.markdown(f"**RSI(14):** {rsi:.2f}  •  **ATR(14):** {atr:.5f}")

        # -------------------------
        # --- Signal logic (kept simple & robust) ---
        # -------------------------
        # Decision rules:
        #  - BUY if EMA20 > EMA50 AND RSI < 30 (oversold in uptrend)
        #  - SELL if EMA20 < EMA50 AND RSI > 70 (overbought in downtrend)
        signal = "WAIT"
        reasons = []
        if ema20 > ema50 and rsi < 40:
            signal = "BUY"
            reasons.append("EMA20 > EMA50 + low RSI (momentum confirmation)")
        elif ema20 < ema50 and rsi > 60:
            signal = "SELL"
            reasons.append("EMA20 < EMA50 + high RSI (momentum confirmation)")
        else:
            signal = "WAIT"
            reasons.append("No clear multi-factor confirmation")

        # -------------------------
        # --- TP/SL calculation (adaptive to market movement) ---
        # -------------------------
        # Use ATR-based SL distance and conservative multiplier depending on asset type
        asset_type = MARKETS[selected_market]["type"]
        # base multiplier per market type (smaller for forex, larger for crypto)
        type_mult = {"forex":1.5, "crypto":2.5, "commodity":1.8, "index":1.5}
        mult = type_mult.get(asset_type, 1.7)
        sl_distance = max(atr * mult, MARKETS[selected_market]["pip"] * 5)  # ensure minimum
        # Use RR 1:3 default (can be adjusted later)
        rr = 3.0
        if signal == "BUY":
            entry = price
            sl = entry - sl_distance
            tp = entry + sl_distance * rr
        elif signal == "SELL":
            entry = price
            sl = entry + sl_distance
            tp = entry - sl_distance * rr
        else:
            entry = price
            sl = None
            tp = None

        # position sizing: risk_amount = account_balance * risk_percent/100
        risk_amount = account_balance * (risk_percent/100)
        # compute pip distance in units for lot calc
        pip_value = {"forex":10, "crypto":1, "commodity":1, "index":1}.get(asset_type,1)
        pip_size = MARKETS[selected_market]["pip"]
        if sl:
            sl_pips = abs((entry - sl) / pip_size)
            # avoid divide by zero
            lot = (risk_amount / (sl_pips * pip_value)) if sl_pips > 0 else 0
            lot = max(round(lot,4), 0)
        else:
            lot = 0

        # -------------------------
        # --- Show results & save history (button) ---
        # -------------------------
        col_a, col_b = st.columns([2,1])
        with col_a:
            st.markdown(f"**Signal:** `{signal}`")
            st.progress(100 if signal != "WAIT" else 5)
            st.markdown(f"**Entry:** {entry:.5f}" + (f"  |  **SL:** {sl:.5f}  |  **TP:** {tp:.5f}" if sl else ""))
            st.markdown(f"**Risk ${risk_amount:.2f}**  •  **Estimated Lot:** {lot}")
            st.markdown("**Reasons:** " + ("; ".join(reasons)))
            st.caption(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        with col_b:
            if st.button("Save Signal to History"):
                df_hist = pd.read_csv(HISTORY_FILE)
                new = {
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "market": selected_market,
                    "timeframe": timeframe,
                    "signal": signal,
                    "entry": round(entry,5),
                    "sl": round(sl,5) if sl else "",
                    "tp": round(tp,5) if tp else "",
                    "risk_usd": round(risk_amount,2),
                    "lot": lot,
                    "result": ""
                }
                df_hist = pd.concat([df_hist, pd.DataFrame([new])], ignore_index=True)
                df_hist.to_csv(HISTORY_FILE, index=False)
                st.success("Saved to history.")

        st.markdown("---")
        st.subheader("Recent Signal History")
        try:
            hist = pd.read_csv(HISTORY_FILE).tail(10).reset_index(drop=True)
            st.dataframe(hist)
        except Exception:
            st.info("No history yet.")

# optional auto-refresh (handled by streamlit's rerun)
if auto_refresh > 0:
    st.experimental_rerun()
