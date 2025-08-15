import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# -------------------------------
# Streamlit config
# -------------------------------
st.set_page_config(page_title="Rayner Bot - Trading Assistant", layout="wide")

# -------------------------------
# Helpers
# -------------------------------
def get_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        df = df.dropna().copy()
        if df.empty:
            return df
        # Ensure standard OHLC names
        if {"Open","High","Low","Close"}.issubset(df.columns):
            return df
        # Sometimes yfinance returns lowercase
        cols = {c.lower(): c for c in df.columns}
        df.rename(columns={
            cols.get("open","Open"): "Open",
            cols.get("high","High"): "High",
            cols.get("low","Low"): "Low",
            cols.get("close","Close"): "Close",
        }, inplace=True)
        return df.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return float(atr.iloc[-1])

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    return df

def generate_signal(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data"
    latest = df.iloc[-1]
    if latest["EMA20"] > latest["EMA50"] and latest["RSI"] < 70:
        return "BUY"
    elif latest["EMA20"] < latest["EMA50"] and latest["RSI"] > 30:
        return "SELL"
    else:
        return "HOLD"

# ATR multipliers tuned per market type (tighter for FX, wider for crypto/commodities)
ATR_SETTINGS = {
    "Forex":      {"sl_mult": 1.0, "tp_mult": 2.0},   # R:R = 1:2 default
    "Crypto":     {"sl_mult": 1.5, "tp_mult": 3.0},
    "Commodities":{"sl_mult": 1.2, "tp_mult": 2.4},
    "Indian":     {"sl_mult": 1.0, "tp_mult": 2.0},
}

def calc_tp_sl(entry: float, signal: str, atr: float, market_type: str) -> tuple[float|None,float|None,float]:
    s = ATR_SETTINGS.get(market_type, ATR_SETTINGS["Forex"])
    sl_mult = s["sl_mult"]
    tp_mult = s["tp_mult"]
    if atr is None or atr <= 0:
        return None, None, 0.0
    if signal == "BUY":
        sl = entry - sl_mult * atr
        tp = entry + tp_mult * atr
    elif signal == "SELL":
        sl = entry + sl_mult * atr
        tp = entry - tp_mult * atr
    else:
        return None, None, tp_mult / max(sl_mult, 1e-9)
    return float(tp), float(sl), float(tp_mult / sl_mult)

# -------------------------------
# Market maps (Yahoo → TradingView)
# -------------------------------
MARKETS = {
    "Forex": {
        "EUR/USD": ("EURUSD=X", "FX:EURUSD"),
        "GBP/JPY": ("GBPJPY=X", "FX:GBPJPY"),
        "USD/JPY": ("JPY=X",     "FX:USDJPY"),
        "AUD/USD": ("AUDUSD=X",  "FX:AUDUSD"),
        "XAU/USD": ("XAUUSD=X",  "FX_IDC:XAUUSD")
    },
    "Crypto": {
        "Bitcoin":  ("BTC-USD", "CRYPTO:BTCUSD"),
        "Ethereum": ("ETH-USD", "CRYPTO:ETHUSD"),
        "Litecoin": ("LTC-USD", "CRYPTO:LTCUSD")
    },
    "Commodities": {
        "Gold Futures":  ("GC=F", "CME:GC1!"),
        "Silver Futures":("SI=F", "COMEX:SI1!"),
        "Crude Oil WTI": ("CL=F", "NYMEX:CL1!")
    },
    "Indian": {
        "NIFTY 50": ("^NSEI",     "NSE:NIFTY"),
        "BANKNIFTY":("^NSEBANK",  "NSE:BANKNIFTY"),
        "SENSEX":   ("^BSESN",    "BSE:SENSEX")
    }
}

YF_INTERVALS = ["1m","5m","15m","30m","60m","1d"]
TV_INTERVAL_MAP = {
    "1m":"1", "5m":"5", "15m":"15", "30m":"30", "60m":"60", "1d":"D"
}

# -------------------------------
# UI
# -------------------------------
st.title("📈 Rayner Bot - Trading Assistant")
st.caption("EMA20 vs EMA50 + RSI filter with ATR-based TP/SL. TradingView live chart + Risk/Reward box.")

c1, c2, c3 = st.columns(3)
with c1:
    mtype = st.selectbox("Market Type", list(MARKETS.keys()))
with c2:
    sym_name = st.selectbox("Select Symbol", list(MARKETS[mtype].keys()))
    yf_symbol, tv_symbol = MARKETS[mtype][sym_name]
with c3:
    interval = st.selectbox("Time Interval", YF_INTERVALS, index=2)  # default 15m

period = st.selectbox("Data Period", ["1d","5d","1mo","3mo","6mo","1y"], index=1)

# -------------------------------
# Generate
# -------------------------------
if st.button("Generate Signal"):
    df = get_data(yf_symbol, interval, period)
    if df.empty or len(df) < 30:
        st.error("No data or too few candles. Try a longer period.")
    else:
        df = calculate_indicators(df)
        atr = compute_atr(df, 14)
        direction = generate_signal(df)
        entry = float(df["Close"].iloc[-1])

        tp, sl, rr = calc_tp_sl(entry, direction, atr, mtype)

        # Summary
        st.subheader(f"📊 Signal: {direction}")
        colA, colB, colC, colD = st.columns(4)
        with colA:
            st.metric("Entry", f"{entry:.5f}")
        with colB:
            st.metric("ATR(14)", f"{atr:.5f}")
        with colC:
            st.metric("R:R", f"1:{rr:.2f}" if rr else "—")
        with colD:
            st.metric("Timeframe", interval)

        if direction in ("BUY","SELL"):
            st.write(f"🛑 **Stop Loss:** `{sl:.5f}` &nbsp;&nbsp; 🎯 **Take Profit:** `{tp:.5f}`")

        # Last rows
        with st.expander("Last 10 rows"):
            st.dataframe(df.tail(10))

        # TradingView live chart
        tv_int = TV_INTERVAL_MAP.get(interval, "15")
        st.markdown(
            f"""
            <iframe
              src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval={tv_int}&hidesidetoolbar=1&symboledit=1&saveimage=0&toolbarbg=F1F3F6&studies=[]&theme=dark&style=1&timezone=Etc/UTC"
              width="100%" height="520" frameborder="0" allowtransparency="true" scrolling="no">
            </iframe>
            """,
            unsafe_allow_html=True
        )

        # Plotly candle with Risk/Reward box
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Price"
        ))
        # EMAs
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], mode="lines", name="EMA20"))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], mode="lines", name="EMA50"))

        # RR box for latest bar (visual)
        if direction in ("BUY","SELL") and tp is not None and sl is not None:
            x0 = df.index[-25] if len(df) > 25 else df.index[0]
            x1 = df.index[-1]
            y0 = min(sl, entry)
            y1 = max(tp, entry) if direction == "BUY" else max(entry, sl)
            # Box
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=x0, x1=x1, y0=y0, y1=y1,
                line=dict(width=1), fillcolor="rgba(0,150,0,0.10)" if direction=="BUY" else "rgba(200,0,0,0.10)"
            )
            # Entry/SL/TP lines
            fig.add_hline(y=entry, line_dash="dot", annotation_text=f"Entry {entry:.5f}")
            fig.add_hline(y=sl, line_dash="dot", annotation_text=f"SL {sl:.5f}")
            fig.add_hline(y=tp, line_dash="dot", annotation_text=f"TP {tp:.5f}")

        fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
