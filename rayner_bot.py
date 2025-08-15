import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# --------------------------- #
# App Config
# --------------------------- #
st.set_page_config(page_title="Rayner Bot – Pro Signal", layout="wide")
st.markdown(
    """
    <style>
      .metric-small { font-size:13px; opacity:.7 }
      .good {color:#22c55e;font-weight:600}
      .bad {color:#ef4444;font-weight:600}
      .neutral {color:#a3a3a3;font-weight:600}
      .rrbar {height:8px;border-radius:6px;background:linear-gradient(90deg,#ef4444 0 33%,#f59e0b 33% 66%,#22c55e 66% 100%);}
      .box{border:1px solid #333;border-radius:10px;padding:14px;margin-bottom:10px}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------- #
# Helpers: Mapping
# --------------------------- #

MARKET_MAP = {
    "Forex": {
        "EUR/USD": {"yf": "EURUSD=X", "tv": "FX_IDC:EURUSD"},
        "GBP/JPY": {"yf": "GBPJPY=X", "tv": "FX_IDC:GBPJPY"},
        "USD/JPY": {"yf": "USDJPY=X", "tv": "FX_IDC:USDJPY"},
        "AUD/USD": {"yf": "AUDUSD=X", "tv": "FX_IDC:AUDUSD"},
        "XAU/USD": {"yf": "XAUUSD=X", "tv": "OANDA:XAUUSD"},
    },
    "Crypto": {
        "BTC/USDT": {"yf": "BTC-USD", "tv": "BINANCE:BTCUSDT"},
        "ETH/USDT": {"yf": "ETH-USD", "tv": "BINANCE:ETHUSDT"},
    },
    "Commodities": {
        "Gold (XAUUSD)": {"yf": "XAUUSD=X", "tv": "OANDA:XAUUSD"},
        "Silver (XAGUSD)": {"yf": "XAGUSD=X", "tv": "OANDA:XAGUSD"},
        "Crude Oil (WTI)": {"yf": "CL=F", "tv": "TVC:USOIL"},
    },
    "Indian": {
        "NIFTY 50": {"yf": "^NSEI", "tv": "NSE:NIFTY"},
        "BANKNIFTY": {"yf": "^NSEBANK", "tv": "NSE:BANKNIFTY"},
        "SENSEX": {"yf": "^BSESN", "tv": "BSE:SENSEX"},
    },
}

INTERVALS = {
    "1m": {"yf": "1m", "period": "5d"},
    "5m": {"yf": "5m", "period": "30d"},
    "15m": {"yf": "15m", "period": "60d"},
    "1h": {"yf": "60m", "period": "730d"},
    "4h": {"yf": "60m", "period": "730d"},  # we’ll resample
    "1d": {"yf": "1d", "period": "max"},
}

def tradingview_iframe(tv_symbol: str, interval: str = "15") -> str:
    """Return TradingView embedded iframe HTML."""
    # Map to TradingView interval codes
    tv_iv = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}.get(interval, "15")
    html = f"""
    <div class="tradingview-widget-container">
      <iframe
        src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval={tv_iv}&theme=dark&style=1&locale=en#"
        style="width:100%;height:520px;border:none;border-radius:10px"
        allowtransparency="true" scrolling="no">
      </iframe>
    </div>
    """
    return html

# --------------------------- #
# Data
# --------------------------- #

@st.cache_data(ttl=60)
def fetch_data(yf_symbol: str, interval_key: str) -> pd.DataFrame:
    """Fetch and return OHLCV with correct interval and resampling if needed."""
    spec = INTERVALS[interval_key]
    df = yf.download(yf_symbol, period=spec["period"], interval=spec["yf"], auto_adjust=True, progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.dropna().copy()

    # Resample for 4h from 60m
    if interval_key == "4h":
        df = (
            df.resample("4H")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna()
        )
    return df

# --------------------------- #
# Indicators & Patterns
# --------------------------- #

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_dn = pd.Series(down, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_dn.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def macd(series: pd.Series) -> pd.DataFrame:
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return pd.DataFrame({"macd": line, "signal": signal, "hist": hist})

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()

def detect_engulfing(df: pd.DataFrame) -> str:
    """Simple last-candle engulfing detection."""
    if len(df) < 3:
        return "none"
    c1o, c1c = df["Open"].iloc[-2], df["Close"].iloc[-2]
    c2o, c2c = df["Open"].iloc[-1], df["Close"].iloc[-1]
    # Bullish engulfing
    if (c1c < c1o) and (c2c > c2o) and (c2c >= c1o) and (c2o <= c1c):
        return "bullish"
    # Bearish engulfing
    if (c1c > c1o) and (c2c < c2o) and (c2c <= c1o) and (c2o >= c1c):
        return "bearish"
    return "none"

def detect_pinbar(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "none"
    o = df["Open"].iloc[-1]
    h = df["High"].iloc[-1]
    l = df["Low"].iloc[-1]
    c = df["Close"].iloc[-1]
    body = abs(c - o)
    upper = h - max(c, o)
    lower = min(c, o) - l
    if body == 0:
        body = 1e-9
    # Bullish pin
    if lower > body * 2 and upper < body * 1.2 and c > o:
        return "bullish"
    # Bearish pin
    if upper > body * 2 and lower < body * 1.2 and c < o:
        return "bearish"
    return "none"

def market_snapshot(df: pd.DataFrame) -> dict:
    out = {}
    out["ema21"] = ema(df["Close"], 21)
    out["ema50"] = ema(df["Close"], 50)
    out["rsi"] = rsi(df["Close"], 14)
    mac = macd(df["Close"])
    out["macd"] = mac["macd"]
    out["hist"] = mac["hist"]
    out["atr"] = atr(df, 14)
    out["trend"] = "uptrend" if out["ema21"].iloc[-1] > out["ema50"].iloc[-1] else "downtrend"
    out["momentum"] = "strong" if abs(out["hist"].iloc[-1]) > abs(out["hist"]).rolling(20).mean().iloc[-1] else "normal"
    out["volatility"] = int((out["atr"].iloc[-1] / df["Close"].iloc[-1]) * 100000)
    out["support"] = float(df["Low"].tail(30).min())
    out["resistance"] = float(df["High"].tail(30).max())
    return out

# --------------------------- #
# Strategy (LTF + HTF confirm)
# --------------------------- #

def higher_tf(interval_key: str) -> str:
    return {"1m": "15m", "5m": "15m", "15m": "1h", "1h": "4h", "4h": "1d", "1d": "1d"}[interval_key]

def compute_signal(df_ltf: pd.DataFrame, df_htf: pd.DataFrame) -> dict:
    """Return dict with signal, entry/sl/tp, reasons, confidence."""
    res = {
        "signal": "HOLD",
        "entry": None,
        "sl": None,
        "tp": None,
        "rr": None,
        "reasons": [],
        "confidence": 0,
    }
    if df_ltf.empty or df_htf.empty:
        res["reasons"].append("Insufficient data")
        return res

    snap_ltf = market_snapshot(df_ltf)
    snap_htf = market_snapshot(df_htf)

    close = float(df_ltf["Close"].iloc[-1])
    last_atr = float(snap_ltf["atr"].iloc[-1])
    ema_ok_up = snap_ltf["ema21"].iloc[-1] > snap_ltf["ema50"].iloc[-1]
    ema_ok_dn = snap_ltf["ema21"].iloc[-1] < snap_ltf["ema50"].iloc[-1]
    rsi_v = float(snap_ltf["rsi"].iloc[-1])
    mac_hist = float(snap_ltf["hist"].iloc[-1])

    pat_eng = detect_engulfing(df_ltf)
    pat_pin = detect_pinbar(df_ltf)

    # HTF trend alignment
    htf_up = snap_htf["ema21"].iloc[-1] > snap_htf["ema50"].iloc[-1]
    htf_dn = not htf_up

    long_ok = ema_ok_up and htf_up and rsi_v > 48 and mac_hist > 0
    short_ok = ema_ok_dn and htf_dn and rsi_v < 52 and mac_hist < 0

    # Decide direction
    if long_ok:
        res["signal"] = "BUY"
        res["reasons"] += ["HTF uptrend", "EMA21>EMA50 (LTF)", "MACD momentum > 0", "RSI above 50"]
        if pat_eng == "bullish":
            res["reasons"].append("Bullish engulfing")
            res["confidence"] += 10
        if pat_pin == "bullish":
            res["reasons"].append("Bullish pin bar")
            res["confidence"] += 8

        # ATR based SL/TP
        sl = close - 1.2 * last_atr
        tp = close + 2.2 * last_atr
    elif short_ok:
        res["signal"] = "SELL"
        res["reasons"] += ["HTF downtrend", "EMA21<EMA50 (LTF)", "MACD momentum < 0", "RSI below 50"]
        if pat_eng == "bearish":
            res["reasons"].append("Bearish engulfing")
            res["confidence"] += 10
        if pat_pin == "bearish":
            res["reasons"].append("Bearish pin bar")
            res["confidence"] += 8

        sl = close + 1.2 * last_atr
        tp = close - 2.2 * last_atr
    else:
        res["reasons"].append("Filters not aligned")
        return res

    res["entry"] = round(close, 5)
    res["sl"] = round(sl, 5)
    res["tp"] = round(tp, 5)

    risk = abs(close - sl)
    reward = abs(tp - close)
    rr = reward / risk if risk > 0 else None
    res["rr"] = round(rr, 2) if rr else None

    # Confidence base
    base_conf = 80 if res["signal"] in ["BUY", "SELL"] else 0
    res["confidence"] = base_conf + res["confidence"]
    res["confidence"] = min(res["confidence"], 120)

    return res

# --------------------------- #
# Risk & Lot Size (Forex-style)
# --------------------------- #

def pip_size(symbol_name: str) -> float:
    # JPY pairs use 0.01 pip, most others 0.0001
    if any(k in symbol_name.replace("/", "") for k in ["JPY"]):
        return 0.01
    return 0.0001

def lot_size_forex(account_balance: float, risk_pct: float, entry: float, sl: float, symbol_name: str) -> float:
    risk_amount = max(account_balance * (risk_pct / 100.0), 0.0)
    pips = abs(entry - sl) / pip_size(symbol_name)
    if pips <= 0:
        return 0.0
    # ~ $10 per pip per 1.00 lot on majors
    lots = risk_amount / (pips * 10.0)
    return max(round(lots, 2), 0.01)

# --------------------------- #
# UI
# --------------------------- #

st.title("📈 Rayner Bot – Pro Signal (EMA + RSI + MACD + ATR)")
st.caption("Multi-confirmation strategy with HTF trend, ATR-based TP/SL, TradingView live chart, and dynamic risk engine.")

with st.sidebar:
    st.header("Account & Risk")
    balance = st.number_input("Account Balance ($)", min_value=100.0, value=1000.0, step=50.0)
    risk_pct = st.slider("Risk per trade (%)", 0.5, 2.0, 1.0, 0.1)

# Selection
c1, c2, c3 = st.columns([1, 1.2, 1])
with c1:
    market = st.selectbox("Market Category", list(MARKET_MAP.keys()))
with c2:
    symbol = st.selectbox("Select Market", list(MARKET_MAP[market].keys()))
with c3:
    interval_key = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2)

yf_symbol = MARKET_MAP[market][symbol]["yf"]
tv_symbol = MARKET_MAP[market][symbol]["tv"]

# Live Chart
st.subheader(f"📺 Live Chart — {symbol}")
st.components.v1.html(tradingview_iframe(tv_symbol, interval_key), height=540)

# Generate Signal
if st.button("🚦 Generate Signal", use_container_width=True):
    with st.spinner("Fetching data & computing signal..."):
        df_ltf = fetch_data(yf_symbol, interval_key)
        df_htf = fetch_data(yf_symbol, higher_tf(interval_key))

    if df_ltf.empty or df_htf.empty:
        st.error("No data or too few candles. Try a different interval.")
        st.stop()

    sig = compute_signal(df_ltf, df_htf)

    # Snapshot
    snap = market_snapshot(df_ltf)
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown("**Trend**")
        st.write("🟢 Uptrend" if snap["trend"] == "uptrend" else "🔴 Downtrend")
    with mcol2:
        st.markdown("**Momentum**")
        st.write(snap["momentum"])
    with mcol3:
        st.markdown("**Support / Resistance**")
        st.write(f"{snap['support']:.4f} / {snap['resistance']:.4f}")
    with mcol4:
        st.markdown("**Volatility (ATR%)**")
        atr_pct = (snap["atr"].iloc[-1] / df_ltf["Close"].iloc[-1]) * 100
        st.write(f"{atr_pct:.2f}%")

    st.markdown("---")

    # Signal block
    if sig["signal"] in ["BUY", "SELL"]:
        # Risk / Lot
        lots = lot_size_forex(balance, risk_pct, sig["entry"], sig["sl"], symbol)

        st.markdown("### 🚨 Signal")
        st.markdown(
            f"**Signal:** **{sig['signal']}** with **{sig['confidence']}%** confirmation  \n"
            f"**Entry:** `{sig['entry']}`  |  **SL:** `{sig['sl']}`  |  **TP:** `{sig['tp']}`  \n"
            f"**R:R:** `{sig['rr']}`  |  **Risk:** `{risk_pct:.2f}%` of `${balance:,.2f}`  |  **Lot size:** `{lots}`"
        )
        st.markdown('<div class="rrbar"></div>', unsafe_allow_html=True)

        st.markdown("**Reasons:**")
        for r in sig["reasons"]:
            st.write(f"• {r}")

        st.markdown("---")

        # Show tail of data for transparency
        with st.expander("Show last 10 candles (debug)"):
            st.dataframe(df_ltf.tail(10))
    else:
        st.info("No trade: filters not aligned.")
