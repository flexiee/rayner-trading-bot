# app.py — Next-Level Trading Bot (single file)
# - Generate button triggers analysis
# - Fixed pro indicator settings (no sliders)
# - TradingView chart embed (visual only)
# - Data via yfinance
# - Signal + TP/SL + R:R + Confidence + Strength + Position sizing
# - Saves signal history to bot_data/signal_history.csv

import os
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------- UI/APP SETUP ----------------
st.set_page_config(page_title="Next Level Trading Bot", layout="wide")
DATA_DIR = os.path.join(os.getcwd(), "bot_data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(DATA_DIR, "signal_history.csv")

# Markets: yfinance ticker for data, TradingView symbol for the embedded chart
MARKETS = {
    "EUR/USD": {"yf": "EURUSD=X", "tv": "OANDA:EURUSD"},
    "GBP/USD": {"yf": "GBPUSD=X", "tv": "OANDA:GBPUSD"},
    "USD/JPY": {"yf": "JPY=X",     "tv": "OANDA:USDJPY"},
    "XAU/USD": {"yf": "GC=F",      "tv": "OANDA:XAUUSD"},
    "BTC/USD": {"yf": "BTC-USD",   "tv": "BITSTAMP:BTCUSD"},
    "ETH/USD": {"yf": "ETH-USD",   "tv": "BITSTAMP:ETHUSD"},
}

# Interval mapping
INTERVALS = {
    "1m": {"yf_interval": "1m",  "yf_period": "7d",  "tv_interval": "1"},
    "5m": {"yf_interval": "5m",  "yf_period": "30d", "tv_interval": "5"},
    "15m":{"yf_interval": "15m", "yf_period": "60d", "tv_interval": "15"},
    "1h": {"yf_interval": "60m", "yf_period": "730d","tv_interval": "60"},
}

# ---------------- HELPERS ----------------
def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[cols].copy().dropna()
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")]
    return df

def fetch_yf(symbol: str, yf_interval: str, yf_period: str) -> pd.DataFrame:
    try:
        df = yf.download(tickers=symbol, interval=yf_interval, period=yf_period, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return ensure_ohlcv(flatten_yf_columns(df))
    except Exception as e:
        st.error(f"Data fetch error for {symbol}: {e}")
        return pd.DataFrame()

# ----- indicators (no external TA package) -----
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd_series(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def resample_to(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    ohlc = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    return df.resample(rule).agg(ohlc).dropna()

def pip_size(symbol_key: str) -> float:
    s = symbol_key.upper()
    if "JPY" in s and "XAU" not in s:
        return 0.01
    if "XAU" in s or "GOLD" in s:
        return 0.01
    return 0.0001

def position_size(balance: float, risk_pct: float, entry: float, sl: float, symbol_key: str):
    if entry is None or sl is None:
        return 0.0, 0.0
    risk_amount = max(0.0, balance * (risk_pct/100.0))
    pip = pip_size(symbol_key)
    pip_distance = abs(entry - sl) / pip if pip > 0 else 0
    if pip_distance <= 0:
        return 0.0, round(risk_amount, 2)
    approx_pip_value_per_lot = 10.0  # heuristic for USD acct on majors
    lots = risk_amount / (pip_distance * approx_pip_value_per_lot)
    return round(lots, 4), round(risk_amount, 2)

def market_strength(df: pd.DataFrame) -> int:
    if df is None or len(df) < 20:
        return 0
    closes = df["Close"].dropna()
    x = np.arange(len(closes[-20:]))
    y = closes[-20:].values
    if len(x) < 2:
        return 0
    slope = np.polyfit(x, y, 1)[0]
    momentum_score = np.clip((slope / (np.mean(y) + 1e-9)) * 10000, -50, 50)
    vol = df["Close"].pct_change().rolling(14).std().iloc[-1]
    vol_score = np.clip(vol * 1000, 0, 50)
    score = 50 + momentum_score + vol_score
    return int(np.clip(score, 0, 100))

# ----- strategy core (fixed pro settings) -----
DEFAULTS = dict(
    ema_short=20, ema_long=50,
    rsi_period=14, rsi_overbought=70, rsi_oversold=30,
    macd_fast=12, macd_slow=26, macd_signal=9,
    bb_period=20, atr_period=14,
    sl_atr_mult=1.5, tp_atr_mult=3.0
)

def generate_signal(df: pd.DataFrame, symbol_key: str, cfg=DEFAULTS) -> dict:
    out = {"signal":"NONE","entry":None,"sl":None,"tp":None,"rr":None,
           "confidence":0,"reasons":[],"strength":0}
    if df is None or df.empty:
        return out

    price = float(df["Close"].iat[-1])
    out["entry"] = price

    ema_s = float(ema(df["Close"], cfg["ema_short"]).iat[-1])
    ema_l = float(ema(df["Close"], cfg["ema_long"]).iat[-1])
    r = float(rsi_wilder(df["Close"], cfg["rsi_period"]).iat[-1])
    macd_line, macd_sig, _ = macd_series(df["Close"], cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
    macd_dir = float(macd_line.iat[-1] - macd_sig.iat[-1])

    mid = df["Close"].rolling(cfg["bb_period"]).mean().iat[-1]
    std = df["Close"].rolling(cfg["bb_period"]).std().iat[-1]
    z = (price - mid) / std if (not math.isnan(std) and std > 0) else 0.0

    votes_bull = votes_bear = 0

    if ema_s > ema_l:
        votes_bull += 1; out["confidence"] += 15; out["reasons"].append("EMA20>EMA50 (trend up)")
    else:
        votes_bear += 1; out["reasons"].append("EMA20<EMA50 (trend down)")

    if r < cfg["rsi_oversold"]:
        votes_bull += 1; out["confidence"] += 12; out["reasons"].append("RSI oversold")
    elif r > cfg["rsi_overbought"]:
        votes_bear += 1; out["reasons"].append("RSI overbought")
    else:
        out["confidence"] += 4; out["reasons"].append("RSI neutral")

    if macd_dir > 0:
        votes_bull += 1; out["confidence"] += 10; out["reasons"].append("MACD bullish")
    else:
        votes_bear += 1; out["reasons"].append("MACD bearish")

    if z < -2:
        votes_bull += 1; out["confidence"] += 6; out["reasons"].append("Below -2σ (mean-revert up)")
    elif z > 2:
        votes_bear += 1; out["reasons"].append("Above +2σ (mean-revert down)")

    # Simple engulfing confirmation
    if len(df) >= 2:
        prev, cur = df.iloc[-2], df.iloc[-1]
        if (cur["Close"] > cur["Open"]) and (prev["Close"] < prev["Open"]) and (cur["Close"] > prev["Open"]) and (cur["Open"] < prev["Close"]):
            votes_bull += 1; out["confidence"] += 8; out["reasons"].append("Bullish engulfing")
        if (cur["Close"] < cur["Open"]) and (prev["Close"] > prev["Open"]) and (cur["Close"] < prev["Open"]) and (cur["Open"] > prev["Close"]):
            votes_bear += 1; out["confidence"] -= 5; out["reasons"].append("Bearish engulfing")

    out["signal"] = "BUY" if votes_bull > votes_bear else ("SELL" if votes_bear > votes_bull else "NONE")

    atr_v = float(atr(df, cfg["atr_period"]).iat[-1]) if len(df) >= cfg["atr_period"] else None
    if atr_v is None or math.isnan(atr_v) or atr_v == 0:
        recent_std = df["Close"].pct_change().rolling(14).std().iat[-1] if len(df) >= 14 else 0.001
        vol_px = recent_std * price
    else:
        vol_px = atr_v

    if out["signal"] == "BUY":
        out["sl"] = price - vol_px * cfg["sl_atr_mult"]
        out["tp"] = price + vol_px * cfg["tp_atr_mult"]
    elif out["signal"] == "SELL":
        out["sl"] = price + vol_px * cfg["sl_atr_mult"]
        out["tp"] = price - vol_px * cfg["tp_atr_mult"]

    if out["sl"] and out["tp"]:
        denom = abs(price - out["sl"])
        out["rr"] = round(abs((out["tp"] - price) / denom), 2) if denom > 0 else None

    out["confidence"] = int(np.clip(out["confidence"], 0, 100))
    out["strength"] = market_strength(df)
    return out

# ---------------- UI ----------------
st.title("⚡ Next-Level Trading Bot")

left, right = st.columns([1.0, 2.0])

with left:
    market_name = st.selectbox("Market", list(MARKETS.keys()), index=0)
    interval_key = st.selectbox("Interval", list(INTERVALS.keys()), index=1)
    account_balance = st.number_input("Account Balance (USD)", min_value=10.0, value=1000.0, step=50.0)
    risk_percent = st.slider("Risk % per trade", 0.1, 5.0, 1.0, 0.1)

    st.markdown("---")
    st.subheader("Signal History")
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        st.dataframe(hist.sort_values("timestamp", ascending=False).head(25))
    else:
        st.write("No history yet.")

with right:
    tv_symbol = MARKETS[market_name]["tv"]
    yf_symbol = MARKETS[market_name]["yf"]
    settings = INTERVALS[interval_key]

    st.subheader("TradingView Chart (visual)")
    st.components.v1.html(
        f"""
        <iframe src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval={settings['tv_interval']}&hidesidetoolbar=1&symboledit=1&hideideas=1&theme=dark"
                width="100%" height="520" frameborder="0"></iframe>
        """,
        height=540
    )

    st.markdown("---")
    if st.button("🔮 Generate Signal"):
        with st.spinner("Analyzing..."):
            df = fetch_yf(yf_symbol, settings["yf_interval"], settings["yf_period"])

        if df.empty:
            st.error("No data returned. Try a different symbol/interval.")
        else:
            sig = generate_signal(df, market_name)

            st.success(f"Signal: {sig['signal']}  |  Confidence: {sig['confidence']}  |  Strength: {sig['strength']}/100")
            st.write(f"**Entry:** {sig['entry']}")
            st.write(f"**SL:** {sig['sl']}   |   **TP:** {sig['tp']}   |   **R:R:** {sig.get('rr')}")

            lots, risk_amt = position_size(
                account_balance, risk_percent,
                sig["entry"], sig["sl"] if sig["sl"] else sig["entry"], market_name
            )
            st.write(f"**Suggested lots (approx):** {lots}   |   **Risk amount:** ${risk_amt}")

            if sig.get("reasons"):
                with st.expander("Why this signal?"):
                    for r in sig["reasons"]:
                        st.write("•", r)

            # Save history row
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "market": market_name,
                "yf_symbol": yf_symbol,
                "interval": settings["yf_interval"],
                "signal": sig["signal"],
                "confidence": sig["confidence"],
                "strength": sig["strength"],
                "entry": sig["entry"],
                "sl": sig["sl"],
                "tp": sig["tp"],
                "rr": sig.get("rr"),
                "risk_percent": risk_percent,
                "lots": lots,
                "risk_amount": risk_amt,
            }
            if os.path.exists(HISTORY_FILE):
                old = pd.read_csv(HISTORY_FILE)
                pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(HISTORY_FILE, index=False)
            else:
                pd.DataFrame([row]).to_csv(HISTORY_FILE, index=False)
            st.info("Signal saved to history.")

st.caption("Educational tool. Signals are not financial advice. Trade responsibly.")
