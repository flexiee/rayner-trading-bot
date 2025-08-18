# app.py
# Pro Trading Bot — TradingView-only (no exchange API)
# Single-file Streamlit app. Click "Generate Signal" to analyze the current market
# using TradingView candles via tvDatafeed (unofficial TV data access).
# Educational use only.

import os
import math
from datetime import datetime
import time

import numpy as np
import pandas as pd
import streamlit as st
from tvDatafeed import TvDatafeed, Interval  # unofficial TradingView data

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="Pro Trading Bot — TradingView (1:3 RR)", layout="wide")

# ---------------------------
# Markets (symbol + exchange for tvdatafeed)
# ---------------------------
MARKETS = {
    "EUR/USD":        {"symbol": "EURUSD",  "exchange": "OANDA"},
    "GBP/JPY":        {"symbol": "GBPJPY",  "exchange": "OANDA"},
    "USD/JPY":        {"symbol": "USDJPY",  "exchange": "OANDA"},
    "XAU/USD":        {"symbol": "XAUUSD",  "exchange": "OANDA"},
    "BTC/USDT":       {"symbol": "BTCUSDT", "exchange": "BINANCE"},
    "ETH/USDT":       {"symbol": "ETHUSDT", "exchange": "BINANCE"},
    "BNB/USDT":       {"symbol": "BNBUSDT", "exchange": "BINANCE"},
    "SOL/USDT":       {"symbol": "SOLUSDT", "exchange": "BINANCE"},
    "NIFTY 50":       {"symbol": "NIFTY",   "exchange": "NSE"},
    "BANKNIFTY":      {"symbol": "BANKNIFTY","exchange":"NSE"},
}

TV_IFRAME_TEMPLATE = (
    '<iframe src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval={interval}'
    '&hidesidetoolbar=1&symboledit=1&hideideas=1&theme=dark" width="100%" height="520" '
    'frameborder="0"></iframe>'
)

# Strategy defaults
CFG = {
    "ema_short": 20, "ema_long": 50, "ema_trend": 200,
    "rsi_p": 14, "rsi_oversold": 30, "rsi_overbought": 70,
    "macd_fast": 12, "macd_slow": 26, "macd_sig": 9,
    "atr_p": 14, "bb_p": 20,
    "rr_target": 3.0,            # fixed 1:3 RR
    "min_confidence": 60,        # show signal only if >= this
    "min_volume_ratio": 0.4      # current vol must be >= 40% of 20-bar avg
}

# ---------------------------
# Helpers: indicators (pure pandas)
# ---------------------------
def ema(s: pd.Series, p: int):
    return s.ewm(span=p, adjust=False).mean()

def rsi_wilder(s: pd.Series, p: int = 14):
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/p, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/p, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd_series(s: pd.Series, fast=12, slow=26, sig=9):
    line = ema(s, fast) - ema(s, slow)
    signal = line.ewm(span=sig, adjust=False).mean()
    hist = line - signal
    return line, signal, hist

def atr(df: pd.DataFrame, p: int = 14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def resample_to(df: pd.DataFrame, rule: str):
    ohlc = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    return df.resample(rule).agg(ohlc).dropna()

def market_strength(df: pd.DataFrame):
    if df is None or len(df) < 20: return 0
    closes = df["Close"].tail(50).to_numpy()
    x = np.arange(len(closes))
    if len(x) < 2: return 0
    slope = np.polyfit(x, closes, 1)[0]
    momentum_score = np.clip((slope / (np.mean(closes) + 1e-9)) * 9000, -50, 50)
    vol = df["Close"].pct_change().rolling(20).std().iloc[-1]
    vol_score = np.clip((vol or 0) * 1200, 0, 50)
    return int(np.clip(50 + momentum_score + vol_score, 0, 100))

# ---------------------------
# TradingView data (tvdatafeed)
# ---------------------------
INTERVAL_MAP = {
    "1m":  Interval.in_1_minute,
    "5m":  Interval.in_5_minute,
    "15m": Interval.in_15_minute,
}

def tv_symbol_str(market_name: str):
    m = MARKETS[market_name]
    return f"{m['exchange']}:{m['symbol']}"

@st.cache_data(show_spinner=False, ttl=30)  # cache briefly to avoid hammering
def fetch_tv_bars(market_name: str, interval_key: str, n_bars: int):
    m = MARKETS[market_name]
    iv = INTERVAL_MAP[interval_key]
    tv = TvDatafeed()  # anonymous login (works for many symbols)
    df = tv.get_hist(symbol=m["symbol"], exchange=m["exchange"], interval=iv, n_bars=n_bars)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)  # tv returns open/high/low/close/volume
    df.reset_index(inplace=True)
    df = df.rename(columns={"Datetime":"Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df[["Datetime","Open","High","Low","Close","Volume"]].set_index("Datetime").sort_index()
    return df

# ---------------------------
# Signal generation (1m with MTF 5m/15m alignment)
# ---------------------------
def generate_signal_tv(df1m: pd.DataFrame, cfg=CFG):
    out = {"signal":"NONE","entry":None,"sl":None,"tp":None,"rr":None,"confidence":0,"reasons":[],"strength":0}
    if df1m is None or df1m.empty or len(df1m) < 210:
        out["reasons"].append("Not enough data")
        return out

    # Build MTF by resampling 1m → 5m/15m
    df5m = resample_to(df1m, "5T") if len(df1m) >= 5 else None
    df15m = resample_to(df1m, "15T") if len(df1m) >= 15 else None

    price = float(df1m["Close"].iat[-1])
    out["entry"] = round(price, 8)

    # 1m indicators
    ema_s = float(ema(df1m["Close"], cfg["ema_short"]).iat[-1])
    ema_l = float(ema(df1m["Close"], cfg["ema_long"]).iat[-1])
    ema_t = float(ema(df1m["Close"], cfg["ema_trend"]).iat[-1])
    r = float(rsi_wilder(df1m["Close"], cfg["rsi_p"]).iat[-1])
    macd_l, macd_sig, macd_h = macd_series(df1m["Close"], cfg["macd_fast"], cfg["macd_slow"], cfg["macd_sig"])
    macd_dir = float(macd_l.iat[-1] - macd_sig.iat[-1])
    bb_mid = df1m["Close"].rolling(cfg["bb_p"]).mean().iat[-1]
    bb_std = df1m["Close"].rolling(cfg["bb_p"]).std().iat[-1]
    z = (price - bb_mid) / bb_std if (not math.isnan(bb_std) and bb_std > 0) else 0.0
    atr_v = float(atr(df1m, cfg["atr_p"]).iat[-1]) if len(df1m) >= cfg["atr_p"] else None

    # Volume filter
    vol_ok = True
    try:
        avg_vol = df1m["Volume"].rolling(20).mean().iat[-1]
        cur_vol = df1m["Volume"].iat[-1]
        if avg_vol > 0 and cur_vol < avg_vol * cfg["min_volume_ratio"]:
            vol_ok = False
            out["reasons"].append("Low volume vs 20-bar avg")
    except Exception:
        pass

    # MTF alignment
    mtf_ok = True
    for mdf, label in [(df5m,"5m"), (df15m,"15m")]:
        if mdf is None or mdf.empty:
            mtf_ok = False; out["reasons"].append(f"{label} missing"); continue
        m_ema_s = float(ema(mdf["Close"], cfg["ema_short"]).iat[-1])
        m_ema_l = float(ema(mdf["Close"], cfg["ema_long"]).iat[-1])
        aligned = (ema_s > ema_l and m_ema_s > m_ema_l) or (ema_s < ema_l and m_ema_s < m_ema_l)
        if aligned:
            out["reasons"].append(f"{label} aligned")
        else:
            mtf_ok = False; out["reasons"].append(f"{label} not aligned")

    # Votes
    votes_buy = votes_sell = 0
    if ema_s > ema_l: votes_buy += 1; out["confidence"] += 18; out["reasons"].append("EMA20>EMA50")
    else:            votes_sell += 1; out["reasons"].append("EMA20<EMA50")

    if ema_s > ema_t: votes_buy += 1; out["confidence"] += 8; out["reasons"].append("Above EMA200")
    else:             votes_sell += 1; out["reasons"].append("Below EMA200")

    if r < cfg["rsi_oversold"]: votes_buy += 1; out["confidence"] += 10; out["reasons"].append("RSI oversold")
    elif r > cfg["rsi_overbought"]: votes_sell += 1; out["reasons"].append("RSI overbought")
    else: out["confidence"] += 4; out["reasons"].append("RSI neutral")

    if macd_dir > 0: votes_buy += 1; out["confidence"] += 8; out["reasons"].append("MACD bullish")
    else:            votes_sell += 1; out["reasons"].append("MACD bearish")

    if z < -2: votes_buy += 1; out["confidence"] += 5; out["reasons"].append("Below -2σ")
    elif z > 2: votes_sell += 1; out["reasons"].append("Above +2σ")

    # Simple engulfing
    if len(df1m) >= 2:
        prev = df1m.iloc[-2]; cur = df1m.iloc[-1]
        bull_engulf = (cur["Close"] > cur["Open"]) and (prev["Close"] < prev["Open"]) and (cur["Close"] > prev["Open"]) and (cur["Open"] < prev["Close"])
        bear_engulf = (cur["Close"] < cur["Open"]) and (prev["Close"] > prev["Open"]) and (cur["Close"] < prev["Open"]) and (cur["Open"] > prev["Close"])
        if bull_engulf: votes_buy += 1; out["confidence"] += 6; out["reasons"].append("Bullish engulfing")
        if bear_engulf: votes_sell += 1; out["confidence"] -= 3; out["reasons"].append("Bearish engulfing")

    if not vol_ok or not mtf_ok:
        final = "NONE"
    else:
        final = "BUY" if votes_buy > votes_sell else ("SELL" if votes_sell > votes_buy else "NONE")
    out["signal"] = final

    # SL/TP fixed RR = 1:3
    if final in ["BUY", "SELL"]:
        if atr_v is None or atr_v == 0 or math.isnan(atr_v):
            volatility = df1m["Close"].pct_change().rolling(14).std().iat[-1] if len(df1m) >= 14 else 0.001
            atr_px = max(volatility * price, 1e-6)
        else:
            atr_px = atr_v
        if final == "BUY":
            sl = price - atr_px
            tp = price + (price - sl) * CFG["rr_target"]
        else:
            sl = price + atr_px
            tp = price - (sl - price) * CFG["rr_target"]
        out["sl"] = round(sl, 8)
        out["tp"] = round(tp, 8)
        denom = abs(price - sl)
        out["rr"] = round(abs((tp - price) / denom), 2) if denom > 0 else None
    out["confidence"] = int(np.clip(out["confidence"], 0, 100))
    out["strength"] = market_strength(df1m)
    return out

# ---------------------------
# UI
# ---------------------------
st.title("🚀 Pro Trading Bot — TradingView (1:3 RR)")

left, right = st.columns([1, 2])

with left:
    st.subheader("Configuration")
    market_name = st.selectbox("Market", list(MARKETS.keys()), index=0)
    interval_choice = st.selectbox("Entry timeframe", ["1m","5m","15m"], index=0)
    account_balance = st.number_input("Account balance (USD)", min_value=10.0, value=1000.0, step=50.0)
    risk_pct = st.number_input("Risk % per trade", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

with right:
    st.subheader("TradingView Chart")
    # TV iframe interval: "1" for 1m, "5" for 5m, etc.
    tv_interval_token = interval_choice.replace("m", "")
    st.components.v1.html(
        TV_IFRAME_TEMPLATE.format(tv_symbol=tv_symbol_str(market_name), interval=tv_interval_token),
        height=540
    )

st.markdown("---")

if st.button("🔮 Generate Signal"):
    with st.spinner("Fetching live candles from TradingView and analyzing…"):
        # Always pull 1m bars and resample internally (lets us do MTF confirmation)
        df1m = fetch_tv_bars(market_name, "1m", n_bars=1500)
        if df1m is None or df1m.empty:
            st.error("Failed to fetch market data from TradingView (tvdatafeed). Try again or change market.")
        else:
            result = generate_signal_tv(df1m, cfg=CFG)
            if result["signal"] == "NONE" or result["confidence"] < CFG["min_confidence"]:
                st.warning(f"No high-confidence signal. Confidence: {result['confidence']} | Strength: {result['strength']}")
                if result.get("reasons"):
                    with st.expander("Diagnostics"):
                        for r in result["reasons"]:
                            st.write("•", r)
            else:
                st.success(f"SIGNAL: {result['signal']} | Confidence: {result['confidence']} | Strength: {result['strength']}/100")
                st.markdown(f"**Entry:** {result['entry']}  \n**SL:** {result['sl']}  |  **TP:** {result['tp']}  |  **R:R:** {result.get('rr')}")

                # simple position sizing (units) – uses price distance, no broker API
                price = result["entry"]; sl = result["sl"]
                risk_amt = round(account_balance * (risk_pct / 100.0), 2)
                diff = abs(price - sl)
                size_units = round(risk_amt / diff, 6) if diff > 0 else 0.0
                st.info(f"Suggested size: {size_units} units   |   Risk amount: ${risk_amt}")

                if result.get("reasons"):
                    with st.expander("Why this signal?"):
                        for r in result["reasons"]:
                            st.write("•", r)

st.caption("Notes: Uses TradingView data via tvdatafeed (unofficial). Real-time quality depends on TV availability. Not financial advice.")
