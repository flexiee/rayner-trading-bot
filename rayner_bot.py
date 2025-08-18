# app.py
# Fixed all-in-one app (TradingView summary + Backtest + Scanner)
# - Uses tradingview-ta + yfinance (no Selenium, no exchange API keys)
# - Single file, Streamlit Cloud compatible
# - Click \"Generate Signal\" -> will fetch TV summary -> builds SL/TP (1:3 RR)
# - Backtest uses yfinance OHLC

import os
import time
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from tradingview_ta import TA_Handler, Interval

# ---------------------------
# Config + storage
# ---------------------------
st.set_page_config(page_title="Fixed Pro Trading Bot", layout="wide")
st.title("🔧 Fixed — Pro Trading Bot (TradingView + Backtest)")

DATA_DIR = "bot_data"
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(DATA_DIR, "signal_history.csv")

# ---------------------------
# Markets mapping
# ---------------------------
MARKETS = {
    "BTC/USDT": {"tv_symbol": "BTCUSDT", "exchange": "BINANCE", "screener": "CRYPTO", "yf": "BTC-USD", "type": "crypto"},
    "ETH/USDT": {"tv_symbol": "ETHUSDT", "exchange": "BINANCE", "screener": "CRYPTO", "yf": "ETH-USD", "type": "crypto"},
    "EUR/USD":  {"tv_symbol": "EURUSD",  "exchange": "OANDA",   "screener": "FOREX",  "yf": "EURUSD=X", "type": "forex"},
    "USD/JPY":  {"tv_symbol": "USDJPY",  "exchange": "OANDA",   "screener": "FOREX",  "yf": "JPY=X",    "type": "forex"},
    "XAU/USD":  {"tv_symbol": "XAUUSD",  "exchange": "OANDA",   "screener": "FOREX",  "yf": "GC=F",     "type": "commodity"},
}

TV_IFRAME_TEMPLATE = '<iframe src="https://s.tradingview.com/widgetembed/?symbol={exchange}:{symbol}&interval={iv}&hidesidetoolbar=1&hideideas=1&theme=dark" width="100%" height="520" frameborder="0"></iframe>'

TV_INTERVALS = {
    "1m": Interval.INTERVAL_1_MINUTE,
    "5m": Interval.INTERVAL_5_MINUTES,
    "15m": Interval.INTERVAL_15_MINUTES,
    "1h": Interval.INTERVAL_1_HOUR,
    "4h": Interval.INTERVAL_4_HOURS,
    "1d": Interval.INTERVAL_1_DAY,
}

CFG = {
    "ema_short": 20, "ema_long": 50, "ema_trend": 200,
    "rsi_p": 14, "rsi_oversold": 30, "rsi_overbought": 70,
    "macd_fast": 12, "macd_slow": 26, "macd_sig": 9,
    "atr_p": 14, "bb_p": 20,
    "rr_target": 3.0,
    "min_confidence": 55,
    "min_volume_ratio": 0.4
}

# ---------------------------
# Small indicator helpers (pandas)
# ---------------------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi_wilder(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100/(1+rs))

def atr(df, p=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def resample_to(df, rule):
    ohlc = {"Open":"first", "High":"max", "Low":"min", "Close":"last", "Volume":"sum"}
    return df.resample(rule).agg(ohlc).dropna()

# ---------------------------
# TradingView analysis wrapper
# ---------------------------
def get_tv_analysis(tv_symbol, exchange, screener, interval_key):
    handler = TA_Handler(symbol=tv_symbol, exchange=exchange, screener=screener, interval=TV_INTERVALS[interval_key])
    try:
        analysis = handler.get_analysis()
        return analysis
    except Exception as e:
        return None

# ---------------------------
# Build signal from TV summary (primary)
# ---------------------------
def build_signal_from_tv(market_key, interval_key, rr_target=3.0):
    m = MARKETS[market_key]
    analysis = get_tv_analysis(m["tv_symbol"], m["exchange"], m["screener"], interval_key)
    if analysis is None:
        return {"error":"TradingView analysis failed for this symbol/timeframe."}
    summary = analysis.summary or {}
    indicators = analysis.indicators or {}

    rec = (summary.get("RECOMMENDATION") or "NEUTRAL").upper()
    direction = "NONE"
    if rec.startswith("BUY"): direction = "BUY"
    if rec.startswith("SELL"): direction = "SELL"

    # Try to get a current price from indicators; else fallback to yfinance last close
    entry = None
    for key in ("close","Close","c"):
        if key in indicators:
            try:
                entry = float(indicators[key]); break
            except Exception:
                entry = None
    if entry is None and m.get("yf"):
        try:
            df_tmp = yf.download(m["yf"], period="1d", interval="1m", progress=False, threads=False)
            if not df_tmp.empty:
                entry = float(df_tmp["Close"].iat[-1])
        except Exception:
            entry = None

    # compute confidence from summary counts
    b = summary.get("BUY",0); s = summary.get("SELL",0); n = summary.get("NEUTRAL",0)
    total = max(b+s+n, 1)
    if direction == "BUY":
        confidence = int(np.clip((b/total)*100, 0, 100))
    elif direction == "SELL":
        confidence = int(np.clip((s/total)*100, 0, 100))
    else:
        confidence = int(np.clip((max(b,s)/total)*100, 0, 100))

    reasons = []
    if b> s: reasons.append(f"More BUY indicators ({b})")
    elif s> b: reasons.append(f"More SELL indicators ({s})")
    else: reasons.append(f"Balanced indicators B:{b}/S:{s}/N:{n}")

    # SL/TP: attempt to use ATR indicator if present; else fallback percent
    atr_val = None
    for key in ("ATR","atr"):
        try:
            if key in indicators:
                atr_val = float(indicators[key]); break
        except Exception:
            atr_val = None
    fallback_pct = 0.01 if m["type"]=="crypto" else 0.001
    if entry is None:
        sl = tp = rr = None
    else:
        sl_move = atr_val if atr_val and 0<atr_val<entry*0.5 else entry * fallback_pct
        if direction=="BUY":
            sl = round(entry - sl_move, 8)
            tp = round(entry + sl_move * rr_target, 8)
        elif direction=="SELL":
            sl = round(entry + sl_move, 8)
            tp = round(entry - sl_move * rr_target, 8)
        else:
            sl = tp = None
        rr = round(abs((tp-entry)/(entry-sl)), 2) if sl and entry!=sl else None

    return {
        "direction": direction, "entry": entry, "sl": sl, "tp": tp,
        "rr": rr, "confidence": confidence, "reasons": reasons, "tv_summary": summary, "tv_indicators": indicators
    }

# ---------------------------
# Backtest engine (yfinance OHLC)
# ---------------------------
def fetch_ohlc_yf(yf_symbol, interval, period="90d"):
    try:
        df = yf.download(tickers=yf_symbol, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=lambda c: c.capitalize())
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()

def simple_backtest(df):
    # quick sliding-window backtest using the same scoring rules as the app
    if df is None or df.empty or len(df) < 300:
        return {"error":"Not enough data for backtest (need >=300 bars)"}
    wins = losses = timeouts = 0
    trades = []
    n = len(df)
    for i in range(200, n-2):
        window = df.iloc[:i+1].copy()
        # lightweight scoring: EMA crossover + RSI threshold + MACD sign
        ema_s = ema(window["Close"], 20).iat[-1]
        ema_l = ema(window["Close"], 50).iat[-1]
        r = rsi_wilder(window["Close"], 14).iat[-1]
        macd_line = ema(window["Close"], 12) - ema(window["Close"], 26)
        macd_sig = macd_line.ewm(span=9, adjust=False).mean().iat[-1]
        signal = None
        if ema_s > ema_l and r > 50 and (macd_line.iat[-1] - macd_sig) > 0:
            signal = "BUY"
        elif ema_s < ema_l and r < 50 and (macd_line.iat[-1] - macd_sig) < 0:
            signal = "SELL"
        if signal:
            entry = float(df["Open"].iat[i+1])
            atr_v = float(atr(window, 14).iat[-1]) if len(window) >= 14 else max(0.001, entry*0.001)
            sl = entry - atr_v if signal=="BUY" else entry + atr_v
            tp = entry + 3*(entry-sl) if signal=="BUY" else entry - 3*(sl-entry)
            # scan forward for hit
            outcome = None
            for j in range(i+1, min(n, i+1+240)):
                high = float(df["High"].iat[j]); low = float(df["Low"].iat[j])
                if signal=="BUY":
                    if high >= tp: outcome = "TP"; break
                    if low <= sl: outcome = "SL"; break
                else:
                    if low <= tp: outcome = "TP"; break
                    if high >= sl: outcome = "SL"; break
            trades.append({"entry_idx":i+1,"outcome": outcome or "TIMEOUT"})
            if outcome=="TP": wins +=1
            elif outcome=="SL": losses +=1
            else: timeouts +=1
    total = wins + losses
    winrate = (wins/total*100) if total>0 else None
    return {"trades": len(trades), "wins": wins, "losses": losses, "timeouts": timeouts, "winrate": winrate}

# ---------------------------
# Scanner
# ---------------------------
def scanner(interval_key):
    results=[]
    for mk in MARKETS.keys():
        cfg = MARKETS[mk]
        try:
            handler = TA_Handler(symbol=cfg["tv_symbol"], exchange=cfg["exchange"], screener=cfg["screener"], interval=TV_INTERVALS[interval_key])
            analysis = handler.get_analysis()
            summary = analysis.summary or {}
            b = summary.get("BUY",0); s=summary.get("SELL",0); n=summary.get("NEUTRAL",0)
            conf = int(np.clip((max(b,s)/max(b+s+n,1))*100, 0, 100))
            results.append({"market": mk, "rec": summary.get("RECOMMENDATION","NEUTRAL"), "buy": b, "sell": s, "neutral": n, "confidence": conf})
            time.sleep(0.25)
        except Exception as e:
            results.append({"market": mk, "error": str(e)})
    return pd.DataFrame(results)

# ---------------------------
# UI
# ---------------------------
left, right = st.columns([1,2])

with left:
    st.subheader("Config")
    market_key = st.selectbox("Market", list(MARKETS.keys()), index=0)
    tv_interval = st.selectbox("TradingView TF", ["1m","5m","15m","1h","4h"], index=0)
    backtest_interval = st.selectbox("Backtest interval (yfinance)", ["1m","5m","15m","60m","1d"], index=4)
    account_balance = st.number_input("Account balance (USD)", min_value=10.0, value=1000.0, step=50.0)
    risk_pct = st.number_input("Risk % per trade", min_value=0.1, value=1.0, max_value=10.0, step=0.1)

    st.markdown("---")
    st.subheader("Scanner / Backtest")
    if st.button("Run Market Scanner (quick)"):
        with st.spinner("Scanning..."):
            dfscan = scanner(tv_interval)
            st.dataframe(dfscan.sort_values("confidence", ascending=False).reset_index(drop=True))

    if st.button("Run Backtest (yfinance)"):
        cfg = MARKETS[market_key]
        yf_sym = cfg.get("yf")
        if not yf_sym:
            st.error("Backtest not available for selected market (no yfinance mapping).")
        else:
            with st.spinner("Running backtest (this may take ~1 minute)..."):
                stats = simple_backtest(fetch_ohlc_yf(yf_sym, backtest_interval, period="90d"))
                if isinstance(stats, dict) and stats.get("error"):
                    st.error(stats["error"])
                else:
                    st.metric("Trades", stats["trades"])
                    st.metric("Wins", stats["wins"])
                    st.metric("Losses", stats["losses"])
                    st.metric("Timeouts", stats["timeouts"])
                    st.metric("Win rate", f\"{stats['winrate']:.1f}%\" if stats.get("winrate") else "N/A")

with right:
    st.subheader("Chart (TradingView)")
    cfg = MARKETS[market_key]
    # safe iframe: compute pieces first
    tv_ex = cfg["exchange"]; tv_sym = cfg["tv_symbol"]; iv = tv_interval.replace("m","")
    iframe_html = TV_IFRAME_TEMPLATE.format(exchange=tv_ex, symbol=tv_sym, iv=iv)
    st.components.v1.html(iframe_html, height=540)

    st.markdown("---")
    if st.button("🔮 Generate Signal (TradingView summary)"):
        with st.spinner("Fetching TradingView summary..."):
            sig = build_signal_from_tv(market_key, tv_interval, rr_target=CFG["rr_target"])
            if "error" in sig:
                st.error(sig["error"])
            else:
                if sig["direction"]=="NONE" or sig["confidence"] < CFG["min_confidence"]:
                    st.warning(f"No strong signal. Direction: {sig['direction']} | Confidence: {sig['confidence']}%")
                    st.write("Reasons:", sig["reasons"])
                else:
                    st.success(f\"SIGNAL: {sig['direction']}  |  Confidence: {sig['confidence']}%`)
                    st.write(f\"Entry: {sig['entry']}  \\nSL: {sig['sl']}  |  TP: {sig['tp']}  |  R:R: {sig['rr']}\")
                    # suggested size
                    if sig['entry'] and sig['sl']:
                        risk_amount = round(account_balance * (risk_pct/100.0), 2)
                        size = round(risk_amount / abs(sig['entry'] - sig['sl']), 6)
                    else:
                        risk_amount = 0.0; size = 0.0
                    st.info(f\"Suggested size: {size} units  |  Risk amount: ${risk_amount}\")
                    with st.expander("Why this signal?"):
                        for r in sig["reasons"]:
                            st.write("•", r)
                    # save history
                    row = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "market": market_key,
                        "interval": tv_interval,
                        "signal": sig["direction"],
                        "confidence": sig["confidence"],
                        "entry": sig["entry"],
                        "sl": sig["sl"],
                        "tp": sig["tp"],
                        "rr": sig["rr"],
                        "size": size,
                        "risk_amount": risk_amount
                    }
                    if os.path.exists(HISTORY_FILE):
                        old = pd.read_csv(HISTORY_FILE)
                        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(HISTORY_FILE, index=False)
                    else:
                        pd.DataFrame([row]).to_csv(HISTORY_FILE, index=False)
                    st.success("Signal saved to bot_data/signal_history.csv")

st.markdown("---")
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "rb") as f:
        st.download_button("Download signal history CSV", data=f, file_name="signal_history.csv")

st.caption("Educational tool. Paper-test first. Not financial advice.")
