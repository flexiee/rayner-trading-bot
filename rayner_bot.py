# app.py
# Pro Trading Bot — TradingView Technicals (stable) | Fixed RR = 1:3
# No API keys, no exchange accounts. Data via tradingview-ta (TV technical summary).
# Educational use only.

import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from tradingview_ta import TA_Handler, Interval

# ------------- Streamlit setup -------------
st.set_page_config(page_title="Pro Trading Bot — TradingView (1:3 RR)", layout="wide")
st.title("🚀 Pro Trading Bot — TradingView (1:3 RR)")

# ------------- Markets you can trade -------------
# Each entry: TV handler params + a default SL distance model (if ATR unavailable)
# sl_model: percentage-based fallback for SL (of price) if ATR not present.
MARKETS = {
    "EUR/USD":  {
        "symbol": "EURUSD", "exchange": "OANDA",   "screener": "forex",
        "tv_symbol": "OANDA:EURUSD", "class": "forex", "sl_model": 0.0010  # 0.10%
    },
    "GBP/JPY":  {
        "symbol": "GBPJPY", "exchange": "OANDA",   "screener": "forex",
        "tv_symbol": "OANDA:GBPJPY", "class": "forex", "sl_model": 0.0012
    },
    "USD/JPY":  {
        "symbol": "USDJPY", "exchange": "OANDA",   "screener": "forex",
        "tv_symbol": "OANDA:USDJPY", "class": "forex", "sl_model": 0.0010
    },
    "XAU/USD":  {
        "symbol": "XAUUSD", "exchange": "OANDA",   "screener": "forex",
        "tv_symbol": "OANDA:XAUUSD", "class": "metal", "sl_model": 0.0025
    },
    "BTC/USDT": {
        "symbol": "BTCUSDT","exchange": "BINANCE", "screener": "crypto",
        "tv_symbol": "BINANCE:BTCUSDT","class": "crypto","sl_model": 0.0100  # 1.0%
    },
    "ETH/USDT": {
        "symbol": "ETHUSDT","exchange": "BINANCE", "screener": "crypto",
        "tv_symbol": "BINANCE:ETHUSDT","class": "crypto","sl_model": 0.0120
    },
    "NIFTY 50": {
        "symbol": "NIFTY",  "exchange": "NSE",     "screener": "india",
        "tv_symbol": "NSE:NIFTY", "class": "index","sl_model": 0.0050
    },
    "BANKNIFTY": {
        "symbol": "BANKNIFTY", "exchange": "NSE",  "screener": "india",
        "tv_symbol": "NSE:BANKNIFTY","class": "index","sl_model": 0.0060
    },
}

# ------------- Timeframe mapping -------------
TV_INTERVALS = {
    "1m":  Interval.INTERVAL_1_MINUTE,
    "5m":  Interval.INTERVAL_5_MINUTES,
    "15m": Interval.INTERVAL_15_MINUTES,
    "1h":  Interval.INTERVAL_1_HOUR,
    "4h":  Interval.INTERVAL_4_HOURS,
    "1d":  Interval.INTERVAL_1_DAY,
}

# ------------- TradingView Chart Embed -------------
TV_IFRAME = (
    '<iframe src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval={iv}'
    '&hidesidetoolbar=1&hideideas=1&theme=dark" width="100%" height="520" frameborder="0"></iframe>'
)

# ------------- Utility -------------
def pct_to_price_move(price: float, pct: float) -> float:
    return float(price) * float(pct)

def safe_indicator(indicators: dict, key: str, default=None):
    try:
        v = indicators.get(key, default)
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def summarize_reasons(summary: dict) -> list:
    """
    Convert TradingView summary counts into human-friendly reasons.
    """
    reasons = []
    b, s, n = summary.get("BUY", 0), summary.get("SELL", 0), summary.get("NEUTRAL", 0)
    if b > s:
        reasons.append(f"More BUY signals ({b}) than SELL ({s})")
    elif s > b:
        reasons.append(f"More SELL signals ({s}) than BUY ({b})")
    else:
        reasons.append(f"Balanced signals — Buy {b}, Sell {s}, Neutral {n}")

    # Add a few popular indicator hints if present
    # tradingview_ta returns keys like "RSI", "MACD.macd", "MACD.signal", "ADX", "ATR", "close"
    # We won't fail if absent.
    return reasons

def make_signal(handler: TA_Handler, rr: float):
    """
    Use TradingView technical summary (live) to produce a signal, entry, SL/TP (1:RR), confidence & reasons.
    """
    analysis = handler.get_analysis()
    summary = analysis.summary or {}
    indicators = analysis.indicators or {}

    # Direction
    rec = summary.get("RECOMMENDATION", "NEUTRAL").upper()
    if rec.startswith("BUY"):
        direction = "BUY"
    elif rec.startswith("SELL"):
        direction = "SELL"
    else:
        direction = "NONE"

    # Entry price from TV indicators if available
    entry = safe_indicator(indicators, "close", None)

    # Confidence: scale by distribution of BUY/SELL/NEUTRAL
    b, s, n = summary.get("BUY", 0), summary.get("SELL", 0), summary.get("NEUTRAL", 0)
    total_votes = max(b + s + n, 1)
    if direction == "BUY":
        conf = int(np.clip((b / total_votes) * 100, 0, 100))
    elif direction == "SELL":
        conf = int(np.clip((s / total_votes) * 100, 0, 100))
    else:
        conf = int(np.clip((max(b, s) / total_votes) * 100, 0, 100))

    # Reasons (simple)
    reasons = summarize_reasons(summary)

    return {
        "direction": direction,
        "entry": entry,
        "summary": summary,
        "indicators": indicators,
        "confidence": conf,
        "reasons": reasons,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "rr": rr,
    }

def sl_tp_from_tv(mkt_cfg: dict, signal: dict):
    """
    Build SL/TP with fixed 1:RR (RR=3 by default).
    Prefer ATR from indicators if present; otherwise fallback to percent model.
    """
    rr = signal["rr"]
    entry = signal["entry"]
    if entry is None or math.isnan(entry):
        return None, None, None  # cannot size without price

    # 1) Try ATR from TV (points)
    atr = safe_indicator(signal["indicators"], "ATR", None)
    # Sometimes TV returns ATR on higher TFs only; if small/None, fallback to % model
    if atr is None or atr <= 0 or atr > entry * 0.5:
        # 2) Percent model fallback (asset-specific)
        sl_move = pct_to_price_move(entry, mkt_cfg["sl_model"])
    else:
        sl_move = atr

    if signal["direction"] == "BUY":
        sl = round(entry - sl_move, 8)
        tp = round(entry + sl_move * rr, 8)
    elif signal["direction"] == "SELL":
        sl = round(entry + sl_move, 8)
        tp = round(entry - sl_move * rr, 8)
    else:
        return None, None, None

    actual_rr = round(abs((tp - entry) / (entry - sl)), 2) if (entry - sl) != 0 else None
    return sl, tp, actual_rr

# ------------- Sidebar / top controls -------------
left, right = st.columns([1, 2])

with left:
    market = st.selectbox("Market", list(MARKETS.keys()), index=0)
    tf_key = st.selectbox("Timeframe", list(TV_INTERVALS.keys()), index=0)
    balance = st.number_input("Account balance (USD)", min_value=10.0, value=1000.0, step=50.0)
    risk_pct = st.number_input("Risk % per trade", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    rr_fixed = 3.0  # fixed 1:3 RR as requested

with right:
    st.subheader("TradingView Chart")
    st.components.v1.html(
        TV_IFRAME.format(tv_symbol=MARKETS[market]["tv_symbol"], iv=tf_key.replace("m", "")),
        height=540
    )

st.markdown("---")

# ------------- Generate Button -------------
if st.button("🔮 Generate Signal"):
    with st.spinner("Reading TradingView technicals and computing signal…"):
        cfg = MARKETS[market]
        handler = TA_Handler(
            symbol=cfg["symbol"],
            exchange=cfg["exchange"],
            screener=cfg["screener"],
            interval=TV_INTERVALS[tf_key],
        )

        try:
            sig = make_signal(handler, rr_fixed)
        except Exception as e:
            st.error(f"Failed to get TradingView analysis: {e}")
            st.stop()

        # Build SL/TP (1:3 RR)
        sl, tp, rr = sl_tp_from_tv(cfg, sig)

        # If direction NONE or missing prices
        if sig["direction"] == "NONE" or sig["entry"] is None:
            st.warning(f"No actionable signal right now. (Confidence {sig['confidence']}%)")
            with st.expander("Diagnostics"):
                st.write("Summary:", sig["summary"])
            st.stop()

        # Show signal card
        st.success(
            f"SIGNAL: **{sig['direction']}**  |  Confidence: **{sig['confidence']}%**  |  Strength votes (B/S/N): "
            f"{sig['summary'].get('BUY',0)}/{sig['summary'].get('SELL',0)}/{sig['summary'].get('NEUTRAL',0)}"
        )
        st.write(f"**Entry:** {sig['entry']}  \n**SL:** {sl}  |  **TP:** {tp}  |  **R:R:** {rr or rr_fixed}  \n**Time:** {sig['timestamp']}")

        # Sizing (units) using price distance to SL
        if sl is not None and sig["entry"] is not None:
            risk_amount = round(balance * (risk_pct / 100.0), 2)
            distance = abs(sig["entry"] - sl)
            size_units = round(risk_amount / distance, 6) if distance > 0 else 0
            st.info(f"Suggested position size: **{size_units} units**  |  Risk amount: **${risk_amount}**")

        # Reasons
        with st.expander("Why this signal?"):
            for r in sig["reasons"]:
                st.write("•", r)

        # Raw TV summary (optional)
        with st.expander("Raw TradingView summary"):
            st.json(sig["summary"])

st.caption(
    "Powered by TradingView (tradingview-ta). This tool provides educational signals only — "
    "always manage risk. No broker/exchange connections are used."
)
