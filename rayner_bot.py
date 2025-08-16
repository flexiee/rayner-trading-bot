# app.py
# ------------------------------------------------------------
# Next-Level Trading Bot (single file)
# - TradingView chart embed (visual)
# - Data via yfinance (1m..1h)
# - Signal engine: EMA trend, RSI, MACD, candle engulfing,
#   Bollinger z-score bias, ATR-based SL/TP, MTF confirmation
# - Risk: pip calc + lot size + % risk
# - Market strength score + signal history CSV
# ------------------------------------------------------------
import os
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Technical indicators (ta)
import ta  # pip install ta

# Optional plotting (internal R:R visualization)
try:
    import mplfinance as mpf
except Exception:
    mpf = None

st.set_page_config(page_title="Next Level Trading Bot", layout="wide")

# ------------------- Paths -------------------
DATA_DIR = os.path.join(os.getcwd(), "bot_data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(DATA_DIR, "signal_history.csv")

# ------------------- Symbol maps -------------------
# yfinance tickers for data + TradingView symbols for the iframe
MARKETS = {
    "EUR/USD": {"yf": "EURUSD=X",   "tv": "OANDA:EURUSD"},
    "GBP/USD": {"yf": "GBPUSD=X",   "tv": "OANDA:GBPUSD"},
    "USD/JPY": {"yf": "JPY=X",      "tv": "OANDA:USDJPY"},
    "XAU/USD": {"yf": "GC=F",       "tv": "OANDA:XAUUSD"},
    "BTC/USD": {"yf": "BTC-USD",    "tv": "BITSTAMP:BTCUSD"},
    "ETH/USD": {"yf": "ETH-USD",    "tv": "BITSTAMP:ETHUSD"},
    "NIFTY 50": {"yf": "^NSEI",     "tv": "NSE:NIFTY"},
}

# ------------------- Utilities -------------------
def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns MultiIndex columns; flatten to single level."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only OHLCV and drop rows with NA."""
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].copy()
    df = df.dropna()
    # Convert to float for ta
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna()
    # Enforce DatetimeIndex
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")]
    return df

def fetch_data_yf(yf_symbol: str, interval: str = "5m", period: str = "7d") -> pd.DataFrame:
    """Fetch bars from yfinance and sanitize."""
    try:
        df = yf.download(tickers=yf_symbol, interval=interval, period=period, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = flatten_yf_columns(df)
        df = ensure_ohlcv(df)
        return df
    except Exception as e:
        st.error(f"Data fetch error for {yf_symbol}: {e}")
        return pd.DataFrame()

def resample_to(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1m→5m (or any) OHLCV."""
    if df.empty:
        return df
    ohlc = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = df.resample(rule).agg(ohlc).dropna()
    return out

# ------------------- Indicators -------------------
def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14):
    return ta.momentum.RSIIndicator(series, window=period).rsi()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    m = ta.trend.MACD(series, window_slow=slow, window_fast=fast, window_sign=signal)
    return m.macd(), m.macd_signal(), m.macd_diff()

def atr(df: pd.DataFrame, period: int = 14):
    a = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=period)
    return a.average_true_range()

# ------------------- Risk / sizing -------------------
def pip_size(symbol_key: str) -> float:
    s = symbol_key.upper()
    if "JPY" in s and "XAU" not in s:
        return 0.01
    if "XAU" in s or "GOLD" in s:
        return 0.01
    return 0.0001  # default forex majors

def calc_position_size(account_balance: float, risk_percent: float, entry: float, sl: float, symbol_key: str):
    if entry is None or sl is None:
        return 0.0, 0.0
    risk_amount = max(0.0, account_balance * (risk_percent / 100.0))
    pip = pip_size(symbol_key)
    pip_distance = abs(entry - sl) / pip if pip > 0 else 0
    if pip_distance <= 0:
        return 0.0, round(risk_amount, 2)
    approx_pip_value_per_lot = 10.0  # heuristic for USD acct on majors
    lots = risk_amount / (pip_distance * approx_pip_value_per_lot)
    return round(lots, 4), round(risk_amount, 2)

# ------------------- Market strength -------------------
def market_strength(df: pd.DataFrame) -> int:
    """0..100 score using momentum (slope) + volatility."""
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

# ------------------- Signal engine -------------------
def generate_signal(df: pd.DataFrame, symbol_key: str, cfg: dict, mtf_df: pd.DataFrame | None) -> dict:
    """
    Returns:
      dict(signal, entry, sl, tp, rr, confidence, reasons(list), strength)
    """
    out = {"signal": "NONE", "entry": None, "sl": None, "tp": None, "rr": None,
           "confidence": 0, "reasons": [], "strength": 0}
    if df is None or df.empty:
        return out

    price = float(df["Close"].iat[-1])
    out["entry"] = price

    # Indicators
    ema_s = float(ema(df["Close"], cfg["ema_short"]).iat[-1])
    ema_l = float(ema(df["Close"], cfg["ema_long"]).iat[-1])
    r = float(rsi(df["Close"], cfg["rsi_period"]).iat[-1])
    macd_line, macd_sig, macd_hist = macd(df["Close"], cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
    macd_dir = float(macd_line.iat[-1] - macd_sig.iat[-1])

    # Bollinger z-score (bias)
    mid = df["Close"].rolling(cfg["bb_period"]).mean().iat[-1]
    std = df["Close"].rolling(cfg["bb_period"]).std().iat[-1]
    if not math.isnan(std) and std > 0:
        z = (price - mid) / std
    else:
        z = 0.0

    # Votes
    votes_bull = 0
    votes_bear = 0

    # EMA trend
    if ema_s > ema_l:
        votes_bull += 1
        out["confidence"] += 15
        out["reasons"].append("EMA short > EMA long")
    else:
        votes_bear += 1
        out["reasons"].append("EMA short < EMA long")

    # RSI
    if r < cfg["rsi_oversold"]:
        votes_bull += 1
        out["confidence"] += 12
        out["reasons"].append("RSI oversold")
    elif r > cfg["rsi_overbought"]:
        votes_bear += 1
        out["reasons"].append("RSI overbought")
    else:
        out["confidence"] += 4
        out["reasons"].append("RSI neutral")

    # MACD momentum
    if macd_dir > 0:
        votes_bull += 1
        out["confidence"] += 10
        out["reasons"].append("MACD bullish")
    else:
        votes_bear += 1
        out["reasons"].append("MACD bearish")

    # Bollinger bias
    if z < -2:
        votes_bull += 1
        out["confidence"] += 6
        out["reasons"].append("Below -2σ (mean reversion bias up)")
    elif z > 2:
        votes_bear += 1
        out["reasons"].append("Above +2σ (mean reversion bias down)")

    # Candle engulfing confirmation
    if len(df) >= 2:
        prev = df.iloc[-2]
        cur = df.iloc[-1]
        # bullish engulfing
        if (cur["Close"] > cur["Open"]) and (prev["Close"] < prev["Open"]) and (cur["Close"] > prev["Open"]) and (cur["Open"] < prev["Close"]):
            votes_bull += 1
            out["confidence"] += 8
            out["reasons"].append("Bullish engulfing")
        # bearish engulfing
        if (cur["Close"] < cur["Open"]) and (prev["Close"] > prev["Open"]) and (cur["Close"] < prev["Open"]) and (cur["Open"] > prev["Close"]):
            votes_bear += 1
            out["confidence"] -= 5
            out["reasons"].append("Bearish engulfing")

    # Multi-timeframe: 5m confirmation (if available)
    if mtf_df is not None and len(mtf_df) >= max(cfg["ema_long"], cfg["rsi_period"]):
        mtf_ema_s = float(ema(mtf_df["Close"], cfg["ema_short"]).iat[-1])
        mtf_ema_l = float(ema(mtf_df["Close"], cfg["ema_long"]).iat[-1])
        if mtf_ema_s > mtf_ema_l:
            out["confidence"] += 6
            out["reasons"].append("MTF(5m) aligned up")
        else:
            out["reasons"].append("MTF(5m) not aligned up")

    # Decision
    if votes_bull > votes_bear:
        out["signal"] = "BUY"
    elif votes_bear > votes_bull:
        out["signal"] = "SELL"
    else:
        out["signal"] = "NONE"

    # ATR-based SL/TP
    atr_series = atr(df, cfg["atr_period"])
    atr_v = float(atr_series.iat[-1]) if len(atr_series) else None
    if atr_v is None or math.isnan(atr_v) or atr_v == 0:
        recent_std = df["Close"].pct_change().rolling(14).std().iat[-1] if len(df) >= 14 else 0.001
        vol_px = recent_std * price
    else:
        vol_px = atr_v

    if out["signal"] == "BUY":
        out["sl"] = price - vol_px * float(cfg["sl_atr_mult"])
        out["tp"] = price + vol_px * float(cfg["tp_atr_mult"])
    elif out["signal"] == "SELL":
        out["sl"] = price + vol_px * float(cfg["sl_atr_mult"])
        out["tp"] = price - vol_px * float(cfg["tp_atr_mult"])

    if out["sl"] and out["tp"]:
        denom = abs(price - out["sl"])
        out["rr"] = round(abs((out["tp"] - price) / denom), 2) if denom > 0 else None

    out["confidence"] = int(np.clip(out["confidence"], 0, 100))
    out["strength"] = market_strength(df)
    return out

# ------------------- Plot internal R:R box -------------------
def plot_rr_box(df: pd.DataFrame, entry: float, sl: float, tp: float, title: str | None = None):
    if df is None or df.empty:
        return
    if mpf is None:
        # simple fallback
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df["Close"])
        ax.axhline(entry, linestyle="--", label="Entry")
        if sl: ax.axhline(sl, linestyle=":", label="SL")
        if tp: ax.axhline(tp, linestyle="-.", label="TP")
        if title: ax.set_title(title)
        ax.legend()
        st.pyplot(fig)
        return

    mc = mpf.make_marketcolors(up="g", down="r", wick="inherit", edge="inherit", volume="in")
    style = mpf.make_mpf_style(marketcolors=mc)
    fig, axes = mpf.plot(df.tail(200), type="candle", style=style, volume=True, returnfig=True, figsize=(12, 6))
    ax = axes[0]
    ax.axhline(entry, color="blue", linestyle="--", linewidth=1.1, label="Entry")
    if sl: ax.axhline(sl, color="red", linestyle="--", linewidth=1.0, label="SL")
    if tp: ax.axhline(tp, color="green", linestyle="--", linewidth=1.0, label="TP")
    try:
        if entry and sl and tp:
            ymin = min(entry, sl, tp)
            ymax = max(entry, sl, tp)
            ax.fill_between(df.tail(200).index, ymin, ymax, alpha=0.08)
    except Exception:
        pass
    if title:
        ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

# ------------------- UI -------------------
st.title("📈 Next Level Trading Bot (Single File)")

left, right = st.columns([1.05, 1.95])

with left:
    st.subheader("Settings")
    market_name = st.selectbox("Market", list(MARKETS.keys()), index=0)
    intervals = {"1m": ("1m", "7d"), "5m": ("5m", "30d"), "15m": ("15m", "60d"), "1h": ("1h", "730d")}
    interval_label = st.selectbox("Interval", list(intervals.keys()), index=1)
    yf_interval, yf_period = intervals[interval_label]

    account_balance = st.number_input("Account Balance (USD)", min_value=10.0, value=1000.0, step=50.0)
    risk_percent = st.slider("Risk % per trade", 0.1, 5.0, 1.0, 0.1)

    st.markdown("---")
    st.caption("Strategy Parameters (adjust if you want)")
    ema_short = st.number_input("EMA short", 3, 100, 9)
    ema_long = st.number_input("EMA long", 5, 400, 21)
    rsi_p = st.number_input("RSI period", 5, 50, 14)
    rsi_over = st.number_input("RSI overbought", 55, 95, 70)
    rsi_under = st.number_input("RSI oversold", 5, 45, 30)
    macd_fast = st.number_input("MACD fast", 5, 50, 12)
    macd_slow = st.number_input("MACD slow", 10, 100, 26)
    macd_sig = st.number_input("MACD signal", 5, 30, 9)
    bb_period = st.number_input("Bollinger period", 10, 50, 20)
    atr_period = st.number_input("ATR period", 5, 50, 14)
    sl_mult = st.number_input("SL ATR multiplier", 0.5, 10.0, 1.5, 0.1)
    tp_mult = st.number_input("TP ATR multiplier", 0.5, 15.0, 3.0, 0.1)
    threshold = st.slider("Confidence threshold", 10, 100, 50)

    config = {
        "ema_short": int(ema_short), "ema_long": int(ema_long),
        "rsi_period": int(rsi_p), "rsi_overbought": int(rsi_over), "rsi_oversold": int(rsi_under),
        "macd_fast": int(macd_fast), "macd_slow": int(macd_slow), "macd_signal": int(macd_sig),
        "bb_period": int(bb_period), "atr_period": int(atr_period),
        "sl_atr_mult": float(sl_mult), "tp_atr_mult": float(tp_mult),
    }

    st.markdown("---")
    st.subheader("History")
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        st.dataframe(hist.sort_values("timestamp", ascending=False).head(25))
    else:
        st.write("No history yet.")

with right:
    st.subheader("Chart & Signal")
    tv_symbol = MARKETS[market_name]["tv"]
    yf_symbol = MARKETS[market_name]["yf"]

    # TradingView chart embed (visual)
    st.markdown("**TradingView (visual reference)**")
    tv_iframe = f"""
    <iframe src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval=1&symboledit=1&hideideas=1&hidetoptoolbar=1"
            width="100%" height="520" style="border:1px solid rgba(255,255,255,0.1)"></iframe>
    """
    st.components.v1.html(tv_iframe, height=540)

    st.markdown("---")

    if st.button("🔮 Generate Signal"):
        with st.spinner("Fetching data and computing..."):
            df = fetch_data_yf(yf_symbol, interval=yf_interval, period=yf_period)

        if df.empty:
            st.error("No data returned. Try a different symbol/interval.")
        else:
            # Build MTF (5-minute) from the base data if base is 1m; otherwise still create 5m for alignment
            try:
                mtf_df = resample_to(df, "5T")
            except Exception:
                mtf_df = None

            signal = generate_signal(df, market_name, config, mtf_df)
            if signal["confidence"] < threshold:
                signal["signal"] = "NONE"

            st.success(f"Signal: {signal['signal']}  |  Confidence: {signal['confidence']}  |  Strength: {signal['strength']}/100")

            st.write(
                f"**Entry:** {signal['entry']}\n\n"
                f"**SL:** {signal['sl']}   |   **TP:** {signal['tp']}   |   **R:R:** {signal.get('rr')}"
            )

            lots, risk_amt = calc_position_size(
                account_balance, risk_percent, signal["entry"], signal["sl"] if signal["sl"] else signal["entry"], market_name
            )
            st.write(f"**Suggested lots (approx):** {lots}   |   **Risk amount:** ${risk_amt}")

            if signal.get("reasons"):
                with st.expander("Why this signal?", expanded=False):
                    for r in signal["reasons"]:
                        st.write("•", r)

            # Internal candlestick with R:R box
            st.markdown("**Internal candlestick with R:R visualization**")
            plot_rr_box(df, signal["entry"], signal["sl"], signal["tp"], title=f"{market_name} — {signal['signal']} (R:R {signal.get('rr')})")

            # Save history
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "market": market_name,
                "yf_symbol": yf_symbol,
                "tv_symbol": tv_symbol,
                "interval": yf_interval,
                "signal": signal["signal"],
                "confidence": signal["confidence"],
                "strength": signal["strength"],
                "entry": signal["entry"],
                "sl": signal["sl"],
                "tp": signal["tp"],
                "rr": signal.get("rr"),
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

# ------------------- Footer -------------------
st.caption(
    "This tool is for educational use only and does not execute trades. "
    "Trading involves risk; use proper risk management and test before going live."
)
