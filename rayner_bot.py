# rayner_bot.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Rayner Bot (yfinance)", layout="wide")
st.title("📈 Rayner Bot — yfinance version (Stable for Streamlit Cloud)")

# ---------- Markets mapping (for iframe vs yfinance symbol) ----------
MARKETS = {
    "EUR/USD": {"yf": "EURUSD=X", "tv": "OANDA:EURUSD"},
    "GBP/JPY": {"yf": "GBPJPY=X", "tv": "OANDA:GBPJPY"},
    "USD/JPY": {"yf": "JPY=X",    "tv": "OANDA:USDJPY"},   # fallback
    "AUD/USD": {"yf": "AUDUSD=X","tv": "OANDA:AUDUSD"},
    "XAU/USD": {"yf": "GC=F",    "tv": "OANDA:XAUUSD"},   # gold futures symbol for yfinance
    "BTC/USD": {"yf": "BTC-USD", "tv": "BINANCE:BTCUSDT"},
    "ETH/USD": {"yf": "ETH-USD", "tv": "BINANCE:ETHUSDT"},
    "NIFTY 50": {"yf": "^NSEI",  "tv": "NSE:NIFTY"},      # approximate
    "BANKNIFTY": {"yf": "^NSEBANK","tv": "NSE:BANKNIFTY"},
}

# ---------- Helpers ----------
def safe_series(x):
    if isinstance(x, pd.Series):
        return x.astype(float)
    try:
        return pd.Series(x).astype(float)
    except Exception:
        return pd.Series(dtype=float)

def compute_rsi(series, period=14):
    s = safe_series(series).dropna()
    if len(s) < period + 1:
        return pd.Series(dtype=float)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def atr(df, period=14):
    dfc = df.copy()
    if not all(k in dfc.columns for k in ['High','Low','Close','Adj Close']) and not all(k.lower() in dfc.columns for k in ['high','low','close']):
        # try lowercase
        dfc.columns = [c.lower() for c in dfc.columns]
    # ensure names
    try:
        high = safe_series(dfc['High'] if 'High' in dfc.columns else dfc['high'])
        low = safe_series(dfc['Low'] if 'Low' in dfc.columns else dfc['low'])
        close = safe_series(dfc['Close'] if 'Close' in dfc.columns else dfc['close'])
    except Exception:
        return pd.Series(dtype=float)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def generate_signal_from_df(df):
    out = {"signal":"WAIT","confidence":0,"entry":None,"sl":None,"tp":None,"rr":None,"reasons":[]}
    if df is None or df.empty or 'Close' not in df.columns and 'close' not in df.columns:
        return out
    # normalize
    df2 = df.copy()
    cols_lower = [c.lower() for c in df2.columns]
    df2.columns = cols_lower
    close = safe_series(df2['close'])
    if len(close) < 20:
        return out

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    rsi = compute_rsi(close, 14)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    atr_series = atr(df2, 14)

    try:
        last = close.iloc[-1]
        last_ema9 = ema9.iloc[-1]
        last_ema21 = ema21.iloc[-1]
        last_rsi = float(rsi.iloc[-1]) if not rsi.empty else np.nan
        last_macd = float(macd.iloc[-1])
        last_macd_sig = float(macd_signal.iloc[-1])
        last_atr = float(atr_series.iloc[-1]) if not atr_series.empty else np.nan
    except Exception:
        return out

    score = 0
    if last_ema9 > last_ema21:
        score += 10; out["reasons"].append("EMA9 > EMA21 (trend up)")
    else:
        score -= 6; out["reasons"].append("EMA9 < EMA21 (trend down)")

    if not np.isnan(last_rsi):
        if last_rsi < 35:
            score += 12; out["reasons"].append("RSI low (oversold)")
        elif last_rsi > 65:
            score -= 8; out["reasons"].append("RSI high (overbought)")

    if last_macd > last_macd_sig:
        score += 6; out["reasons"].append("MACD bullish")
    else:
        score -= 4; out["reasons"].append("MACD bearish")

    # decide
    if score >= 10:
        out["signal"]="BUY"
    elif score <= -5:
        out["signal"]="SELL"
    else:
        out["signal"]="WAIT"

    out["confidence"] = int(max(10, min(120, 50 + score*4)))

    # build TP/SL using ATR
    if not np.isnan(last_atr) and last_atr > 0:
        sl_points = last_atr * 1.2
        tp_points = sl_points * 2  # 1:2
    else:
        # fallback tiny
        sl_points = last * 0.001
        tp_points = sl_points * 2

    if out["signal"] == "BUY":
        out["entry"] = float(round(last,5))
        out["sl"] = float(round(last - sl_points, 5))
        out["tp"] = float(round(last + tp_points, 5))
    elif out["signal"] == "SELL":
        out["entry"] = float(round(last,5))
        out["sl"] = float(round(last + sl_points, 5))
        out["tp"] = float(round(last - tp_points, 5))
    else:
        out["entry"] = float(round(last,5))
        out["sl"] = None
        out["tp"] = None

    try:
        rr = abs((out["tp"] - out["entry"]) / (out["entry"] - out["sl"])) if out["sl"] and out["tp"] else None
        out["rr"] = round(rr,2) if rr else None
    except Exception:
        out["rr"] = None
    return out

# ---------- UI ----------
left, right = st.columns([1,2])
with left:
    st.header("Account & Settings")
    account_balance = st.number_input("Account balance ($)", value=1000.0, step=10.0)
    risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
    market = st.selectbox("Choose Market", list(MARKETS.keys()), index=0)
    interval = st.selectbox("Interval (for chart)", ["1m","5m","15m","30m","1h","1d"], index=0)

with right:
    st.header("Live Chart")
    tv_symbol = MARKETS.get(market, {}).get("tv", None)
    # tradingview iframe embed (public widget)
    if tv_symbol:
        iframe = f"""<iframe src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval={interval}&theme=dark&hidesidetoolbar=1" width="900" height="480" frameborder="0"></iframe>"""
        st.components.v1.html(iframe, height=520)
    else:
        st.info("No TradingView symbol for this market.")

# Generate Signal
st.markdown("---")
if st.button("Generate Signal"):
    st.info("Fetching OHLC from yfinance and computing indicators...")
    yf_sym = MARKETS.get(market, {}).get("yf", None)
    if yf_sym is None:
        st.error("No yfinance symbol found for this market.")
    else:
        try:
            # Use interval -> map to yfinance period/interval accepted values
            # yfinance handles 1m only for recent period; get last 500 bars for safety
            yf_interval = interval if interval != "1h" else "60m"
            df = yf.download(yf_sym, period="7d", interval=yf_interval, progress=False)
            if df is None or df.empty:
                st.error("No data or too few candles. Try a different market or check symbol.")
            else:
                sig = generate_signal_from_df(df)
                st.success(f"Signal: **{sig['signal']}**  | Confidence: {sig['confidence']}%")
                st.write(f"Entry: {sig['entry']}, SL: {sig['sl']}, TP: {sig['tp']}, R:R: {sig['rr']}")
                st.write("Reasons:")
                for r in sig["reasons"]:
                    st.write("- " + r)

                # risk/lot calc (very simplified)
                if sig['sl'] and sig['entry']:
                    price_diff = abs(sig['entry'] - sig['sl'])
                    # pip value heuristic: forex small pairs => 10$/pip for 1 lot, else scale
                    pip_value = 10.0
                    if sig['entry'] >= 100:  # indices/crypto large price -> treat differently
                        pip_equiv = price_diff  # use price diff as pip-equivalent
                    else:
                        pip_equiv = price_diff * 10000
                    risk_amount = account_balance * (risk_pct / 100)
                    lot_size = 0
                    if pip_equiv > 0:
                        lot_size = round(risk_amount / (pip_equiv * pip_value), 4)
                    st.write(f"Risk Amount: ${risk_amount:.2f} | Suggested lot size (approx): {lot_size}")
                st.markdown("#### Last candles (tail)")
                st.dataframe(df.tail(8))
        except Exception as e:
            st.error(f"Error fetching/processing data: {e}")

st.caption(f"App time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
