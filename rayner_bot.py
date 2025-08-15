# rayner_bot.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from tvDatafeed import TvDatafeed, Interval

# ---------- Helpers ----------
def safe_series(series_like):
    """
    Ensure series_like becomes a 1-D pandas Series suitable for indicators.
    """
    if series_like is None:
        return pd.Series(dtype=float)
    if isinstance(series_like, pd.DataFrame):
        # prefer a numeric column if present, else first column
        numeric_cols = [c for c in series_like.columns if np.issubdtype(series_like[c].dtype, np.number)]
        if numeric_cols:
            s = series_like[numeric_cols[0]]
        else:
            s = series_like.iloc[:, 0]
    elif isinstance(series_like, pd.Series):
        s = series_like
    else:
        s = pd.Series(series_like)
    # flatten any multiindex columns etc.
    return s.astype(float).copy()

def compute_rsi(series, period=14):
    s = safe_series(series).dropna()
    if s.empty or len(s) < period + 1:
        return pd.Series(dtype=float)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    # for first non-null use simple ratio, then fill using EMA approach
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # if initial values NaN because of rolling, compute manually for first valid index
    rsi = rsi.fillna(method='bfill')
    return rsi

def normalize_df_cols(df):
    df = df.copy()
    # lowercase column names
    df.columns = [c.lower() for c in df.columns]
    return df

def atr(df, period=14):
    """
    Simple ATR using high, low, close. df must contain 'high','low','close'
    """
    df = normalize_df_cols(df)
    if not all(k in df.columns for k in ('high', 'low', 'close')):
        return pd.Series(dtype=float)
    high = safe_series(df['high'])
    low = safe_series(df['low'])
    close = safe_series(df['close'])
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(window=period, min_periods=period).mean()
    return atr_series

# ---------- TradingView / Data feed ----------
tv = None
try:
    tv = TvDatafeed()  # guest mode: leave blank. In Cloud ensure tvDatafeed installed.
except Exception as e:
    # We'll show a message in UI if tv is None
    tv = None

# Map a market key to (exchange, symbol_for_tvdata, tradingview_symbol_for_iframe)
MARKETS = {
    "EUR/USD": ("OANDA", "EURUSD", "OANDA:EURUSD"),
    "GBP/JPY": ("OANDA", "GBPJPY", "OANDA:GBPJPY"),
    "USD/JPY": ("OANDA", "USDJPY", "OANDA:USDJPY"),
    "AUD/USD": ("OANDA", "AUDUSD", "OANDA:AUDUSD"),
    "XAU/USD": ("OANDA", "XAUUSD", "OANDA:XAUUSD"),
    "BTC/USD": ("BINANCE", "BTCUSDT", "BINANCE:BTCUSDT"),
    "ETH/USD": ("BINANCE", "ETHUSDT", "BINANCE:ETHUSDT"),
    # Add more maps as needed
}

# ---------- Signal Logic ----------
def generate_signal_from_df(df):
    """
    Returns: dict with keys: signal ('BUY'/'SELL'/'WAIT'),
             confidence (int 0-100+), sl, tp, rr_ratio
    """
    out = {"signal": "WAIT", "confidence": 0, "sl": None, "tp": None, "rr": None, "reasons": []}
    if df is None or df.empty:
        return out

    df = normalize_df_cols(df)
    # ensure we have close/open/high/low
    if 'close' not in df.columns or len(df) < 20:
        return out

    close = safe_series(df['close'])
    rsi = compute_rsi(close, period=14)
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    atr_series = atr(df, period=14)

    # Last values
    try:
        last_rsi = float(rsi.iloc[-1])
        last_ema9 = float(ema9.iloc[-1])
        last_ema21 = float(ema21.iloc[-1])
        last_macd = float(macd_line.iloc[-1])
        last_macd_signal = float(signal_line.iloc[-1])
        last_atr = float(atr_series.iloc[-1]) if not atr_series.empty else np.nan
        last_close = float(close.iloc[-1])
    except Exception:
        return out

    score = 0
    # EMA trend
    if last_ema9 > last_ema21:
        score += 10
        out["reasons"].append("EMA short above EMA long (trend up)")
    else:
        score -= 5
        out["reasons"].append("EMA short below EMA long (trend down)")

    # RSI confirmation
    if last_rsi < 35:
        score += 12
        out["reasons"].append("RSI oversold")
    elif last_rsi > 65:
        score -= 8
        out["reasons"].append("RSI overbought")

    # MACD confirmation
    if last_macd > last_macd_signal:
        score += 8
        out["reasons"].append("MACD bullish")
    else:
        score -= 6
        out["reasons"].append("MACD bearish")

    # Build final signal
    if score >= 10:
        out["signal"] = "BUY"
    elif score <= -5:
        out["signal"] = "SELL"
    else:
        out["signal"] = "WAIT"

    # Confidence: scale and clip 10..120
    out["confidence"] = int(max(10, min(120, 50 + score * 4)))

    # Set SL/TP based on recent ATR or fixed fallback
    if not np.isnan(last_atr) and last_atr > 0:
        # choose a multiple for SL and TP (market specific can be tuned)
        sl = last_atr * 1.2
        tp = sl * 2  # 1:2 base target
    else:
        sl = (last_close * 0.0010) if last_close > 0 else 10  # fallback small SL
        tp = sl * 2

    # finalize
    out["sl"] = float(round(last_close - sl, 5)) if out["signal"] == "BUY" else float(round(last_close + sl, 5))
    out["tp"] = float(round(last_close + tp, 5)) if out["signal"] == "BUY" else float(round(last_close - tp, 5))
    out["rr"] = round((tp) / (sl if sl != 0 else 1), 2)

    return out

# ---------- UI ----------
st.set_page_config(page_title="Rayner Bot (Fixed)", layout="wide")
st.markdown("# 📈 Rayner Bot — Fixed & Robust")

col1, col2 = st.columns([1,3])
with col1:
    st.subheader("Settings")
    market = st.selectbox("Market", list(MARKETS.keys()))
    timeframe = st.selectbox("Interval", ["1m","5m","15m","30m","60m"])
    account_balance = st.number_input("Account balance ($)", value=1000.0, step=10.0)
    risk_pct = st.slider("Risk % per trade", 0.1, 5.0, 1.0, 0.1)
    st.write("Press Generate Signal to fetch live data & analyze.")

with col2:
    st.subheader("Live Chart & Signal")
    # TradingView iframe area
    tv_symbol = MARKETS[market][2] if market in MARKETS else None
    iframe_html = ""
    if tv_symbol:
        iframe_html = f"""<iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_{tv_symbol.replace(':','_')}&symbol={tv_symbol}&interval={timeframe}&hidesidetoolbar=1&theme=dark" width="900" height="450" frameborder="0"></iframe>"""
    else:
        iframe_html = "<div style='color:yellow'>No frame available for this market.</div>"

    st.components.v1.html(iframe_html, height=480)

    if st.button("Generate Signal"):
        if tv is None:
            st.error("tvDatafeed not initialized. Install and configure tvDatafeed in the environment.")
        else:
            exch, sym_tvdata, _ = MARKETS[market]
            # fetch data carefully
            try:
                df = tv.get_hist(symbol=sym_tvdata, exchange=exch, interval=getattr(Interval, f"in_{timeframe}"), n_bars=300)
            except Exception as e:
                st.error(f"Error fetching data from tvDatafeed: {e}")
                df = None

            if df is None or df.empty:
                st.error("No data or too few candles. Try different interval / check tvDatafeed credentials.")
            else:
                # some tvDatafeed returns dataframe with columns e.g. ['open','high','low','close','volume']
                df_norm = normalize_df_cols(df)
                sig = generate_signal_from_df(df_norm)
                st.markdown("### Signal Result")
                st.write(f"**Signal:** {sig['signal']}  |  **Confidence:** {sig['confidence']}%")
                st.write(f"Entry ~ {float(df_norm['close'].iloc[-1]):.5f}")
                st.write(f"Stop Loss (price): {sig['sl']}  |  Take Profit (price): {sig['tp']}  |  R:R: {sig['rr']}:1")
                st.write("Reasons:")
                for r in sig["reasons"]:
                    st.write("- " + r)

                # risk & lot calc (assuming pip value simplified)
                # pip_value placeholder: for Forex 1 pip=10 USD per lot (for simplicity)
                pip_value = 10.0
                # compute stop_loss in pips approx (for Forex small pair: difference * 10000)
                try:
                    last_price = float(df_norm['close'].iloc[-1])
                    sl_in_price = abs(last_price - sig['sl'])
                    # approximate pips: if last_price > 10 (like indices), scale different; keep a simple conversion:
                    if last_price >= 100:  # indices/crypto large price
                        pip_est = sl_in_price  # use price units as 'pips' for simplicity
                    else:
                        pip_est = sl_in_price * 10000
                except Exception:
                    pip_est = 1.0

                risk_amount, lot_size = (account_balance * (risk_pct/100)), 0
                if pip_est > 0:
                    lot_size = round((account_balance * (risk_pct/100)) / (pip_est * pip_value), 4)
                st.write(f"Risk Amount: ${risk_amount:.2f}   |   Suggested Lot Size: {lot_size}")

                # show last few candles
                st.markdown("#### Last candles (tail)")
                st.dataframe(df_norm.tail(8))

st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
