# app.py — Next-Level Trading Bot (single file)
# - Generate Signal button triggers analysis
# - Fixed pro indicator settings (no sliders)
# - TradingView (tvDatafeed) primary, yfinance fallback
# - Risk/Reward fixed to 1:3
# - Position sizing: lots for forex, units for crypto
# - Signal history saved to bot_data/signal_history.csv

import os
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# try tvDatafeed first (TradingView)
try:
    from tvDatafeed import TvDatafeed, Interval
    TV_AVAILABLE = True
    tv = TvDatafeed()
except Exception:
    TV_AVAILABLE = False
    Interval = None
    tv = None

# fallback to yfinance
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

st.set_page_config(page_title="Next-Level Trading Bot (1:3 RR + Sizing)", layout="wide")

# -------------------------
# Data & Persistence paths
# -------------------------
DATA_DIR = os.path.join(os.getcwd(), "bot_data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(DATA_DIR, "signal_history.csv")

# -------------------------
# Market list (TV + yf)
# -------------------------
MARKETS = {
    "EUR/USD": {"tv": ("OANDA", "EURUSD"), "yf": "EURUSD=X", "type": "forex"},
    "GBP/JPY": {"tv": ("OANDA", "GBPJPY"), "yf": "GBPJPY=X", "type": "forex"},
    "USD/JPY": {"tv": ("OANDA", "USDJPY"), "yf": "JPY=X", "type": "forex"},
    "XAU/USD": {"tv": ("OANDA", "XAUUSD"), "yf": "GC=F", "type": "commodity"},
    "BTC/USD": {"tv": ("BINANCE", "BTCUSDT"), "yf": "BTC-USD", "type": "crypto"},
    "ETH/USD": {"tv": ("BINANCE", "ETHUSDT"), "yf": "ETH-USD", "type": "crypto"},
}

# -------------------------
# Utility functions
# -------------------------
def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].copy().dropna()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")]
    return df

def fetch_tv_hist(exchange: str, symbol: str, interval_obj, n_bars: int = 500):
    if not TV_AVAILABLE:
        return None
    try:
        df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval_obj, n_bars=n_bars)
        if df is None or df.empty:
            return None
        # tvDatafeed returns lower-case columns
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        return df[['Open','High','Low','Close','Volume']]
    except Exception:
        return None

def fetch_yf_hist(ticker: str, interval: str = '5m', period: str = '30d'):
    if not YF_AVAILABLE:
        return None
    try:
        df = yf.download(tickers=ticker, interval=interval, period=period, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        df = flatten_yf_columns(df)
        return ensure_ohlcv(df)
    except Exception:
        return None

def fetch_bars(market_key: str, interval_key: str = '5m'):
    """
    Attempts tvDatafeed first (TradingView). If unavailable or fails, falls back to yfinance.
    interval_key: '1m','5m','15m','1h'
    """
    mapping = {'1m': ('1m', '7d', Interval.in_1_minute if Interval else None),
               '5m': ('5m', '30d', Interval.in_5_minute if Interval else None),
               '15m':('15m','60d', Interval.in_15_minute if Interval else None),
               '1h': ('60m','730d', Interval.in_hour if Interval else None)}
    yf_interval, yf_period, tv_interval = mapping[interval_key]
    market = MARKETS[market_key]
    # try tv
    if TV_AVAILABLE and tv_interval is not None:
        tv_exchange, tv_symbol = market['tv']
        df = fetch_tv_hist(tv_exchange, tv_symbol, tv_interval, n_bars=800)
        if df is not None and not df.empty:
            return df
    # fallback yf
    if YF_AVAILABLE:
        df = fetch_yf_hist(market['yf'], interval=yf_interval, period=yf_period)
        if df is not None and not df.empty:
            return df
    return pd.DataFrame()

# -------------------------
# Indicators (fixed defaults)
# -------------------------
def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def rsi_wilder(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100/(1+rs))

def macd_series(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def atr(df: pd.DataFrame, period: int = 14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# -------------------------
# Risk / pip / sizing
# -------------------------
def pip_size_for(market_key: str) -> float:
    t = MARKETS[market_key]['type']
    if t == 'forex':
        # JPY handled by yf ticker naming; we'll approximate by checking key
        if 'JPY' in market_key.upper():
            return 0.01
        return 0.0001
    if t == 'commodity':
        return 0.01
    if t == 'crypto':
        # crypto price movements are larger; treat as price units (we'll compute units)
        return None
    return 0.0001

def calc_size(account_balance: float, risk_percent: float, entry: float, sl: float, market_key: str):
    """
    Returns (size, size_type, risk_amount)
    - For forex: size in standard lots (approx), type='lots'
    - For crypto: size in units (approx), type='units'
    """
    if entry is None or sl is None:
        return 0.0, 'none', 0.0
    risk_amount = account_balance * (risk_percent / 100.0)
    market_type = MARKETS[market_key]['type']
    if market_type == 'crypto':
        # units = risk_amount / (abs(entry - sl))
        price_diff = abs(entry - sl)
        if price_diff <= 0:
            return 0.0, 'units', round(risk_amount,2)
        units = risk_amount / price_diff
        return round(units, 6), 'units', round(risk_amount, 2)
    else:
        pip = pip_size_for(market_key)
        if pip is None or pip == 0:
            return 0.0, 'lots', round(risk_amount,2)
        pip_distance = abs(entry - sl) / pip
        if pip_distance <= 0:
            return 0.0, 'lots', round(risk_amount,2)
        # heuristic pip value per standard lot (USD acc) ~ $10 for majors
        pip_value_per_lot = 10.0
        lots = risk_amount / (pip_distance * pip_value_per_lot)
        return round(lots, 4), 'lots', round(risk_amount, 2)

# -------------------------
# Market strength (0..100)
# -------------------------
def market_strength(df: pd.DataFrame) -> int:
    if df is None or len(df) < 20:
        return 0
    closes = df['Close'].dropna()
    x = np.arange(len(closes[-20:]))
    y = closes[-20:].values
    if len(x) < 2:
        return 0
    slope = np.polyfit(x, y, 1)[0]
    momentum = np.clip((slope / (np.mean(y)+1e-9)) * 10000, -50, 50)
    vol = df['Close'].pct_change().rolling(14).std().iloc[-1]
    vol_score = np.clip(vol * 1000, 0, 50)
    score = 50 + momentum + vol_score
    return int(np.clip(score, 0, 100))

# -------------------------
# Strategy core (fixed settings) + 1:3 RR
# -------------------------
DEFAULTS = dict(
    ema_short=20, ema_long=50,
    rsi_period=14, rsi_overbought=70, rsi_oversold=30,
    macd_fast=12, macd_slow=26, macd_signal=9,
    atr_period=14,
    rr_target=3.0,  # fixed 1:3
)

def generate_signal_from_df(df: pd.DataFrame, market_key: str, cfg=DEFAULTS) -> dict:
    out = {"signal":"NONE","entry":None,"sl":None,"tp":None,"rr":None,"confidence":0,"reasons":[],"strength":0}
    if df is None or df.empty:
        return out

    price = float(df['Close'].iat[-1])
    out['entry'] = round(price, 8)

    # indicators
    ema_s = float(ema(df['Close'], cfg['ema_short']).iat[-1])
    ema_l = float(ema(df['Close'], cfg['ema_long']).iat[-1])
    r = float(rsi_wilder(df['Close'], cfg['rsi_period']).iat[-1])
    macd_line, macd_sig, macd_hist = macd_series(df['Close'], cfg['macd_fast'], cfg['macd_slow'], cfg['macd_signal'])
    macd_dir = float(macd_line.iat[-1] - macd_sig.iat[-1])

    # votes
    vb = 0; vs = 0
    if ema_s > ema_l:
        vb += 1; out['confidence'] += 15; out['reasons'].append("EMA trend up")
    else:
        vs += 1; out['reasons'].append("EMA trend down")

    if r < cfg['rsi_oversold']:
        vb += 1; out['confidence'] += 12; out['reasons'].append("RSI oversold")
    elif r > cfg['rsi_overbought']:
        vs += 1; out['reasons'].append("RSI overbought")
    else:
        out['confidence'] += 4; out['reasons'].append("RSI neutral")

    if macd_dir > 0:
        vb += 1; out['confidence'] += 10; out['reasons'].append("MACD bullish")
    else:
        vs += 1; out['reasons'].append("MACD bearish")

    # simple candle confirmation
    if len(df) >= 2:
        prev = df.iloc[-2]; cur = df.iloc[-1]
        if (cur['Close'] > cur['Open']) and (prev['Close'] < prev['Open']) and (cur['Close'] > prev['Open']) and (cur['Open'] < prev['Close']):
            vb += 1; out['confidence'] += 8; out['reasons'].append("Bullish engulfing")
        if (cur['Close'] < cur['Open']) and (prev['Close'] > prev['Open']) and (cur['Close'] < prev['Open']) and (cur['Open'] > prev['Close']):
            vs += 1; out['confidence'] -= 5; out['reasons'].append("Bearish engulfing")

    out['signal'] = 'BUY' if vb > vs else ('SELL' if vs > vb else 'NONE')

    # ATR for SL distance
    atr_v = float(atr(df, cfg['atr_period']).iat[-1]) if len(df) >= cfg['atr_period'] else None
    if atr_v is None or math.isnan(atr_v) or atr_v == 0:
        recent_std = df['Close'].pct_change().rolling(14).std().iat[-1] if len(df) >= 14 else 0.001
        vol_px = recent_std * price
    else:
        vol_px = atr_v

    if out['signal'] == 'BUY':
        sl = price - vol_px * 1.0  # base SL = 1 ATR
        tp = price + (price - sl) * cfg['rr_target']  # ensure RR = 1:3
    elif out['signal'] == 'SELL':
        sl = price + vol_px * 1.0
        tp = price - (sl - price) * cfg['rr_target']
    else:
        sl = None; tp = None

    out['sl'] = round(sl, 8) if sl is not None else None
    out['tp'] = round(tp, 8) if tp is not None else None
    if out['sl'] and out['tp']:
        denom = abs(price - out['sl'])
        out['rr'] = round(abs((out['tp'] - price) / denom), 2) if denom > 0 else None

    out['confidence'] = int(np.clip(out['confidence'], 0, 100))
    out['strength'] = market_strength(df)
    return out

# -------------------------
# Streamlit UI (no parameter sliders)
# -------------------------
st.title("📈 Next-Level Trading Bot — Fixed 1:3 RR + Auto Sizing")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Trade Settings")
    market_key = st.selectbox("Market", list(MARKETS.keys()), index=0)
    interval_choice = st.selectbox("Interval", ['1m','5m','15m','1h'], index=1)
    account_balance = st.number_input("Account balance (USD)", min_value=10.0, value=1000.0, step=50.0)
    risk_percent = st.number_input("Risk % per trade", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    st.markdown("----")
    st.write("No manual indicator inputs — strategy uses pro defaults (EMA20/50, RSI14, MACD12/26/9, ATR14).")
    st.markdown("Signals generate only when you click **Generate Signal**.")

    st.markdown("----")
    st.subheader("Signal History")
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        st.dataframe(hist.sort_values("timestamp", ascending=False).head(25))
    else:
        st.write("No history yet.")

with col2:
    st.subheader("Chart & Signal")
    # Show interactive TradingView iframe (visual only)
    tv_symbol = "%s:%s" % MARKETS[market_key]['tv']
    tv_interval_map = {'1m':'1','5m':'5','15m':'15','1h':'60'}
    tv_interval = tv_interval_map.get(interval_choice, '5')
    iframe = f"""
        <iframe src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval={tv_interval}&hidesidetoolbar=1&symboledit=1&hideideas=1&theme=dark"
                width="100%" height="520" frameborder="0"></iframe>
    """
    st.components.v1.html(iframe, height=540)

    st.markdown("---")
    if st.button("🔮 Generate Signal"):
        with st.spinner("Fetching data and analyzing..."):
            df = fetch_bars(market_key, interval_choice)
        if df is None or df.empty:
            st.error("Failed to fetch bars. Try again or change the interval/market.")
        else:
            sig = generate_signal_from_df(df, market_key)
            if sig['signal'] == 'NONE':
                st.warning("No clear trade signal right now.")
            else:
                st.success(f"Signal: {sig['signal']}  |  Confidence: {sig['confidence']}  |  Strength: {sig['strength']}/100")
                st.write(f"**Entry:** {sig['entry']}")
                st.write(f"**SL:** {sig['sl']}")
                st.write(f"**TP:** {sig['tp']}  (1:3 RR target)")
                st.write(f"**R:R (calc):** {sig.get('rr')}")

                # sizing
                size, size_type, risk_amount = calc_size(account_balance, risk_percent, sig['entry'], sig['sl'], market_key)
                if size_type == 'lots':
                    st.write(f"**Suggested size:** {size} lots (approx)  |  **Risk amount:** ${risk_amount}")
                elif size_type == 'units':
                    st.write(f"**Suggested size:** {size} units (approx)  |  **Risk amount:** ${risk_amount}")
                else:
                    st.write(f"**Suggested size:** {size} ({size_type})  |  **Risk amount:** ${risk_amount}")

                if sig.get('reasons'):
                    with st.expander("Why this signal?"):
                        for r in sig['reasons']:
                            st.write("•", r)

                # Save history row
                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "market": market_key,
                    "interval": interval_choice,
                    "signal": sig['signal'],
                    "confidence": sig['confidence'],
                    "strength": sig['strength'],
                    "entry": sig['entry'],
                    "sl": sig['sl'],
                    "tp": sig['tp'],
                    "rr": sig.get('rr'),
                    "risk_percent": risk_percent,
                    "size": size,
                    "size_type": size_type,
                    "risk_amount": risk_amount
                }
                if os.path.exists(HISTORY_FILE):
                    old = pd.read_csv(HISTORY_FILE)
                    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(HISTORY_FILE, index=False)
                else:
                    pd.DataFrame([row]).to_csv(HISTORY_FILE, index=False)
                st.info("Signal saved to bot_data/signal_history.csv")

st.caption("This is an educational tool. Always paper-test before using real money.")
