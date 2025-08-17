# app.py — Pro Version Trading Bot (single file)
# - Multi-strategy & multi-timeframe confirmation
# - Fixed RR 1:3, position sizing, yfinance primary (tvDatafeed optional)
# - News avoidance via NewsAPI (optional API key)
# - Generates signal only when you click 'Generate Signal'
# - Saves history to bot_data/signal_history.csv

import os
import math
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
import streamlit as st

# Try tvDatafeed optionally (not required)
try:
    from tvDatafeed import TvDatafeed, Interval
    TV_AVAILABLE = True
    tv = TvDatafeed()
except Exception:
    TV_AVAILABLE = False
    Interval = None
    tv = None

# yfinance fallback (recommended)
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

st.set_page_config(page_title="Pro Trading Bot — MultiTF Confirm (1:3 RR)", layout="wide")
DATA_DIR = os.path.join(os.getcwd(), "bot_data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(DATA_DIR, "signal_history.csv")

# Markets mapping (yfinance + TradingView embed symbol)
MARKETS = {
    "EUR/USD": {"yf": "EURUSD=X", "tv": "OANDA:EURUSD", "type": "forex"},
    "GBP/USD": {"yf": "GBPUSD=X", "tv": "OANDA:GBPUSD", "type": "forex"},
    "USD/JPY": {"yf": "JPY=X", "tv": "OANDA:USDJPY", "type": "forex"},
    "XAU/USD": {"yf": "GC=F", "tv": "OANDA:XAUUSD", "type": "commodity"},
    "BTC/USD": {"yf": "BTC-USD", "tv": "BINANCE:BTCUSDT", "type": "crypto"},
    "ETH/USD": {"yf": "ETH-USD", "tv": "BINANCE:ETHUSDT", "type": "crypto"},
}

# Defaults (fixed pro settings)
DEFAULTS = {
    "ema_short": 20, "ema_long": 50, "ema_trend": 200,
    "rsi_p": 14, "macd_fast": 12, "macd_slow": 26, "macd_sig": 9,
    "bb_p": 20, "atr_p": 14,
    "rr_target": 3.0,
    "min_confidence": 70,   # require at least 70% confidence to accept a signal
    "min_volume_ratio": 1.0 # require current volume >= avg_volume * this
}

# --------------------------- Helpers & Indicators ---------------------------
def flatten_yf_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def ensure_ohlcv(df):
    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[cols].copy().dropna()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")]
    return df

def fetch_yf(ticker, interval="5m", period="30d"):
    if not YF_AVAILABLE:
        return pd.DataFrame()
    df = yf.download(tickers=ticker, interval=interval, period=period, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return ensure_ohlcv(flatten_yf_columns(df))

def fetch_tv_hist(exchange, symbol, interval_obj, n_bars=800):
    if not TV_AVAILABLE:
        return None
    try:
        df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval_obj, n_bars=n_bars)
        if df is None or df.empty:
            return None
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        return df[['Open','High','Low','Close','Volume']]
    except Exception:
        return None

def fetch_bars(market_key, interval_key='5m'):
    mapping = {'1m':('1m','7d', Interval.in_1_minute if Interval else None),
               '5m':('5m','30d', Interval.in_5_minute if Interval else None),
               '15m':('15m','60d', Interval.in_15_minute if Interval else None),
               '1h':('60m','730d', Interval.in_hour if Interval else None)}
    yf_interval, yf_period, tv_interval = mapping[interval_key]
    m = MARKETS[market_key]
    # try tv
    if TV_AVAILABLE and tv_interval is not None:
        df = fetch_tv_hist(m['tv'][0], m['tv'][1], tv_interval, n_bars=800)
        if df is not None and not df.empty:
            return df
    # fallback to yfinance
    df = fetch_yf(m['yf'], interval=yf_interval, period=yf_period)
    return df

def resample_to(df, rule):
    ohlc = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    return df.resample(rule).agg(ohlc).dropna()

def ema(series, p): return series.ewm(span=p, adjust=False).mean()

def rsi_wilder(series, p=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/p, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/p, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100/(1+rs))

def macd_series(series, fast=12, slow=26, sig=9):
    macd_line = ema(series, fast) - ema(series, slow)
    macd_sig = macd_line.ewm(span=sig, adjust=False).mean()
    macd_hist = macd_line - macd_sig
    return macd_line, macd_sig, macd_hist

def atr(df, period=14):
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def pip_size_for(market_key):
    t = MARKETS[market_key]['type']
    if t == 'forex':
        return 0.01 if 'JPY' in market_key.upper() else 0.0001
    if t == 'commodity':
        return 0.01
    return None  # crypto: use absolute units

def calc_size(balance, risk_pct, entry, sl, market_key):
    if entry is None or sl is None:
        return 0.0, 'none', 0.0
    risk_amount = balance * (risk_pct/100.0)
    mtype = MARKETS[market_key]['type']
    if mtype == 'crypto':
        price_diff = abs(entry - sl)
        if price_diff <= 0: return 0.0, 'units', round(risk_amount,2)
        units = risk_amount / price_diff
        return round(units,6), 'units', round(risk_amount,2)
    pip = pip_size_for(market_key)
    if pip is None or pip==0: return 0.0, 'lots', round(risk_amount,2)
    pip_distance = abs(entry - sl) / pip
    if pip_distance <= 0: return 0.0, 'lots', round(risk_amount,2)
    pip_value_per_lot = 10.0
    lots = risk_amount / (pip_distance * pip_value_per_lot)
    return round(lots,4), 'lots', round(risk_amount,2)

# --------------------------- News avoidance (optional) ---------------------------
def check_news(news_api_key, country='us', minutes_forward=30):
    """Optional: uses NewsAPI to check headlines (not FX-specific). Returns True if high-impact news upcoming."""
    if not news_api_key:
        return False
    try:
        # get top headlines for now — this is a heuristic; NewsAPI doesn't label 'impact' so user should use a dedicated economic calendar
        url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=20&apiKey={news_api_key}"
        r = requests.get(url, timeout=6)
        if r.status_code != 200:
            return False
        data = r.json()
        # If any headline contains words like 'inflation', 'CPI', 'Fed', 'rate', 'unemployment' in title -> treat as high impact
        keywords = ['inflation','cpi','fed','rate','unemployment','interest rate','nfp','non farm payroll','cpi']
        for a in data.get('articles', []):
            title = (a.get('title') or '').lower()
            for kw in keywords:
                if kw in title:
                    return True
        return False
    except Exception:
        return False

# --------------------------- Core Signal Logic ---------------------------
def score_and_generate(df1m, df5m, df15m, market_key, cfg=DEFAULTS):
    """Return dict with signal, entry, sl, tp, rr, confidence, reasons, strength"""
    out = {"signal":"NONE","entry":None,"sl":None,"tp":None,"rr":None,"confidence":0,"reasons":[],"strength":0}
    if df1m is None or df1m.empty:
        return out

    price = float(df1m['Close'].iat[-1])
    out['entry'] = round(price, 8)

    # indicators on 1m
    ema_s = float(ema(df1m['Close'], cfg['ema_short']).iat[-1])
    ema_l = float(ema(df1m['Close'], cfg['ema_long']).iat[-1])
    ema_trend = float(ema(df1m['Close'], cfg.get('ema_trend',200)).iat[-1])
    r = float(rsi_wilder(df1m['Close'], cfg['rsi_p']).iat[-1])
    macd_l, macd_s, macd_h = macd_series(df1m['Close'], cfg['macd_fast'], cfg['macd_slow'], cfg['macd_sig'])
    macd_dir = float(macd_l.iat[-1] - macd_s.iat[-1])
    bb_mid = df1m['Close'].rolling(cfg['bb_p']).mean().iat[-1]
    bb_std = df1m['Close'].rolling(cfg['bb_p']).std().iat[-1]
    z = (price - bb_mid)/bb_std if not math.isnan(bb_std) and bb_std>0 else 0.0
    atr_v = float(atr(df1m, cfg['atr_p']).iat[-1]) if len(df1m) >= cfg['atr_p'] else None

    # MTF trend checks: require the MTF trend direction to align with 1m (simple EMA short>long)
    mtf_ok = True
    mtf_reasons = []
    for mdf, label in [(df5m, '5m'), (df15m, '15m')]:
        if mdf is None or mdf.empty:
            mtf_ok = False
            mtf_reasons.append(f"{label} missing")
            continue
        try:
            m_ema_s = float(ema(mdf['Close'], cfg['ema_short']).iat[-1])
            m_ema_l = float(ema(mdf['Close'], cfg['ema_long']).iat[-1])
            if (ema_s > ema_l and m_ema_s > m_ema_l) or (ema_s < ema_l and m_ema_s < m_ema_l):
                mtf_reasons.append(f"{label} aligned")
            else:
                mtf_ok = False
                mtf_reasons.append(f"{label} not aligned")
        except Exception:
            mtf_ok = False
            mtf_reasons.append(f"{label} error")

    # volume filter: current volume vs average (1m)
    vol_ok = True
    try:
        avg_vol = df1m['Volume'].rolling(20).mean().iat[-1]
        cur_vol = df1m['Volume'].iat[-1]
        if avg_vol > 0 and cur_vol < avg_vol * cfg['min_volume_ratio']:
            vol_ok = False
    except Exception:
        vol_ok = True

    # Voting system
    votes_buy = votes_sell = 0

    # EMA trend bias
    if ema_s > ema_l:
        votes_buy += 1; out['confidence'] += 15; out['reasons'].append("EMA20>EMA50")
    else:
        votes_sell += 1; out['reasons'].append("EMA20<EMA50")

    # Longer-term trend confirmation (EMA200)
    if ema_s > ema_trend:
        votes_buy += 1; out['confidence'] += 6; out['reasons'].append("Above EMA200")
    else:
        votes_sell += 1; out['reasons'].append("Below EMA200")

    # RSI
    if r < cfg['rsi_oversold']:
        votes_buy += 1; out['confidence'] += 12; out['reasons'].append("RSI oversold")
    elif r > cfg['rsi_overbought']:
        votes_sell += 1; out['reasons'].append("RSI overbought")
    else:
        out['confidence'] += 4; out['reasons'].append("RSI neutral")

    # MACD
    if macd_dir > 0:
        votes_buy += 1; out['confidence'] += 10; out['reasons'].append("MACD bullish")
    else:
        votes_sell += 1; out['reasons'].append("MACD bearish")

    # Bollinger z
    if z < -2:
        votes_buy += 1; out['confidence'] += 6; out['reasons'].append("Below -2σ")
    elif z > 2:
        votes_sell += 1; out['reasons'].append("Above +2σ")

    # Candle pattern (engulfing)
    if len(df1m) >= 2:
        prev = df1m.iloc[-2]; cur = df1m.iloc[-1]
        if (cur['Close'] > cur['Open']) and (prev['Close'] < prev['Open']) and (cur['Close'] > prev['Open']) and (cur['Open'] < prev['Close']):
            votes_buy += 1; out['confidence'] += 8; out['reasons'].append("Bullish engulfing")
        if (cur['Close'] < cur['Open']) and (prev['Close'] > prev['Open']) and (cur['Close'] < prev['Open']) and (cur['Open'] > prev['Close']):
            votes_sell += 1; out['confidence'] -= 5; out['reasons'].append("Bearish engulfing")

    # Apply filters: volume & MTF
    if not vol_ok:
        out['reasons'].append("Low volume -> skip")
    if not mtf_ok:
        out['reasons'].append("MTF not aligned -> skip")

    # Final decision only if filters ok
    signal = 'NONE'
    if vol_ok and mtf_ok:
        if votes_buy > votes_sell:
            signal = 'BUY'
        elif votes_sell > votes_buy:
            signal = 'SELL'
        else:
            signal = 'NONE'
    else:
        signal = 'NONE'

    out['signal'] = signal

    # If signal, compute SL/TP with ATR and force RR = 1:3
    if signal in ['BUY','SELL']:
        if atr_v is None or atr_v == 0 or math.isnan(atr_v):
            atr_px = (df1m['Close'].pct_change().rolling(14).std().iat[-1] if len(df1m)>=14 else 0.001) * price
        else:
            atr_px = atr_v
        # base SL distance = 1 * ATR
        if signal == 'BUY':
            sl = price - atr_px * 1.0
            tp = price + (price - sl) * cfg['rr_target']  # ensures RR = 1:3
        else:
            sl = price + atr_px * 1.0
            tp = price - (sl - price) * cfg['rr_target']
        out['sl'] = round(sl, 8); out['tp'] = round(tp, 8)
        denom = abs(price - sl)
        out['rr'] = round(abs((tp - price) / denom), 2) if denom>0 else None
    else:
        out['sl']=out['tp']=out['rr']=None

    out['confidence'] = int(np.clip(out['confidence'], 0, 100))
    out['strength'] = market_strength(df1m)
    return out

# --------------------------- Market strength (same as earlier) ---------------------------
def market_strength(df):
    if df is None or len(df) < 20:
        return 0
    closes = df['Close'].dropna()
    x = np.arange(len(closes[-20:]))
    y = closes[-20:].values
    if len(x) < 2:
        return 0
    slope = np.polyfit(x, y, 1)[0]
    momentum_score = np.clip((slope / (np.mean(y)+1e-9)) * 10000, -50, 50)
    vol = df['Close'].pct_change().rolling(14).std().iloc[-1]
    vol_score = np.clip(vol * 1000, 0, 50)
    score = 50 + momentum_score + vol_score
    return int(np.clip(score, 0, 100))

# --------------------------- UI ---------------------------
st.title("🚀 Pro Trading Bot — MultiTF Confirmation (Fixed 1:3 RR)")

left, right = st.columns([1,2])
with left:
    st.subheader("Configuration (minimal)")
    market = st.selectbox("Market", list(MARKETS.keys()), index=0)
    interval = st.selectbox("Interval (entry timeframe)", ['1m','5m','15m'], index=0)
    account_balance = st.number_input("Account balance (USD)", min_value=10.0, value=1000.0, step=50.0)
    risk_percent = st.number_input("Risk % per trade", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    news_api_key = st.text_input("NewsAPI.org API key (optional for news avoidance)", value="", type='password')
    st.caption("NewsAPI is optional. If provided, bot will attempt to avoid obvious economic headlines.")
    st.markdown("---")
    st.write("Notes: fixed strategy settings (EMA20/50/200, RSI14, MACD12/26/9, ATR14). No sliders.")

with right:
    st.subheader("TradingView Chart (visual)")
    tv_symbol = MARKETS[market]['tv'][0] + ':' + MARKETS[market]['tv'][1]
    tv_int_map = {'1m':'1','5m':'5','15m':'15'}
    st.components.v1.html(f'<iframe src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval={tv_int_map[interval]}&hidesidetoolbar=1&symboledit=1&hideideas=1&theme=dark" width="100%" height="520" frameborder="0"></iframe>', height=540)

    st.markdown("---")
    if st.button("🔮 Generate Signal — Pro (MultiTF)"):
        # news avoidance check (optional)
        with st.spinner("Checking news & fetching data..."):
            news_block = False
            if news_api_key:
                try:
                    news_block = check_news(news_api_key, minutes_forward=30)
                except Exception:
                    news_block = False

            if news_block:
                st.warning("High-impact news found — skipping signal generation (news avoidance).")
            else:
                # fetch 1m bars for entry, and create 5m/15m MTF via resample (we fetch 1m to resample reliably)
                base_interval = '1m' if interval == '1m' else '1m'  # always fetch 1m and resample; gives best MTF alignment
                df1m = fetch_bars(market, '1m')
                if df1m is None or df1m.empty:
                    st.error("Failed to fetch data. Try later or check connection.")
                else:
                    # build MTFs (resample)
                    df5m = resample_to(df1m, '5T') if len(df1m) >= 5 else None
                    df15m = resample_to(df1m, '15T') if len(df1m) >= 15 else None

                    result = score_and_generate(df1m, df5m, df15m, market, cfg=DEFAULTS)

                    if result['signal'] == 'NONE' or result['confidence'] < DEFAULTS['min_confidence']:
                        st.warning(f"No high-confidence signal. (Confidence: {result['confidence']})")
                        # still show some diagnostics
                        st.write("Strength:", result['strength'])
                        if result.get('reasons'):
                            with st.expander("Why no trade? (diagnostics)"):
                                for r in result['reasons']:
                                    st.write("•", r)
                    else:
                        st.success(f"✅ SIGNAL: {result['signal']}  |  Confidence: {result['confidence']}  |  Strength: {result['strength']}/100")
                        st.write(f"Entry: {result['entry']}")
                        st.write(f"SL: {result['sl']}  |  TP: {result['tp']}  | R:R: {result.get('rr')}")
                        # sizing
                        size, size_type, risk_amount = calc_size(account_balance, risk_percent, result['entry'], result['sl'], market)
                        if size_type == 'lots':
                            st.write(f"Suggested size: {size} lots (approx)  |  Risk amount: ${risk_amount}")
                        else:
                            st.write(f"Suggested size: {size} {size_type} (approx)  |  Risk amount: ${risk_amount}")
                        if result.get('reasons'):
                            with st.expander("Why this signal?"):
                                for r in result['reasons']:
                                    st.write("•", r)
                        # save history
                        row = {
                            "timestamp": datetime.utcnow().isoformat(),
                            "market": market,
                            "interval": interval,
                            "signal": result['signal'],
                            "confidence": result['confidence'],
                            "strength": result['strength'],
                            "entry": result['entry'],
                            "sl": result['sl'],
                            "tp": result['tp'],
                            "rr": result.get('rr'),
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

st.caption("This is an advanced signal assistant. Always paper-test and use risk management. Not financial advice.")
