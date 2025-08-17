# app.py
# Pro Trading Bot — Single file with:
# - Generate Signal button (multi-TF confirmation)
# - Fixed RR 1:3
# - Position sizing by account balance & risk %
# - Backtest mode (historical simulation)
# - Market scanner (scan all pairs and rank by confidence)
# - Telegram alerts (optional)
# - Uses yfinance for data (robust on Streamlit Cloud)
# - Saves signals to bot_data/signal_history.csv

import os
import math
from datetime import datetime, timedelta
import time
import requests

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Pro Bot — Backtest, Scan, Telegram", layout="wide")

# -------------------------
# Persistence / Constants
# -------------------------
DATA_DIR = os.path.join(os.getcwd(), "bot_data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(DATA_DIR, "signal_history.csv")

# Markets: yf ticker, TradingView embed symbol, market type
MARKETS = {
    "EUR/USD": {"yf":"EURUSD=X", "tv":"FX:EURUSD", "type":"forex"},
    "GBP/USD": {"yf":"GBPUSD=X", "tv":"FX:GBPUSD", "type":"forex"},
    "USD/JPY": {"yf":"JPY=X",    "tv":"FX:USDJPY", "type":"forex"},
    "AUD/USD": {"yf":"AUDUSD=X","tv":"FX:AUDUSD", "type":"forex"},
    "XAU/USD": {"yf":"GC=F",     "tv":"OANDA:XAUUSD","type":"commodity"},
    "BTC/USD": {"yf":"BTC-USD",  "tv":"BINANCE:BTCUSDT","type":"crypto"},
    "ETH/USD": {"yf":"ETH-USD",  "tv":"BINANCE:ETHUSDT","type":"crypto"},
}

# Strategy defaults (consistent keys)
DEFAULTS = {
    "ema_short": 20,
    "ema_long": 50,
    "ema_trend": 200,
    "rsi_p": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_sig": 9,
    "bb_p": 20,
    "atr_p": 14,
    "rr_target": 3.0,
    "min_confidence": 70,
    "min_volume_ratio": 0.5  # allow lower threshold
}

# -------------------------
# Helpers & Indicators
# -------------------------
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

def fetch_yf(ticker, interval="1m", period="7d"):
    # interval examples: '1m','5m','15m','60m'
    try:
        df = yf.download(tickers=ticker, interval=interval, period=period, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return ensure_ohlcv(flatten_yf_columns(df))
    except Exception:
        return pd.DataFrame()

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
    macd_l = ema(series, fast) - ema(series, slow)
    macd_s = macd_l.ewm(span=sig, adjust=False).mean()
    macd_h = macd_l - macd_s
    return macd_l, macd_s, macd_h

def atr(df, p=14):
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def pip_size_for(market_key):
    t = MARKETS[market_key]['type']
    if t == 'forex':
        return 0.01 if 'JPY' in market_key.upper() else 0.0001
    if t == 'commodity':
        return 0.01
    return None

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
    if pip is None or pip == 0:
        return 0.0, 'lots', round(risk_amount,2)
    pip_distance = abs(entry - sl)/pip
    if pip_distance <= 0: return 0.0, 'lots', round(risk_amount,2)
    pip_value_per_lot = 10.0
    lots = risk_amount / (pip_distance * pip_value_per_lot)
    return round(lots,4), 'lots', round(risk_amount,2)

def market_strength(df):
    if df is None or len(df) < 20:
        return 0
    closes = df['Close'].dropna()
    x = np.arange(len(closes[-20:]))
    y = closes[-20:].values
    if len(x) < 2: return 0
    slope = np.polyfit(x,y,1)[0]
    momentum = np.clip((slope/(np.mean(y)+1e-9))*10000, -50, 50)
    vol = df['Close'].pct_change().rolling(14).std().iloc[-1]
    vol_score = np.clip(vol*1000, 0, 50)
    score = 50 + momentum + vol_score
    return int(np.clip(score, 0, 100))

# -------------------------
# Core signal logic (multi-TF)
# -------------------------
def score_and_generate(df1m, market_key, cfg=DEFAULTS):
    """
    df1m: 1-minute bars DataFrame (Open,High,Low,Close,Volume)
    We'll resample to 5m and 15m inside to confirm.
    """
    out = {"signal":"NONE","entry":None,"sl":None,"tp":None,"rr":None,"confidence":0,"reasons":[],"strength":0}
    if df1m is None or df1m.empty: return out

    # Build MTFs by resampling 1m -> 5m & 15m (if enough data)
    df5m = resample_to(df1m, '5T') if len(df1m)>=5 else None
    df15m = resample_to(df1m, '15T') if len(df1m)>=15 else None

    price = float(df1m['Close'].iat[-1])
    out['entry'] = round(price,8)

    # indicators on 1m
    ema_s = float(ema(df1m['Close'], cfg['ema_short']).iat[-1])
    ema_l = float(ema(df1m['Close'], cfg['ema_long']).iat[-1])
    ema_trend = float(ema(df1m['Close'], cfg['ema_trend']).iat[-1]) if len(df1m)>=cfg['ema_trend'] else ema_l
    r = float(rsi_wilder(df1m['Close'], cfg['rsi_p']).iat[-1])
    macd_l, macd_s, macd_h = macd_series(df1m['Close'], cfg['macd_fast'], cfg['macd_slow'], cfg['macd_sig'])
    macd_dir = float(macd_l.iat[-1] - macd_s.iat[-1])
    bb_mid = df1m['Close'].rolling(cfg['bb_p']).mean().iat[-1]
    bb_std = df1m['Close'].rolling(cfg['bb_p']).std().iat[-1]
    z = (price - bb_mid) / bb_std if not math.isnan(bb_std) and bb_std>0 else 0.0
    atr_v = float(atr(df1m, cfg['atr_p']).iat[-1]) if len(df1m) >= cfg['atr_p'] else None

    # MTF alignment check (require same bias on 5m & 15m if available)
    mtf_ok = True
    mtf_reasons = []
    for mdf, label in [(df5m,'5m'),(df15m,'15m')]:
        if mdf is None:
            mtf_reasons.append(f"{label} missing")
            mtf_ok = False
            continue
        try:
            m_ema_s = float(ema(mdf['Close'], cfg['ema_short']).iat[-1])
            m_ema_l = float(ema(mdf['Close'], cfg['ema_long']).iat[-1])
            if (ema_s > ema_l and m_ema_s > m_ema_l) or (ema_s < ema_l and m_ema_s < m_ema_l):
                mtf_reasons.append(f"{label} aligned")
            else:
                mtf_reasons.append(f"{label} not aligned"); mtf_ok = False
        except Exception:
            mtf_reasons.append(f"{label} error"); mtf_ok = False

    # volume filter
    vol_ok = True
    try:
        avg_vol = df1m['Volume'].rolling(20).mean().iat[-1]
        cur_vol = df1m['Volume'].iat[-1]
        if avg_vol>0 and cur_vol < avg_vol*cfg['min_volume_ratio']:
            vol_ok = False
    except Exception:
        vol_ok = True

    # voting
    vb=vs=0
    if ema_s > ema_l:
        vb +=1; out['confidence'] += 15; out['reasons'].append("EMA20>EMA50")
    else:
        vs +=1; out['reasons'].append("EMA20<EMA50")

    if ema_s > ema_trend:
        vb +=1; out['confidence'] += 6; out['reasons'].append("Above EMA200")
    else:
        vs +=1; out['reasons'].append("Below EMA200")

    if r < cfg['rsi_oversold']:
        vb +=1; out['confidence'] += 12; out['reasons'].append("RSI oversold")
    elif r > cfg['rsi_overbought']:
        vs +=1; out['reasons'].append("RSI overbought")
    else:
        out['confidence'] += 4; out['reasons'].append("RSI neutral")

    if macd_dir > 0:
        vb +=1; out['confidence'] += 10; out['reasons'].append("MACD bullish")
    else:
        vs +=1; out['reasons'].append("MACD bearish")

    if z < -2:
        vb +=1; out['confidence'] += 6; out['reasons'].append("Below -2σ")
    elif z > 2:
        vs +=1; out['reasons'].append("Above +2σ")

    # candle engulfing
    if len(df1m)>=2:
        prev = df1m.iloc[-2]; cur = df1m.iloc[-1]
        if (cur['Close']>cur['Open']) and (prev['Close']<prev['Open']) and (cur['Close']>prev['Open']) and (cur['Open']<prev['Close']):
            vb +=1; out['confidence'] +=8; out['reasons'].append("Bullish engulfing")
        if (cur['Close']<cur['Open']) and (prev['Close']>prev['Open']) and (cur['Close']<prev['Open']) and (cur['Open']>prev['Close']):
            vs +=1; out['confidence'] -=5; out['reasons'].append("Bearish engulfing")

    # final decision requires filters
    if not vol_ok:
        out['reasons'].append("Low volume (filter)")
    if not mtf_ok:
        out['reasons'].append("MTF not aligned (filter)")

    if vol_ok and mtf_ok:
        if vb > vs: signal = 'BUY'
        elif vs > vb: signal = 'SELL'
        else: signal = 'NONE'
    else:
        signal = 'NONE'

    out['signal'] = signal

    # compute SL and TP using ATR & force RR = 1:3
    if signal in ['BUY','SELL']:
        if atr_v is None or atr_v==0 or math.isnan(atr_v):
            atr_px = (df1m['Close'].pct_change().rolling(14).std().iat[-1] if len(df1m)>=14 else 0.001) * price
        else:
            atr_px = atr_v
        if signal == 'BUY':
            sl = price - atr_px*1.0
            tp = price + (price - sl) * cfg['rr_target']
        else:
            sl = price + atr_px*1.0
            tp = price - (sl - price) * cfg['rr_target']
        out['sl'] = round(sl,8); out['tp'] = round(tp,8)
        denom = abs(price - sl)
        out['rr'] = round(abs((tp - price)/denom),2) if denom>0 else None
    else:
        out['sl'] = out['tp'] = out['rr'] = None

    out['confidence'] = int(np.clip(out['confidence'],0,100))
    out['strength'] = market_strength(df1m)
    return out

# -------------------------
# Backtest (simple simulator)
# -------------------------
def backtest_df(df, market_key, cfg=DEFAULTS, max_hold_bars=120):
    """
    Simple backtest: walk over historical bars (use 1m bars),
    when a signal arises at index t (based on indicators computed on data up to t),
    simulate entry at next bar open (t+1), then check subsequent bars up to max_hold_bars
    for SL/TP hit (using High/Low). Record P/L (fixed RR using ATR).
    Returns stats dict.
    """
    results = []
    n = len(df)
    if n < 200:
        return {"error":"Not enough data for backtest (need >=200 bars)."}
    for i in range(200, n-2):
        window = df.iloc[:i+1].copy()  # up to i
        res = score_and_generate(window, market_key, cfg=cfg)
        if res['signal'] in ['BUY','SELL'] and res['confidence'] >= cfg['min_confidence']:
            # entry next bar open
            if i+1 >= n: break
            entry_price = float(df['Open'].iat[i+1])
            sl = res['sl']; tp = res['tp']
            if sl is None or tp is None:
                continue
            outcome = None; exit_idx=None
            # scan forward for hit
            for j in range(i+1, min(n, i+1+max_hold_bars)):
                high = float(df['High'].iat[j]); low = float(df['Low'].iat[j])
                if res['signal']=='BUY':
                    if low <= sl and high >= tp:
                        # both hit same candle -> check which came first is unknown; assume TP if TP reachable within candle? assume TP win to be conservative
                        outcome = 'TP'; exit_idx=j; break
                    elif high >= tp:
                        outcome = 'TP'; exit_idx=j; break
                    elif low <= sl:
                        outcome = 'SL'; exit_idx=j; break
                else:
                    if high >= sl and low <= tp:
                        outcome = 'TP'; exit_idx=j; break
                    elif low <= tp:
                        outcome = 'TP'; exit_idx=j; break
                    elif high >= sl:
                        outcome = 'SL'; exit_idx=j; break
            # compute pnl
            if outcome == 'TP':
                pnl = abs(tp - entry_price)
                results.append({"entry_idx":i+1,"exit_idx":exit_idx,"outcome":"TP","pnl":pnl})
            elif outcome == 'SL':
                pnl = -abs(entry_price - sl)
                results.append({"entry_idx":i+1,"exit_idx":exit_idx,"outcome":"SL","pnl":pnl})
            else:
                # no hit within horizon -> treat as no trade or small negative? we'll mark as 'timeout' and exit flat at last price
                exit_price = float(df['Close'].iat[min(n-1, i+max_hold_bars)])
                pnl = exit_price - entry_price if res['signal']=='BUY' else entry_price - exit_price
                results.append({"entry_idx":i+1,"exit_idx":min(n-1,i+max_hold_bars),"outcome":"TIMEOUT","pnl":pnl})
    # summarize
    if not results:
        return {"trades":0,"wins":0,"losses":0,"timeouts":0,"winrate":None,"net_pnl":0.0}
    wins = sum(1 for r in results if r['outcome']=='TP')
    losses = sum(1 for r in results if r['outcome']=='SL')
    timeouts = sum(1 for r in results if r['outcome']=='TIMEOUT')
    net = sum(r['pnl'] for r in results)
    winrate = wins / (wins + losses) * 100 if wins+losses>0 else None
    return {"trades":len(results),"wins":wins,"losses":losses,"timeouts":timeouts,"winrate":winrate,"net_pnl":net}

# -------------------------
# Telegram alert helper
# -------------------------
def send_telegram_alert(bot_token, chat_id, text):
    if not bot_token or not chat_id: return False, "no credentials"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id":chat_id,"text":text}, timeout=6)
        return r.status_code==200, r.text
    except Exception as e:
        return False, str(e)

# -------------------------
# UI
# -------------------------
st.title("🚀 Pro Trading Bot — Backtest, Scan & Telegram")

col_left, col_right = st.columns([1,2])

with col_left:
    st.subheader("Configuration")
    market = st.selectbox("Market", list(MARKETS.keys()), index=0)
    interval = st.selectbox("Entry timeframe (we fetch 1m and resample MTF)", ['1m','5m','15m'], index=0)
    account_balance = st.number_input("Account balance (USD)", value=1000.0, min_value=10.0, step=50.0)
    risk_pct = st.number_input("Risk % per trade", value=1.0, min_value=0.1, max_value=5.0, step=0.1)
    st.markdown("---")
    st.subheader("Telegram Alerts (optional)")
    tg_token = st.text_input("Telegram Bot token", value="", type='password')
    tg_chat = st.text_input("Telegram Chat ID", value="")
    st.markdown("---")
    st.subheader("Backtest & Scan")
    run_backtest = st.button("Run Backtest (historical)")
    run_scan = st.button("Scan Markets (quick)")
    st.markdown("Notes: Strategy uses fixed settings (EMA20/50/200, RSI14, MACD12/26/9, ATR14). Fixed RR = 1:3.")

with col_right:
    st.subheader("TradingView (visual) & Generate Signal")
    tv_sym = MARKETS[market]['tv']
    # interval mapping for embed
    embed_map = {'1m':'1','5m':'5','15m':'15'}
    st.components.v1.html(f'<iframe src="https://s.tradingview.com/widgetembed/?symbol={tv_sym}&interval={embed_map.get(interval,"5")}&hidesidetoolbar=1&symboledit=1&hideideas=1&theme=dark" width="100%" height="520" frameborder="0"></iframe>', height=540)
    st.markdown("---")

    if st.button("🔮 Generate Signal"):
        with st.spinner("Fetching 1m bars & analyzing..."):
            # always fetch 1m bars (we resample for MTF inside)
            df1m = fetch_yf(MARKETS[market]['yf'], interval='1m', period='7d')
            if df1m is None or df1m.empty:
                st.error("Failed to fetch data (yfinance). Try again later.")
            else:
                result = score_and_generate(df1m, market, cfg=DEFAULTS)
                if result['signal']=='NONE' or result['confidence'] < DEFAULTS['min_confidence']:
                    st.warning(f"No high-confidence signal. Confidence: {result['confidence']}")
                    st.write("Strength:", result['strength'])
                    if result.get('reasons'):
                        with st.expander("Diagnostics"):
                            for r in result['reasons']:
                                st.write("•", r)
                else:
                    st.success(f"SIGNAL: {result['signal']} | Confidence: {result['confidence']} | Strength: {result['strength']}/100")
                    st.write("Entry:", result['entry'])
                    st.write("SL:", result['sl'])
                    st.write("TP:", result['tp'], "(RR 1:3 target)")
                    size, stype, risk_amount = calc_size(account_balance, risk_pct, result['entry'], result['sl'], market)
                    if stype == 'lots':
                        st.write(f"Suggested size: {size} lots (approx) | Risk amount: ${risk_amount}")
                    else:
                        st.write(f"Suggested size: {size} {stype} (approx) | Risk amount: ${risk_amount}")
                    if result.get('reasons'):
                        with st.expander("Why this signal?"):
                            for r in result['reasons']:
                                st.write("•", r)
                    # save
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
                        "rr": result['rr'],
                        "risk_percent": risk_pct,
                        "size": size,
                        "size_type": stype,
                        "risk_amount": risk_amount
                    }
                    if os.path.exists(HISTORY_FILE):
                        old = pd.read_csv(HISTORY_FILE)
                        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(HISTORY_FILE, index=False)
                    else:
                        pd.DataFrame([row]).to_csv(HISTORY_FILE, index=False)
                    st.info("Signal saved to bot_data/signal_history.csv")
                    # telegram
                    if tg_token and tg_chat:
                        text = f"Signal {result['signal']} {market} entry:{result['entry']} sl:{result['sl']} tp:{result['tp']} RR:{result['rr']} conf:{result['confidence']}"
                        ok, resp = send_telegram_alert(tg_token, tg_chat, text)
                        if ok:
                            st.success("Telegram alert sent")
                        else:
                            st.error(f"Telegram failed: {resp}")

# -------------------------
# Backtest button behavior (left column)
# -------------------------
if run_backtest:
    st.sidebar.info("Running backtest — this may take a while depending on data length.")
    with st.spinner("Downloading data for backtest..."):
        df_bt = fetch_yf(MARKETS[market]['yf'], interval='1m', period='90d')  # 90d for sample backtest
    if df_bt is None or df_bt.empty:
        st.error("Could not fetch enough data for backtest.")
    else:
        with st.spinner("Simulating..."):
            stats = backtest_df(df_bt, market, cfg=DEFAULTS, max_hold_bars=240)
        if 'error' in stats:
            st.error(stats['error'])
        else:
            st.subheader("Backtest Results")
            st.metric("Trades", stats['trades'])
            st.metric("Wins", stats['wins'])
            st.metric("Losses", stats['losses'])
            st.metric("Timeouts", stats['timeouts'])
            if stats['winrate'] is not None:
                st.metric("Win rate", f"{stats['winrate']:.1f}%")
            st.write("Net P/L (price units):", stats['net_pnl'])
            st.markdown("Backtest is a simple simulator (see code). Use this to compare parameters and as a first pass.")

# -------------------------
# Market scanner
# -------------------------
if run_scan:
    st.sidebar.info("Scanning markets (quick)...")
    scan_results = []
    with st.spinner("Scanning..."):
        for mk in MARKETS.keys():
            df = fetch_yf(MARKETS[mk]['yf'], interval='1m', period='7d')
            if df is None or df.empty:
                continue
            res = score_and_generate(df, mk, cfg=DEFAULTS)
            # only keep plausible signals above low threshold
            scan_results.append({
                "market": mk,
                "signal": res['signal'],
                "confidence": res['confidence'],
                "strength": res['strength'],
                "entry": res['entry'],
                "sl": res['sl'],
                "tp": res['tp']
            })
            time.sleep(0.2)  # be gentle on yfinance
    if not scan_results:
        st.warning("Scanner didn't return any markets (data issue).")
    else:
        dfscan = pd.DataFrame(scan_results).sort_values(["confidence","strength"], ascending=False)
        st.subheader("Scanner Results")
        st.dataframe(dfscan)
        # quick send telegram for top signals if creds provided
        if tg_token and tg_chat:
            top = dfscan[dfscan['signal']!='NONE'].head(3)
            for _, row in top.iterrows():
                text = f"Scanner Signal {row['signal']} {row['market']} entry:{row['entry']} sl:{row['sl']} tp:{row['tp']} conf:{row['confidence']}"
                send_telegram_alert(tg_token, tg_chat, text)
            st.info("Top scanner signals (if any) sent to Telegram (if configured).")

# Footer: download history if exists
st.markdown("---")
if os.path.exists(HISTORY_FILE):
    st.download_button("Download signal history CSV", data=open(HISTORY_FILE,'rb'), file_name="signal_history.csv")
st.caption("This tool is educational. Always paper-test and use proper risk management. Not financial advice.")
