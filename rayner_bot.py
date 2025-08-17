# app.py
# Pro Trading Bot — Single-file Streamlit app
# - Real-time Binance public data (no API required)
# - Multi-timeframe confirmation (1m -> 5m,15m)
# - EMA20/50/200, RSI14, MACD12/26/9, ATR14, Bollinger z-score, Candles
# - Fixed Risk:Reward 1:3 (TP set to ensure RR)
# - Position sizing (account balance + risk%)
# - Backtest (simple simulator), Market Scanner
# - Telegram alerts (optional)
# - Signal history saved to bot_data/signal_history.csv
# - Uses Option A iframe fix for TradingView embed

import os, math, time
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------
# Config & persistence
# ---------------------------
st.set_page_config(page_title="Pro Trading Bot — Real-time (1:3 RR)", layout="wide")
DATA_DIR = os.path.join(os.getcwd(), "bot_data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(DATA_DIR, "signal_history.csv")

# Markets (Binance symbol, friendly label, type)
MARKETS = {
    "BTC/USDT": {"sym": "BTCUSDT", "type": "crypto"},
    "ETH/USDT": {"sym": "ETHUSDT", "type": "crypto"},
    "BNB/USDT": {"sym": "BNBUSDT", "type": "crypto"},
    "SOL/USDT": {"sym": "SOLUSDT", "type": "crypto"},
    "ADA/USDT": {"sym": "ADAUSDT", "type": "crypto"},
}

# Strategy defaults (keys used by code)
DEFAULTS = {
    "ema_short": 20, "ema_long": 50, "ema_trend": 200,
    "rsi_p": 14, "rsi_oversold": 30, "rsi_overbought": 70,
    "macd_fast": 12, "macd_slow": 26, "macd_sig": 9,
    "atr_p": 14, "bb_p": 20,
    "rr_target": 3.0,
    "min_confidence": 70,
    "min_volume_ratio": 0.5
}

# ---------------------------
# Binance public klines fetch
# ---------------------------
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

def fetch_klines_binance(symbol: str, interval: str = "1m", limit: int = 1000) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(BINANCE_KLINES, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "open_time","Open","High","Low","Close","Volume","close_time","q","n","taker_base","taker_quote","ignore"
        ])
        df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
        df["High"] = pd.to_numeric(df["High"], errors="coerce")
        df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df[["open_time","Open","High","Low","Close","Volume"]].rename(columns={"open_time":"Datetime"})
        df = df.set_index("Datetime").dropna()
        return df
    except Exception:
        return pd.DataFrame()

# ---------------------------
# Indicators (pure pandas)
# ---------------------------
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

def macd_series(series: pd.Series, fast=12, slow=26, sig=9):
    macd_line = ema(series, fast) - ema(series, slow)
    macd_sig = macd_line.ewm(span=sig, adjust=False).mean()
    macd_hist = macd_line - macd_sig
    return macd_line, macd_sig, macd_hist

def atr(df: pd.DataFrame, period: int = 14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def resample_to(df: pd.DataFrame, rule: str):
    ohlc = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    return df.resample(rule).agg(ohlc).dropna()

# ---------------------------
# Market utilities: sizing, strength
# ---------------------------
def pip_size_for_market(market_type: str):
    if market_type == "crypto":
        return None
    return 0.0001

def position_size(account_balance: float, risk_pct: float, entry: float, sl: float, market_type: str):
    if entry is None or sl is None:
        return 0.0, "none", 0.0
    risk_amount = account_balance * (risk_pct / 100.0)
    if market_type == "crypto":
        price_diff = abs(entry - sl)
        if price_diff <= 0:
            return 0.0, "units", round(risk_amount, 2)
        units = risk_amount / price_diff
        return round(units, 6), "units", round(risk_amount, 2)
    pip = pip_size_for_market(market_type)
    if pip is None or pip == 0:
        return 0.0, "lots", round(risk_amount, 2)
    pip_distance = abs(entry - sl) / pip
    if pip_distance <= 0:
        return 0.0, "lots", round(risk_amount, 2)
    pip_value_per_lot = 10.0
    lots = risk_amount / (pip_distance * pip_value_per_lot)
    return round(lots, 4), "lots", round(risk_amount, 2)

def market_strength(df: pd.DataFrame):
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

# ---------------------------
# Core scoring + signal generation (Multi-TF)
# ---------------------------
def score_and_generate(df1m: pd.DataFrame, market_label: str, cfg=DEFAULTS):
    out = {"signal":"NONE","entry":None,"sl":None,"tp":None,"rr":None,"confidence":0,"reasons":[],"strength":0}
    if df1m is None or df1m.empty:
        return out

    # build MTFs
    df5m = resample_to(df1m, "5T") if len(df1m) >= 5 else None
    df15m = resample_to(df1m, "15T") if len(df1m) >= 15 else None

    price = float(df1m["Close"].iat[-1])
    out["entry"] = round(price, 8)

    # indicators on 1m
    ema_s = float(ema(df1m["Close"], cfg["ema_short"]).iat[-1])
    ema_l = float(ema(df1m["Close"], cfg["ema_long"]).iat[-1])
    ema_trend = float(ema(df1m["Close"], cfg["ema_trend"]).iat[-1]) if len(df1m) >= cfg["ema_trend"] else ema_l
    r = float(rsi_wilder(df1m["Close"], cfg["rsi_p"]).iat[-1])
    macd_l, macd_s, macd_h = macd_series(df1m["Close"], cfg["macd_fast"], cfg["macd_slow"], cfg["macd_sig"])
    macd_dir = float(macd_l.iat[-1] - macd_s.iat[-1])
    bb_mid = df1m["Close"].rolling(cfg["bb_p"]).mean().iat[-1]
    bb_std = df1m["Close"].rolling(cfg["bb_p"]).std().iat[-1]
    z = (price - bb_mid) / bb_std if (not math.isnan(bb_std) and bb_std > 0) else 0.0
    atr_v = float(atr(df1m, cfg["atr_p"]).iat[-1]) if len(df1m) >= cfg["atr_p"] else None

    # MTF alignment check
    mtf_ok = True
    for mdf, label in [(df5m, "5m"), (df15m, "15m")]:
        if mdf is None or mdf.empty:
            mtf_ok = False
            out["reasons"].append(f"{label} missing")
            continue
        try:
            m_ema_s = float(ema(mdf["Close"], cfg["ema_short"]).iat[-1])
            m_ema_l = float(ema(mdf["Close"], cfg["ema_long"]).iat[-1])
            if (ema_s > ema_l and m_ema_s > m_ema_l) or (ema_s < ema_l and m_ema_s < m_ema_l):
                out["reasons"].append(f"{label} aligned")
            else:
                mtf_ok = False
                out["reasons"].append(f"{label} not aligned")
        except Exception:
            mtf_ok = False
            out["reasons"].append(f"{label} error")

    # volume filter
    vol_ok = True
    try:
        avg_vol = df1m["Volume"].rolling(20).mean().iat[-1]
        cur_vol = df1m["Volume"].iat[-1]
        if avg_vol > 0 and cur_vol < avg_vol * cfg["min_volume_ratio"]:
            vol_ok = False
            out["reasons"].append("Low volume")
    except Exception:
        vol_ok = True

    # voting
    votes_buy = votes_sell = 0
    if ema_s > ema_l:
        votes_buy += 1; out["confidence"] += 15; out["reasons"].append("EMA20>EMA50")
    else:
        votes_sell += 1; out["reasons"].append("EMA20<EMA50")

    if ema_s > ema_trend:
        votes_buy += 1; out["confidence"] += 6; out["reasons"].append("Above EMA200")
    else:
        votes_sell += 1; out["reasons"].append("Below EMA200")

    if r < cfg["rsi_oversold"]:
        votes_buy += 1; out["confidence"] += 12; out["reasons"].append("RSI oversold")
    elif r > cfg["rsi_overbought"]:
        votes_sell += 1; out["reasons"].append("RSI overbought")
    else:
        out["confidence"] += 4; out["reasons"].append("RSI neutral")

    if macd_dir > 0:
        votes_buy += 1; out["confidence"] += 10; out["reasons"].append("MACD bullish")
    else:
        votes_sell += 1; out["reasons"].append("MACD bearish")

    if z < -2:
        votes_buy += 1; out["confidence"] += 6; out["reasons"].append("Below -2σ")
    elif z > 2:
        votes_sell += 1; out["reasons"].append("Above +2σ")

    if len(df1m) >= 2:
        prev = df1m.iloc[-2]; cur = df1m.iloc[-1]
        if (cur["Close"] > cur["Open"]) and (prev["Close"] < prev["Open"]) and (cur["Close"] > prev["Open"]) and (cur["Open"] < prev["Close"]):
            votes_buy += 1; out["confidence"] += 8; out["reasons"].append("Bullish engulfing")
        if (cur["Close"] < cur["Open"]) and (prev["Close"] > prev["Open"]) and (cur["Close"] < prev["Open"]) and (cur["Open"] > prev["Close"]):
            votes_sell += 1; out["confidence"] -= 5; out["reasons"].append("Bearish engulfing")

    # final decision requires filters
    if not vol_ok or not mtf_ok:
        final_signal = "NONE"
    else:
        final_signal = "BUY" if votes_buy > votes_sell else ("SELL" if votes_sell > votes_buy else "NONE")

    out["signal"] = final_signal

    # compute SL/TP as 1 ATR and force RR = 1:3
    if final_signal in ["BUY", "SELL"]:
        if atr_v is None or atr_v == 0 or math.isnan(atr_v):
            atr_px = (df1m["Close"].pct_change().rolling(14).std().iat[-1] if len(df1m) >= 14 else 0.001) * price
        else:
            atr_px = atr_v
        if final_signal == "BUY":
            sl = price - atr_px * 1.0
            tp = price + (price - sl) * cfg["rr_target"]
        else:
            sl = price + atr_px * 1.0
            tp = price - (sl - price) * cfg["rr_target"]
        out["sl"] = round(sl, 8)
        out["tp"] = round(tp, 8)
        denom = abs(price - sl)
        out["rr"] = round(abs((tp - price) / denom), 2) if denom > 0 else None
    else:
        out["sl"] = out["tp"] = out["rr"] = None

    out["confidence"] = int(np.clip(out["confidence"], 0, 100))
    out["strength"] = market_strength(df1m)
    return out

# ---------------------------
# Backtest (simple)
# ---------------------------
def backtest_df(df: pd.DataFrame, market_label: str, cfg=DEFAULTS, max_hold_bars: int = 240):
    results = []
    n = len(df)
    if n < 200:
        return {"error": "Not enough data for backtest (need >=200 bars)."}
    for i in range(200, n - 2):
        window = df.iloc[: i + 1].copy()
        res = score_and_generate(window, market_label, cfg=cfg)
        if res["signal"] in ["BUY", "SELL"] and res["confidence"] >= cfg["min_confidence"]:
            if i + 1 >= n:
                break
            entry_price = float(df["Open"].iat[i + 1])
            sl = res["sl"]; tp = res["tp"]
            if sl is None or tp is None:
                continue
            outcome = None; exit_idx = None
            for j in range(i + 1, min(n, i + 1 + max_hold_bars)):
                high = float(df["High"].iat[j]); low = float(df["Low"].iat[j])
                if res["signal"] == "BUY":
                    if high >= tp:
                        outcome = "TP"; exit_idx = j; break
                    if low <= sl:
                        outcome = "SL"; exit_idx = j; break
                else:
                    if low <= tp:
                        outcome = "TP"; exit_idx = j; break
                    if high >= sl:
                        outcome = "SL"; exit_idx = j; break
            if outcome == "TP":
                pnl = abs(tp - entry_price)
            elif outcome == "SL":
                pnl = -abs(entry_price - sl)
            else:
                exit_price = float(df["Close"].iat[min(n - 1, i + max_hold_bars)])
                pnl = (exit_price - entry_price) if res["signal"] == "BUY" else (entry_price - exit_price)
            results.append({"outcome": outcome or "TIMEOUT", "pnl": pnl})
    if not results:
        return {"trades": 0, "wins": 0, "losses": 0, "timeouts": 0, "winrate": None, "net_pnl": 0.0}
    wins = sum(1 for r in results if r["outcome"] == "TP")
    losses = sum(1 for r in results if r["outcome"] == "SL")
    timeouts = sum(1 for r in results if r["outcome"] == "TIMEOUT")
    net = sum(r["pnl"] for r in results)
    winrate = wins / (wins + losses) * 100 if wins + losses > 0 else None
    return {"trades": len(results), "wins": wins, "losses": losses, "timeouts": timeouts, "winrate": winrate, "net_pnl": net}

# ---------------------------
# Telegram helper
# ---------------------------
def send_telegram(bot_token: str, chat_id: str, text: str):
    if not bot_token or not chat_id:
        return False, "missing credentials"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=8)
        return r.status_code == 200, r.text
    except Exception as e:
        return False, str(e)

# ---------------------------
# Streamlit UI (single file)
# ---------------------------
st.title("🚀 Pro Trading Bot — Live Signals (1:3 RR)")

left, right = st.columns([1, 2])
with left:
    st.subheader("Configuration")
    market_label = st.selectbox("Market", list(MARKETS.keys()), index=0)
    interval_choice = st.selectbox("Interval (entry TF)", ["1m", "5m", "15m"], index=0)
    account_balance = st.number_input("Account balance (USD)", min_value=10.0, value=1000.0, step=50.0)
    risk_pct = st.number_input("Risk % per trade", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    st.markdown("---")
    st.subheader("Telegram Alerts (optional)")
    tg_token = st.text_input("Telegram Bot token", value="", type="password")
    tg_chat = st.text_input("Telegram chat id", value="")
    st.markdown("---")
    st.subheader("History")
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        st.dataframe(hist.sort_values("timestamp", ascending=False).head(25))
    else:
        st.write("No history yet.")

with right:
    st.subheader("TradingView (visual)")
    tv_sym = MARKETS[market_label]["sym"]
    # Option A: compute interval token (no backslashes inside f-string)
    interval_no_m = interval_choice.replace("m", "")
    iframe = f'<iframe src="https://s.tradingview.com/widgetembed/?symbol=BINANCE:{tv_sym}&interval={interval_no_m}&hidesidetoolbar=1&symboledit=1&hideideas=1&theme=dark" width="100%" height="520" frameborder="0"></iframe>'
    st.components.v1.html(iframe, height=540)
    st.markdown("---")

    if st.button("🔮 Generate Signal"):
        with st.spinner("Fetching data and analyzing..."):
            # Always fetch 1m bars and resample for multi-TF
            df1m = fetch_klines_binance(MARKETS[market_label]["sym"], interval="1m", limit=1500)
            if df1m is None or df1m.empty:
                st.error("Failed to fetch market data from Binance public API.")
            else:
                result = score_and_generate(df1m, market_label, cfg=DEFAULTS)
                if result["signal"] == "NONE" or result["confidence"] < DEFAULTS["min_confidence"]:
                    st.warning(f"No high-confidence signal. Confidence: {result['confidence']} | Strength: {result['strength']}")
                    if result.get("reasons"):
                        with st.expander("Diagnostics"):
                            for r in result["reasons"]:
                                st.write("•", r)
                else:
                    st.success(f"SIGNAL: {result['signal']}  |  Confidence: {result['confidence']}  |  Strength: {result['strength']}/100")
                    st.markdown(f"**Entry:** {result['entry']}   \n**SL:** {result['sl']}   |   **TP:** {result['tp']}   |   **R:R:** {result.get('rr')}")
                    size, stype, risk_amt = position_size(account_balance, risk_pct, result["entry"], result["sl"], MARKETS[market_label]["type"])
                    st.write(f"Suggested size: {size} {stype}   |   Risk amount: ${risk_amt}")
                    if result.get("reasons"):
                        with st.expander("Why this signal?"):
                            for r in result["reasons"]:
                                st.write("•", r)
                    # save to history
                    row = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "market": market_label,
                        "interval": interval_choice,
                        "signal": result["signal"],
                        "confidence": result["confidence"],
                        "strength": result["strength"],
                        "entry": result["entry"],
                        "sl": result["sl"],
                        "tp": result["tp"],
                        "rr": result.get("rr"),
                        "risk_pct": risk_pct,
                        "size": size,
                        "size_type": stype,
                        "risk_amount": risk_amt
                    }
                    if os.path.exists(HISTORY_FILE):
                        old = pd.read_csv(HISTORY_FILE)
                        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(HISTORY_FILE, index=False)
                    else:
                        pd.DataFrame([row]).to_csv(HISTORY_FILE, index=False)
                    st.info("Signal saved to bot_data/signal_history.csv")
                    # send telegram
                    if tg_token and tg_chat:
                        text = f"Signal {result['signal']} {market_label}\\nEntry:{result['entry']}\\nSL:{result['sl']}\\nTP:{result['tp']}\\nRR:{result.get('rr')}\\nConf:{result['confidence']}"
                        ok, resp = send_telegram(tg_token, tg_chat, text)
                        if ok:
                            st.success("Telegram alert sent")
                        else:
                            st.error(f"Telegram failed: {resp}")

# ---------------------------
# Backtest & Market Scan
# ---------------------------
st.markdown("---")
st.subheader("Backtest & Market Scanner")
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Run Backtest (90d approx)"):
        st.info("Running backtest (may take 1-3 minutes)...")
        df_bt = fetch_klines_binance(MARKETS[market_label]["sym"], interval="1m", limit=90*24*60)
        if df_bt is None or df_bt.empty:
            st.error("Not enough data for backtest.")
        else:
            stats = backtest_df(df_bt, market_label, cfg=DEFAULTS, max_hold_bars=240)
            if "error" in stats:
                st.error(stats["error"])
            else:
                st.metric("Trades", stats["trades"])
                st.metric("Wins", stats["wins"])
                st.metric("Losses", stats["losses"])
                st.metric("Timeouts", stats["timeouts"])
                st.metric("Win rate", f\"{stats['winrate']:.1f}%\" if stats['winrate'] else "N/A")
                st.write("Net P/L (price units):", stats["net_pnl"])

with col_b:
    if st.button("Scan Markets (quick)"):
        scan = []
        for mk in MARKETS.keys():
            df = fetch_klines_binance(MARKETS[mk]["sym"], interval="1m", limit=500)
            if df is None or df.empty:
                continue
            r = score_and_generate(df, mk, cfg=DEFAULTS)
            scan.append({"market": mk, "signal": r["signal"], "confidence": r["confidence"], "strength": r["strength"], "entry": r["entry"], "sl": r["sl"], "tp": r["tp"]})
            time.sleep(0.2)
        if not scan:
            st.warning("Scanner returned no results.")
        else:
            dfscan = pd.DataFrame(scan).sort_values(["confidence", "strength"], ascending=False)
            st.dataframe(dfscan)
            st.info("Scanner complete. Use Telegram fields to auto-send top signals if desired.")

# Download history CSV
st.markdown("---")
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "rb") as f:
        st.download_button("Download signal history CSV", data=f, file_name="signal_history.csv")
st.caption("Educational tool. Paper-test extensively. Not financial advice.")
