import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from urllib.parse import quote

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Fenil Pro Trading Bot", layout="wide")

# --------------------------
# Market List (Forex, Crypto, Commodities, Indian Indices)
# --------------------------
MARKET_SYMBOLS = {
    "Forex": {
        "EUR/USD": "EURUSD=X",
        "GBP/JPY": "GBPJPY=X",
        "USD/JPY": "JPY=X",
        "AUD/USD": "AUDUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/CHF": "CHF=X",
        "USD/CAD": "CAD=X"
    },
    "Crypto": {
        "Bitcoin (BTC/USD)": "BTC-USD",
        "Ethereum (ETH/USD)": "ETH-USD",
        "Litecoin (LTC/USD)": "LTC-USD",
        "BNB (BNB/USD)": "BNB-USD"
    },
    "Commodities": {
        "Gold (GC)": "GC=F",
        "Silver (SI)": "SI=F",
        "Crude Oil (CL)": "CL=F",
        "Natural Gas (NG)": "NG=F"
    },
    "Indian Indices": {
        "NIFTY 50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN"
    }
}

# --------------------------
# Helpers
# --------------------------
def map_interval_to_period(interval: str) -> str:
    return {
        "1m": "1d",
        "5m": "5d",
        "15m": "1mo",
        "30m": "1mo",
        "1h": "3mo",
        "1d": "6mo",
    }.get(interval, "1mo")

def map_interval_to_tv(interval: str) -> str:
    return {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "1d": "D",
    }.get(interval, "15")  # fallback 15

def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance output to columns: date, open, high, low, close, volume.
    Works for single-level and multi-level columns.
    """
    if df.empty:
        return df

    # Flatten MultiIndex if any
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c != ""]).lower() for col in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    # Helper to find first column that starts with a key
    def pick(starts_with: str):
        for c in df.columns:
            if c.startswith(starts_with):
                return c
        return None

    o_col = pick("open")
    h_col = pick("high")
    l_col = pick("low")
    c_col = pick("close")
    v_col = pick("volume")

    required = [o_col, h_col, l_col, c_col]
    if any(x is None for x in required):
        # Try alternate names if any
        raise ValueError("Unable to locate OHLC columns in downloaded data.")

    out = pd.DataFrame({
        "date": df.index if "date" not in df.columns else df["date"],
        "open": df[o_col].astype("float64"),
        "high": df[h_col].astype("float64"),
        "low": df[l_col].astype("float64"),
        "close": df[c_col].astype("float64"),
    })
    if v_col is not None:
        out["volume"] = pd.to_numeric(df[v_col], errors="coerce")
    else:
        out["volume"] = np.nan

    out = out.reset_index(drop=True)
    out.dropna(subset=["open", "high", "low", "close"], inplace=True)
    return out

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    period = map_interval_to_period(interval)
    try:
        raw = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = normalize_ohlc(raw)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol} ({interval}): {e}")
        return pd.DataFrame()

# --------------------------
# Strategy Logic (unchanged core; RSI-based)
# --------------------------
def generate_signal(df: pd.DataFrame):
    if df is None or df.empty or len(df) < 20:
        return "No Data", np.nan

    df = df.copy()
    df["change"] = df["close"].pct_change()
    df["gain"] = np.where(df["change"] > 0, df["change"], 0.0)
    df["loss"] = np.where(df["change"] < 0, -df["change"], 0.0)
    avg_gain = df["gain"].rolling(14).mean()
    avg_loss = df["loss"].rolling(14).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    last_rsi = float(df["rsi"].iloc[-1])

    if last_rsi >= 70:
        return "SELL", last_rsi
    elif last_rsi <= 30:
        return "BUY", last_rsi
    else:
        return "HOLD", last_rsi

# --------------------------
# Sidebar
# --------------------------
st.sidebar.header("Market Selection")
market_type = st.sidebar.selectbox("Market Category", list(MARKET_SYMBOLS.keys()))
symbol_name = st.sidebar.selectbox("Select Market", list(MARKET_SYMBOLS[market_type].keys()))
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d"])
symbol = MARKET_SYMBOLS[market_type][symbol_name]

# --------------------------
# Fetch Data & Signal
# --------------------------
df = get_data(symbol, interval)
signal, rsi_value = generate_signal(df)

# --------------------------
# Live TradingView Chart
# --------------------------
st.subheader(f"📈 Live Chart — {symbol_name}")
tv_symbol = quote(symbol)  # handle ^ and = characters
tv_int = map_interval_to_tv(interval)

st.markdown(
    f"""
<iframe src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval={tv_int}&hidesidetoolbar=1&symboledit=1&saveimage=0&toolbarbg=f1f3f6&theme=dark&style=1&timezone=Etc%2FUTC&hideideas=1"
width="100%" height="520" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
""",
    unsafe_allow_html=True
)

# --------------------------
# Output
# --------------------------
st.subheader("Trading Signal")
if np.isnan(rsi_value):
    st.info("Not enough candles to compute RSI. Try a longer period (e.g., change interval).")
else:
    st.write(f"**Signal:** {signal}")
    st.write(f"**RSI:** {rsi_value:.2f}")

st.subheader("Recent Market Data")
if df.empty:
    st.warning("No data returned for this symbol/interval. Try a different interval.")
else:
    st.dataframe(df.tail(30), use_container_width=True)
