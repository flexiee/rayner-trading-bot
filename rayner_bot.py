import sys
from datetime import datetime
import pytz

try:
    import streamlit as st
    from streamlit.components.v1 import iframe
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    sys.exit("Please install tvDatafeed: pip install git+https://github.com/rongardF/tvdatafeed.git")

tv = TvDatafeed()

MARKET_SYMBOLS = {
    "EUR/USD": ("OANDA", "EURUSD"),
    "GBP/JPY": ("OANDA", "GBPJPY"),
    "USD/JPY": ("OANDA", "USDJPY"),
    "AUD/USD": ("OANDA", "AUDUSD"),
    "XAU/USD": ("OANDA", "XAUUSD"),
    "BTC/USD": ("BINANCE", "BTCUSDT"),
    "ETH/USD": ("BINANCE", "ETHUSDT"),
    "Gold": ("OANDA", "XAUUSD"),
    "Silver": ("OANDA", "XAGUSD"),
    "Oil WTI": ("OANDA", "WTICOUSD"),
    "NIFTY 50": ("NSE", "NIFTY"),
    "BANKNIFTY": ("NSE", "BANKNIFTY"),
}

def get_live_data(symbol_info):
    exchange, symbol = symbol_info
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_minute, n_bars=20)
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = round(last['close'], 5)
    prev_price = round(prev['close'], 5)
    support = round(df['low'].min(), 5)
    resistance = round(df['high'].max(), 5)
    momentum = "strong" if abs(price - prev_price) > 0.0008 else "weak"
    volatility = round(df['high'].std() * 10000)
    trend = "uptrend" if price > df['close'].rolling(5).mean().iloc[-1] else "downtrend"
    return {
        "price": price,
        "trend": trend,
        "support": support,
        "resistance": resistance,
        "momentum": momentum,
        "volatility": volatility,
        "signal_strength": min(100, max(10, volatility))
    }

def calculate_lot_size(entry_price, stop_loss, account_balance, risk_percent=1):
    pip_diff = abs(entry_price - stop_loss)
    risk_amount = account_balance * (risk_percent / 100)
    pip_value_per_lot = 10  # for standard forex pairs
    if pip_diff == 0:
        return 0
    lot_size = risk_amount / (pip_diff * pip_value_per_lot)
    return round(lot_size, 2), round(risk_amount, 2), round(risk_amount * 3, 2)

def generate_signal(data, account_balance):
    entry = data["price"]
    sl, tp = None, None
    signal = "WAIT"
    reasons = []

    if data["trend"] == "uptrend" and entry > data["support"]:
        if data["momentum"] == "strong" and data["volatility"] > 50:
            sl = entry - 0.0015
            tp = entry + (entry - sl) * 3
            signal = "BUY"
            reasons.append("Strong uptrend breakout")
    elif data["trend"] == "downtrend" and entry < data["resistance"]:
        if data["momentum"] == "strong" and data["volatility"] > 50:
            sl = entry + 0.0015
            tp = entry - (sl - entry) * 3
            signal = "SELL"
            reasons.append("Strong downtrend breakout")

    if sl:
        lot_size, risk_amt, reward_amt = calculate_lot_size(entry, sl, account_balance)
    else:
        lot_size, risk_amt, reward_amt = 0, 0, 0

    return {
        "signal": signal,
        "entry": round(entry, 5),
        "stop_loss": round(sl, 5) if sl else None,
        "take_profit": round(tp, 5) if tp else None,
        "confidence": data["signal_strength"],
        "reasons": reasons,
        "risk_amount": risk_amt,
        "reward_amount": reward_amt,
        "lot_size": lot_size
    }

if STREAMLIT_AVAILABLE:
    def run_ui():
        st.set_page_config(layout="wide", page_title="Trading Bot with Lot Size & Risk Mgmt")
        st.title("📊 Professional Forex Signal Bot")

        if "selected_market" not in st.session_state:
            st.session_state.selected_market = "EUR/USD"

        market = st.selectbox("Select Market", list(MARKET_SYMBOLS.keys()), index=list(MARKET_SYMBOLS.keys()).index(st.session_state.selected_market))
        st.session_state.selected_market = market

        st.markdown("### 📈 Chart View")
        exch, sym = MARKET_SYMBOLS[market]
        iframe(f"https://s.tradingview.com/widgetembed/?symbol={exch}:{sym}&interval=1&theme=dark", height=400)

        account_balance = st.number_input("💰 Account Balance ($)", min_value=10, value=1000)
        st.markdown("---")

        data = get_live_data((exch, sym))
        if data:
            signal = generate_signal(data, account_balance)

            st.subheader("📌 Market Snapshot")
            st.markdown(f"- Trend: **{data['trend']}**")
            st.markdown(f"- Momentum: **{data['momentum']}**")
            st.markdown(f"- Volatility: **{data['volatility']}**")
            st.markdown(f"- Support: **{data['support']}**")
            st.markdown(f"- Resistance: **{data['resistance']}**")

            st.subheader("✅ Signal Result")
            st.markdown(f"- Signal: `{signal['signal']}`")
            st.markdown(f"- Confidence: **{signal['confidence']}%**")
            st.progress(signal['confidence'])

            st.markdown(f"- Entry Price: **{signal['entry']}**")
            st.markdown(f"- Stop Loss: **{signal['stop_loss']}**  |  Take Profit: **{signal['take_profit']}**")
            st.markdown(f"- 💸 Risk: `${signal['risk_amount']}` | 🟢 Reward: `${signal['reward_amount']}`")
            st.markdown(f"- 📦 Recommended Lot Size: **{signal['lot_size']} lot(s)**")

            if signal['reasons']:
                st.markdown(f"**Reason:** {' | '.join(signal['reasons'])}")
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.error("❌ Failed to fetch live data.")

    if __name__ == "__main__":
        run_ui()
else:
    print("Streamlit is not installed. Install with: pip install streamlit")
