import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import yfinance as yf  # For stock/ETF data
import requests  # For forex/commodity data

# =============================
# PURE PYTHON TECHNICAL INDICATORS
# =============================
def calculate_ema(prices, period):
    """Exponential Moving Average (Pure Python)"""
    alpha = 2 / (period + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    return ema

def calculate_rsi(prices, period=14):
    """Relative Strength Index (Pure Python)"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Initialize arrays
    avg_gain = np.zeros(len(prices))
    avg_loss = np.zeros(len(prices))
    rsi = np.zeros(len(prices))
    
    # First values
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    # Smoothing
    for i in range(period+1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
        
    # Calculate RSI
    for i in range(period, len(prices)):
        rs = avg_gain[i] / (avg_loss[i] + 1e-10)  # Avoid division by zero
        rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range (Pure Python)"""
    tr = np.zeros(len(high))
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros(len(high))
    atr[period] = np.mean(tr[1:period+1])
    
    for i in range(period+1, len(high)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
    return atr

# =============================
# TRADING STRATEGIES
# =============================
def enhanced_rr_strategy(data, risk_percent=1.0):
    """
    Enhanced 1:3 Risk-Reward Strategy for all markets
    - Triple confirmation entry (EMA + RSI + Volume)
    - ATR-based dynamic stop loss
    - Fibonacci-based take profit levels
    """
    df = data.copy()
    
    # 1. Calculate indicators
    df['ema_20'] = calculate_ema(df['close'].values, 20)
    df['ema_50'] = calculate_ema(df['close'].values, 50)
    df['rsi'] = calculate_rsi(df['close'].values, 14)
    df['atr'] = calculate_atr(df['high'].values, df['low'].values, df['close'].values, 14)
    
    if 'volume' in df.columns:
        df['volume_ma'] = df['volume'].rolling(20).mean().fillna(0)
        volume_condition = (df['volume'] > df['volume_ma'] * 1.2)
    else:
        volume_condition = True  # Skip volume check if not available
    
    # 2. Entry Logic with Triple Confirmation
    trend_condition = (df['ema_20'] > df['ema_50'])
    rsi_condition = (df['rsi'] > 40) & (df['rsi'] < 70) & (df['rsi'].diff() > 0)
    df['entry_signal'] = trend_condition & rsi_condition & volume_condition
    
    # 3. Dynamic Risk Management
    # Stop loss: 1.5x ATR below recent low
    df['recent_low'] = df['low'].rolling(5).min()
    df['stop_loss'] = df['recent_low'] - (df['atr'] * 1.5)
    
    # Calculate risk distance
    df['risk_distance'] = df['close'] - df['stop_loss']
    
    # 1:3 Risk-Reward Take Profit Levels (Fibonacci-based)
    df['tp_1'] = df['close'] + df['risk_distance'] * 1.618  # 61.8%
    df['tp_2'] = df['close'] + df['risk_distance'] * 2.618  # 161.8%
    df['tp_3'] = df['close'] + df['risk_distance'] * 4.236  # 261.8%
    
    # 4. Position Sizing
    risk_per_trade = risk_percent / 100
    df['position_size'] = risk_per_trade / (df['risk_distance'] / df['close'])
    
    return df

# =============================
# DATA FETCHING FOR ALL MARKETS
# =============================
def fetch_market_data(market_type, symbol, period='1d', interval='1h'):
    """Fetch data for any market type"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        if market_type == "Crypto":
            # Format symbol for crypto (BTC-USD format)
            crypto_symbol = symbol.replace("/", "-")
            df = yf.download(crypto_symbol, start=start_date, end=end_date, interval=interval)
            return df
        
        elif market_type == "Stocks":
            df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            return df
        
        elif market_type == "Forex":
            # Format symbol for forex (EURUSD=X format)
            forex_symbol = symbol.replace("/", "") + "=X"
            df = yf.download(forex_symbol, start=start_date, end=end_date, interval=interval)
            return df
        
        elif market_type == "Commodities":
            # Map commodities to their Yahoo Finance symbols
            commodity_map = {
                "Gold": "GC=F",
                "Silver": "SI=F",
                "Oil": "CL=F",
                "Natural Gas": "NG=F",
                "Copper": "HG=F"
            }
            if symbol in commodity_map:
                df = yf.download(commodity_map[symbol], start=start_date, end=end_date, interval=interval)
                return df
            else:
                st.error(f"Commodity {symbol} not supported")
                return None
        
        elif market_type == "Indices":
            # Map indices to their Yahoo Finance symbols
            index_map = {
                "S&P 500": "^GSPC",
                "NASDAQ": "^IXIC",
                "Dow Jones": "^DJI",
                "FTSE 100": "^FTSE",
                "DAX": "^GDAXI",
                "Nikkei 225": "^N225"
            }
            if symbol in index_map:
                df = yf.download(index_map[symbol], start=start_date, end=end_date, interval=interval)
                return df
            else:
                st.error(f"Index {symbol} not supported")
                return None
                
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# =============================
# TRADINGVIEW CHART EMBED
# =============================
def get_tradingview_chart(symbol, market_type, interval="1H"):
    """Generate TradingView chart embed code for any market"""
    # Map market types to TradingView formats
    if market_type == "Crypto":
        exchange = "BINANCE"
        tv_symbol = f"{exchange}:{symbol.replace('/', '')}"
    elif market_type == "Stocks":
        tv_symbol = symbol
    elif market_type == "Forex":
        tv_symbol = f"FX:{symbol.replace('/', '')}"
    elif market_type == "Commodities":
        tv_symbol = f"TVC:{symbol.upper()}"
    elif market_type == "Indices":
        tv_symbol = f"INDICES:{symbol}"
    else:
        tv_symbol = symbol
    
    # Map intervals
    interval_map = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "4h": "240",
        "1d": "1D",
        "1w": "1W"
    }
    tv_interval = interval_map.get(interval, "60")
    
    return f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart" style="height:600px; width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget(
          {{
            "autosize": true,
            "symbol": "{tv_symbol}",
            "interval": "{tv_interval}",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "hide_top_toolbar": false,
            "hide_legend": false,
            "save_image": false,
            "container_id": "tradingview_chart"
          }}
        );
      </script>
    </div>
    """

# =============================
# STREAMLIT APP
# =============================
def main():
    st.set_page_config(
        page_title="Universal Trading Bot",
        layout="wide",
        page_icon="📊"
    )
    
    st.title("🌎 Universal Trading Bot")
    st.subheader("Trade All Markets with 1:3 Risk-Reward Strategy")
    
    with st.sidebar:
        st.header("⚙️ Trading Configuration")
        
        # Market Type Selection
        market_types = ["Crypto", "Stocks", "Forex", "Commodities", "Indices"]
        market_type = st.selectbox("Market Type", market_types, index=0)
        
        # Symbol Selection based on market type
        symbol_options = {
            "Crypto": ["BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD", "XRP/USD"],
            "Stocks": ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NFLX"],
            "Forex": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"],
            "Commodities": ["Gold", "Silver", "Oil", "Natural Gas", "Copper"],
            "Indices": ["S&P 500", "NASDAQ", "Dow Jones", "FTSE 100", "DAX"]
        }
        symbol = st.selectbox("Trading Symbol", symbol_options[market_type], index=0)
        
        # Strategy Parameters
        st.subheader("📈 Strategy Parameters")
        risk_percent = st.slider("Risk per Trade (%)", 0.1, 10.0, 1.0, 0.1)
        interval = st.selectbox("Chart Interval", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"], index=4)
        
        # Risk Management
        st.subheader("🛡️ Risk Management")
        volatility_filter = st.checkbox("Enable Volatility Filter", True)
        min_volatility = st.slider("Min Volatility (ATR %)", 0.5, 5.0, 1.5, 0.1) if volatility_filter else 0
        
        # Data Refresh
        st.subheader("🔄 System Settings")
        refresh_rate = st.selectbox("Refresh Rate (seconds)", [10, 30, 60, 300], index=2)
    
    # Initialize session state
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now() - timedelta(seconds=refresh_rate+1)
    
    # Auto-refresh logic
    if (datetime.now() - st.session_state.last_refresh).seconds >= refresh_rate:
        st.session_state.last_refresh = datetime.now()
        st.experimental_rerun()
    
    # Fetch market data
    data = fetch_market_data(market_type, symbol, interval=interval)
    
    if data is None or data.empty:
        st.warning("Failed to fetch market data. Please try another symbol or market.")
        return
    
    # Run trading strategy
    results = enhanced_rr_strategy(data, risk_percent)
    last_signal = results.iloc[-1]
    
    # Display TradingView chart
    st.header(f"📊 Live TradingView Chart: {symbol}")
    tradingview_html = get_tradingview_chart(symbol, market_type, interval)
    st.components.v1.html(tradingview_html, height=600)
    
    # Display key metrics
    st.header("📈 Trading Signals & Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${last_signal['close']:.4f}" if market_type != "Forex" else f"${last_signal['close']:.6f}")
    col2.metric("Volatility (ATR)", f"{last_signal['atr']:.4f}", 
                f"{last_signal['atr']/last_signal['close']*100:.2f}%")
    col3.metric("Stop Loss", f"{last_signal['stop_loss']:.4f}", 
                f"-{(last_signal['close']-last_signal['stop_loss'])/last_signal['close']*100:.2f}%")
    col4.metric("Position Size", f"{last_signal['position_size']:.4f} units")
    
    # Strategy visualization
    st.header("🔍 Strategy Analysis")
    
    # Create chart data
    chart_data = results[['close', 'ema_20', 'ema_50']].copy()
    chart_data = chart_data.rename(columns={
        'close': 'Price',
        'ema_20': 'EMA 20',
        'ema_50': 'EMA 50'
    })
    
    # Plot price and EMAs
    st.line_chart(chart_data)
    
    # Visualize entry signals
    st.subheader("🚦 Entry Signals")
    entry_points = results[results['entry_signal'] == True]
    if not entry_points.empty:
        st.write("Recent entry signals:")
        st.dataframe(entry_points[['close', 'rsi']].tail(5).style.format({
            'close': '{:.4f}',
            'rsi': '{:.2f}'
        }))
        
        last_entry = entry_points.iloc[-1]
        st.success(f"✅ Last Entry Signal: {last_entry.name.strftime('%Y-%m-%d %H:%M')} at ${last_entry['close']:.4f}")
    else:
        st.warning("⚠️ No recent entry signals")
    
    # Risk management visualization
    st.subheader("🎯 Risk Management Levels")
    
    # Create a dataframe for risk levels
    risk_data = pd.DataFrame({
        'Level': ['Entry', 'Stop Loss', 'TP 1 (61.8%)', 'TP 2 (161.8%)', 'TP 3 (261.8%)'],
        'Price': [
            last_signal['close'],
            last_signal['stop_loss'],
            last_signal['tp_1'],
            last_signal['tp_2'],
            last_signal['tp_3']
        ]
    })
    
    # Calculate percentage changes
    risk_data['Change'] = (risk_data['Price'] - last_signal['close']) / last_signal['close'] * 100
    
    # Display risk levels
    st.dataframe(risk_data.style.format({
        'Price': '${:.4f}',
        'Change': '{:.2f}%'
    }))
    
    # Visualize as a bar chart
    st.bar_chart(risk_data.set_index('Level')['Price'])
    
    # Strategy details
    with st.expander("📖 Strategy Details & Conditions"):
        st.subheader("Strategy Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Entry Conditions:**")
            st.write(f"- Trend: EMA(20) > EMA(50): {'✅' if last_signal['ema_20'] > last_signal['ema_50'] else '❌'}")
            st.write(f"- Momentum: RSI(14) 40-70 & Rising: {'✅' if 40 < last_signal['rsi'] < 70 and last_signal['rsi'] > results['rsi'].iloc[-2] else '❌'}")
            if 'volume_ma' in last_signal:
                st.write(f"- Volume > 120% of 20D MA: {'✅' if last_signal['volume'] > last_signal['volume_ma'] * 1.2 else '❌'}")
            else:
                st.write("- Volume condition: N/A for this market")
            
            st.write(f"**🚀 Entry Signal:** {'✅ ACTIVE' if last_signal['entry_signal'] else '❌ INACTIVE'}")
            
        with col2:
            st.write("**⚖️ Risk Parameters:**")
            st.write(f"- Risk per Trade: {risk_percent}%")
            st.write(f"- Stop Loss: ${last_signal['stop_loss']:.4f}")
            st.write(f"- Take Profit 1: ${last_signal['tp_1']:.4f} (61.8%)")
            st.write(f"- Take Profit 2: ${last_signal['tp_2']:.4f} (161.8%)")
            st.write(f"- Take Profit 3: ${last_signal['tp_3']:.4f} (261.8%)")
            st.write(f"- Position Size: {last_signal['position_size']:.4f} units")
    
    # Auto-refresh countdown
    refresh_count = refresh_rate - (datetime.now() - st.session_state.last_refresh).seconds
    st.write(f"⏱️ Next refresh in: {refresh_count} seconds")
    time.sleep(1)
    st.experimental_rerun()

if __name__ == "__main__":
    main()
