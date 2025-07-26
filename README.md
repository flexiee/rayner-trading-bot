# Rayner Trading Bot

📈 A trading signal generator bot based on Rayner Teo's price action strategies using real-time TradingView data.

## Features

- Live data from TradingView using `tvDatafeed`
- Streamlit web interface (optional)
- BUY/SELL signal logic with confidence levels
- Displays support/resistance, trend, momentum & volatility

## How to Run

```bash
pip install -r requirements.txt
python rayner_bot.py
```

If Streamlit is installed, the bot will launch a web UI. Otherwise, it will run in console mode.

## Dependencies

- streamlit
- tvDatafeed
