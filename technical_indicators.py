import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Load data
df = pd.read_csv("data/AAPL.csv")

# Moving Averages
df["SMA_20"] = SMAIndicator(df["Close"], window=20).sma_indicator()
df["EMA_20"] = EMAIndicator(df["Close"], window=20).ema_indicator()

# RSI
df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()

# MACD
macd = MACD(df["Close"])
df["MACD"] = macd.macd()
df["MACD_Signal"] = macd.macd_signal()

# Bollinger Bands
bb = BollingerBands(df["Close"])
df["BB_High"] = bb.bollinger_hband()
df["BB_Low"] = bb.bollinger_lband()

# Save
df.to_csv("data/AAPL_with_indicators.csv", index=False)

print(df.tail())
print("\nTechnical Indicators Added Successfully!")