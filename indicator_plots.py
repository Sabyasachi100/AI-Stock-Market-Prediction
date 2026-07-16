import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/AAPL_with_indicators.csv")

plt.figure(figsize=(14,6))

plt.plot(df["Close"], label="Close")
plt.plot(df["SMA_20"], label="SMA 20")
plt.plot(df["EMA_20"], label="EMA 20")

plt.title("Stock Price with Moving Averages")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14,4))
plt.plot(df["RSI"], label="RSI")
plt.axhline(70, linestyle="--")
plt.axhline(30, linestyle="--")
plt.title("RSI")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14,4))
plt.plot(df["MACD"], label="MACD")
plt.plot(df["MACD_Signal"], label="Signal")
plt.title("MACD")
plt.legend()
plt.grid(True)
plt.show()