import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/AAPL.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

plt.figure(figsize=(12,6))

plt.plot(df["Date"], df["Close"])

plt.title("Apple Stock Price")
plt.xlabel("Date")
plt.ylabel("Closing Price")

plt.grid(True)

plt.show()