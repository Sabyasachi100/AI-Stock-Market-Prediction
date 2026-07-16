import yfinance as yf
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

ticker = "AAPL"

# Download stock data
df = yf.download(
    ticker,
    start="2015-01-01",
    end="2025-01-01",
    auto_adjust=False
)

# Make Date a normal column
df.reset_index(inplace=True)

# Remove MultiIndex if present
if hasattr(df.columns, "droplevel"):
    try:
        df.columns = df.columns.droplevel(1)
    except:
        pass

# Save CSV
df.to_csv(f"data/{ticker}.csv", index=False)

print(df.head())
print("\nData downloaded successfully!")