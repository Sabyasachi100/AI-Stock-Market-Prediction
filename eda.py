import pandas as pd

df = pd.read_csv("data/AAPL.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nColumns")
print(df.columns.tolist())

print("\nShape")
print(df.shape)

print("\nInfo")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistics")
print(df.describe())