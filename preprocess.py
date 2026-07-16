import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("data/AAPL.csv")

print("Columns:", df.columns.tolist())

# ----------------------------
# Keep only Close Price
# ----------------------------
data = df[['Close']]

# ----------------------------
# Scale Data
# ----------------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

print("\nScaled Data Shape:", scaled_data.shape)

# ----------------------------
# Create Sequences
# ----------------------------
sequence_length = 60

X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

print("\nX Shape :", X.shape)
print("Y Shape :", y.shape)

# ----------------------------
# Train Test Split
# ----------------------------
split = int(len(X)*0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("\nTraining Data")
print(X_train.shape)
print(y_train.shape)

print("\nTesting Data")
print(X_test.shape)
print(y_test.shape)

# ----------------------------
# Save Arrays
# ----------------------------
np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)

print("\nPreprocessing Completed Successfully!")

import joblib

joblib.dump(scaler, "data/scaler.pkl")

print("Scaler Saved Successfully!")