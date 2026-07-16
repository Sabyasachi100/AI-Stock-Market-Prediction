import numpy as np
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ----------------------------
# Load Model
# ----------------------------

model = load_model("models/final_lstm.keras")

# ----------------------------
# Load Data
# ----------------------------

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

scaler = joblib.load("data/scaler.pkl")

# ----------------------------
# Predict
# ----------------------------

predictions = model.predict(X_test)

# Convert back to original prices
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test)

# ----------------------------
# Evaluation
# ----------------------------

mae = mean_absolute_error(actual, predictions)
mse = mean_squared_error(actual, predictions)
rmse = np.sqrt(mse)

mape = np.mean(np.abs((actual - predictions) / actual)) * 100

r2 = r2_score(actual, predictions)

print("\n========== MODEL PERFORMANCE ==========")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")

# ----------------------------
# Plot
# ----------------------------

plt.figure(figsize=(14,6))

plt.plot(actual, label="Actual Price")
plt.plot(predictions, label="Predicted Price")

plt.title("Actual vs Predicted Stock Price")

plt.xlabel("Days")
plt.ylabel("Price")

plt.legend()

plt.grid(True)

plt.show()