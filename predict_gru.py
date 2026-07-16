import numpy as np
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

model = load_model("models/gru_model.keras")

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

scaler = joblib.load("data/scaler.pkl")

prediction = model.predict(X_test)

prediction = scaler.inverse_transform(prediction)
actual = scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(actual, prediction))
mae = mean_absolute_error(actual, prediction)

print("RMSE:", rmse)
print("MAE:", mae)

plt.figure(figsize=(12,6))
plt.plot(actual, label="Actual")
plt.plot(prediction, label="GRU Prediction")
plt.legend()
plt.show()