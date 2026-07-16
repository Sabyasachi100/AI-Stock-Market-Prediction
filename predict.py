import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

model = load_model("models/final_lstm.keras")

print("Model Loaded Successfully!")


# ==================================================
# LOAD TEST DATA
# ==================================================

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

scaler = joblib.load("data/scaler.pkl")

print("Test Data Loaded Successfully!")


# ==================================================
# MAKE PREDICTIONS
# ==================================================

predictions = model.predict(X_test)


# ==================================================
# CONVERT BACK TO ORIGINAL PRICE SCALE
# ==================================================

predictions = scaler.inverse_transform(predictions)

actual = scaler.inverse_transform(y_test)


# ==================================================
# MODEL EVALUATION
# ==================================================

mae = mean_absolute_error(
    actual,
    predictions
)

mse = mean_squared_error(
    actual,
    predictions
)

rmse = np.sqrt(mse)


mape = np.mean(
    np.abs((actual - predictions) / actual)
) * 100


r2 = r2_score(
    actual,
    predictions
)


# ==================================================
# DIRECTIONAL ACCURACY
# ==================================================

actual_direction = np.diff(
    actual.flatten()
) > 0


predicted_direction = np.diff(
    predictions.flatten()
) > 0


direction_accuracy = np.mean(
    actual_direction == predicted_direction
) * 100



# ==================================================
# PRINT PERFORMANCE
# ==================================================

print("\n===================================")
print("       MODEL PERFORMANCE")
print("===================================")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")
print(f"Direction Accuracy : {direction_accuracy:.2f}%")



# ==================================================
# PLOT ACTUAL VS PREDICTED
# ==================================================

plt.figure(figsize=(14,6))


plt.plot(
    actual,
    label="Actual Price",
    linewidth=2
)


plt.plot(
    predictions,
    label="Predicted Price",
    linewidth=2
)


plt.title(
    "LSTM Stock Price Prediction\nActual vs Predicted"
)


plt.xlabel(
    "Days"
)


plt.ylabel(
    "Stock Price"
)


plt.legend()


plt.grid(True)


plt.show()



# ==================================================
# SAVE PREDICTIONS CSV
# ==================================================

results = pd.DataFrame({

    "Day": range(
        len(actual.flatten())
    ),

    "Actual_Price": actual.flatten(),

    "Predicted_Price": predictions.flatten(),

    "Error":
    (
        actual.flatten()
        -
        predictions.flatten()
    )

})


results.to_csv(
    "predictions.csv",
    index=False
)


print("\nPrediction file saved as predictions.csv")