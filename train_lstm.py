import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ----------------------------
# Load Preprocessed Data
# ----------------------------

X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ----------------------------
# Build LSTM Model
# ----------------------------

model = Sequential()

model.add(LSTM(
    units=100,
    return_sequences=True,
    input_shape=(X_train.shape[1], 1)
))
model.add(Dropout(0.2))

model.add(LSTM(
    units=100,
    return_sequences=False
))
model.add(Dropout(0.2))

model.add(Dense(50))
model.add(Dense(1))

# ----------------------------
# Compile
# ----------------------------

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# ----------------------------
# Callbacks
# ----------------------------

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/lstm_model.keras",
    save_best_only=True
)

# ----------------------------
# Train
# ----------------------------

history = model.fit(

    X_train,
    y_train,

    epochs=50,

    batch_size=32,

    validation_data=(X_test, y_test),

    callbacks=[early_stop, checkpoint]
)

# ----------------------------
# Save Model
# ----------------------------

model.save("models/final_lstm.keras")

print("\nModel Saved Successfully!")

# ----------------------------
# Plot Loss
# ----------------------------

plt.figure(figsize=(10,5))

plt.plot(history.history['loss'], label="Training Loss")

plt.plot(history.history['val_loss'], label="Validation Loss")

plt.legend()

plt.title("LSTM Training")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.show()