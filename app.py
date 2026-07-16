from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os


app = Flask(__name__)


# -------------------------------
# Load Model
# -------------------------------

model = load_model(
    "models/final_lstm.keras"
)


# -------------------------------
# Load Scaler
# -------------------------------

scaler = joblib.load(
    "data/scaler.pkl"
)


# -------------------------------
# Home Page
# -------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )



# -------------------------------
# Prediction Route
# -------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        ticker = request.form["ticker"].upper().strip()


        # Download Stock Data
        data = yf.download(
            ticker,
            period="1y",
            progress=False
        )


        # Check data exists

        if data.empty:

            return """
            <h2>
            Invalid Stock Symbol or Yahoo Finance Error
            </h2>
            <a href="/">Go Back</a>
            """



        # Extract Closing Price

        close = data["Close"]


        # Handle yfinance new format

        if len(close.shape) > 1:

            close = close.iloc[:,0]


        close = close.values.reshape(
            -1,1
        )



        # Need minimum 60 days

        if len(close) < 60:

            return """
            <h2>
            Not enough data for prediction
            </h2>
            """



        # Scale Data

        scaled = scaler.transform(
            close
        )



        # Last 60 days

        x = scaled[-60:]


        x = np.reshape(
            x,
            (1,60,1)
        )



        # Prediction

        prediction = model.predict(
            x
        )


        # Convert back

        predicted_price = scaler.inverse_transform(
            prediction
        )



        result = round(
            float(predicted_price[0][0]),
            2
        )


        return f"""

        <html>

        <body>

        <h1>
        📈 Stock Prediction Result
        </h1>


        <h2>
        Stock: {ticker}
        </h2>


        <h2>
        Predicted Next Price:
        ${result}
        </h2>


        <br>

        <a href="/">
        Predict Another
        </a>


        </body>

        </html>

        """



    except Exception as e:


        return f"""

        <h2>
        Error Occurred:
        </h2>

        <p>
        {str(e)}
        </p>

        <a href="/">
        Go Back
        </a>

        """



# -------------------------------
# Run Server
# -------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )