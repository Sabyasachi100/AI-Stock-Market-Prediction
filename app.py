import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from keras.models import load_model

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="AI Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Market Prediction System")

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
st.sidebar.header("⚙️ Settings")

stock = st.sidebar.selectbox(
    "📈 Select Stock",
    ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
)

# User-friendly time periods
period_options = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "Maximum": "max"
}

selected_period = st.sidebar.selectbox(
    "📅 Select Time Period",
    list(period_options.keys())
)

period = period_options[selected_period]

# ----------------------------------------------------
# DOWNLOAD DATA
# ----------------------------------------------------
with st.spinner("Downloading stock data..."):
    df = yf.download(stock, period=period, auto_adjust=False)

# Fix for latest yfinance MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# ----------------------------------------------------
# METRICS
# ----------------------------------------------------
current_price = float(df["Close"].iloc[-1])
highest_price = float(df["High"].max())
lowest_price = float(df["Low"].min())

col1, col2, col3 = st.columns(3)

col1.metric("💲 Current Price", f"${current_price:.2f}")
col2.metric("📈 Highest Price", f"${highest_price:.2f}")
col3.metric("📉 Lowest Price", f"${lowest_price:.2f}")

# ----------------------------------------------------
# STOCK PRICE CHART
# ----------------------------------------------------
st.subheader(f"📊 {stock} Closing Price")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Close Price",
        line=dict(width=2)
    )
)

fig.update_layout(
    template="plotly_white",
    height=500,
    xaxis_title="Date",
    yaxis_title="Price ($)"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# RECENT DATA
# ----------------------------------------------------
st.subheader("📋 Recent Stock Data")

st.dataframe(df.tail())

# ----------------------------------------------------
# AI PREDICTION
# ----------------------------------------------------
st.subheader("🤖 AI Prediction")

try:

    model = load_model("models/final_lstm.keras")
    scaler = joblib.load("data/scaler.pkl")

    close_prices = df["Close"].values.reshape(-1, 1)

    scaled_data = scaler.transform(close_prices)

    if len(scaled_data) >= 60:

        last_60 = scaled_data[-60:]

        X_test = np.array([last_60])

        prediction = model.predict(X_test, verbose=0)

        predicted_price = scaler.inverse_transform(prediction)

        st.success(
            f"📈 Predicted Next Closing Price: ${predicted_price[0][0]:.2f}"
        )

    else:

        st.warning(
            "⚠️ At least 60 days of data are required for prediction."
        )

except Exception as e:

    st.warning("⚠️ LSTM model or scaler not found.")

    st.error(str(e))

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.markdown("---")

st.markdown(
    """
### 📌 Technologies Used

- Python
- TensorFlow (LSTM)
- Streamlit
- Plotly
- Yahoo Finance
- Scikit-Learn

---
Developed as a **Minor Project** 🚀
"""
)