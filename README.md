# 📈 AI-Based Stock Market Prediction Using LSTM

## 📌 Project Overview

This project predicts the next day's stock closing price using a Long Short-Term Memory (LSTM) neural network. It also provides an interactive dashboard built with Streamlit to visualize historical stock prices and AI predictions.

The application downloads live stock data from Yahoo Finance, preprocesses it, trains an LSTM model, and predicts future stock prices.

---

## 🚀 Features

- 📊 Live Stock Data from Yahoo Finance
- 🤖 LSTM-Based Stock Price Prediction
- 📈 Interactive Plotly Charts
- 📉 Historical Stock Analysis
- 📅 Multiple Time Periods
  - 1 Month
  - 3 Months
  - 6 Months
  - 1 Year
  - 2 Years
  - 5 Years
  - Maximum Available Data
- 📋 Recent Stock Data Display
- 🎨 Streamlit Web Dashboard

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
- Scikit-Learn
- Yahoo Finance (yfinance)

---

## 📁 Project Structure

StockPrediction/

├── app.py

├── download_data.py

├── visualization.py

├── preprocess.py

├── train_lstm.py

├── train_gru.py

├── predict.py

├── technical_indicators.py

├── data/

│ ├── AAPL.csv

│ └── scaler.pkl

├── models/

│ ├── final_lstm.keras

│ └── gru_model.keras

├── requirements.txt

└── README.md

---