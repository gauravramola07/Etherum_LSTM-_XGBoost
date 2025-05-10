import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Set Streamlit page config
st.set_page_config(page_title="Ethereum Price Prediction", layout="wide")

# Sidebar for user inputs
st.sidebar.header("Model Parameters")
start_date = st.sidebar.date_input("Start date", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.today())
window_size = st.sidebar.number_input("Window size (days)", min_value=1, max_value=180, value=60, step=1)
epochs = st.sidebar.number_input("LSTM epochs", min_value=1, max_value=50, value=10, step=1)
train_split = st.sidebar.slider("Train/test split ratio", min_value=0.5, max_value=0.9, value=0.8, step=0.05)

st.sidebar.write("This app uses an LSTM model and an XGBoost model to predict Ethereum prices (ETH-USD).")
st.sidebar.write("Data is fetched from Yahoo Finance.")
st.sidebar.markdown("Model Developed and Created by:<br>Gaurav Ramola<br>MBA AI and Data Science<br>Graphic Era University", unsafe_allow_html=True)
# Main title
st.title("Ethereum Price Prediction: LSTM vs XGBoost")

# Load Ethereum data
@st.cache_data(ttl=86400)
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

data = load_data("ETH-USD", start_date, end_date)
if data is None or data.empty:
    st.error("No data fetched. Check date range.")
    st.stop()

# Use only 'Close' price and reset index
data = data[['Close']].dropna().reset_index()

# Prepare data for modeling
prices = data['Close'].values.reshape(-1,1)

# Scale data to [0,1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Function to create sequences for LSTM
def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Create sequences for LSTM
X, y = create_sequences(scaled_prices, window_size)
split_idx = int(len(X) * train_split)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Reshape for LSTM [samples, timesteps, features]
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define and train LSTM model
with st.spinner("Training LSTM model..."):
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=32, verbose=0)

# LSTM Predictions
pred_lstm_scaled = lstm_model.predict(X_test_lstm)
pred_lstm = scaler.inverse_transform(pred_lstm_scaled)
y_test_lstm = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate LSTM metrics
if len(y_test_lstm) > 0:
    mse_lstm = mean_squared_error(y_test_lstm, pred_lstm)
    mae_lstm = mean_absolute_error(y_test_lstm, pred_lstm)
    rmse_lstm = np.sqrt(mse_lstm)
else:
    mse_lstm = mae_lstm = rmse_lstm = None

# Prepare lagged features for XGBoost
lag_data = pd.DataFrame({'Close': scaled_prices.flatten()})
for i in range(1, window_size+1):
    lag_data[f'lag_{i}'] = lag_data['Close'].shift(i)
lag_data = lag_data.dropna().reset_index(drop=True)

features = [f'lag_{i}' for i in range(1, window_size+1)]
X_xgb = lag_data[features].values
y_xgb = lag_data['Close'].values

split_idx_xgb = int(len(X_xgb) * train_split)
X_train_xgb, X_test_xgb = X_xgb[:split_idx_xgb], X_xgb[split_idx_xgb:]
y_train_xgb, y_test_xgb = y_xgb[:split_idx_xgb], y_xgb[split_idx_xgb:]

# Train XGBoost model if possible
xgb_available = False
if len(X_train_xgb) > 0 and len(X_test_xgb) > 0:
    try:
        with st.spinner("Training XGBoost model..."):
            xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
            xgb_model.fit(X_train_xgb, y_train_xgb)
        xgb_available = True
    except Exception as e:
        st.warning(f"XGBoost training failed: {e}")
        xgb_available = False
else:
    st.warning("Insufficient data for XGBoost model training.")

# XGBoost Predictions
if xgb_available:
    pred_xgb_scaled = xgb_model.predict(X_test_xgb)
    pred_xgb = scaler.inverse_transform(pred_xgb_scaled.reshape(-1, 1))
    y_test_xgb_orig = scaler.inverse_transform(y_test_xgb.reshape(-1, 1))
    if len(y_test_xgb_orig) > 0:
        mse_xgb = mean_squared_error(y_test_xgb_orig, pred_xgb)
        mae_xgb = mean_absolute_error(y_test_xgb_orig, pred_xgb)
        rmse_xgb = np.sqrt(mse_xgb)
    else:
        mse_xgb = mae_xgb = rmse_xgb = None
else:
    pred_xgb = None
    mse_xgb = mae_xgb = rmse_xgb = None

# Plot actual vs predicted for LSTM
st.subheader("Actual vs LSTM Predicted Prices")
test_dates_lstm = data['Date'].iloc[split_idx+window_size:].reset_index(drop=True)
df_plot_lstm = pd.DataFrame({
    'Date': test_dates_lstm,
    'Actual': y_test_lstm.flatten(),
    'LSTM_Predicted': pred_lstm.flatten()
})
df_plot_lstm.set_index('Date', inplace=True)
st.line_chart(df_plot_lstm)

# Plot actual vs predicted for XGBoost if available
if xgb_available:
    st.subheader("Actual vs XGBoost Predicted Prices")
    test_dates_xgb = data['Date'].iloc[split_idx_xgb+window_size:].reset_index(drop=True)
    df_plot_xgb = pd.DataFrame({
        'Date': test_dates_xgb,
        'Actual': y_test_xgb_orig.flatten(),
        'XGB_Predicted': pred_xgb.flatten()
    })
    df_plot_xgb.set_index('Date', inplace=True)
    st.line_chart(df_plot_xgb)

# Display performance metrics
st.subheader("Model Performance Metrics")
col1, col2 = st.columns(2)
with col1:
    st.write("**LSTM Model**")
    if mse_lstm is not None:
        st.metric("MAE (USD)", f"{mae_lstm:.2f}")
        st.metric("MSE (USD)", f"{mse_lstm:.2f}")
        st.metric("RMSE (USD)", f"{rmse_lstm:.2f}")
    else:
        st.write("Insufficient data for LSTM metrics.")
with col2:
    st.write("**XGBoost Model**")
    if xgb_available and mse_xgb is not None:
        st.metric("MAE (USD)", f"{mae_xgb:.2f}")
        st.metric("MSE (USD)", f"{mse_xgb:.2f}")
        st.metric("RMSE (USD)", f"{rmse_xgb:.2f}")
    else:
        st.write("Insufficient data for XGBoost metrics.")

# Next-day prediction
st.subheader("Next-Day Price Prediction")
last_window = scaled_prices[-window_size:]
# LSTM next-day
next_pred_lstm_scaled = lstm_model.predict(last_window.reshape(1, window_size, 1))
next_pred_lstm = scaler.inverse_transform(next_pred_lstm_scaled)[0][0]
col1, col2 = st.columns(2)
col1.metric("LSTM Prediction (next day)", f"${next_pred_lstm:,.2f}")

if xgb_available:
    next_pred_xgb_scaled = xgb_model.predict(last_window.reshape(1, -1))
    next_pred_xgb = scaler.inverse_transform(next_pred_xgb_scaled.reshape(-1, 1))[0][0]
    col2.metric("XGBoost Prediction (next day)", f"${next_pred_xgb:,.2f}")
else:
    col2.write("XGBoost prediction not available.")
