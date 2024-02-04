import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Modularization of code
def fetch_data(symbol, start_date, end_date):
    return yf.download(tickers=symbol, start=start_date, end=end_date)

def preprocess_data(data, feature_col='Close', timesteps=60):
    close_prices = data[feature_col].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_prices)
    return create_sequences(scaled_data, timesteps), scaler

def create_sequences(data, timesteps):
    X, y = [], []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    return model

# Fetch and preprocess data
data = fetch_data('BTC-USD', '2013-1-30', '2024-1-30')
(X, y), scaler = preprocess_data(data)

# Split data into train and test sets
split = int(0.9 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build and compile the model
model = build_model((X_train.shape[1], 1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
history = model.fit(X_train, y_train, epochs=25, batch_size=32, callbacks=[early_stopping], validation_split=0.1)

# Make Predictions for the test set and the next day price
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Inverse scaling the actual test data back to original prices
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Predict the next day's price
last_60_days = scaler.transform(data['Close'][-60:].values.reshape(-1, 1))
next_day_price_input = last_60_days.reshape(1, last_60_days.shape[0], 1)
next_day_price = model.predict(next_day_price_input)
next_day_price = scaler.inverse_transform(next_day_price)

# Calculate MAE, RMSE, MAPE
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

# Print MAE, RMSE, MAPE, and Next Day's Price
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Predicted Next Day's Price: {next_day_price[0][0]}")

# Visualization
plt.figure(figsize=(10,6))
plt.plot(actual_prices, color='red', label='Actual Bitcoin Price')
plt.plot(predicted_prices, color='blue', label='Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()
