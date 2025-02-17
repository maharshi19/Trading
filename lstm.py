import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import psycopg2
import keras_tuner as kt

# Database connection details
DB_DETAILS = {
    "dbname": os.getenv("POSTGRES_DB", "ie7945"),
    "user": os.getenv("POSTGRES_USER", "ie7945"),
    "password": os.getenv("POSTGRES_PASSWORD", "AgKpAmRePTUUZ9j"),
    "host": os.getenv("POSTGRES_HOST", "ie7945.postgres.database.azure.com"),
    "port": "5432",
}

# Connect to the database and fetch data
def get_data(query):
    conn = psycopg2.connect(**DB_DETAILS)
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

# Fetch historical data
query = """
SELECT timestamp, price, volume
FROM public.alpaca_historical_trades
ORDER BY timestamp;
"""
data = get_data(query)

# Preprocess the data
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Feature Engineering
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek
data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
data['rolling_mean_price'] = data['price'].rolling(window=10).mean()
data['rolling_std_price'] = data['price'].rolling(window=10).std()
data['price_volume_ratio'] = data['price'] / (data['volume'] + 1e-6)
data['price_lag_1'] = data['price'].shift(1)
data['volume_lag_1'] = data['volume'].shift(1)

# Drop rows with NaN values created by rolling and shifting
data.dropna(inplace=True)

# Scale features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length, :-1])  # Features
        y.append(data[i + seq_length, 0])     # Target (price)
    return np.array(x), np.array(y)

SEQ_LENGTH = 50
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Split into training and validation sets
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

# Hyperparameter Tuning
def build_model(hp):
    model = Sequential([
        LSTM(hp.Int("units_1", min_value=32, max_value=128, step=16),
             return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])),
        Dropout(hp.Float("dropout_1", min_value=0.1, max_value=0.5, step=0.1)),
        LSTM(hp.Int("units_2", min_value=32, max_value=128, step=16)),
        Dropout(hp.Float("dropout_2", min_value=0.1, max_value=0.5, step=0.1)),
        Dense(1)
    ])
    model.compile(
        optimizer=hp.Choice("optimizer", values=["adam", "rmsprop"]),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"])
    return model

# Initialize tuner
tuner = kt.RandomSearch(
    build_model,
    objective="val_mean_absolute_error",
    max_trials=3,
    directory="lstm_tuning",
    project_name="crypto_price_prediction"
)

# Search for best hyperparameters
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Retrieve best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

# Train the model with the best hyperparameters
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

# Save the model
model.save("lstm_crypto_model.h5")

# Load the model for inference
model = tf.keras.models.load_model("lstm_crypto_model.h5")

# Evaluate the model
y_pred = model.predict(X_val)
y_val_rescaled = scaler.inverse_transform(np.hstack((y_val.reshape(-1, 1), np.zeros((len(y_val), scaled_data.shape[1] - 1)))))
y_pred_rescaled = scaler.inverse_transform(np.hstack((y_pred, np.zeros((len(y_pred), scaled_data.shape[1] - 1)))))

# Metrics
mae = mean_absolute_error(y_val_rescaled[:, 0], y_pred_rescaled[:, 0])
rmse = np.sqrt(mean_squared_error(y_val_rescaled[:, 0], y_pred_rescaled[:, 0]))
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Predict on live data from crypto_table
live_query = """
SELECT timestamp, price, volume
FROM public.crypto_table
ORDER BY timestamp DESC
LIMIT 50;
"""
live_data = get_data(live_query)
live_data['timestamp'] = pd.to_datetime(live_data['timestamp'])
live_data.set_index('timestamp', inplace=True)

# Preprocess live data
live_data['hour'] = live_data.index.hour
live_data['day_of_week'] = live_data.index.dayofweek
live_data['is_weekend'] = (live_data['day_of_week'] >= 5).astype(int)
live_data['rolling_mean_price'] = live_data['price'].rolling(window=10).mean()
live_data['rolling_std_price'] = live_data['price'].rolling(window=10).std()
live_data['price_volume_ratio'] = live_data['price'] / (live_data['volume'] + 1e-6)
live_data['price_lag_1'] = live_data['price'].shift(1)
live_data['volume_lag_1'] = live_data['volume'].shift(1)
live_data.dropna(inplace=True)

# Scale and create sequence for live data
scaled_live_data = scaler.transform(live_data)
live_seq = np.expand_dims(scaled_live_data[-SEQ_LENGTH:, :-1], axis=0)

# Predict future price
predicted_price_scaled = model.predict(live_seq)[0][0]
predicted_price = scaler.inverse_transform([[predicted_price_scaled] + [0] * (scaled_data.shape[1] - 1)])[0][0]
print(f"Predicted Price: {predicted_price}")

# Compare with actual price
actual_price = live_data.iloc[-1]['price']
print(f"Actual Price: {actual_price}")
