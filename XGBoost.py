import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib


import matplotlib.pyplot as plt

# Database connection details
DB_DETAILS = {
    "dbname": "ie7945",
    "user": "ie7945",
    "password": "AgKpAmRePTUUZ9j",
    "host": "ie7945.postgres.database.azure.com",
    "port": "5432",
}

# Create SQLAlchemy engine
engine = create_engine(
    f"postgresql+psycopg2://{DB_DETAILS['user']}:{DB_DETAILS['password']}@{DB_DETAILS['host']}:{DB_DETAILS['port']}/{DB_DETAILS['dbname']}"
)

# Fetch historical data for training
query = """
SELECT price, volume, timestamp, symbol, 
       EXTRACT(MINUTE FROM timestamp) AS minute,
       EXTRACT(MONTH FROM timestamp) AS month,
       (price - LAG(price) OVER (PARTITION BY symbol ORDER BY timestamp)) AS price_change
FROM public.alpaca_historical_trades
WHERE price IS NOT NULL AND volume IS NOT NULL
LIMIT 50000;
"""
data = pd.read_sql_query(query, engine)

# Data preprocessing
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['price_change'] = data['price_change'].fillna(0)  # Fill NaN with 0 for price change

# Feature engineering
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['rolling_mean_5'] = data['price'].rolling(5).mean().fillna(data['price'].mean())
data['rolling_std_5'] = data['price'].rolling(5).std().fillna(data['price'].std())

# Define features and target variable
X = data[['volume', 'hour', 'day_of_week', 'minute', 'month', 'price_change', 'rolling_mean_5', 'rolling_std_5']]
y = data['price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define and train the XGBoost model
xgb_model = xgb.XGBRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=42
)
xgb_model.fit(X_train, y_train)

# Evaluate on the test set
y_pred = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Test Set MAE: {mae}, RMSE: {rmse}, MAPE: {mape * 100}%")

# Fetch and preprocess live data
def get_live_data():
    query = """
    SELECT price, volume, timestamp, symbol, 
           EXTRACT(MINUTE FROM timestamp) AS minute,
           EXTRACT(MONTH FROM timestamp) AS month,
           (price - LAG(price) OVER (PARTITION BY symbol ORDER BY timestamp)) AS price_change
    FROM public.crypto_table
    WHERE price IS NOT NULL AND volume IS NOT NULL
    ORDER BY "timestamp" DESC
    LIMIT 10000;
    """
    live_data = pd.read_sql_query(query, engine)
    live_data['timestamp'] = pd.to_datetime(live_data['timestamp'])
    live_data['price_change'] = live_data['price_change'].fillna(0)
    live_data['hour'] = live_data['timestamp'].dt.hour
    live_data['day_of_week'] = live_data['timestamp'].dt.dayofweek
    live_data['rolling_mean_5'] = live_data['price'].rolling(5).mean().fillna(live_data['price'].mean())
    live_data['rolling_std_5'] = live_data['price'].rolling(5).std().fillna(live_data['price'].std())
    
    return live_data

live_data = get_live_data()
X_live = live_data[['volume', 'hour', 'day_of_week', 'minute', 'month', 'price_change', 'rolling_mean_5', 'rolling_std_5']]
X_live_scaled = scaler.transform(X_live)
y_live = live_data['price']

# Predict on live data
y_live_pred = xgb_model.predict(X_live)
mae_live = mean_absolute_error(y_live, y_live_pred)
rmse_live = np.sqrt(mean_squared_error(y_live, y_live_pred))
mape_live = mean_absolute_percentage_error(y_live, y_live_pred)
print(f"Live Data MAE: {mae_live}, RMSE: {rmse_live}, MAPE: {mape_live * 100}%")

# Print predicted vs actual for live data
for pred, actual in zip(y_live_pred, y_live):
    print(f"Predicted: {pred:.2f}, Actual: {actual:.2f}")

# Test set metrics summary
print("\n--- Test Set Metrics ---")
print(f"Test Set MAE: {mae:.2f}")
print(f"Test Set RMSE: {rmse:.2f}")
print(f"Test Set MAPE: {mape * 100:.2f}%")

# Live data metrics summary
print("\n--- Live Data Metrics ---")
print(f"Live Data MAE: {mae_live:.2f}")
print(f"Live Data RMSE: {rmse_live:.2f}")
print(f"Live Data MAPE: {mape_live * 100:.2f}%")

# Calculate accuracy as a percentage
def calculate_accuracy(y_true, y_pred):
    accuracy = np.mean(1 - (np.abs(y_true - y_pred) / y_true)) * 100
    return accuracy

test_accuracy = calculate_accuracy(y_test, y_pred)
live_accuracy = calculate_accuracy(y_live, y_live_pred)

print("\n--- Accuracy ---")
print(f"Test Set Accuracy: {test_accuracy:.2f}%")
print(f"Live Data Accuracy: {live_accuracy:.2f}%")
# Save the model to a file
joblib.dump(xgb_model, 'xgb_model_final.pkl')

