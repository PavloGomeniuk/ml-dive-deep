#!pip install meteostat tensorflow scikit-learn matplotlib pandas /Necessary to use with Google CoLab
from datetime import datetime
from meteostat import Point, Daily
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Define location: Warsaw
location = Point(52.2297, 21.0122)
# Define date range
start = datetime(2015, 1, 1)
end = datetime(2025, 5, 9)
# Fetch daily data
data = Daily(location, start, end).fetch()

# --- Fill missing values ---
data = data.ffill().bfill()  # Avoid dropna

# --- Rename and add time features ---
df = data[['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']].copy()
df.columns = ['temp_avg', 'temp_min', 'temp_max', 'precip', 'wind_speed', 'pressure']
df['month'] = df.index.month
df['dayofyear'] = df.index.dayofyear
df['year'] = df.index.year

# --- Normalize weather features ---
weather_features = ['temp_avg', 'temp_min', 'temp_max', 'precip', 'wind_speed', 'pressure']
scaler = MinMaxScaler()
df_scaled_weather = pd.DataFrame(
    scaler.fit_transform(df[weather_features]),
    columns=weather_features,
    index=df.index
)

# --- Combine scaled weather with raw time features ---
df_final = pd.concat([df_scaled_weather, df[['month', 'dayofyear', 'year']]], axis=1)

# --- Save to CSV ---
#df_final.to_csv('warsaw_weather_cleaned.csv')

# Use 30 days to predict the next day
sequence_length = 30

# Use only numerical data (not datetime index)
data = df_final.values

X = []
y = []

for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])       # Sequence of 30 days
    y.append(data[i+sequence_length][0])      # Next day's temp_avg (index 0)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)  # (samples, time_steps, features)
print("y shape:", y.shape)  # (samples,)


# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # no shuffle for time-series
)

# Input shape: (timesteps, features)
n_timesteps = X_train.shape[1]
n_features = X_train.shape[2]

model = Sequential([
    LSTM(64, input_shape=(n_timesteps, n_features), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # output: temperature
])

model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training History")
plt.show()

# Predict on test set
y_pred = model.predict(X_test)

# Start from the last 30 days in the dataset
last_sequence = data[-sequence_length:].copy()  # shape (30, features)

predictions = []

for _ in range(2):  # Predict 2 future days
    # Reshape to (1, 30, features) for model input
    input_seq = np.expand_dims(last_sequence, axis=0)

    # Predict the next day's normalized temperature
    next_temp_norm = model.predict(input_seq)[0][0]
    predictions.append(next_temp_norm)

    # Create next day's input row
    next_day_features = np.copy(last_sequence[-1])
    next_day_features[0] = next_temp_norm  # update temp_avg

    # Optional: Keep other features like month/dayofyear fixed or estimated
    # Here, we simply shift dayofyear and keep others same (you can improve this)
    next_day_features[7] += 1  # dayofyear index = 7

    # Append and shift window
    last_sequence = np.vstack([last_sequence[1:], next_day_features])

# Convert to 2D array for inverse transform
predictions_array = np.array(predictions).reshape(-1, 1)

# Only inverse transform temp_avg (index 0 feature)
temp_avg_minmax_scaler = MinMaxScaler()
temp_avg_minmax_scaler.min_, temp_avg_minmax_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
temp_preds_celsius = temp_avg_minmax_scaler.inverse_transform(predictions_array)
# Show predicted temperature
for i, temp in enumerate(temp_preds_celsius, 1):
    print(f"Predicted Day {i}: {temp[0]:.2f} °C")