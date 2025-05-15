#!pip install meteostat tensorflow scikit-learn matplotlib pandas /Necessary to use with Google CoLab
from datetime import datetime
import numpy as np
import pandas as pd
from meteostat import Point, Daily
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# 1. Fetch data
location = Point(52.2297, 21.0122)
start = datetime(2015, 1, 1)
end   = datetime(2025, 5, 9)
data = Daily(location, start, end).fetch().ffill().bfill()

# 2. Rename & basic features
df = data[['tavg','tmin','tmax','prcp','wspd','pres']].copy()
df.columns = ['temp_avg','temp_min','temp_max','precip','wind_speed','pressure']
df['month']     = df.index.month
df['dayofyear'] = df.index.dayofyear
df['year']      = df.index.year

# 3. Cyclical (sin/cos) time features
df['sin_doy']   = np.sin(2*np.pi * df['dayofyear']/365)
df['cos_doy']   = np.cos(2*np.pi * df['dayofyear']/365)
df['sin_mon']   = np.sin(2*np.pi * df['month']/12)
df['cos_mon']   = np.cos(2*np.pi * df['month']/12)

# 4. Lag & rolling features
for lag in [1, 7, 14]:
    df[f'temp_avg_lag{lag}'] = df['temp_avg'].shift(lag)
df['roll7_mean'] = df['temp_avg'].rolling(7).mean()
df['roll7_std']  = df['temp_avg'].rolling(7).std()
df = df.dropna()

# 5. Separate scalers per weather feature
weather_feats = ['temp_avg','temp_min','temp_max','precip','wind_speed','pressure']
scalers = {}
scaled_parts = []
for feat in weather_feats:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[[feat]])
    scalers[feat] = scaler
    scaled_parts.append(pd.DataFrame(scaled, columns=[feat], index=df.index))

# 6. Scale & combine all features
df_scaled = pd.concat(scaled_parts + [
    df[['sin_doy','cos_doy','sin_mon','cos_mon',
        'temp_avg_lag1','temp_avg_lag7','temp_avg_lag14',
        'roll7_mean','roll7_std']]
], axis=1)

# 7. Build sequences & targets (multi-step = 2-day forecast)
SEQ_LEN = 30
HORIZON = 2
X, y = [], []
arr = df_scaled.values
for i in range(len(arr) - SEQ_LEN - HORIZON + 1):
    X.append(arr[i:i+SEQ_LEN])
    # collect next HORIZON days of temp_avg only
    future = arr[i+SEQ_LEN:i+SEQ_LEN+HORIZON, df_scaled.columns.get_loc('temp_avg')]
    y.append(future.reshape(-1,1))
X = np.array(X)                # (samples, 30, features)
y = np.array(y)                # (samples, 2, 1)

# 8. Time-series cross-validation (rolling)
tscv = TimeSeriesSplit(n_splits=3)
train_idx, test_idx = list(tscv.split(X))[-1]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 9. Define Seq2Seq model
n_timesteps, n_feats = X_train.shape[1], X_train.shape[2]
model = Sequential([
    LSTM(64, input_shape=(n_timesteps, n_feats)),
    Dropout(0.2),
    RepeatVector(HORIZON),
    LSTM(64, return_sequences=True),
    TimeDistributed(Dense(32, activation='relu')),
    TimeDistributed(Dense(1))
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# 10. Callbacks for training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# 11. Fit
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50, batch_size=16,
    callbacks=callbacks
)

# 12. Plot loss
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend(); plt.title('Loss over Epochs'); plt.show()

# 13. Predict & inverse-transform
y_pred_norm = model.predict(X_test)  # shape (samples, 2, 1)
# convert both days back to °C
preds = []
for h in range(HORIZON):
    feat_idx = df_scaled.columns.get_loc('temp_avg')
    col_scaler = scalers['temp_avg']
    norm_vals = y_pred_norm[:,h,0].reshape(-1,1)
    preds.append(col_scaler.inverse_transform(norm_vals).flatten())
preds = np.stack(preds, axis=1)

# 14. Print first 5 forecasts
for i in range(5):
    print(f"Sample {i+1} → Day+1: {preds[i,0]:.2f} °C, Day+2: {preds[i,1]:.2f} °C")