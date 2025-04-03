import pandas as pd
import numpy as np
import tensorflow as tf

# Load model and scaler
model = tf.keras.models.load_model('model.h5')
train_mean = np.load('train_mean.npy')
train_std = np.load('train_std.npy')

# Generate 2022 dates
dates_2022 = pd.date_range('2022-01-01', '2022-12-31')
df_2021 = pd.read_csv('../data/daily_receipts.csv', parse_dates=['date'], index_col='date')

# Initialize combined data
combined = df_2021[['receipts']].copy()

for date in dates_2022:
    # Compute features
    prev_day = date - pd.Timedelta(days=1)
    prev_week = date - pd.Timedelta(days=7)

    lag_1 = combined.loc[prev_day, 'receipts'] if prev_day in combined.index else 0
    lag_7 = combined.loc[prev_week, 'receipts'] if prev_week in combined.index else 0
    rolling_7 = combined.loc[prev_week:prev_day, 'receipts'].mean() if not combined.loc[prev_week:prev_day].empty else 0

    features = np.array([
        date.dayofweek,
        date.month,
        date.day,
        lag_1,
        lag_7,
        rolling_7
    ]).reshape(1, -1)

    # Normalize and predict
    features_scaled = (features - train_mean) / train_std
    pred = model.predict(features_scaled)[0][0]
    combined.loc[date] = pred

# Aggregate monthly predictions
monthly_2022 = combined.loc['2022'].resample('M').sum()
monthly_2022.to_csv('2022_predictions.csv')
