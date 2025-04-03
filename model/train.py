import pandas as pd
import numpy as np
import tensorflow as tf

# Load data
df = pd.read_csv('../data/daily_receipts.csv', parse_dates=['date'])
df = df.set_index('date').sort_index().asfreq('D').fillna(0)

# Feature engineering
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['day'] = df.index.day
df['lag_1'] = df['receipts'].shift(1)
df['lag_7'] = df['receipts'].shift(7)
df['rolling_7'] = df['receipts'].rolling(7).mean().shift(1)
df = df.dropna()

# Split features and target
X = df.drop('receipts', axis=1).values
y = df['receipts'].values.reshape(-1, 1)

# Normalize
train_mean = np.mean(X, axis=0)
train_std = np.std(X, axis=0)
X = (X - train_mean) / train_std

# Save normalization parameters
np.save('train_mean.npy', train_mean)
np.save('train_std.npy', train_std)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],),
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Added learning rate scheduler and early stopping
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-3 * 10**(epoch / 20))
early_stop = tf.keras.callbacks.EarlyStopping(patience=10)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='mse',
              metrics=['mae'])

history = model.fit(X, y,
                   epochs=200,
                   validation_split=0.2,
                   batch_size=32,
                   callbacks=[lr_scheduler, early_stop])

