import tensorflow as tf
import pandas as pd
import numpy as np

class ReceiptPredictor:
    def __init__(self, load_pretrained=False, model_path=None, train_mean_path=None, 
                 train_std_path=None, y_mean_path=None, y_std_path=None):
        self.train_mean = None
        self.train_std = None
        self.y_mean = None
        self.y_std = None
        
        if load_pretrained and model_path:
            # Load pre-trained model
            self.model = tf.keras.models.load_model(model_path, compile=False)
            
            # Load normalization parameters if provided
            if train_mean_path and train_std_path:
                self.train_mean = np.load(train_mean_path)
                self.train_std = np.load(train_std_path)
            
            if y_mean_path and y_std_path:
                self.y_mean = np.load(y_mean_path)
                self.y_std = np.load(y_std_path)
        else:
            # Build a new model
            self.model = self.build_enhanced_model()

    def build_enhanced_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu',
                                kernel_initializer='he_normal',
                                input_shape=(6,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l1_l2(0.01, 0.01)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001,
            clipvalue=1.0
        )

        model.compile(optimizer=optimizer,
                    loss='mse',
                    metrics=['mae'])
        return model

    def validate_data(self, df):
        if 'Date' not in df.columns:
            raise ValueError("CSV must contain 'Date' column (case-sensitive)")
        if 'Receipt_Count' not in df.columns:
            raise ValueError("CSV must contain 'Receipt_Count' column (case-sensitive)")

        try:
            dates = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
            if dates.isnull().any():
                bad_dates = df[dates.isnull()]['Date'].tolist()
                raise ValueError(f"Invalid date formats: {bad_dates[:5]}")
        except Exception as e:
            raise ValueError(f"Date parsing failed: {str(e)}")

        date_range = pd.date_range(start=dates.min(), end=dates.max())
        missing = date_range.difference(dates)
        if len(missing) > 0:
            raise ValueError(f"Missing {len(missing)} dates. First 5: {missing[:5]}")

    def preprocess_data(self, df):
        self.validate_data(df)

        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df = df.set_index('Date').sort_index().asfreq('D').fillna(0)

        # Feature engineering
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['lag_1'] = df['Receipt_Count'].shift(1)
        df['lag_7'] = df['Receipt_Count'].shift(7)
        df['rolling_7'] = df['Receipt_Count'].rolling(7).mean().shift(1)

        df = df.dropna()
        return df

    def train(self, file_path):
        df = pd.read_csv(file_path)
        df = self.preprocess_data(df)

        X = df.drop('Receipt_Count', axis=1).values
        y = df['Receipt_Count'].values.reshape(-1, 1)

        # Normalize target
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        y_norm = (y - self.y_mean) / self.y_std

        # Normalize features
        self.train_mean = np.mean(X, axis=0)
        self.train_std = np.std(X, axis=0)
        X_norm = (X - self.train_mean) / self.train_std

        # Train with early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True
        )
        self.model.fit(X_norm, y_norm,
                      epochs=200,
                      validation_split=0.2,
                      batch_size=32,
                      callbacks=[early_stop])

    def predict(self, start_date, end_date):
        dates = pd.date_range(start_date, end_date)
        predictions = []
        history = []

        # Initialize with last 7 real values
        try:
            train_data = pd.read_csv('../data/daily_receipts.csv')
            if not train_data.empty and 'Receipt_Count' in train_data.columns:
                history = train_data['Receipt_Count'].tail(7).tolist()
            else:
                print("Warning: Training data is empty or missing Receipt_Count column")
        except Exception as e:
            print(f"Warning: Could not load training data: {str(e)}")
            
        # If we couldn't get any history data, return empty predictions
        if not history:
            print("Warning: No historical data available for prediction")
            return pd.DataFrame({
                'Date': dates,
                'Predicted_Receipts': [0] * len(dates)  # Default to 0 for all dates
            })

        for date in dates:
            # Generate features from history
            if not history:
                # If no history data is available, use reasonable defaults
                lag_1 = 0
                lag_7 = 0
                rolling_7 = 0
            else:
                lag_1 = history[-1] if len(history) >= 1 else np.mean(history)
                lag_7 = history[-7] if len(history) >= 7 else np.mean(history)
                rolling_7 = np.mean(history[-7:]) if len(history) >= 7 else np.mean(history)

            features = np.array([
                date.dayofweek,
                date.month,
                date.day,
                lag_1,
                lag_7,
                rolling_7
            ])

            # Normalize and predict
            if self.train_mean is None or self.train_std is None:
                raise ValueError("Training normalization parameters (train_mean, train_std) are not set")
                
            features_norm = (features - self.train_mean) / self.train_std
            pred_norm = self.model.predict(features_norm.reshape(1, -1))[0][0]
            
            # Denormalize prediction
            if self.y_mean is None or self.y_std is None:
                raise ValueError("Target normalization parameters (y_mean, y_std) are not set")
                
            # Ensure we're not operating on None values
            pred = pred_norm * self.y_std + self.y_mean if self.y_std is not None and self.y_mean is not None else 0

            predictions.append(pred)
            history.append(pred)

        return pd.DataFrame({
            'Date': dates,
            'Predicted_Receipts': predictions
        })
