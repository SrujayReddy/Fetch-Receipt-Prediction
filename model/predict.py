import tensorflow as tf
from model_utils import ReceiptPredictor
import numpy as np
import pandas as pd

if __name__ == "__main__":
    try:
        # Load saved parameters with absolute paths
        train_mean = np.load('/app/model/train_mean.npy')
        train_std = np.load('/app/model/train_std.npy')
        y_mean = np.load('/app/model/y_mean.npy')
        y_std = np.load('/app/model/y_std.npy')

        predictor = ReceiptPredictor(
            load_pretrained=True,
            model_path='/app/model/receipt_model.h5',
            train_mean_path='/app/model/train_mean.npy',
            train_std_path='/app/model/train_std.npy',
            y_mean_path='/app/model/y_mean.npy',
            y_std_path='/app/model/y_std.npy'
        )

        predictions = predictor.predict('2022-01-01', '2022-12-31')
        predictions.to_csv('/app/model/2022_predictions.csv', index=False)

    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        exit(1)
