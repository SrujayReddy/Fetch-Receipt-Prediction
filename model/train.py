from model_utils import ReceiptPredictor
import numpy as np

if __name__ == "__main__":
    try:
        predictor = ReceiptPredictor()
        predictor.train('/app/data/daily_receipts.csv')  # Absolute path

        # Save model and parameters
        predictor.model.save('/app/model/receipt_model.h5')
        np.save('/app/model/train_mean.npy', predictor.train_mean)
        np.save('/app/model/train_std.npy', predictor.train_std)
        np.save('/app/model/y_mean.npy', predictor.y_mean)
        np.save('/app/model/y_std.npy', predictor.y_std)

    except Exception as e:
        print(f"Training failed: {str(e)}")
        exit(1)
