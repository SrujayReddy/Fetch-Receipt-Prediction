# Fetch Receipt Prediction System

![Dashboard Screenshot](/app/static/dashboard-preview.png)


An end-to-end machine learning solution for predicting monthly scanned receipts in 2022 using daily 2021 data, developed for Fetch Rewards' Machine Learning Engineer take-home exercise.

## 🚀 Quick Start (Docker)

```bash
# 1. Clone repository
git clone https://github.com/SrujayReddy/Fetch-Receipt-Prediction.git
cd fetch-receipt-prediction

# 2. Build and run container
docker build -t fetch-app . && docker run -p 5001:5001 fetch-app

# 3. Access dashboard
open http://localhost:5001
```

## 📋 Project Overview

### Key Features
- **Custom Neural Network** built with TensorFlow
- **Temporal Feature Engineering** (lag features, rolling averages)
- **Interactive Web Dashboard** with comparative visualizations
- **Docker Containerization** for reproducible execution
- **Production-Grade Pipeline**:
  - Automated data validation
  - Model serialization/deserialization
  - Comprehensive error handling

### Technical Highlights
- **Validation MAE**: 0.2607 (normalized units)
- **Training Time**: <2 minutes on CPU
- **Prediction Accuracy**: ±3% of monthly averages

## 📂 Repository Structure

```
.
├── data/                   # Input data
│   └── daily_receipts.csv  # 2021 daily receipt counts
├── model/                  # Machine learning components
│   ├── model_utils.py      # Core ML logic
│   ├── train.py            # Training pipeline
│   ├── predict.py          # Prediction script
│   └── *.npy               # Normalization parameters
├── app/                    # Web application
│   ├── app.py              # Flask server
│   ├── static/             # CSS/JS assets
│   └── templates/          # HTML templates
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
└── README.md               # This documentation
```

## 🧠 Machine Learning Implementation

### Model Architecture
```python
tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu', 
                        kernel_regularizer=tf.keras.regularizers.l1_l2(0.01)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
```

### Feature Engineering
| Feature Type         | Description                          |
|----------------------|--------------------------------------|
| Temporal Features    | Day of week, month, day of month     |
| Lag Features         | 1-day and 7-day previous values      |
| Rolling Window       | 7-day moving average                 |

## 🛠️ Installation & Usage

### Docker Deployment (Recommended)
```bash
docker build -t fetch-app .  # Build image
docker run -p 5001:5001 fetch-app  # Start container
```

### Local Execution
```bash
# Create virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train model and generate predictions
cd model && python train.py && python predict.py

# Start web server
cd ../app && python app.py
```

## 🔍 Verification

After running the pipeline:
```bash
# Check generated predictions
head model/2022_predictions.csv

# Expected output:
Date,Predicted_Receipts
2022-01-01,7564766.32
2022-01-02,7455524.15
...
```

## 📊 Web Interface

Access the dashboard at `http://localhost:5001` to view:
- Interactive comparison of 2021 vs 2022 data
- Monthly prediction tables
- Detailed trend visualizations

## 🚨 Troubleshooting

| Issue                  | Solution                              |
|------------------------|---------------------------------------|
| Port 5001 occupied     | Use `-p 5002:5001` in docker run      |
| Docker build failures  | Run `docker system prune -a`          |
| Missing predictions    | Verify CSV file in `data/` directory  |
| Model loading errors   | Check `model/*.npy` files exist       |

## ❓ FAQ

**Q: How do I modify the prediction period?**  
A: Edit `start_date` and `end_date` in `model/predict.py`

**Q: Where are the model parameters stored?**  
A: `model/receipt_model.h5` (model weights) and `*.npy` (normalization)

**Q: How is monthly aggregation calculated?**  
A: Simple sum of daily predictions for each month

## 📄 Documentation

| Component              | File Path                  |
|------------------------|----------------------------|
| Core Model Logic       | [model/model_utils.py](model/model_utils.py) |
| Training Pipeline      | [model/train.py](model/train.py) |
| Web Interface          | [app/app.py](app/app.py)   |

---

*Developed with ❤️ for Fetch Rewards Machine Learning Engineer Position*
``
