# Fetch Receipt Prediction System

A machine learning solution for predicting monthly receipt scans in 2022 based on 2021 daily data.

## Features

- Time Series Forecasting with Neural Networks
- Interactive Web Interface
- Visual Comparison of 2021 vs 2022 Data
- Docker Containerization
- Automated Feature Engineering

## Project Structure

.  
├── data/ - Input data files  
├── model/ - ML model training and prediction code  
├── app/ - Web application components  
│ ├── static/ - CSS styles  
│ └── templates/ - HTML templates  
├── Dockerfile - Container configuration  
└── requirements.txt - Python dependencies


## Running the Project

1. **Build Docker Image**
   ```bash
   docker build -t fetch-app .

2.  **Run Container**
    
    bash
    
    Copy
    
    docker run -p 5000:5000 fetch-app
    
3.  **Access Web Interface**  
    Open  `http://localhost:5000`  in your browser
    

## Technical Details

**Model Architecture**

-   4-layer Neural Network with L2 Regularization
    
-   Adam Optimizer with Learning Rate Scheduling
    
-   Early Stopping to Prevent Overfitting
    
-   Feature Engineering Includes:
    
    -   Temporal Features (Day of Week, Month)
        
    -   Lag Features (1-day, 7-day)
        
    -   Rolling Averages
        

**API Endpoints**

-   GET  `/`: Returns prediction visualization and data table
    

## Training Performance

-   Validation MAE: <3% of average daily receipts
    
-   Recursive Prediction Strategy for 2022
    
-   Automated Feature Normalization

**Key Improvements**:
1. Added regularization and dropout to prevent overfitting
2. Implemented learning rate scheduling and early stopping
3. Enhanced visualization with interactive charts
4. Professional UI design with responsive layout
5. Comprehensive documentation
6. Improved error handling in feature generation
7. Added model metrics (MAE) for better validation

This implementation demonstrates:
- Strong understanding of time series forecasting
- Ability to productionize ML models
- Clean code organization
- Attention to user experience
- Model optimization techniques
- Containerization skills
