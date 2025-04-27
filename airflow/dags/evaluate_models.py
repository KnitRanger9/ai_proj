#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import mlflow.tensorflow
import logging
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add the ml directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../ml'))

from models.lstm import StockPredictor
from utils.data_processor import load_and_preprocess_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model(symbol, data_path, model_path, scaler_path):
    """
    Evaluate a trained model.
    
    Args:
        symbol (str): Stock symbol
        data_path (str): Path to the test data
        model_path (str): Path to the trained model
        scaler_path (str): Path to the saved scaler
    """
    try:
        # Load and preprocess data
        logger.info(f"Loading test data for {symbol}")
        df = load_and_preprocess_data(data_path)
        
        # Prepare features
        features = ['Close', 'Open', 'High', 'Low', 'Volume']
        data = df[features].values
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        scaled_data = scaler.transform(data)
        
        # Create model and load weights
        model = StockPredictor()
        model.load_model(model_path)
        
        # Prepare test data
        X, y = model.prepare_data(scaled_data)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        # Log metrics to MLflow
        with mlflow.start_run(run_name=f"evaluate_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            # Log predictions
            results_df = pd.DataFrame({
                'actual': y.flatten(),
                'predicted': predictions.flatten()
            })
            results_path = f"/opt/airflow/results/{symbol}_predictions.csv"
            results_df.to_csv(results_path, index=False)
            mlflow.log_artifact(results_path)
        
        logger.info(f"Evaluation completed for {symbol}")
        logger.info(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        return False

def main():
    """Main function to evaluate all models."""
    # List of stock symbols to evaluate
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Create results directory
    os.makedirs("/opt/airflow/results", exist_ok=True)
    
    for symbol in symbols:
        data_path = f"/opt/airflow/data/processed/{symbol}_processed.csv"
        model_path = f"/opt/airflow/models/saved_models/{symbol}_model.h5"
        scaler_path = f"/opt/airflow/models/saved_models/{symbol}_scaler.pkl"
        
        if not all(os.path.exists(path) for path in [data_path, model_path, scaler_path]):
            logger.warning(f"Required files not found for {symbol}")
            continue
        
        logger.info(f"Evaluating model for {symbol}")
        success = evaluate_model(symbol, data_path, model_path, scaler_path)
        
        if success:
            logger.info(f"Successfully evaluated model for {symbol}")
        else:
            logger.error(f"Failed to evaluate model for {symbol}")

if __name__ == "__main__":
    main() 