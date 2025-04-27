#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import mlflow.tensorflow
import logging
from sklearn.preprocessing import MinMaxScaler

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm import StockPredictor
from utils.data_processor import load_and_preprocess_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(symbol, data_path, model_path, sequence_length=60, n_features=5):
    """
    Train the LSTM model for stock price prediction.
    
    Args:
        symbol (str): Stock symbol
        data_path (str): Path to the data file
        model_path (str): Path to save the trained model
        sequence_length (int): Number of time steps to look back
        n_features (int): Number of features to use for prediction
    """
    try:
        # Load and preprocess data
        logger.info(f"Loading data for {symbol}")
        df = load_and_preprocess_data(data_path)
        
        # Prepare features
        features = ['Close', 'Open', 'High', 'Low', 'Volume']
        data = df[features].values
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Create and train the model
        model = StockPredictor(sequence_length=sequence_length, n_features=n_features)
        X, y = model.prepare_data(scaled_data)
        
        # Split the data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"train_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params({
                "symbol": symbol,
                "sequence_length": sequence_length,
                "n_features": n_features,
                "train_size": train_size
            })
            
            # Train the model
            logger.info("Training model...")
            history = model.train(X_train, y_train)
            
            # Log metrics
            for metric in history.history:
                mlflow.log_metric(metric, history.history[metric][-1])
            
            # Evaluate on test set
            test_loss = model.model.evaluate(X_test, y_test, verbose=0)
            mlflow.log_metric("test_loss", test_loss[0])
            mlflow.log_metric("test_mae", test_loss[1])
            
            # Save the model
            model.save_model(model_path)
            mlflow.log_artifact(model_path)
            
            # Save the scaler
            scaler_path = os.path.join(os.path.dirname(model_path), f"{symbol}_scaler.pkl")
            import joblib
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(scaler_path)
        
        logger.info(f"Training completed for {symbol}")
        return True
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return False

def main():
    """Main function to train models for multiple stocks."""
    # List of stock symbols to train
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    for symbol in symbols:
        data_path = f"data/processed/{symbol}_processed.csv"
        model_path = f"models/saved_models/{symbol}_model.h5"
        
        if not os.path.exists(data_path):
            logger.warning(f"Data file not found for {symbol}: {data_path}")
            continue
        
        logger.info(f"Training model for {symbol}")
        success = train_model(symbol, data_path, model_path)
        
        if success:
            logger.info(f"Successfully trained model for {symbol}")
        else:
            logger.error(f"Failed to train model for {symbol}")

if __name__ == "__main__":
    main() 