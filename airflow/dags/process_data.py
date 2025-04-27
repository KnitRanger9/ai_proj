#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add the ml directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../ml'))

from utils.data_processor import load_and_preprocess_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_data(symbol):
    """
    Process raw stock data.
    
    Args:
        symbol (str): Stock symbol
    """
    try:
        # Define paths
        raw_data_path = f"/opt/airflow/data/raw/{symbol}_raw.csv"
        processed_data_path = f"/opt/airflow/data/processed/{symbol}_processed.csv"
        
        # Create processed data directory if it doesn't exist
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        
        # Load and preprocess data
        logger.info(f"Processing data for {symbol}")
        df = load_and_preprocess_data(raw_data_path)
        
        # Save processed data
        df.to_csv(processed_data_path)
        logger.info(f"Saved processed data to {processed_data_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing data for {symbol}: {str(e)}")
        return False

def main():
    """Main function to process data for all stocks."""
    # List of stock symbols to process
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    for symbol in symbols:
        raw_data_path = f"/opt/airflow/data/raw/{symbol}_raw.csv"
        
        if not os.path.exists(raw_data_path):
            logger.warning(f"Raw data file not found for {symbol}: {raw_data_path}")
            continue
        
        logger.info(f"Processing data for {symbol}")
        success = process_data(symbol)
        
        if success:
            logger.info(f"Successfully processed data for {symbol}")
        else:
            logger.error(f"Failed to process data for {symbol}")

if __name__ == "__main__":
    main() 