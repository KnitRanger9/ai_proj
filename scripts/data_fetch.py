#!/usr/bin/env python3

import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_stock_data(symbol, start_date, end_date=None):
    """
    Fetch stock data for a given symbol and date range.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to today.
    
    Returns:
        pd.DataFrame: DataFrame containing stock data
    """
    try:
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return None
        
        # Add symbol column
        df['Symbol'] = symbol
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def save_data(df, symbol, data_type='raw'):
    """
    Save stock data to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        symbol (str): Stock symbol
        data_type (str): Type of data ('raw', 'processed', or 'features')
    """
    if df is None or df.empty:
        return
    
    # Create directory if it doesn't exist
    os.makedirs(f'ml/data/{data_type}', exist_ok=True)
    
    # Save to CSV
    filename = f'ml/data/{data_type}/{symbol}_{datetime.now().strftime("%Y%m%d")}.csv'
    df.to_csv(filename, index=False)
    logger.info(f"Saved data to {filename}")

def main():
    # List of stock symbols to fetch
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Date range (last 5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    for symbol in symbols:
        # Fetch raw data
        df = fetch_stock_data(symbol, start_date.strftime('%Y-%m-%d'))
        if df is not None:
            save_data(df, symbol, 'raw')
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)

if __name__ == "__main__":
    main() 