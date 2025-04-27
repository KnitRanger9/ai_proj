import pandas as pd
import numpy as np
import logging
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(file_path):
    """
    Load and preprocess stock data.
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Handle missing values
        df = handle_missing_values(df)
        
        # Remove outliers
        df = remove_outliers(df)
        
        return df
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    try:
        # Simple Moving Averages
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        
        # Exponential Moving Averages
        df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
        
        # Relative Strength Index
        df['RSI'] = RSIIndicator(close=df['Close']).rsi()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        
        return df
    
    except Exception as e:
        logger.error(f"Error adding technical indicators: {str(e)}")
        raise

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    try:
        # Forward fill missing values
        df.fillna(method='ffill', inplace=True)
        
        # Backward fill any remaining missing values
        df.fillna(method='bfill', inplace=True)
        
        return df
    
    except Exception as e:
        logger.error(f"Error handling missing values: {str(e)}")
        raise

def remove_outliers(df, columns=None, n_std=3):
    """
    Remove outliers from the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to check for outliers
        n_std (int): Number of standard deviations for outlier detection
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    try:
        if columns is None:
            columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for column in columns:
            mean = df[column].mean()
            std = df[column].std()
            df[column] = df[column].clip(lower=mean - n_std * std, upper=mean + n_std * std)
        
        return df
    
    except Exception as e:
        logger.error(f"Error removing outliers: {str(e)}")
        raise 