import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, sequence_length=60, n_features=5):
        """
        Initialize the stock predictor model.
        
        Args:
            sequence_length (int): Number of time steps to look back
            n_features (int): Number of features to use for prediction
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build the LSTM model architecture.
        
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, data):
        """
        Prepare data for LSTM model.
        
        Args:
            data (np.array): Input data array
            
        Returns:
            tuple: (X, y) where X is the input sequences and y is the target values
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, 0])  # Predict the next closing price
        
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """
        Train the model.
        
        Args:
            X_train (np.array): Training input sequences
            y_train (np.array): Training target values
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        try:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1
            )
            return history
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (np.array): Input sequences
            
        Returns:
            np.array: Predicted values
        """
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def save_model(self, filepath):
        """
        Save the model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath):
        """
        Load a saved model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        try:
            self.model = tf.keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 