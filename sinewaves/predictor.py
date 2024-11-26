import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import logging

from .data_processor import SensorDataProcessor
from .anomaly_detector import AnomalyDetector

class EquipmentPredictor:
    """
    Predictive maintenance model for equipment failure prediction
    """
    def __init__(self):
        """
        Initialize predictor components
        """
        self.data_processor = SensorDataProcessor()
        self.anomaly_detector = AnomalyDetector()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.logger = logging.getLogger(__name__)
    
    def load_sensor_data(self, filepath):
        """
        Load sensor data from CSV file
        
        Args:
            filepath (str): Path to sensor data CSV
        
        Returns:
            pd.DataFrame: Loaded sensor data
        """
        try:
            sensor_data = pd.read_csv(filepath)
            self.logger.info(f"Loaded sensor data from {filepath}")
            return sensor_data
        except Exception as e:
            self.logger.error(f"Error loading sensor data: {e}")
            raise
    
    def preprocess_data(self, sensor_data):
        """
        Preprocess sensor data and extract features
        
        Args:
            sensor_data (pd.DataFrame): Raw sensor measurements
        
        Returns:
            np.array: Processed feature matrix
        """
        # Assuming sensor_data has columns representing different sensor measurements
        processed_features = []
        
        for column in sensor_data.columns:
            if column != 'failure_label':  # Exclude target variable
                processed_signal = self.data_processor.preprocess_signal(sensor_data[column].values)
                features = self.data_processor.extract_features(processed_signal)
                processed_features.append(list(features.values()))
        
        return np.array(processed_features).T
    
    def train_model(self, sensor_data):
        """
        Train predictive maintenance model
        
        Args:
            sensor_data (pd.DataFrame): Sensor data with failure labels
        """
        # Preprocess data
        X = self.preprocess_data(sensor_data)
        y = sensor_data['failure_label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X_train)
        
        # Train predictive model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        self.logger.info("Model Training Report:\n" + classification_report(y_test, y_pred))
    
    def predict_failures(self, sensor_data):
        """
        Predict potential equipment failures
        
        Args:
            sensor_data (pd.DataFrame): New sensor measurements
        
        Returns:
            dict: Prediction results with failure probabilities
        """
        # Preprocess data
        X = self.preprocess_data(sensor_data)
        
        # Detect anomalies
        anomaly_scores = self.anomaly_detector.get_anomaly_scores(X)
        anomaly_labels = self.anomaly_detector.predict(X)
        
        # Predict failure probabilities
        failure_probs = self.model.predict_proba(X)
        
        return {
            'failure_probabilities': failure_probs,
            'anomaly_scores': anomaly_scores,
            'anomaly_labels': anomaly_labels
        }
    
    def save_model(self, filepath):
        """
        Save trained model to file
        
        Args:
            filepath (str): Path to save model
        """
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load pre-trained model
        
        Args:
            filepath (str): Path to model file
        """
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from {filepath}")
