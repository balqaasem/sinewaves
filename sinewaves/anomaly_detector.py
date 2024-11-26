import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AnomalyDetector:
    """
    Detect anomalies in sensor data using machine learning techniques
    """
    def __init__(self, contamination=0.1):
        """
        Initialize anomaly detector
        
        Args:
            contamination (float): Expected proportion of anomalies in dataset
        """
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination, 
            random_state=42
        )
    
    def fit(self, feature_matrix):
        """
        Train anomaly detection model
        
        Args:
            feature_matrix (np.array): Normalized feature matrix
        """
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        # Train isolation forest
        self.isolation_forest.fit(scaled_features)
    
    def predict(self, feature_matrix):
        """
        Predict anomalies in new data
        
        Args:
            feature_matrix (np.array): Normalized feature matrix
        
        Returns:
            np.array: Anomaly labels (-1 for anomaly, 1 for normal)
        """
        # Scale features
        scaled_features = self.scaler.transform(feature_matrix)
        
        # Predict anomalies
        return self.isolation_forest.predict(scaled_features)
    
    def get_anomaly_scores(self, feature_matrix):
        """
        Get anomaly scores for input features
        
        Args:
            feature_matrix (np.array): Normalized feature matrix
        
        Returns:
            np.array: Anomaly scores
        """
        scaled_features = self.scaler.transform(feature_matrix)
        return -self.isolation_forest.score_samples(scaled_features)
