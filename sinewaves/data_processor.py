import numpy as np
import pandas as pd
from scipy import signal
import logging

class SensorDataProcessor:
    """
    Process and prepare sensor data for predictive maintenance analysis
    """
    def __init__(self, sampling_rate=1000):
        """
        Initialize data processor with sampling rate
        
        Args:
            sampling_rate (int): Sampling frequency of sensor data
        """
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(__name__)
    
    def preprocess_signal(self, raw_data):
        """
        Preprocess raw sensor signal
        
        Args:
            raw_data (np.array): Raw sensor measurements
        
        Returns:
            np.array: Processed signal
        """
        # Remove baseline drift
        processed_signal = signal.detrend(raw_data)
        
        # Apply low-pass filter to remove high-frequency noise
        nyquist_freq = 0.5 * self.sampling_rate
        normalized_cutoff = 100 / nyquist_freq
        b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)
        filtered_signal = signal.filtfilt(b, a, processed_signal)
        
        return filtered_signal
    
    def extract_features(self, signal_data):
        """
        Extract statistical features from signal
        
        Args:
            signal_data (np.array): Processed signal
        
        Returns:
            dict: Extracted signal features
        """
        features = {
            'mean': np.mean(signal_data),
            'std': np.std(signal_data),
            'rms': np.sqrt(np.mean(signal_data**2)),
            'peak_to_peak': np.ptp(signal_data),
            'kurtosis': np.mean(((signal_data - np.mean(signal_data)) / np.std(signal_data))**4)
        }
        return features
