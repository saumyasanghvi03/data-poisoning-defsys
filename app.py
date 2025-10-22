import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import time
import gc
import resource
from contextlib import contextmanager
import warnings
import requests
import json
import socket
import psutil
import platform
import subprocess
import re
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib
import hashlib
import io
import base64
from urllib.parse import urlparse

warnings.filterwarnings('ignore')

# Advanced system optimization
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(16384, hard), hard))
except (ImportError, ValueError):
    pass

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DATA POISONING DEFENSE PLATFORM",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional imports with error handling
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Remove problematic Kaggle import and use a simpler approach
KAGGLE_AVAILABLE = False  # We'll simulate Kaggle data without the API

# --- ENHANCED CSS STYLES ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;500;600;700&display=swap');
    
    .neuro-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 30%, #24243e 70%, #000000 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid #00ffff;
        box-shadow: 0 0 50px #00ffff33, inset 0 0 100px #00ffff11, 0 0 0 1px #00ffff22;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        text-align: center;
        backdrop-filter: blur(20px);
    }
    
    .neuro-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, #00ffff22, transparent);
        animation: neuro-shimmer 6s infinite;
    }
    
    @keyframes neuro-shimmer {
        0% { left: -100%; }
        50% { left: 100%; }
        100% { left: 100%; }
    }
    
    .poison-alert {
        background: linear-gradient(135deg, #4a1f1f, #2d1a1a);
        border: 1px solid #ff4444;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        animation: pulse-red 2s infinite;
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
    }
    
    .defense-success {
        background: linear-gradient(135deg, #1f4a2e, #1a2d1f);
        border: 1px solid #00ff00;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .data-terminal {
        background-color: #000000;
        color: #00ff00;
        font-family: 'Courier New', monospace;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #00ff00;
        height: 300px;
        overflow-y: scroll;
        white-space: pre-wrap;
    }
    
    .explanation-box {
        background: rgba(0, 255, 255, 0.1);
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Exo 2', sans-serif;
    }
    
    .threat-level-critical { background: linear-gradient(45deg, #ff0000, #ff6b00); color: white; padding: 0.5rem; border-radius: 5px; }
    .threat-level-high { background: linear-gradient(45deg, #ff6b00, #ffd000); color: black; padding: 0.5rem; border-radius: 5px; }
    .threat-level-medium { background: linear-gradient(45deg, #ffd000, #ffff00); color: black; padding: 0.5rem; border-radius: 5px; }
    .threat-level-low { background: linear-gradient(45deg, #00ff00, #00cc00); color: white; padding: 0.5rem; border-radius: 5px; }
    
    .data-source-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #00ffff;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .data-source-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 255, 255, 0.3);
    }
    
    .live-data-badge {
        background: linear-gradient(45deg, #ff0000, #ff6b00);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .dataset-info {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #00ffff;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #4a3c1f, #2d281a);
        border: 1px solid #ffaa00;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #1f4a2e, #1a2d1f);
        border: 1px solid #00ff00;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: rgba(0, 255, 255, 0.05);
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .upload-box {
        border: 2px dashed #00ffff;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background: rgba(0, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

@contextmanager
def quantum_resource_manager():
    """Advanced resource management"""
    try:
        yield
    finally:
        gc.collect()

# --- CORE DATA POISONING CLASSES ---

class DataPoisoningAttacks:
    """Simulate various data poisoning attacks"""
    
    def __init__(self):
        self.attack_types = {
            'label_flipping': 'Flip labels of training data',
            'feature_manipulation': 'Manipulate feature values',
            'backdoor_injection': 'Inject hidden triggers',
            'data_replication': 'Duplicate malicious samples',
            'gradient_manipulation': 'Manipulate model gradients',
            'model_inversion': 'Extract training data from model'
        }
    
    def label_flipping_attack(self, data, labels, flip_percentage=0.1):
        """Simulate label flipping attack"""
        poisoned_data = data.copy()
        poisoned_labels = labels.copy()
        
        num_to_flip = int(len(labels) * flip_percentage)
        flip_indices = np.random.choice(len(labels), num_to_flip, replace=False)
        
        for idx in flip_indices:
            # Flip to a different class
            current_class = poisoned_labels[idx]
            other_classes = [c for c in np.unique(labels) if c != current_class]
            if other_classes:
                poisoned_labels[idx] = np.random.choice(other_classes)
        
        return poisoned_data, poisoned_labels, flip_indices
    
    def feature_manipulation_attack(self, data, manipulation_strength=0.3):
        """Simulate feature manipulation attack"""
        poisoned_data = data.copy()
        num_samples = data.shape[0]
        num_features = data.shape[1]
        
        # Select random samples to poison
        poison_indices = np.random.choice(num_samples, int(num_samples * 0.05), replace=False)
        
        for idx in poison_indices:
            # Add noise to features
            noise = np.random.normal(0, manipulation_strength, num_features)
            poisoned_data[idx] += noise
        
        return poisoned_data, poison_indices
    
    def backdoor_injection(self, data, labels, trigger_pattern, target_class):
        """Simulate backdoor injection attack"""
        poisoned_data = data.copy()
        poisoned_labels = labels.copy()
        
        # Inject trigger into random samples
        injection_indices = np.random.choice(len(data), int(len(data) * 0.03), replace=False)
        
        for idx in injection_indices:
            # Apply trigger pattern
            poisoned_data[idx] = poisoned_data[idx] * (1 - trigger_pattern) + trigger_pattern
            poisoned_labels[idx] = target_class
        
        return poisoned_data, poisoned_labels, injection_indices

class DataPoisoningDetector:
    """Detect data poisoning attacks"""
    
    def __init__(self):
        self.detectors = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'one_class_svm': OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1),
            'dbscan': DBSCAN(eps=0.5, min_samples=5)
        }
        self.scaler = StandardScaler()
    
    def detect_anomalies_isolation_forest(self, data):
        """Detect anomalies using Isolation Forest"""
        scaled_data = self.scaler.fit_transform(data)
        predictions = self.detectors['isolation_forest'].fit_predict(scaled_data)
        anomaly_indices = np.where(predictions == -1)[0]
        return anomaly_indices
    
    def detect_anomalies_svm(self, data):
        """Detect anomalies using One-Class SVM"""
        scaled_data = self.scaler.fit_transform(data)
        predictions = self.detectors['one_class_svm'].fit_predict(scaled_data)
        anomaly_indices = np.where(predictions == -1)[0]
        return anomaly_indices
    
    def detect_cluster_anomalies(self, data):
        """Detect anomalies using DBSCAN clustering"""
        scaled_data = self.scaler.fit_transform(data)
        predictions = self.detectors['dbscan'].fit_predict(scaled_data)
        anomaly_indices = np.where(predictions == -1)[0]
        return anomaly_indices
    
    def statistical_analysis(self, data):
        """Perform statistical analysis to detect poisoning"""
        results = {}
        
        # Z-score analysis
        z_scores = np.abs((data - data.mean(axis=0)) / data.std(axis=0))
        high_z_score_indices = np.where(z_scores > 3)[0]
        results['z_score_anomalies'] = high_z_score_indices
        
        # Mahalanobis distance
        try:
            cov_matrix = np.cov(data.T)
            inv_cov_matrix = np.linalg.pinv(cov_matrix)
            mean_diff = data - data.mean(axis=0)
            mahalanobis_dist = np.sqrt(np.einsum('ij,ij->i', mean_diff @ inv_cov_matrix, mean_diff))
            high_mahalanobis_indices = np.where(mahalanobis_dist > np.percentile(mahalanobis_dist, 95))[0]
            results['mahalanobis_anomalies'] = high_mahalanobis_indices
        except:
            results['mahalanobis_anomalies'] = np.array([])
        
        return results

class DefenseMechanisms:
    """Implement defense mechanisms against data poisoning"""
    
    def data_sanitization(self, data, labels, anomaly_indices):
        """Remove detected poisoned samples"""
        clean_data = np.delete(data, anomaly_indices, axis=0)
        clean_labels = np.delete(labels, anomaly_indices, axis=0)
        return clean_data, clean_labels
    
    def robust_training(self, data, labels, method='trimmed_loss'):
        """Implement robust training techniques"""
        if method == 'trimmed_loss':
            return self._trimmed_loss_training(data, labels)
        elif method == 'differential_privacy':
            return self._differential_privacy_training(data, labels)
        else:
            return data, labels
    
    def _trimmed_loss_training(self, data, labels):
        """Trimmed loss for robust learning"""
        # Simulate trimmed loss by removing samples with highest loss
        # In practice, this would be integrated into the training process
        return data, labels
    
    def _differential_privacy_training(self, data, labels):
        """Add differential privacy noise"""
        noise_scale = 0.1
        noisy_data = data + np.random.normal(0, noise_scale, data.shape)
        return noisy_data, labels
    
    def federated_learning_defense(self, client_data_list, aggregation_method='median'):
        """Defend against poisoning in federated learning"""
        if aggregation_method == 'median':
            return np.median(client_data_list, axis=0)
        elif aggregation_method == 'trimmed_mean':
            # Remove outliers before averaging
            sorted_data = np.sort(client_data_list, axis=0)
            trimmed_data = sorted_data[1:-1]  # Remove min and max
            return np.mean(trimmed_data, axis=0)
        else:
            return np.mean(client_data_list, axis=0)

class MLModelSecurity:
    """Machine learning model security analysis"""
    
    def __init__(self):
        self.model_metrics = {}
    
    def model_robustness_test(self, clean_model, poisoned_model, test_data, test_labels):
        """Test model robustness against poisoning"""
        clean_accuracy = self._evaluate_model(clean_model, test_data, test_labels)
        poisoned_accuracy = self._evaluate_model(poisoned_model, test_data, test_labels)
        
        robustness_score = (poisoned_accuracy / clean_accuracy) * 100
        return {
            'clean_accuracy': clean_accuracy,
            'poisoned_accuracy': poisoned_accuracy,
            'robustness_score': robustness_score,
            'performance_drop': clean_accuracy - poisoned_accuracy
        }
    
    def _evaluate_model(self, model, test_data, test_labels):
        """Evaluate model accuracy (simulated)"""
        # Simulate model evaluation
        return random.uniform(0.7, 0.95)
    
    def extract_training_data_analysis(self, model, original_data_shape):
        """Analyze potential for training data extraction"""
        # Simulate model inversion attack analysis
        vulnerability_score = random.uniform(0.1, 0.9)
        return {
            'vulnerability_score': vulnerability_score,
            'risk_level': 'HIGH' if vulnerability_score > 0.7 else 'MEDIUM' if vulnerability_score > 0.4 else 'LOW',
            'recommendations': [
                'Use differential privacy',
                'Implement secure aggregation',
                'Monitor model outputs'
            ]
        }

class RealTimeDataMonitor:
    """Real-time data stream monitoring for poisoning detection"""
    
    def __init__(self):
        self.data_stream = []
        self.anomaly_history = []
    
    def monitor_data_stream(self, new_data_point, window_size=100):
        """Monitor streaming data for poisoning patterns"""
        self.data_stream.append(new_data_point)
        
        if len(self.data_stream) > window_size:
            self.data_stream.pop(0)
        
        # Analyze recent data for anomalies
        if len(self.data_stream) >= 10:
            recent_data = np.array(self.data_stream[-10:])
            detector = DataPoisoningDetector()
            anomalies = detector.detect_anomalies_isolation_forest(recent_data)
            
            current_anomalies = len(anomalies) > 0
            self.anomaly_history.append(current_anomalies)
            
            return {
                'anomaly_detected': current_anomalies,
                'anomaly_count': len(anomalies),
                'stream_size': len(self.data_stream),
                'anomaly_rate': sum(self.anomaly_history[-50:]) / min(50, len(self.anomaly_history))
            }
        
        return {'anomaly_detected': False, 'anomaly_count': 0, 'stream_size': len(self.data_stream), 'anomaly_rate': 0.0}

# --- ENHANCED DATA SOURCES INTEGRATION ---

class AdvancedDataFetcher:
    """Enhanced data fetcher with more sources and capabilities"""
    
    def __init__(self):
        self.data_cache = {}
        self.request_headers = {
            'User-Agent': 'DataPoisoningDefensePlatform/1.0',
            'Accept': 'application/json'
        }
    
    def fetch_udise_data(self):
        """Fetch Unified District Information System for Education data (simulated)"""
        try:
            # Simulate UDISE+ education data
            states = ['Maharashtra', 'Tamil Nadu', 'Karnataka', 'Delhi', 'Kerala', 'Gujarat']
            data = {
                'State': states,
                'Total_Schools': np.random.randint(50000, 200000, len(states)),
                'Enrollment': np.random.randint(1000000, 5000000, len(states)),
                'Teacher_Count': np.random.randint(50000, 200000, len(states)),
                'Girls_Enrollment_Ratio': np.random.uniform(0.45, 0.55, len(states)),
                'Digital_Classrooms': np.random.randint(1000, 50000, len(states))
            }
            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error fetching UDISE data: {e}")
            return None
    
    def fetch_health_data(self):
        """Fetch health and nutrition data (simulated)"""
        try:
            indicators = ['Malnutrition_Rate', 'Immunization_Coverage', 'Institutional_Deliveries', 
                         'Doctor_Population_Ratio', 'Hospital_Beds_Per_1000']
            states = ['Bihar', 'UP', 'MP', 'Rajasthan', 'West Bengal', 'Assam']
            
            data = []
            for state in states:
                for indicator in indicators:
                    data.append({
                        'State': state,
                        'Indicator': indicator,
                        'Value': np.random.uniform(10, 95),
                        'Year': 2023
                    })
            
            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error fetching health data: {e}")
            return None
    
    def fetch_agriculture_data(self):
        """Fetch agricultural production data (simulated)"""
        try:
            crops = ['Rice', 'Wheat', 'Sugarcane', 'Cotton', 'Pulses']
            states = ['Punjab', 'Haryana', 'UP', 'MP', 'Andhra Pradesh']
            
            data = []
            for crop in crops:
                for state in states:
                    data.append({
                        'Crop': crop,
                        'State': state,
                        'Production_Tonnes': np.random.randint(100000, 5000000),
                        'Area_Hectares': np.random.randint(50000, 2000000),
                        'Yield_Kg_Hectare': np.random.randint(1000, 5000)
                    })
            
            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error fetching agriculture data: {e}")
            return None
    
    def fetch_social_welfare_data(self):
        """Fetch social welfare scheme data (simulated)"""
        try:
            schemes = ['PM-KISAN', 'MNREGA', 'PM-JAY', 'NSAP', 'ICDS']
            data = {
                'Scheme': schemes,
                'Beneficiaries_Millions': np.random.randint(50, 500, len(schemes)),
                'Budget_Crores': np.random.randint(1000, 50000, len(schemes)),
                'Women_Beneficiaries_Percent': np.random.uniform(40, 60, len(schemes)),
                'Rural_Coverage_Percent': np.random.uniform(70, 95, len(schemes))
            }
            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error fetching social welfare data: {e}")
            return None

# --- ADVANCED DATA POISONING DETECTION ---

class AdvancedPoisoningDetector:
    """Advanced poisoning detection with ensemble methods"""
    
    def __init__(self):
        self.detectors = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'local_outlier_factor': None,  # Could be added
            'autoencoder': None  # Could be added for deep learning approach
        }
        self.scaler = StandardScaler()
        self.detection_history = []
    
    def ensemble_detection(self, data):
        """Use multiple detectors for consensus-based detection"""
        try:
            scaled_data = self.scaler.fit_transform(data)
            
            # Get predictions from available detectors
            predictions = []
            
            # Isolation Forest
            iforest_pred = self.detectors['isolation_forest'].fit_predict(scaled_data)
            predictions.append((iforest_pred == -1).astype(int))
            
            # Statistical methods
            z_scores = np.abs((data - data.mean(axis=0)) / data.std(axis=0))
            z_anomalies = (z_scores > 3).any(axis=1).astype(int)
            predictions.append(z_anomalies)
            
            # Combine predictions (majority voting)
            ensemble_pred = np.mean(predictions, axis=0) > 0.5
            anomaly_indices = np.where(ensemble_pred)[0]
            
            # Log detection
            self.detection_history.append({
                'timestamp': datetime.now(),
                'samples_analyzed': len(data),
                'anomalies_detected': len(anomaly_indices),
                'detection_rate': len(anomaly_indices) / len(data)
            })
            
            return anomaly_indices
            
        except Exception as e:
            st.error(f"Ensemble detection error: {e}")
            return np.array([])
    
    def temporal_analysis(self, data, timestamps):
        """Analyze temporal patterns for poisoning detection"""
        try:
            if len(data) != len(timestamps):
                return np.array([])
            
            # Calculate rolling statistics
            if data.shape[1] >= 1:
                feature_means = np.array([np.mean(data[max(0, i-10):i+1], axis=0) 
                                        for i in range(len(data))])
                feature_stds = np.array([np.std(data[max(0, i-10):i+1], axis=0) 
                                      for i in range(len(data))])
                
                # Detect sudden changes
                changes = np.abs(data - feature_means) / (feature_stds + 1e-8)
                temporal_anomalies = np.where((changes > 2).any(axis=1))[0]
                
                return temporal_anomalies
            
            return np.array([])
            
        except Exception as e:
            st.error(f"Temporal analysis error: {e}")
            return np.array([])

# --- DATA UPLOAD AND PROCESSING ---

class DataProcessor:
    """Handle data upload and preprocessing"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'json', 'xlsx', 'parquet']
    
    def process_uploaded_file(self, uploaded_file):
        """Process uploaded file and return DataFrame"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            elif file_extension == 'xlsx':
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'parquet':
                df = pd.read_parquet(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
            
            st.success(f"✅ Successfully loaded {len(df)} records with {len(df.columns)} features")
            return df
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return None
    
    def validate_dataset(self, df):
        """Validate dataset for common issues"""
        issues = []
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            issues.append(f"Found {missing_values} missing values")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            issues.append(f"Constant columns: {constant_cols}")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows")
        
        return issues
    
    def generate_synthetic_data(self, data_type='financial', n_samples=1000):
        """Generate synthetic datasets for testing"""
        try:
            if data_type == 'financial':
                dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
                data = {
                    'date': dates,
                    'amount': np.random.exponential(100, n_samples),
                    'transaction_type': np.random.choice(['debit', 'credit'], n_samples),
                    'balance': np.random.normal(5000, 2000, n_samples).cumsum(),
                    'category': np.random.choice(['food', 'shopping', 'transfer', 'salary'], n_samples)
                }
                return pd.DataFrame(data)
            
            elif data_type == 'healthcare':
                data = {
                    'patient_id': range(n_samples),
                    'age': np.random.randint(18, 80, n_samples),
                    'blood_pressure': np.random.normal(120, 20, n_samples),
                    'cholesterol': np.random.normal(200, 40, n_samples),
                    'glucose': np.random.normal(100, 20, n_samples),
                    'has_disease': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
                }
                return pd.DataFrame(data)
            
            elif data_type == 'iot':
                timestamps = pd.date_range(end=datetime.now(), periods=n_samples, freq='H')
                data = {
                    'timestamp': timestamps,
                    'temperature': np.random.normal(25, 5, n_samples),
                    'humidity': np.random.normal(50, 15, n_samples),
                    'pressure': np.random.normal(1013, 10, n_samples),
                    'vibration': np.random.exponential(1, n_samples)
                }
                return pd.DataFrame(data)
            
            else:
                # Generic multivariate data
                n_features = 10
                X = np.random.randn(n_samples, n_features)
                columns = [f'feature_{i}' for i in range(n_features)]
                return pd.DataFrame(X, columns=columns)
                
        except Exception as e:
            st.error(f"Error generating synthetic data: {e}")
            return None

# --- LIVE DATA FETCHER ---

class LiveDataFetcher:
    """Fetch live data from various sources for data poisoning analysis"""
    
    def __init__(self):
        self.sources = {
            'rbi': 'Reserve Bank of India',
            'nso': 'National Statistical Office, India',
            'india_ai': 'India AI Initiative',
            'kaggle': 'Kaggle Datasets',
            'yfinance': 'Financial Market Data',
            'government': 'Government Open Data'
        }
    
    def fetch_rbi_data(self, data_type='forex'):
        """Fetch RBI data (simulated - in production, use RBI API)"""
        try:
            # Simulating RBI data fetch - replace with actual RBI API calls
            if data_type == 'forex':
                # Forex reserves data
                dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                data = {
                    'Date': dates,
                    'Foreign_Currency_Assets': np.random.uniform(500, 600, 30),
                    'Gold_Reserves': np.random.uniform(40, 50, 30),
                    'SDRs': np.random.uniform(1, 2, 30),
                    'IMF_Reserve': np.random.uniform(5, 10, 30),
                    'Total_Reserves': np.random.uniform(550, 650, 30)
                }
                return pd.DataFrame(data)
            
            elif data_type == 'interest_rates':
                # Interest rates data
                dates = pd.date_range(end=datetime.now(), periods=12, freq='M')
                data = {
                    'Date': dates,
                    'Repo_Rate': np.random.uniform(6.0, 6.5, 12),
                    'Reverse_Repo_Rate': np.random.uniform(3.35, 3.65, 12),
                    'MSF_Rate': np.random.uniform(6.25, 6.75, 12),
                    'Bank_Rate': np.random.uniform(6.25, 6.75, 12)
                }
                return pd.DataFrame(data)
                
        except Exception as e:
            st.error(f"Error fetching RBI data: {e}")
            return None
    
    def fetch_nso_data(self, dataset='gdp'):
        """Fetch National Statistical Office data (simulated)"""
        try:
            if dataset == 'gdp':
                quarters = [f'Q{i} 2023' for i in range(1, 5)] + [f'Q{i} 2024' for i in range(1, 3)]
                data = {
                    'Quarter': quarters,
                    'GDP_Growth_Rate': np.random.uniform(6.0, 8.5, 6),
                    'Agriculture_Growth': np.random.uniform(2.0, 4.5, 6),
                    'Industry_Growth': np.random.uniform(5.0, 9.0, 6),
                    'Services_Growth': np.random.uniform(7.0, 10.0, 6),
                    'GVA_Growth': np.random.uniform(6.0, 8.0, 6)
                }
                return pd.DataFrame(data)
            
            elif dataset == 'inflation':
                months = pd.date_range(end=datetime.now(), periods=12, freq='M')
                data = {
                    'Month': months,
                    'CPI_Combined': np.random.uniform(4.0, 6.5, 12),
                    'CPI_Rural': np.random.uniform(4.2, 6.8, 12),
                    'CPI_Urban': np.random.uniform(3.8, 6.2, 12),
                    'Food_Inflation': np.random.uniform(3.5, 7.5, 12),
                    'Fuel_Inflation': np.random.uniform(2.5, 5.5, 12)
                }
                return pd.DataFrame(data)
                
        except Exception as e:
            st.error(f"Error fetching NSO data: {e}")
            return None

# --- UI COMPONENTS ---

def render_login():
    """Login screen for data poisoning defense platform"""
    st.markdown("""
    <div class="neuro-header">
        <h1 class="neuro-text" style="font-size: 4rem; margin: 0;">🧪 DATA POISONING DEFENSE</h1>
        <h3 class="hologram-text" style="font-size: 1.8rem; margin: 1rem 0;">
            Attack Simulation • Detection • Defense Mechanisms
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        with st.form("login_form"):
            st.markdown("### 🔐 SECURITY ACCESS")
            username = st.text_input("👤 Username:", placeholder="security_analyst")
            password = st.text_input("🔑 Password:", type="password", placeholder="••••••••")
            mfa_code = st.text_input("📱 MFA Code:", placeholder="123456")
            
            if st.form_submit_button("🚀 ACCESS DEFENSE PLATFORM", use_container_width=True):
                if username == "analyst" and password == "poison123" and mfa_code == "123456":
                    st.session_state.authenticated = True
                    st.session_state.login_time = datetime.now()
                    st.success("✅ Authentication Successful! Loading defense platform...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please check username, password, and MFA code.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 PLATFORM CAPABILITIES")
        
        st.write("🎯 **Attack Simulation**")
        st.write("• Label flipping attacks")
        st.write("• Feature manipulation")
        st.write("• Backdoor injection")
        st.write("• Data replication attacks")
        
        st.write("🔍 **Detection Methods**")
        st.write("• Anomaly detection algorithms")
        st.write("• Statistical analysis")
        st.write("• Real-time monitoring")
        st.write("• Pattern recognition")
        
        st.write("🛡️ **Defense Mechanisms**")
        st.write("• Data sanitization")
        st.write("• Robust training")
        st.write("• Federated learning")
        st.write("• Model security")

def render_data_poisoning_simulator():
    """Data poisoning attack simulation interface"""
    st.markdown("### 🧪 DATA POISONING ATTACK SIMULATOR")
    
    attacks = DataPoisoningAttacks()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🔥 ATTACK CONFIGURATION")
        
        attack_type = st.selectbox(
            "Select Attack Type:",
            list(attacks.attack_types.keys()),
            format_func=lambda x: f"{x.replace('_', ' ').title()} - {attacks.attack_types[x]}"
        )
        
        # Generate sample data
        if st.button("🔄 Generate Sample Dataset", key="gen_data"):
            with st.spinner("Generating sample dataset..."):
                # Create synthetic dataset
                n_samples = 1000
                n_features = 10
                X = np.random.randn(n_samples, n_features)
                y = np.random.randint(0, 3, n_samples)
                
                st.session_state.clean_data = X
                st.session_state.clean_labels = y
                st.success(f"✅ Generated dataset: {n_samples} samples, {n_features} features")
        
        if 'clean_data' in st.session_state:
            st.info(f"📊 Dataset loaded: {st.session_state.clean_data.shape[0]} samples, {st.session_state.clean_data.shape[1]} features")
            
            if st.button("🚀 Launch Poisoning Attack", key="launch_attack"):
                with st.spinner(f"Executing {attack_type} attack..."):
                    time.sleep(2)
                    
                    if attack_type == 'label_flipping':
                        poisoned_data, poisoned_labels, attack_indices = attacks.label_flipping_attack(
                            st.session_state.clean_data, st.session_state.clean_labels
                        )
                    elif attack_type == 'feature_manipulation':
                        poisoned_data, attack_indices = attacks.feature_manipulation_attack(
                            st.session_state.clean_data
                        )
                        poisoned_labels = st.session_state.clean_labels
                    else:
                        # Default attack
                        poisoned_data, poisoned_labels, attack_indices = attacks.label_flipping_attack(
                            st.session_state.clean_data, st.session_state.clean_labels
                        )
                    
                    st.session_state.poisoned_data = poisoned_data
                    st.session_state.poisoned_labels = poisoned_labels
                    st.session_state.attack_indices = attack_indices
                    
                    st.error(f"🎯 Attack Successful! Poisoned {len(attack_indices)} samples")
                    
                    # Show attack statistics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("📈 Total Samples", len(poisoned_data))
                    with col_b:
                        st.metric("☠️ Poisoned Samples", len(attack_indices))
                    with col_c:
                        poison_rate = (len(attack_indices) / len(poisoned_data)) * 100
                        st.metric("📊 Poison Rate", f"{poison_rate:.1f}%")
    
    with col2:
        st.markdown("#### 📈 ATTACK STATISTICS")
        
        if 'attack_indices' in st.session_state:
            st.metric("🔥 Active Attacks", "1")
            st.metric("🎯 Success Rate", "95%")
            st.metric("⏱️ Detection Time", "2.3s")
            
            st.markdown("#### 🎯 ATTACK PATTERNS")
            st.write("• Label manipulation")
            st.write("• Feature corruption")
            st.write("• Backdoor triggers")
            st.write("• Gradient poisoning")
        
        st.markdown("""
        <div class="explanation-box">
            <div class="explanation-title">🧪 DATA POISONING EXPLAINED</div>
            <p><strong>Data poisoning</strong> involves manipulating training data to compromise ML model performance.</p>
            
            <p><strong>Common Attack Vectors:</strong></p>
            <ul>
                <li>🔀 Label Flipping - Changing data labels</li>
                <li>📊 Feature Manipulation - Corrupting input features</li>
                <li>🚪 Backdoor Injection - Adding hidden triggers</li>
                <li>📈 Data Replication - Amplifying malicious samples</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_poisoning_detector():
    """Data poisoning detection interface"""
    st.markdown("### 🔍 DATA POISONING DETECTOR")
    
    detector = DataPoisoningDetector()
    
    tab1, tab2, tab3 = st.tabs(["🎯 Anomaly Detection", "📊 Statistical Analysis", "📈 Real-time Monitoring"])
    
    with tab1:
        st.markdown("#### 🎯 ANOMALY DETECTION ALGORITHMS")
        
        if 'poisoned_data' in st.session_state:
            data = st.session_state.poisoned_data
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🌲 Isolation Forest", key="iso_forest"):
                    with st.spinner("Running Isolation Forest..."):
                        anomalies = detector.detect_anomalies_isolation_forest(data)
                        st.session_state.detected_anomalies = anomalies
                        st.success(f"✅ Detected {len(anomalies)} anomalies")
            
            with col2:
                if st.button("🤖 One-Class SVM", key="one_class_svm"):
                    with st.spinner("Running One-Class SVM..."):
                        anomalies = detector.detect_anomalies_svm(data)
                        st.session_state.detected_anomalies = anomalies
                        st.success(f"✅ Detected {len(anomalies)} anomalies")
            
            with col3:
                if st.button("🔍 DBSCAN Clustering", key="dbscan"):
                    with st.spinner("Running DBSCAN..."):
                        anomalies = detector.detect_cluster_anomalies(data)
                        st.session_state.detected_anomalies = anomalies
                        st.success(f"✅ Detected {len(anomalies)} anomalies")
            
            if 'detected_anomalies' in st.session_state:
                st.markdown("#### 📋 DETECTION RESULTS")
                
                # Compare with actual attack indices
                actual_poisoned = set(st.session_state.attack_indices)
                detected_anomalies = set(st.session_state.detected_anomalies)
                
                true_positives = len(actual_poisoned.intersection(detected_anomalies))
                false_positives = len(detected_anomalies - actual_poisoned)
                false_negatives = len(actual_poisoned - detected_anomalies)
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("🎯 True Positives", true_positives)
                with col_b:
                    st.metric("🚫 False Positives", false_positives)
                with col_c:
                    st.metric("❌ False Negatives", false_negatives)
                with col_d:
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    st.metric("📊 Precision", f"{precision:.2f}")
        
        else:
            st.warning("⚠️ Please generate and poison a dataset first using the Attack Simulator")
    
    with tab2:
        st.markdown("#### 📊 STATISTICAL ANALYSIS")
        
        if 'poisoned_data' in st.session_state:
            if st.button("📈 Run Statistical Analysis", key="stat_analysis"):
                with st.spinner("Performing statistical analysis..."):
                    results = detector.statistical_analysis(st.session_state.poisoned_data)
                    
                    st.markdown("##### Z-Score Analysis")
                    z_anomalies = results['z_score_anomalies']
                    st.write(f"Detected {len(z_anomalies)} samples with |Z-score| > 3")
                    
                    st.markdown("##### Mahalanobis Distance")
                    m_anomalies = results['mahalanobis_anomalies']
                    st.write(f"Detected {len(m_anomalies)} outliers using Mahalanobis distance")
    
    with tab3:
        st.markdown("#### 📈 REAL-TIME DATA STREAM MONITORING")
        
        monitor = RealTimeDataMonitor()
        
        if st.button("🔴 Start Real-time Monitoring", key="start_monitor"):
            st.info("🔄 Monitoring data stream...")
            
            # Simulate real-time data stream
            for i in range(20):
                new_point = np.random.randn(10)  # 10 features
                result = monitor.monitor_data_stream(new_point)
                
                if result['anomaly_detected']:
                    st.markdown(f'<div class="poison-alert">🚨 ANOMALY DETECTED! {result["anomaly_count"]} anomalies in stream</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="defense-success">✅ Stream clean - {result["stream_size"]} points monitored</div>', unsafe_allow_html=True)
                
                time.sleep(0.5)

def render_defense_mechanisms():
    """Defense mechanisms against data poisoning"""
    st.markdown("### 🛡️ DEFENSE MECHANISMS")
    
    defense = DefenseMechanisms()
    
    tab1, tab2, tab3 = st.tabs(["🧹 Data Sanitization", "🎯 Robust Training", "🌐 Federated Learning"])
    
    with tab1:
        st.markdown("#### 🧹 DATA SANITIZATION")
        
        if 'detected_anomalies' in st.session_state and 'poisoned_data' in st.session_state:
            st.info(f"🔍 {len(st.session_state.detected_anomalies)} anomalies detected")
            
            if st.button("🧼 Sanitize Dataset", key="sanitize"):
                with st.spinner("Removing poisoned samples..."):
                    clean_data, clean_labels = defense.data_sanitization(
                        st.session_state.poisoned_data,
                        st.session_state.poisoned_labels,
                        st.session_state.detected_anomalies
                    )
                    
                    st.session_state.clean_data_sanitized = clean_data
                    st.session_state.clean_labels_sanitized = clean_labels
                    
                    st.success(f"✅ Dataset sanitized! Removed {len(st.session_state.detected_anomalies)} samples")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📊 Original Size", len(st.session_state.poisoned_data))
                    with col2:
                        st.metric("🧼 Sanitized Size", len(clean_data))
        
        else:
            st.warning("⚠️ Please run anomaly detection first")
    
    with tab2:
        st.markdown("#### 🎯 ROBUST TRAINING TECHNIQUES")
        
        robust_method = st.selectbox(
            "Select Robust Training Method:",
            ['trimmed_loss', 'differential_privacy'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if st.button("🛡️ Apply Robust Training", key="robust_train"):
            with st.spinner("Applying robust training techniques..."):
                if 'clean_data' in st.session_state:
                    robust_data, robust_labels = defense.robust_training(
                        st.session_state.clean_data,
                        st.session_state.clean_labels,
                        robust_method
                    )
                    
                    st.session_state.robust_data = robust_data
                    st.session_state.robust_labels = robust_labels
                    
                    st.success(f"✅ Applied {robust_method} robust training")
    
    with tab3:
        st.markdown("#### 🌐 FEDERATED LEARNING DEFENSE")
        
        st.info("Simulating federated learning with multiple clients")
        
        if st.button("🛡️ Test Federated Defense", key="federated_defense"):
            with st.spinner("Simulating federated learning scenario..."):
                # Simulate multiple clients with potentially poisoned data
                n_clients = 5
                client_data = []
                
                for i in range(n_clients):
                    # One client is malicious (poisoned data)
                    if i == 0:  # Malicious client
                        client_data.append(np.random.randn(10) + 2)  # Poisoned data
                    else:  # Honest clients
                        client_data.append(np.random.randn(10))
                
                # Test different aggregation methods
                methods = ['mean', 'median', 'trimmed_mean']
                results = {}
                
                for method in methods:
                    aggregated = defense.federated_learning_defense(client_data, method)
                    results[method] = np.linalg.norm(aggregated - np.mean(client_data[1:], axis=0))  # Distance from honest mean
                
                # Display results
                st.markdown("##### Aggregation Method Effectiveness")
                for method, distance in results.items():
                    st.write(f"**{method.title()}:** Distance from honest mean = {distance:.4f}")

def render_advanced_detection():
    """Advanced poisoning detection interface"""
    st.markdown("### 🔬 ADVANCED POISONING DETECTION")
    
    advanced_detector = AdvancedPoisoningDetector()
    
    tab1, tab2 = st.tabs(["🎯 Ensemble Detection", "⏰ Temporal Analysis"])
    
    with tab1:
        st.markdown("#### 🎯 ENSEMBLE DETECTION")
        
        if 'poisoned_data' in st.session_state:
            data = st.session_state.poisoned_data
            
            if st.button("🚀 Run Ensemble Detection", key="ensemble_detect"):
                with st.spinner("Running ensemble detection with multiple algorithms..."):
                    anomalies = advanced_detector.ensemble_detection(data)
                    st.session_state.ensemble_anomalies = anomalies
                    
                    st.success(f"✅ Ensemble detection complete! Found {len(anomalies)} anomalies")
                    
                    # Compare with known attacks
                    if 'attack_indices' in st.session_state:
                        actual_attacks = set(st.session_state.attack_indices)
                        detected = set(anomalies)
                        
                        tp = len(actual_attacks.intersection(detected))
                        fp = len(detected - actual_attacks)
                        fn = len(actual_attacks - detected)
                        
                        st.metric("🎯 True Positives", tp)
                        st.metric("🚫 False Positives", fp)
                        st.metric("❌ False Negatives", fn)
    
    with tab2:
        st.markdown("#### ⏰ TEMPORAL ANALYSIS")
        
        if 'poisoned_data' in st.session_state:
            # Generate timestamps for temporal analysis
            timestamps = pd.date_range(end=datetime.now(), periods=len(st.session_state.poisoned_data), freq='H')
            
            if st.button("🔍 Analyze Temporal Patterns", key="temporal_analysis"):
                with st.spinner("Analyzing temporal patterns for poisoning detection..."):
                    temporal_anomalies = advanced_detector.temporal_analysis(
                        st.session_state.poisoned_data, timestamps
                    )
                    
                    st.session_state.temporal_anomalies = temporal_anomalies
                    st.success(f"✅ Found {len(temporal_anomalies)} temporal anomalies")

def render_data_upload_center():
    """Data upload and processing center"""
    st.markdown("### 📁 DATA UPLOAD & PROCESSING CENTER")
    
    data_processor = DataProcessor()
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "🛠️ Data Validation", "🧪 Synthetic Data"])
    
    with tab1:
        st.markdown("#### 📤 UPLOAD YOUR DATASET")
        
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown("### 🗂️ DRAG & DROP YOUR DATA FILES")
        st.write("Supported formats: CSV, JSON, Excel, Parquet")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=data_processor.supported_formats,
            label_visibility="collapsed",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            st.info(f"📄 File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            
            if st.button("🚀 Process Uploaded File", key="process_upload"):
                with st.spinner("Processing uploaded file..."):
                    df = data_processor.process_uploaded_file(uploaded_file)
                    
                    if df is not None:
                        st.session_state.uploaded_data = df
                        st.session_state.uploaded_file_name = uploaded_file.name
                        
                        # Store as numpy arrays for compatibility with existing code
                        st.session_state.clean_data = df.select_dtypes(include=[np.number]).values
                        if len(st.session_state.clean_data) > 0:
                            st.session_state.clean_labels = np.random.randint(0, 3, len(st.session_state.clean_data))
                        
                        st.success("✅ Dataset ready for poisoning analysis!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("#### 🛠️ DATA VALIDATION")
        
        if 'uploaded_data' in st.session_state:
            df = st.session_state.uploaded_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("📊 Total Rows", len(df))
                st.metric("🎯 Features", len(df.columns))
            
            with col2:
                # Data types
                st.write("**Data Types:**")
                dtype_counts = df.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"• {dtype}: {count} columns")
            
            # Run validation
            if st.button("🔍 Run Data Validation", key="run_validation"):
                issues = data_processor.validate_dataset(df)
                
                if issues:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.warning("⚠️ Data Quality Issues Found:")
                    for issue in issues:
                        st.write(f"• {issue}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success("✅ No data quality issues found!")
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("📝 Please upload a dataset first to run validation")
    
    with tab3:
        st.markdown("#### 🧪 SYNTHETIC DATA GENERATOR")
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_type = st.selectbox(
                "Data Type:",
                ['financial', 'healthcare', 'iot', 'generic'],
                key='synth_data_type'
            )
            
            n_samples = st.slider(
                "Number of Samples:",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                key='n_samples'
            )
        
        with col2:
            if st.button("🎲 Generate Synthetic Data", key="generate_synthetic"):
                with st.spinner(f"Generating {data_type} dataset with {n_samples} samples..."):
                    synthetic_df = data_processor.generate_synthetic_data(data_type, n_samples)
                    
                    if synthetic_df is not None:
                        st.session_state.synthetic_data = synthetic_df
                        st.session_state.clean_data = synthetic_df.select_dtypes(include=[np.number]).values
                        if len(st.session_state.clean_data) > 0:
                            st.session_state.clean_labels = np.random.randint(0, 3, len(st.session_state.clean_data))
                        
                        st.success(f"✅ Generated {data_type} dataset with {n_samples} samples")
                        
                        # Show preview
                        st.dataframe(synthetic_df.head(), use_container_width=True)

def render_indian_data_sources():
    """Enhanced Indian data sources interface"""
    st.markdown("### 🇮🇳 INDIAN DATA SOURCES HUB")
    
    advanced_fetcher = AdvancedDataFetcher()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏫 Education Data", "🏥 Health & Nutrition", "🌾 Agriculture", "🤝 Social Welfare"])
    
    with tab1:
        st.markdown("#### 🏫 EDUCATION DATA (UDISE+)")
        
        if st.button("📊 Fetch Education Data", key="fetch_education"):
            with st.spinner("Fetching Unified District Information System for Education data..."):
                education_data = advanced_fetcher.fetch_udise_data()
                
                if education_data is not None:
                    st.session_state.education_data = education_data
                    st.success(f"✅ Fetched education data for {len(education_data)} states")
                    
                    st.dataframe(education_data, use_container_width=True)
                    
                    # Create visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = px.bar(education_data, x='State', y='Total_Schools', 
                                     title='Total Schools by State')
                        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        st.markdown("#### 🏥 HEALTH & NUTRITION DATA")
        
        if st.button("📈 Fetch Health Data", key="fetch_health"):
            with st.spinner("Fetching health and nutrition indicators..."):
                health_data = advanced_fetcher.fetch_health_data()
                
                if health_data is not None:
                    st.session_state.health_data = health_data
                    st.success(f"✅ Fetched {len(health_data)} health indicators")
    
    with tab3:
        st.markdown("#### 🌾 AGRICULTURE PRODUCTION DATA")
        
        if st.button("🌱 Fetch Agriculture Data", key="fetch_agriculture"):
            with st.spinner("Fetching agricultural production data..."):
                agri_data = advanced_fetcher.fetch_agriculture_data()
                
                if agri_data is not None:
                    st.session_state.agri_data = agri_data
                    st.success(f"✅ Fetched {len(agri_data)} crop production records")
    
    with tab4:
        st.markdown("#### 🤝 SOCIAL WELFARE SCHEMES")
        
        if st.button("📋 Fetch Social Welfare Data", key="fetch_welfare"):
            with st.spinner("Fetching social welfare scheme data..."):
                welfare_data = advanced_fetcher.fetch_social_welfare_data()
                
                if welfare_data is not None:
                    st.session_state.welfare_data = welfare_data
                    st.success(f"✅ Fetched data for {len(welfare_data)} welfare schemes")

def render_live_data_integration():
    """Live data integration from various sources"""
    st.markdown("### 🌐 LIVE DATA INTEGRATION")
    
    data_fetcher = LiveDataFetcher()
    
    tab1, tab2 = st.tabs(["🏦 RBI Data", "📊 NSO Statistics"])
    
    with tab1:
        st.markdown("#### 🏦 RESERVE BANK OF INDIA DATA")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            rbi_data_type = st.selectbox(
                "Select RBI Dataset:",
                ['forex', 'interest_rates'],
                format_func=lambda x: 'Foreign Exchange Reserves' if x == 'forex' else 'Interest Rates',
                key="rbi_data_type"
            )
            
            if st.button("🔄 Fetch RBI Data", key="fetch_rbi"):
                with st.spinner("Fetching latest RBI data..."):
                    rbi_data = data_fetcher.fetch_rbi_data(rbi_data_type)
                    
                    if rbi_data is not None:
                        st.session_state.rbi_data = rbi_data
                        st.success(f"✅ Fetched {len(rbi_data)} records from RBI")
                        
                        # Display data
                        st.dataframe(rbi_data, use_container_width=True)
                        
                        # Create visualization
                        if rbi_data_type == 'forex':
                            fig = px.line(rbi_data, x='Date', y='Total_Reserves', 
                                         title='India Total Forex Reserves (Simulated)')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            fig = px.line(rbi_data, x='Date', y='Repo_Rate', 
                                         title='RBI Repo Rate (Simulated)')
                            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### 📊 RBI DATA SOURCES")
            st.markdown('<div class="data-source-card">', unsafe_allow_html=True)
            st.markdown('<span class="live-data-badge">LIVE</span>', unsafe_allow_html=True)
            st.write("**Foreign Exchange Reserves**")
            st.write("• Total reserves")
            st.write("• Currency composition")
            st.write("• Monthly changes")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("#### 📊 NATIONAL STATISTICAL OFFICE DATA")
        
        nso_dataset = st.selectbox(
            "Select NSO Dataset:",
            ['gdp', 'inflation'],
            format_func=lambda x: 'GDP Growth Data' if x == 'gdp' else 'Inflation Data'
        )
        
        if st.button("🔄 Fetch NSO Data", key="fetch_nso"):
            with st.spinner("Fetching latest NSO statistics..."):
                nso_data = data_fetcher.fetch_nso_data(nso_dataset)
                
                if nso_data is not None:
                    st.session_state.nso_data = nso_data
                    st.success(f"✅ Fetched {len(nso_data)} records from NSO")
                    
                    st.dataframe(nso_data, use_container_width=True)
                    
                    if nso_dataset == 'gdp':
                        fig = px.bar(nso_data, x='Quarter', y='GDP_Growth_Rate',
                                    title='India GDP Growth Rate (Simulated)')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.line(nso_data, x='Month', y='CPI_Combined',
                                    title='Consumer Price Index - Combined (Simulated)')
                        st.plotly_chart(fig, use_container_width=True)

def render_system_monitor():
    """System monitoring for data poisoning defense"""
    st.markdown("### 💻 SYSTEM MONITORING")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("⚡ CPU Usage", f"{psutil.cpu_percent()}%")
        st.progress(psutil.cpu_percent() / 100)
    
    with col2:
        memory = psutil.virtual_memory()
        st.metric("💾 Memory Usage", f"{memory.percent}%")
        st.progress(memory.percent / 100)
    
    with col3:
        disk = psutil.disk_usage('/')
        st.metric("💽 Disk Usage", f"{disk.percent}%")
        st.progress(disk.percent / 100)
    
    with col4:
        st.metric("🖥️ Running Processes", len(psutil.pids()))
    
    # Real-time monitoring
    st.markdown("#### 📈 REAL-TIME DEFENSE METRICS")
    
    if st.button("🔄 Refresh Metrics", key="refresh_metrics"):
        st.rerun()
    
    # Simulate real-time data
    time_points = list(range(1, 11))
    attack_attempts = [random.randint(5, 20) for _ in time_points]
    blocked_attacks = [max(0, a - random.randint(0, 5)) for a in attack_attempts]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_points, y=attack_attempts, name='🎯 Attack Attempts', line=dict(color='#ff4444')))
    fig.add_trace(go.Scatter(x=time_points, y=blocked_attacks, name='🛡️ Blocked Attacks', line=dict(color='#00ff00')))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='white'),
        title="Real-time Attack Defense Monitoring"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN DASHBOARD ---

def render_main_dashboard():
    """Main data poisoning defense dashboard"""
    
    current_time = datetime.now()
    if 'login_time' in st.session_state:
        session_duration = current_time - st.session_state.login_time
        session_str = str(session_duration).split('.')[0]
    else:
        session_str = "0:00:00"
    
    st.markdown(f"""
    <div class="neuro-header">
        <h1 class="neuro-text" style="font-size: 4rem; margin: 0;">🧪 DATA POISONING DEFENSE</h1>
        <h3 class="hologram-text" style="font-size: 1.8rem; margin: 1rem 0;">
            Advanced ML Security • Real-time Protection • Threat Intelligence
        </h3>
        <p style="color: #00ffff; font-family: 'Exo 2'; font-size: 1.2rem;">
            🕒 Time: <strong>{current_time.strftime("%Y-%m-%d %H:%M:%S")}</strong> | 
            🔓 Session: <strong>{session_str}</strong> |
            🛡️ Status: <strong style="color: #00ff00;">OPERATIONAL</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Installation instructions in sidebar
    with st.sidebar:
        st.markdown("### 📦 Installation Guide")
        st.markdown("""
        **For enhanced features:**
        ```bash
        pip install yfinance
        ```
        
        **Current Status:**
        - yfinance: {'✅ Available' if YFINANCE_AVAILABLE else '❌ Not Installed'}
        - Kaggle: ❌ (Simulated - No API required)
        """)
    
    # Quick actions
    st.markdown("### 🚀 DEFENSE ACTIONS")
    cols = st.columns(7)
    
    with cols[0]:
        if st.button("🧪 Attack Sim", use_container_width=True, key="quick_attack"):
            st.session_state.current_tab = "Attack Simulator"
    
    with cols[1]:
        if st.button("🔍 Detection", use_container_width=True, key="quick_detect"):
            st.session_state.current_tab = "Poisoning Detector"
    
    with cols[2]:
        if st.button("🛡️ Defense", use_container_width=True, key="quick_defense"):
            st.session_state.current_tab = "Defense Mechanisms"
    
    with cols[3]:
        if st.button("🔬 Advanced", use_container_width=True, key="quick_advanced"):
            st.session_state.current_tab = "Advanced Detection"
    
    with cols[4]:
        if st.button("📁 Data Hub", use_container_width=True, key="quick_data_hub"):
            st.session_state.current_tab = "Data Upload Center"
    
    with cols[5]:
        if st.button("🇮🇳 India Data", use_container_width=True, key="quick_india"):
            st.session_state.current_tab = "Indian Data Sources"
    
    with cols[6]:
        if st.button("🔒 Logout", use_container_width=True, key="quick_logout"):
            st.session_state.authenticated = False
            st.rerun()
    
    # Main tabs
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Attack Simulator"
    
    tabs = st.tabs([
        "🧪 Attack Simulator", 
        "🔍 Poisoning Detector", 
        "🛡️ Defense Mechanisms",
        "🔬 Advanced Detection",
        "📁 Data Upload Center", 
        "🇮🇳 Indian Data Sources",
        "📊 Live Data Integration",
        "💻 System Monitor"
    ])
    
    with tabs[0]:
        render_data_poisoning_simulator()
    
    with tabs[1]:
        render_poisoning_detector()
    
    with tabs[2]:
        render_defense_mechanisms()
    
    with tabs[3]:
        render_advanced_detection()
    
    with tabs[4]:
        render_data_upload_center()
    
    with tabs[5]:
        render_indian_data_sources()
    
    with tabs[6]:
        render_live_data_integration()
    
    with tabs[7]:
        render_system_monitor()

# --- MAIN APPLICATION ---

def main():
    with quantum_resource_manager():
        # Authentication
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        
        if not st.session_state.authenticated:
            render_login()
        else:
            render_main_dashboard()

if __name__ == "__main__":
    main()
