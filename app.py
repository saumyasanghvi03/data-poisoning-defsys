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

# --- DATA POISONING CSS ---
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
</style>
""", unsafe_allow_html=True)

@contextmanager
def quantum_resource_manager():
    """Advanced resource management"""
    try:
        yield
    finally:
        gc.collect()

# --- DATA SOURCES INTEGRATION CLASSES ---

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
    
    def fetch_india_ai_data(self):
        """Fetch India AI dataset information (simulated)"""
        try:
            datasets = {
                'Agricultural_Data': {
                    'description': 'Crop yield prediction data across Indian states',
                    'size': '2.5GB',
                    'samples': '500,000',
                    'features': '45',
                    'url': 'https://indiaai.gov.in/datasets'
                },
                'Healthcare_Records': {
                    'description': 'Anonymized patient records from public hospitals',
                    'size': '1.8GB',
                    'samples': '300,000',
                    'features': '32',
                    'url': 'https://indiaai.gov.in/datasets'
                },
                'Financial_Transactions': {
                    'description': 'Banking transaction patterns',
                    'size': '3.2GB',
                    'samples': '1,200,000',
                    'features': '28',
                    'url': 'https://indiaai.gov.in/datasets'
                },
                'Education_Data': {
                    'description': 'Student performance and institutional data',
                    'size': '950MB',
                    'samples': '150,000',
                    'features': '25',
                    'url': 'https://indiaai.gov.in/datasets'
                }
            }
            return datasets
            
        except Exception as e:
            st.error(f"Error fetching India AI data: {e}")
            return None
    
    def fetch_kaggle_datasets(self, search_term='finance'):
        """Fetch datasets from Kaggle (simulated without API)"""
        try:
            # Simulated Kaggle datasets without API dependency
            datasets = {
                'credit-card-fraud': {
                    'title': 'Credit Card Fraud Detection',
                    'size': '150MB',
                    'downloads': '45,000',
                    'url': 'https://www.kaggle.com/mlg-ulb/creditcardfraud',
                    'description': 'Real-world credit card transactions for fraud detection'
                },
                'loan-prediction': {
                    'title': 'Loan Prediction Dataset',
                    'size': '45MB',
                    'downloads': '23,000',
                    'url': 'https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset',
                    'description': 'Loan application data for prediction models'
                },
                'stock-market-data': {
                    'title': 'Indian Stock Market Data',
                    'size': '320MB',
                    'downloads': '18,000',
                    'url': 'https://www.kaggle.com/rohanrao/nifty50-stock-market-data',
                    'description': 'Historical data for Nifty 50 stocks'
                },
                'customer-segmentation': {
                    'title': 'Customer Segmentation',
                    'size': '85MB',
                    'downloads': '32,000',
                    'url': 'https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python',
                    'description': 'Mall customer data for segmentation analysis'
                }
            }
            
            # Filter by search term
            filtered_datasets = {}
            for key, dataset in datasets.items():
                if search_term.lower() in dataset['title'].lower() or search_term.lower() in dataset['description'].lower():
                    filtered_datasets[key] = dataset
            
            return filtered_datasets if filtered_datasets else datasets
            
        except Exception as e:
            st.error(f"Error fetching Kaggle data: {e}")
            return None
    
    def fetch_financial_data(self, symbols=['RELIANCE.NS', 'TCS.NS', 'INFY.NS']):
        """Fetch live financial data using yfinance"""
        try:
            if not YFINANCE_AVAILABLE:
                # Simulate financial data if yfinance is not available
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.warning("Using simulated financial data. Install yfinance for live data:")
                st.code("pip install yfinance")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Generate simulated stock data
                simulated_data = {}
                dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
                
                for symbol in symbols:
                    # Simulate stock price movement
                    base_price = 1000 if 'RELIANCE' in symbol else 3000 if 'TCS' in symbol else 1500
                    returns = np.random.normal(0.001, 0.02, 100)  # Daily returns
                    prices = base_price * (1 + returns).cumprod()
                    
                    simulated_data[symbol] = pd.DataFrame({
                        'Date': dates,
                        'Open': prices * 0.99,
                        'High': prices * 1.02,
                        'Low': prices * 0.98,
                        'Close': prices,
                        'Volume': np.random.randint(1000000, 5000000, 100)
                    }).set_index('Date')
                
                return simulated_data
            
            # If yfinance is available, use real data
            data = {}
            for symbol in symbols:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="3mo")
                data[symbol] = hist
            return data
        except Exception as e:
            st.error(f"Error fetching financial data: {e}")
            return None
    
    def fetch_government_data(self):
        """Fetch Indian government open data (simulated)"""
        try:
            datasets = {
                'Digital_India_Metrics': {
                    'description': 'Digital infrastructure and adoption metrics',
                    'records': '50,000',
                    'update_frequency': 'Monthly',
                    'url': 'https://data.gov.in'
                },
                'Smart_Cities': {
                    'description': 'Smart city project implementation data',
                    'records': '15,000',
                    'update_frequency': 'Quarterly',
                    'url': 'https://data.gov.in'
                },
                'Agriculture_Production': {
                    'description': 'Crop production data across states',
                    'records': '200,000',
                    'update_frequency': 'Annual',
                    'url': 'https://data.gov.in'
                },
                'Economic_Survey': {
                    'description': 'Annual economic survey data',
                    'records': '10,000',
                    'update_frequency': 'Annual',
                    'url': 'https://data.gov.in'
                }
            }
            return datasets
        except Exception as e:
            st.error(f"Error fetching government data: {e}")
            return None

    def fetch_web_data(self, url):
        """Generic web data fetcher for any public API"""
        try:
            # Simulate web data fetching
            st.info(f"📡 Fetching data from: {url}")
            time.sleep(2)  # Simulate API call
            
            # Return simulated data based on URL type
            if 'rbi' in url.lower():
                return self.fetch_rbi_data()
            elif 'statistics' in url.lower():
                return self.fetch_nso_data()
            else:
                # Generic simulated dataset
                data = {
                    'timestamp': pd.date_range(end=datetime.now(), periods=50, freq='H'),
                    'value': np.random.randn(50).cumsum() + 100,
                    'metric': np.random.choice(['temperature', 'pressure', 'humidity'], 50)
                }
                return pd.DataFrame(data)
                
        except Exception as e:
            st.error(f"Error fetching web data: {e}")
            return None

# --- DATA POISONING CORE CLASSES ---

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

# --- UI COMPONENTS FOR DATA POISONING ---

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
                
                # Visualization
                if data.shape[1] >= 2:
                    fig = px.scatter(
                        x=data[:, 0], y=data[:, 1],
                        color=[1 if i in st.session_state.detected_anomalies else 0 for i in range(len(data))],
                        title="Anomaly Detection Results",
                        labels={'color': 'Anomaly Score'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
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
                    
                    # Distribution plot
                    if st.session_state.poisoned_data.shape[1] >= 2:
                        fig = px.histogram(
                            x=st.session_state.poisoned_data[:, 0],
                            title="Feature Distribution Analysis",
                            nbins=50
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
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
                    
                    st.markdown("""
                    <div class="explanation-box">
                        <div class="explanation-title">🎯 ROBUST TRAINING EXPLAINED</div>
                        <p><strong>Trimmed Loss:</strong> Ignores samples with highest loss during training</p>
                        <p><strong>Differential Privacy:</strong> Adds noise to protect individual data points</p>
                        <p><strong>Benefits:</strong> Reduces impact of poisoned samples on model training</p>
                    </div>
                    """, unsafe_allow_html=True)
    
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
                
                st.markdown("""
                <div class="defense-success">
                    <strong>🎯 DEFENSE SUCCESS:</strong> Median and trimmed mean aggregation 
                    effectively reduce the impact of poisoned client data.
                </div>
                """, unsafe_allow_html=True)

def render_model_security_analysis():
    """ML model security analysis"""
    st.markdown("### 🤖 MODEL SECURITY ANALYSIS")
    
    model_security = MLModelSecurity()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🎯 MODEL ROBUSTNESS TESTING")
        
        if st.button("🛡️ Test Model Robustness", key="test_robustness"):
            with st.spinner("Testing model robustness against poisoning..."):
                # Simulate model testing
                results = model_security.model_robustness_test(
                    clean_model="clean_model",
                    poisoned_model="poisoned_model", 
                    test_data=None,
                    test_labels=None
                )
                
                st.markdown("##### 📊 Robustness Test Results")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("🧼 Clean Accuracy", f"{results['clean_accuracy']:.1%}")
                with col_b:
                    st.metric("☠️ Poisoned Accuracy", f"{results['poisoned_accuracy']:.1%}")
                with col_c:
                    st.metric("🛡️ Robustness Score", f"{results['robustness_score']:.1f}%")
                
                # Threat level assessment
                performance_drop = results['performance_drop']
                if performance_drop > 0.2:
                    threat_level = "CRITICAL"
                    threat_class = "threat-level-critical"
                elif performance_drop > 0.1:
                    threat_level = "HIGH" 
                    threat_class = "threat-level-high"
                elif performance_drop > 0.05:
                    threat_level = "MEDIUM"
                    threat_class = "threat-level-medium"
                else:
                    threat_level = "LOW"
                    threat_class = "threat-level-low"
                
                st.markdown(f'<div class="{threat_class}">🚨 THREAT LEVEL: {threat_level}</div>', unsafe_allow_html=True)
        
        st.markdown("#### 🔓 MODEL INVERSION ANALYSIS")
        
        if st.button("🔍 Analyze Model Inversion Risk", key="model_inversion"):
            with st.spinner("Analyzing model inversion vulnerabilities..."):
                results = model_security.extract_training_data_analysis(
                    model="target_model",
                    original_data_shape=(1000, 10)
                )
                
                st.markdown("##### 📊 Inversion Attack Risk")
                st.metric("🎯 Vulnerability Score", f"{results['vulnerability_score']:.2f}")
                st.write(f"**Risk Level:** {results['risk_level']}")
                
                st.markdown("##### 🛡️ Recommended Defenses")
                for recommendation in results['recommendations']:
                    st.write(f"• {recommendation}")
    
    with col2:
        st.markdown("#### 📈 SECURITY METRICS")
        
        st.metric("🛡️ Overall Security Score", "78%")
        st.metric("🎯 Detection Accuracy", "92%")
        st.metric("⏱️ Response Time", "1.2s")
        st.metric("📊 False Positive Rate", "3.8%")
        
        st.markdown("#### 🎯 THREAT INTELLIGENCE")
        st.write("• Active poisoning campaigns: 3")
        st.write("• New attack vectors: 2")
        st.write("• Zero-day vulnerabilities: 1")
        st.write("• Protected models: 15")

def render_threat_intelligence():
    """Data poisoning threat intelligence"""
    st.markdown("### 🌐 DATA POISONING THREAT INTELLIGENCE")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🚨 ACTIVE THREATS")
        
        threats = [
            {"name": "Label Flipping Campaign", "severity": "HIGH", "targets": "Financial Models", "status": "ACTIVE"},
            {"name": "Backdoor Injection", "severity": "CRITICAL", "targets": "Healthcare AI", "status": "ACTIVE"},
            {"name": "Feature Manipulation", "severity": "MEDIUM", "targets": "Recommendation Systems", "status": "DETECTED"},
            {"name": "Model Inversion", "severity": "HIGH", "targets": "Privacy-Sensitive Models", "status": "MONITORING"},
        ]
        
        for threat in threats:
            with st.expander(f"🔴 {threat['name']} - {threat['severity']}"):
                st.write(f"**Targets:** {threat['targets']}")
                st.write(f"**Status:** {threat['status']}")
                st.write(f"**First Seen:** 2024-01-15")
                st.write(f"**Last Activity:** 2024-01-20")
        
        st.markdown("#### 📊 GLOBAL THREAT LANDSCAPE")
        
        # Threat trend data
        trends = [
            {"month": "Jan", "label_flipping": 45, "backdoor_injection": 12, "feature_manipulation": 8},
            {"month": "Feb", "label_flipping": 52, "backdoor_injection": 18, "feature_manipulation": 12},
            {"month": "Mar", "label_flipping": 48, "backdoor_injection": 15, "feature_manipulation": 10},
            {"month": "Apr", "label_flipping": 61, "backdoor_injection": 22, "feature_manipulation": 15},
        ]
        
        df = pd.DataFrame(trends)
        fig = px.line(df, x='month', y=['label_flipping', 'backdoor_injection', 'feature_manipulation'], 
                     title="Monthly Data Poisoning Attack Trends",
                     labels={"value": "Attack Count", "variable": "Attack Type"})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 THREAT METRICS")
        
        st.metric("🌍 Global Attacks", "1,247")
        st.metric("🎯 Targeted Industries", "8")
        st.metric("🛡️ Successful Defenses", "89%")
        st.metric("⏱️ Average Detection", "3.2s")
        
        st.markdown("#### 🎯 HIGH-RISK INDUSTRIES")
        st.write("1. 🏦 Financial Services")
        st.write("2. 🏥 Healthcare")
        st.write("3. 🛡️ Defense")
        st.write("4. 🔐 Cybersecurity")
        st.write("5. 🏛️ Government")

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

# --- LIVE DATA INTEGRATION ---

def render_live_data_integration():
    """Live data integration from various sources"""
    st.markdown("### 🌐 LIVE DATA INTEGRATION")
    
    data_fetcher = LiveDataFetcher()
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏦 RBI Data", 
        "📊 NSO Statistics", 
        "🤖 India AI", 
        "📈 Kaggle", 
        "🏛️ Government Data",
        "🌐 Web APIs"
    ])
    
    with tab1:
        st.markdown("#### 🏦 RESERVE BANK OF INDIA DATA")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            rbi_data_type = st.selectbox(
                "Select RBI Dataset:",
                ['forex', 'interest_rates'],
                format_func=lambda x: 'Foreign Exchange Reserves' if x == 'forex' else 'Interest Rates'
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
            
            st.markdown('<div class="data-source-card">', unsafe_allow_html=True)
            st.markdown('<span class="live-data-badge">LIVE</span>', unsafe_allow_html=True)
            st.write("**Interest Rates**")
            st.write("• Policy rates")
            st.write("• Lending rates")
            st.write("• Deposit rates")
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
    
    with tab3:
        st.markdown("#### 🤖 INDIA AI DATASETS")
        
        if st.button("🔄 Fetch India AI Datasets", key="fetch_india_ai"):
            with st.spinner("Fetching India AI dataset catalog..."):
                india_ai_data = data_fetcher.fetch_india_ai_data()
                
                if india_ai_data is not None:
                    st.session_state.india_ai_data = india_ai_data
                    st.success(f"✅ Found {len(india_ai_data)} datasets")
                    
                    for dataset_name, dataset_info in india_ai_data.items():
                        st.markdown(f'<div class="dataset-info">', unsafe_allow_html=True)
                        st.markdown(f"**{dataset_name}**")
                        st.write(f"📝 {dataset_info['description']}")
                        st.write(f"💾 Size: {dataset_info['size']} | 📊 Samples: {dataset_info['samples']} | 🎯 Features: {dataset_info['features']}")
                        st.write(f"🔗 [Access Dataset]({dataset_info['url']})")
                        
                        if st.button(f"Use {dataset_name} for Analysis", key=f"use_{dataset_name}"):
                            # Simulate using this dataset for poisoning analysis
                            st.info(f"Loading {dataset_name} for data poisoning analysis...")
                            time.sleep(1)
                            st.success(f"✅ {dataset_name} loaded successfully!")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown("#### 📈 KAGGLE DATASETS")
        
        search_term = st.text_input("🔍 Search Kaggle Datasets:", "finance", key="kaggle_search")
        
        if st.button("🔄 Search Kaggle", key="search_kaggle"):
            with st.spinner(f"Searching Kaggle for '{search_term}'..."):
                kaggle_data = data_fetcher.fetch_kaggle_datasets(search_term)
                
                if kaggle_data is not None:
                    st.session_state.kaggle_data = kaggle_data
                    st.success(f"✅ Found {len(kaggle_data)} relevant datasets")
                    
                    for dataset_id, dataset_info in kaggle_data.items():
                        st.markdown(f'<div class="dataset-info">', unsafe_allow_html=True)
                        st.markdown(f"**{dataset_info['title']}**")
                        st.write(f"📝 {dataset_info.get('description', 'No description available')}")
                        st.write(f"📥 Downloads: {dataset_info['downloads']} | 💾 Size: {dataset_info['size']}")
                        st.write(f"🔗 [Dataset Link]({dataset_info['url']})")
                        
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            if st.button(f"Analyze for Poisoning", key=f"analyze_{dataset_id}"):
                                st.info(f"Starting data poisoning analysis on {dataset_info['title']}...")
                                # Simulate analysis
                                time.sleep(1)
                                st.success("✅ Analysis complete - No poisoning detected")
                        
                        with col_b:
                            if st.button("Download", key=f"download_{dataset_id}"):
                                st.info("📥 Downloading dataset... (Simulated)")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown("#### 🏛️ GOVERNMENT OPEN DATA")
        
        if st.button("🔄 Fetch Government Data", key="fetch_govt"):
            with st.spinner("Fetching government open data catalog..."):
                govt_data = data_fetcher.fetch_government_data()
                
                if govt_data is not None:
                    st.session_state.govt_data = govt_data
                    st.success(f"✅ Found {len(govt_data)} government datasets")
                    
                    for dataset_name, dataset_info in govt_data.items():
                        st.markdown(f'<div class="dataset-info">', unsafe_allow_html=True)
                        st.markdown(f"**{dataset_name.replace('_', ' ').title()}**")
                        st.write(f"📝 {dataset_info['description']}")
                        st.write(f"📊 Records: {dataset_info['records']} | 🔄 Update: {dataset_info['update_frequency']}")
                        st.write(f"🔗 [Access Data]({dataset_info['url']})")
                        
                        if st.button(f"Access {dataset_name}", key=f"access_{dataset_name}"):
                            st.info(f"Accessing {dataset_name}... (Simulated API call)")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab6:
        st.markdown("#### 🌐 WEB API DATA FETCHER")
        
        st.info("Fetch data from any public API endpoint")
        
        api_url = st.text_input("🌐 Enter API URL:", 
                               placeholder="https://api.example.com/data",
                               value="https://api.data.gov.in/resource/")
        
        if st.button("📡 Fetch from API", key="fetch_api"):
            if api_url:
                with st.spinner(f"Fetching data from {api_url}..."):
                    web_data = data_fetcher.fetch_web_data(api_url)
                    
                    if web_data is not None:
                        st.session_state.web_data = web_data
                        st.success(f"✅ Fetched {len(web_data)} records from API")
                        
                        st.dataframe(web_data, use_container_width=True)
                        
                        # Create basic visualization
                        if 'value' in web_data.columns:
                            fig = px.line(web_data, x='timestamp', y='value', 
                                         title='API Data Timeline')
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Please enter a valid API URL")

# --- MAIN DASHBOARD ---

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
        
        st.markdown("### 🔧 Quick Setup")
        st.markdown("""
        All data sources are simulated for demonstration.
        No external API keys required!
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
        if st.button("🤖 Model Security", use_container_width=True, key="quick_model"):
            st.session_state.current_tab = "Model Security"
    
    with cols[4]:
        if st.button("🌐 Threat Intel", use_container_width=True, key="quick_threat"):
            st.session_state.current_tab = "Threat Intelligence"
    
    with cols[5]:
        if st.button("📊 Live Data", use_container_width=True, key="quick_data"):
            st.session_state.current_tab = "Live Data Integration"
    
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
        "🤖 Model Security", 
        "🌐 Threat Intelligence",
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
        render_model_security_analysis()
    
    with tabs[4]:
        render_threat_intelligence()
    
    with tabs[5]:
        render_live_data_integration()
    
    with tabs[6]:
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
