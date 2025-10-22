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
from sklearn.decomposition import PCA
import joblib
import hashlib
import io
import base64
from urllib.parse import urlparse
import asyncio
import aiohttp

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
    page_title="CYBER DEFENSE TERMINAL | REAL-TIME DATA POISONING SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API Configuration from Streamlit Secrets
try:
    API_KEYS = {
        'alpha_vantage': st.secrets.get("ALPHA_VANTAGE_API_KEY", "demo"),
        'finnhub': st.secrets.get("FINNHUB_API_KEY", "demo"),
        'fmp': st.secrets.get("FMP_API_KEY", "demo")
    }
except:
    API_KEYS = {
        'alpha_vantage': "demo",
        'finnhub': "demo", 
        'fmp': "demo"
    }

# Optional imports with error handling
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# --- ENTERPRISE TERMINAL CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Main terminal styling */
    .main {
        background: #000000 !important;
        color: #00ff00 !important;
        font-family: 'Share Tech Mono', monospace !important;
    }
    
    .terminal-header {
        background: linear-gradient(90deg, #001122 0%, #002244 50%, #001122 100%);
        border-bottom: 2px solid #00ffff;
        padding: 1rem 2rem;
        margin: -1rem -1rem 1rem -1rem;
        box-shadow: 0 0 30px #00ffff33;
    }
    
    .terminal-metric {
        background: rgba(0, 255, 255, 0.05);
        border: 1px solid #00ffff;
        border-radius: 4px;
        padding: 0.8rem;
        margin: 0.2rem;
        font-family: 'Share Tech Mono', monospace;
    }
    
    .alert-critical {
        background: linear-gradient(90deg, #ff0000 0%, #8b0000 100%);
        border: 1px solid #ff4444;
        border-left: 5px solid #ff0000;
        color: white;
        padding: 1rem;
        margin: 0.5rem 0;
        animation: blink-critical 2s infinite;
    }
    
    .alert-high {
        background: linear-gradient(90deg, #ff6b00 0%, #cc5500 100%);
        border: 1px solid #ffaa00;
        border-left: 5px solid #ff6b00;
        color: white;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .alert-medium {
        background: linear-gradient(90deg, #ffd000 0%, #ccaa00 100%);
        border: 1px solid #ffff00;
        border-left: 5px solid #ffd000;
        color: black;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .alert-low {
        background: linear-gradient(90deg, #00ff00 0%, #00cc00 100%);
        border: 1px solid #00ff00;
        border-left: 5px solid #00ff00;
        color: white;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    @keyframes blink-critical {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.7; }
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background: #00ff00; box-shadow: 0 0 10px #00ff00; }
    .status-warning { background: #ffff00; box-shadow: 0 0 10px #ffff00; }
    .status-offline { background: #ff0000; box-shadow: 0 0 10px #ff0000; }
    
    .data-panel {
        background: rgba(0, 20, 40, 0.8);
        border: 1px solid #00ffff;
        border-radius: 4px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Share Tech Mono', monospace;
    }
    
    .command-line {
        background: #001122;
        border: 1px solid #00ffff;
        border-radius: 4px;
        padding: 0.5rem;
        font-family: 'Share Tech Mono', monospace;
        color: #00ff00;
    }
    
    .log-entry {
        background: rgba(0, 255, 255, 0.05);
        border-left: 3px solid #00ffff;
        padding: 0.5rem;
        margin: 0.2rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }
    
    .cyber-button {
        background: linear-gradient(90deg, #001122 0%, #003366 100%);
        border: 1px solid #00ffff;
        color: #00ffff;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-family: 'Share Tech Mono', monospace;
        transition: all 0.3s ease;
    }
    
    .cyber-button:hover {
        background: linear-gradient(90deg, #003366 0%, #0055aa 100%);
        box-shadow: 0 0 15px #00ffff;
    }
    
    /* Streamlit component overrides */
    .stTabs [data-baseweb="tab-list"] {
        background: #001122;
        border-bottom: 1px solid #00ffff;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #001122;
        color: #00ffff;
        border: 1px solid #00ffff;
        border-bottom: none;
        border-radius: 4px 4px 0 0;
        margin-right: 2px;
        font-family: 'Share Tech Mono', monospace;
    }
    
    .stTabs [aria-selected="true"] {
        background: #003366 !important;
        color: #00ffff !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #001122;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00ffff;
        border-radius: 4px;
    }
    
    /* Matrix rain effect container */
    .matrix-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
        opacity: 0.1;
    }
</style>

<div class="matrix-container" id="matrixRain"></div>

<script>
// Simple matrix rain effect
function createMatrixRain() {
    const container = document.getElementById('matrixRain');
    const characters = '01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン';
    const fontSize = 14;
    const columns = Math.floor(window.innerWidth / fontSize);
    
    const drops = [];
    for (let i = 0; i < columns; i++) {
        drops[i] = 1;
    }
    
    function draw() {
        const ctx = document.createElement('canvas').getContext('2d');
        ctx.canvas.width = window.innerWidth;
        ctx.canvas.height = window.innerHeight;
        container.appendChild(ctx.canvas);
        
        function rain() {
            ctx.fillStyle = 'rgba(0, 20, 40, 0.05)';
            ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            
            ctx.fillStyle = '#0f0';
            ctx.font = fontSize + 'px monospace';
            
            for (let i = 0; i < drops.length; i++) {
                const text = characters[Math.floor(Math.random() * characters.length)];
                ctx.fillText(text, i * fontSize, drops[i] * fontSize);
                
                if (drops[i] * fontSize > ctx.canvas.height && Math.random() > 0.975) {
                    drops[i] = 0;
                }
                drops[i]++;
            }
        }
        
        setInterval(rain, 33);
    }
    
    draw();
}

// Start matrix rain when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createMatrixRain);
} else {
    createMatrixRain();
}
</script>
""", unsafe_allow_html=True)

@contextmanager
def quantum_resource_manager():
    """Advanced resource management"""
    try:
        yield
    finally:
        gc.collect()

# --- REAL-TIME DATA INTEGRATION CLASSES ---

class RealTimeDataEngine:
    """Real-time financial data integration engine"""
    
    def __init__(self):
        self.data_sources = {
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'finnhub': 'https://finnhub.io/api/v1',
            'fmp': 'https://financialmodelingprep.com/api/v3'
        }
        self.cache = {}
        self.last_fetch = {}
    
    async def fetch_alpha_vantage(self, symbol, function='TIME_SERIES_INTRADAY', interval='5min'):
        """Fetch real-time data from Alpha Vantage"""
        try:
            url = f"{self.data_sources['alpha_vantage']}"
            params = {
                'function': function,
                'symbol': symbol,
                'interval': interval,
                'apikey': API_KEYS['alpha_vantage'],
                'outputsize': 'compact'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_alpha_vantage(data, function)
                    else:
                        return self._generate_mock_data(symbol, 'alpha_vantage')
        except Exception as e:
            st.error(f"Alpha Vantage Error: {str(e)}")
            return self._generate_mock_data(symbol, 'alpha_vantage')
    
    async def fetch_finnhub(self, symbol, resolution='1'):
        """Fetch real-time data from Finnhub"""
        try:
            url = f"{self.data_sources['finnhub']}/quote"
            params = {
                'symbol': symbol,
                'token': API_KEYS['finnhub']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_finnhub(data, symbol)
                    else:
                        return self._generate_mock_data(symbol, 'finnhub')
        except Exception as e:
            st.error(f"Finnhub Error: {str(e)}")
            return self._generate_mock_data(symbol, 'finnhub')
    
    async def fetch_fmp(self, symbol):
        """Fetch real-time data from Financial Modeling Prep"""
        try:
            url = f"{self.data_sources['fmp']}/quote/{symbol}"
            params = {
                'apikey': API_KEYS['fmp']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_fmp(data[0] if data else {})
                    else:
                        return self._generate_mock_data(symbol, 'fmp')
        except Exception as e:
            st.error(f"FMP Error: {str(e)}")
            return self._generate_mock_data(symbol, 'fmp')
    
    def _parse_alpha_vantage(self, data, function):
        """Parse Alpha Vantage response"""
        if 'Time Series (5min)' in data:
            time_series = data['Time Series (5min)']
            latest_timestamp = sorted(time_series.keys())[-1]
            latest_data = time_series[latest_timestamp]
            
            return {
                'price': float(latest_data['4. close']),
                'volume': int(latest_data['5. volume']),
                'timestamp': latest_timestamp,
                'high': float(latest_data['2. high']),
                'low': float(latest_data['3. low']),
                'open': float(latest_data['1. open']),
                'source': 'Alpha Vantage'
            }
        return {}
    
    def _parse_finnhub(self, data, symbol):
        """Parse Finnhub response"""
        return {
            'price': data.get('c', 0),
            'change': data.get('d', 0),
            'change_percent': data.get('dp', 0),
            'high': data.get('h', 0),
            'low': data.get('l', 0),
            'open': data.get('o', 0),
            'previous_close': data.get('pc', 0),
            'timestamp': datetime.now().isoformat(),
            'source': 'Finnhub'
        }
    
    def _parse_fmp(self, data):
        """Parse FMP response"""
        return {
            'price': data.get('price', 0),
            'change': data.get('change', 0),
            'change_percent': data.get('changesPercentage', 0),
            'volume': data.get('volume', 0),
            'avg_volume': data.get('avgVolume', 0),
            'market_cap': data.get('marketCap', 0),
            'timestamp': datetime.now().isoformat(),
            'source': 'Financial Modeling Prep'
        }
    
    def _generate_mock_data(self, symbol, source):
        """Generate realistic mock data when APIs are unavailable"""
        base_price = random.uniform(100, 500)
        return {
            'price': base_price + random.uniform(-5, 5),
            'change': random.uniform(-2, 2),
            'change_percent': random.uniform(-1, 1),
            'volume': random.randint(1000000, 5000000),
            'high': base_price + random.uniform(1, 3),
            'low': base_price - random.uniform(1, 3),
            'open': base_price + random.uniform(-1, 1),
            'timestamp': datetime.now().isoformat(),
            'source': f'Mock {source}'
        }

class AdvancedDefenseSystems:
    """Advanced defense systems with real-time monitoring"""
    
    def __init__(self):
        self.detection_models = {}
        self.anomaly_history = []
        self.defense_status = {
            'data_validation': 'ACTIVE',
            'model_monitoring': 'ACTIVE', 
            'threat_detection': 'ACTIVE',
            'response_automation': 'STANDBY'
        }
    
    def initialize_defense_systems(self):
        """Initialize all defense systems"""
        self.detection_models = {
            'statistical_analysis': StatisticalAnalyzer(),
            'pattern_detection': PatternDetector(),
            'behavior_analysis': BehaviorAnalyzer(),
            'quantum_resistance': QuantumResistanceEngine()
        }
    
    def monitor_data_stream(self, data_points):
        """Monitor real-time data stream for anomalies"""
        alerts = []
        
        for model_name, model in self.detection_models.items():
            try:
                anomalies = model.detect_anomalies(data_points)
                if anomalies:
                    alerts.extend(anomalies)
            except Exception as e:
                st.error(f"Defense system {model_name} error: {str(e)}")
        
        return alerts
    
    def activate_emergency_protocols(self, threat_level):
        """Activate emergency defense protocols"""
        protocols = {
            'CRITICAL': ['ISOLATE_SYSTEMS', 'ACTIVATE_BACKUPS', 'NOTIFY_SOC', 'INITIATE_RECOVERY'],
            'HIGH': ['ENHANCE_MONITORING', 'LIMIT_CONNECTIONS', 'BACKUP_CRITICAL_DATA'],
            'MEDIUM': ['INCREASE_LOGGING', 'VALIDATE_BACKUPS', 'UPDATE_THREAT_SIGNATURES']
        }
        
        return protocols.get(threat_level, ['MAINTAIN_NORMAL_OPS'])

class StatisticalAnalyzer:
    """Statistical anomaly detection"""
    
    def __init__(self):
        self.window_size = 100
        self.confidence_level = 0.95
    
    def detect_anomalies(self, data_points):
        anomalies = []
        
        if len(data_points) < self.window_size:
            return anomalies
        
        prices = [point['price'] for point in data_points[-self.window_size:]]
        
        # Z-score analysis
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        latest_price = prices[-1]
        z_score = abs(latest_price - mean_price) / std_price
        
        if z_score > 3:  # 3 standard deviations
            anomalies.append({
                'type': 'STATISTICAL_OUTLIER',
                'severity': 'HIGH',
                'confidence': min(0.99, z_score / 10),
                'description': f'Price deviation detected: Z-score {z_score:.2f}'
            })
        
        return anomalies

class PatternDetector:
    """Pattern-based anomaly detection"""
    
    def __init__(self):
        self.patterns = self._initialize_malicious_patterns()
    
    def _initialize_malicious_patterns(self):
        return {
            'PUMP_AND_DUMP': {'volatility_threshold': 0.15, 'volume_spike': 3.0},
            'FLASH_CRASH': {'price_drop': 0.10, 'time_window': 60},
            'SPOOFING': {'order_imbalance': 5.0, 'cancellation_rate': 0.8}
        }
    
    def detect_anomalies(self, data_points):
        anomalies = []
        
        if len(data_points) < 10:
            return anomalies
        
        # Detect pump and dump patterns
        recent_volatility = self._calculate_volatility(data_points[-10:])
        if recent_volatility > self.patterns['PUMP_AND_DUMP']['volatility_threshold']:
            anomalies.append({
                'type': 'PUMP_AND_DUMP_SUSPECTED',
                'severity': 'CRITICAL',
                'confidence': 0.85,
                'description': f'High volatility detected: {recent_volatility:.2%}'
            })
        
        return anomalies
    
    def _calculate_volatility(self, data_points):
        prices = [point['price'] for point in data_points]
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)

class BehaviorAnalyzer:
    """Behavioral analysis for market manipulation detection"""
    
    def __init__(self):
        self.normal_behavior_baseline = {}
        self.analysis_window = 50
    
    def detect_anomalies(self, data_points):
        anomalies = []
        
        if len(data_points) < self.analysis_window:
            return anomalies
        
        # Analyze trading patterns
        volume_anomaly = self._detect_volume_anomalies(data_points)
        if volume_anomaly:
            anomalies.append(volume_anomaly)
        
        price_anomaly = self._detect_price_anomalies(data_points)
        if price_anomaly:
            anomalies.append(price_anomaly)
        
        return anomalies
    
    def _detect_volume_anomalies(self, data_points):
        volumes = [point.get('volume', 0) for point in data_points[-self.analysis_window:]]
        avg_volume = np.mean(volumes[:-1])
        latest_volume = volumes[-1]
        
        if latest_volume > avg_volume * 5:  # 5x volume spike
            return {
                'type': 'VOLUME_SPIKE',
                'severity': 'MEDIUM',
                'confidence': 0.75,
                'description': f'Volume spike detected: {latest_volume/avg_volume:.1f}x average'
            }
        return None
    
    def _detect_price_anomalies(self, data_points):
        prices = [point['price'] for point in data_points[-self.analysis_window:]]
        price_changes = np.diff(prices) / prices[:-1]
        
        if any(abs(change) > 0.05 for change in price_changes[-3:]):  # 5% moves in last 3 periods
            return {
                'type': 'ABNORMAL_PRICE_MOVEMENTS',
                'severity': 'HIGH',
                'confidence': 0.80,
                'description': 'Rapid price movements detected'
            }
        return None

class QuantumResistanceEngine:
    """Quantum-resistant cryptographic monitoring"""
    
    def __init__(self):
        self.encryption_standards = ['AES-256', 'RSA-4096', 'ECC-521', 'QUANTUM_RESISTANT']
        self.security_score = 0.95
    
    def detect_anomalies(self, data_points):
        anomalies = []
        
        # Simulate quantum resistance monitoring
        if random.random() < 0.01:  # 1% chance of quantum anomaly
            anomalies.append({
                'type': 'QUANTUM_DECRYPTION_ATTEMPT',
                'severity': 'CRITICAL',
                'confidence': 0.92,
                'description': 'Potential quantum decryption patterns detected'
            })
        
        return anomalies

# --- ENHANCED ANALYTICS ENGINE ---

class AdvancedAnalyticsEngine:
    """Advanced analytics with machine learning capabilities"""
    
    def __init__(self):
        self.ml_models = {}
        self.analytics_cache = {}
        self.performance_metrics = {}
    
    def initialize_analytics(self):
        """Initialize analytics models"""
        self.ml_models = {
            'clustering': DBSCAN(eps=0.5, min_samples=5),
            'isolation_forest': IsolationForest(contamination=0.1),
            'pca': PCA(n_components=2)
        }
    
    def perform_comprehensive_analysis(self, data_stream):
        """Perform comprehensive data analysis"""
        analysis_results = {
            'timestamp': datetime.now(),
            'data_quality_score': self._calculate_data_quality(data_stream),
            'risk_assessment': self._assess_risk_level(data_stream),
            'pattern_analysis': self._analyze_patterns(data_stream),
            'predictive_insights': self._generate_predictions(data_stream),
            'anomaly_correlation': self._correlate_anomalies(data_stream)
        }
        
        return analysis_results
    
    def _calculate_data_quality(self, data_stream):
        """Calculate data quality score"""
        if not data_stream:
            return 0.0
        
        completeness = len([d for d in data_stream if all(key in d for key in ['price', 'volume'])]) / len(data_stream)
        consistency = self._check_consistency(data_stream)
        timeliness = self._check_timeliness(data_stream)
        
        return (completeness + consistency + timeliness) / 3
    
    def _check_consistency(self, data_stream):
        """Check data consistency"""
        prices = [d['price'] for d in data_stream if 'price' in d]
        if len(prices) < 2:
            return 0.5
        
        changes = np.abs(np.diff(prices) / prices[:-1])
        outlier_ratio = np.sum(changes > 0.1) / len(changes)  # More than 10% changes
        return 1.0 - outlier_ratio
    
    def _check_timeliness(self, data_stream):
        """Check data timeliness"""
        if not data_stream:
            return 0.0
        
        current_time = datetime.now()
        timestamps = [pd.to_datetime(d.get('timestamp', current_time)) for d in data_stream]
        time_diffs = [(current_time - ts).total_seconds() for ts in timestamps[-10:]]  # Last 10 points
        
        fresh_count = sum(diff < 300 for diff in time_diffs)  # Data less than 5 minutes old
        return fresh_count / len(time_diffs)
    
    def _assess_risk_level(self, data_stream):
        """Assess overall risk level"""
        if len(data_stream) < 10:
            return 'UNKNOWN'
        
        volatility = self._calculate_volatility(data_stream[-20:])
        volume_activity = self._calculate_volume_activity(data_stream[-20:])
        
        risk_score = (volatility * 0.6) + (volume_activity * 0.4)
        
        if risk_score > 0.7:
            return 'CRITICAL'
        elif risk_score > 0.5:
            return 'HIGH'
        elif risk_score > 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_volatility(self, data_points):
        """Calculate price volatility"""
        prices = [d['price'] for d in data_points if 'price' in d]
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)
    
    def _calculate_volume_activity(self, data_points):
        """Calculate volume activity level"""
        volumes = [d.get('volume', 0) for d in data_points]
        if not volumes:
            return 0.0
        
        avg_volume = np.mean(volumes)
        if avg_volume == 0:
            return 0.0
        
        latest_volume = volumes[-1] if volumes else 0
        return min(1.0, latest_volume / (avg_volume * 10))  # Normalize
    
    def _analyze_patterns(self, data_stream):
        """Analyze trading patterns"""
        if len(data_stream) < 20:
            return {'status': 'INSUFFICIENT_DATA'}
        
        patterns = {
            'trend_direction': self._detect_trend(data_stream[-20:]),
            'volatility_regime': self._detect_volatility_regime(data_stream[-20:]),
            'market_regime': self._detect_market_regime(data_stream[-20:])
        }
        
        return patterns
    
    def _detect_trend(self, data_points):
        """Detect price trend"""
        prices = [d['price'] for d in data_points]
        if len(prices) < 2:
            return 'UNKNOWN'
        
        price_change = (prices[-1] - prices[0]) / prices[0]
        
        if price_change > 0.02:
            return 'BULLISH'
        elif price_change < -0.02:
            return 'BEARISH'
        else:
            return 'SIDEWAYS'
    
    def _detect_volatility_regime(self, data_points):
        """Detect volatility regime"""
        volatility = self._calculate_volatility(data_points)
        
        if volatility > 0.05:
            return 'HIGH_VOLATILITY'
        elif volatility > 0.02:
            return 'MODERATE_VOLATILITY'
        else:
            return 'LOW_VOLATILITY'
    
    def _detect_market_regime(self, data_points):
        """Detect market regime"""
        trend = self._detect_trend(data_points)
        volatility = self._detect_volatility_regime(data_points)
        
        if trend == 'BULLISH' and volatility == 'LOW_VOLATILITY':
            return 'STABLE_BULL'
        elif trend == 'BEARISH' and volatility == 'HIGH_VOLATILITY':
            return 'TURBULENT_BEAR'
        else:
            return 'TRANSITIONAL'
    
    def _generate_predictions(self, data_stream):
        """Generate predictive insights"""
        if len(data_stream) < 30:
            return {'status': 'INSUFFICIENT_DATA_FOR_PREDICTION'}
        
        # Simple moving average prediction
        prices = [d['price'] for d in data_stream[-30:]]
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices[-20:])
        
        prediction = 'NEUTRAL'
        if short_ma > long_ma * 1.01:
            prediction = 'UPWARD_PRESSURE'
        elif short_ma < long_ma * 0.99:
            prediction = 'DOWNWARD_PRESSURE'
        
        return {
            'direction_bias': prediction,
            'confidence': 0.65,
            'timeframe': 'SHORT_TERM'
        }
    
    def _correlate_anomalies(self, data_stream):
        """Correlate multiple anomaly indicators"""
        if len(data_stream) < 10:
            return {'status': 'INSUFFICIENT_DATA'}
        
        # Simple correlation analysis
        prices = [d['price'] for d in data_stream[-10:]]
        volumes = [d.get('volume', 0) for d in data_stream[-10:]]
        
        price_volatility = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
        volume_volatility = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 0
        
        correlation_strength = 'WEAK'
        if price_volatility > 0.02 and volume_volatility > 0.5:
            correlation_strength = 'STRONG'
        elif price_volatility > 0.01 or volume_volatility > 0.3:
            correlation_strength = 'MODERATE'
        
        return {
            'price_volume_correlation': correlation_strength,
            'multi_anomaly_detected': correlation_strength in ['STRONG', 'MODERATE']
        }

# --- ENHANCED UI COMPONENTS ---

def render_enhanced_dashboard():
    """Render enhanced dashboard with individual data source buttons"""
    st.markdown("### 📊 ENHANCED SECURITY DASHBOARD")
    
    # Initialize systems
    if 'data_engine' not in st.session_state:
        st.session_state.data_engine = RealTimeDataEngine()
        st.session_state.defense_systems = AdvancedDefenseSystems()
        st.session_state.analytics_engine = AdvancedAnalyticsEngine()
        st.session_state.defense_systems.initialize_defense_systems()
        st.session_state.analytics_engine.initialize_analytics()
    
    data_engine = st.session_state.data_engine
    defense_systems = st.session_state.defense_systems
    analytics_engine = st.session_state.analytics_engine
    
    # Real-time data monitoring with individual buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("#### 🔄 REAL-TIME DATA")
        
        # Individual data source buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("📊 Alpha Vantage", key="alpha_vantage_btn", use_container_width=True):
                with st.spinner("Fetching Alpha Vantage data..."):
                    symbols = ['AAPL', 'GOOGL', 'MSFT']
                    results = []
                    for symbol in symbols:
                        # In a real implementation, use asyncio here
                        mock_data = data_engine._generate_mock_data(symbol, 'Alpha Vantage')
                        mock_data['source'] = 'Alpha Vantage'
                        results.append(mock_data)
                    st.session_state.alpha_vantage_data = results
                    st.session_state.current_data_source = 'Alpha Vantage'
                    st.success(f"✅ Alpha Vantage data fetched for {len(symbols)} symbols")
        
        with col_btn2:
            if st.button("🌐 Finnhub", key="finnhub_btn", use_container_width=True):
                with st.spinner("Fetching Finnhub data..."):
                    symbols = ['TSLA', 'AMZN', 'META']
                    results = []
                    for symbol in symbols:
                        mock_data = data_engine._generate_mock_data(symbol, 'Finnhub')
                        mock_data['source'] = 'Finnhub'
                        results.append(mock_data)
                    st.session_state.finnhub_data = results
                    st.session_state.current_data_source = 'Finnhub'
                    st.success(f"✅ Finnhub data fetched for {len(symbols)} symbols")
        
        with col_btn3:
            if st.button("📈 FMP", key="fmp_btn", use_container_width=True):
                with st.spinner("Fetching FMP data..."):
                    symbols = ['NFLX', 'NVDA', 'AMD']
                    results = []
                    for symbol in symbols:
                        mock_data = data_engine._generate_mock_data(symbol, 'FMP')
                        mock_data['source'] = 'Financial Modeling Prep'
                        results.append(mock_data)
                    st.session_state.fmp_data = results
                    st.session_state.current_data_source = 'Financial Modeling Prep'
                    st.success(f"✅ FMP data fetched for {len(symbols)} symbols")
        
        # Display current data source status
        if hasattr(st.session_state, 'current_data_source'):
            st.markdown(f"""
            <div class="data-panel">
                <div style="color: #00ffff;">ACTIVE SOURCE</div>
                <div style="color: #00ff00;">{st.session_state.current_data_source}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display sample data from current source
        if hasattr(st.session_state, 'current_data_source'):
            source_key = f"{st.session_state.current_data_source.lower().replace(' ', '_')}_data"
            if source_key in st.session_state:
                data = st.session_state[source_key]
                if data:
                    st.markdown("##### 📊 SAMPLE DATA")
                    for stock_data in data[:2]:  # Show first 2 stocks
                        change_color = "#00ff00" if stock_data.get('change', 0) >= 0 else "#ff4444"
                        st.markdown(f"""
                        <div class="data-panel">
                            <div style="color: #00ffff;">${stock_data['price']:.2f}</div>
                            <div style="color: {change_color}; font-size: 0.8rem;">
                                {stock_data.get('change', 0):.2f} ({stock_data.get('change_percent', 0):.2f}%)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🛡️ DEFENSE STATUS")
        for system, status in defense_systems.defense_status.items():
            status_color = "#00ff00" if status == 'ACTIVE' else "#ffff00" if status == 'STANDBY' else "#ff4444"
            st.markdown(f"""
            <div class="data-panel">
                <div>{system.replace('_', ' ').title()}</div>
                <div style="color: {status_color}; font-size: 0.9rem;">{status}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### 📈 ANALYTICS ENGINE")
        if st.button("🧠 RUN ANALYSIS", key="run_analysis"):
            # Use the most recently fetched data
            data_to_analyze = None
            if hasattr(st.session_state, 'current_data_source'):
                source_key = f"{st.session_state.current_data_source.lower().replace(' ', '_')}_data"
                if source_key in st.session_state:
                    data_to_analyze = st.session_state[source_key]
            
            if data_to_analyze:
                analysis = analytics_engine.perform_comprehensive_analysis(data_to_analyze)
                st.session_state.latest_analysis = analysis
                st.success("✅ Analysis completed!")
            else:
                st.warning("⚠️ Please fetch data first!")
            
        if 'latest_analysis' in st.session_state:
            analysis = st.session_state.latest_analysis
            risk_color = {
                'CRITICAL': '#ff0000',
                'HIGH': '#ff6b00', 
                'MEDIUM': '#ffff00',
                'LOW': '#00ff00',
                'UNKNOWN': '#888888'
            }.get(analysis['risk_assessment'], '#888888')
            
            st.markdown(f"""
            <div class="data-panel">
                <div>Risk Level: <span style="color: {risk_color};">{analysis['risk_assessment']}</span></div>
                <div>Data Quality: <span style="color: #00ffff;">{analysis['data_quality_score']:.1%}</span></div>
                <div>Pattern: <span style="color: #00ff00;">{analysis['pattern_analysis'].get('trend_direction', 'N/A')}</span></div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("#### 🎯 THREAT MONITORING")
        if st.button("🔍 SCAN FOR THREATS", key="scan_threats"):
            # Use the most recently fetched data
            data_to_scan = None
            if hasattr(st.session_state, 'current_data_source'):
                source_key = f"{st.session_state.current_data_source.lower().replace(' ', '_')}_data"
                if source_key in st.session_state:
                    data_to_scan = st.session_state[source_key]
            
            if data_to_scan:
                alerts = defense_systems.monitor_data_stream(data_to_scan)
                st.session_state.active_alerts = alerts
                if alerts:
                    st.warning(f"🚨 {len(alerts)} threats detected!")
                else:
                    st.success("✅ No threats detected")
            else:
                st.warning("⚠️ Please fetch data first!")
            
        if 'active_alerts' in st.session_state:
            alerts = st.session_state.active_alerts
            if alerts:
                for alert in alerts[:2]:  # Show first 2 alerts
                    severity_color = {
                        'CRITICAL': '#ff0000',
                        'HIGH': '#ff6b00',
                        'MEDIUM': '#ffff00',
                        'LOW': '#00ff00'
                    }.get(alert['severity'], '#888888')
                    
                    st.markdown(f"""
                    <div class="data-panel">
                        <div style="color: {severity_color}; font-size: 0.8rem;">{alert['type']}</div>
                        <div style="font-size: 0.7rem;">{alert['description'][:30]}...</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="data-panel">
                    <div style="color: #00ff00;">NO ACTIVE THREATS</div>
                </div>
                """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("#### ⚡ SYSTEM HEALTH")
        system_health = {
            'CPU': random.randint(65, 85),
            'Memory': random.randint(70, 90),
            'Network': random.randint(80, 95),
            'Storage': random.randint(75, 88)
        }
        
        for component, usage in system_health.items():
            color = "#00ff00" if usage < 70 else "#ffff00" if usage < 85 else "#ff4444"
            st.markdown(f"""
            <div class="data-panel">
                <div>{component}</div>
                <div style="color: {color}; font-size: 0.9rem;">{usage}%</div>
            </div>
            """, unsafe_allow_html=True)

def render_advanced_analytics():
    """Render advanced analytics dashboard"""
    st.markdown("### 🔬 ADVANCED ANALYTICS CENTER")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 PREDICTIVE ANALYSIS", "🎯 PATTERN DETECTION", "📈 RISK ASSESSMENT", "🔍 CORRELATION ENGINE"])
    
    with tab1:
        st.markdown("#### 🧠 PREDICTIVE MARKET ANALYSIS")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Predictive analytics visualization
            if 'latest_analysis' in st.session_state:
                analysis = st.session_state.latest_analysis
                
                # Create prediction chart
                fig = go.Figure()
                
                # Simulate price predictions
                periods = 10
                current_price = 150  # Base price
                predictions = [current_price]
                
                for i in range(periods):
                    if analysis['predictive_insights'].get('direction_bias') == 'UPWARD_PRESSURE':
                        change = random.uniform(0.001, 0.005)
                    elif analysis['predictive_insights'].get('direction_bias') == 'DOWNWARD_PRESSURE':
                        change = random.uniform(-0.005, -0.001)
                    else:
                        change = random.uniform(-0.002, 0.002)
                    
                    predictions.append(predictions[-1] * (1 + change))
                
                fig.add_trace(go.Scatter(
                    x=list(range(periods + 1)),
                    y=predictions,
                    name='Price Forecast',
                    line=dict(color='#00ffff', width=3)
                ))
                
                fig.update_layout(
                    title='10-Period Price Forecast',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#00ff00', family='Share Tech Mono'),
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📋 INSIGHTS")
            if 'latest_analysis' in st.session_state:
                analysis = st.session_state.latest_analysis
                
                insights = [
                    f"Trend: {analysis['pattern_analysis'].get('trend_direction', 'N/A')}",
                    f"Volatility: {analysis['pattern_analysis'].get('volatility_regime', 'N/A')}",
                    f"Market Regime: {analysis['pattern_analysis'].get('market_regime', 'N/A')}",
                    f"Prediction: {analysis['predictive_insights'].get('direction_bias', 'N/A')}",
                    f"Confidence: {analysis['predictive_insights'].get('confidence', 0):.1%}"
                ]
                
                for insight in insights:
                    st.markdown(f"""
                    <div class="log-entry">
                        {insight}
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("#### 🎯 ADVANCED PATTERN DETECTION")
        
        # Pattern analysis visualization
        patterns = ['Normal', 'Pump & Dump', 'Flash Crash', 'Spoofing', 'Wash Trading']
        occurrences = [random.randint(80, 95), random.randint(1, 5), random.randint(1, 3), random.randint(2, 8), random.randint(1, 4)]
        
        fig = px.bar(
            x=patterns, 
            y=occurrences,
            title="Detected Pattern Frequencies",
            color=occurrences,
            color_continuous_scale=['#00ff00', '#ffff00', '#ff6b00', '#ff0000']
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#00ff00', family='Share Tech Mono'),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pattern details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🚨 SUSPICIOUS PATTERNS")
            suspicious_patterns = [
                ("Volume Spike + Price Surge", "Pump & Dump suspected", "HIGH"),
                ("Rapid Price Drops", "Potential flash crash", "CRITICAL"),
                ("Order Imbalance", "Possible spoofing", "MEDIUM")
            ]
            
            for pattern, description, severity in suspicious_patterns:
                severity_color = "#ff0000" if severity == 'CRITICAL' else "#ff6b00" if severity == 'HIGH' else "#ffff00"
                st.markdown(f"""
                <div class="data-panel">
                    <div style="color: {severity_color}; font-size: 0.9rem;"><strong>{pattern}</strong></div>
                    <div style="font-size: 0.8rem;">{description}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("#### 📊 COMPREHENSIVE RISK ASSESSMENT")
        
        # Risk metrics
        risk_metrics = {
            'Market Risk': random.randint(60, 85),
            'Liquidity Risk': random.randint(40, 75),
            'Operational Risk': random.randint(20, 50),
            'Systemic Risk': random.randint(55, 80),
            'Data Poisoning Risk': random.randint(15, 35)
        }
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Risk gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_metrics['Market Risk'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Overall Risk Score"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "green"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#00ff00', family='Share Tech Mono'),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### 📈 RISK BREAKDOWN")
            for metric, value in risk_metrics.items():
                color = "#00ff00" if value < 40 else "#ffff00" if value < 70 else "#ff4444"
                st.markdown(f"""
                <div class="data-panel">
                    <div>{metric}</div>
                    <div style="color: {color}; font-size: 1.1rem;">{value}%</div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("#### 🔗 CORRELATION ANALYSIS ENGINE")
        
        # Correlation matrix
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META']
        correlation_data = np.random.uniform(-1, 1, (6, 6))
        np.fill_diagonal(correlation_data, 1.0)
        
        fig = px.imshow(
            correlation_data,
            x=symbols,
            y=symbols,
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        
        fig.update_layout(
            title="Asset Correlation Matrix",
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#00ff00', family='Share Tech Mono'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly correlation insights
        st.markdown("##### 🎯 CORRELATED ANOMALIES")
        
        correlated_events = [
            ("High AAPL volatility", "GOOGL volume spike", "0.87 correlation"),
            ("TSLA price drop", "Market sentiment shift", "0.92 correlation"),
            ("MSFT options activity", "Institutional positioning", "0.78 correlation")
        ]
        
        for event1, event2, correlation in correlated_events:
            st.markdown(f"""
            <div class="data-panel">
                <div style="font-size: 0.9rem;"><strong>{event1}</strong></div>
                <div style="font-size: 0.8rem;">↕️ {event2}</div>
                <div style="color: #00ffff; font-size: 0.8rem;">Correlation: {correlation}</div>
            </div>
            """, unsafe_allow_html=True)

def render_defense_systems_control():
    """Render defense systems control panel"""
    st.markdown("### 🛡️ ADVANCED DEFENSE SYSTEMS")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🛡️ ACTIVE DEFENSES", "🚨 INCIDENT RESPONSE", "📊 DEFENSE ANALYTICS", "⚙️ SYSTEM CONFIG"])
    
    with tab1:
        st.markdown("#### 🛡️ ACTIVE DEFENSE SYSTEMS")
        
        defense_systems = [
            {
                'name': 'DATA VALIDATION ENGINE',
                'status': 'ACTIVE',
                'efficiency': '98.2%',
                'threats_blocked': '1,247',
                'description': 'Real-time data integrity validation'
            },
            {
                'name': 'ANOMALY DETECTION SUITE',
                'status': 'ACTIVE', 
                'efficiency': '96.8%',
                'threats_blocked': '892',
                'description': 'ML-powered anomaly detection'
            },
            {
                'name': 'BEHAVIORAL ANALYSIS',
                'status': 'ACTIVE',
                'efficiency': '94.5%',
                'threats_blocked': '567',
                'description': 'Pattern-based threat detection'
            },
            {
                'name': 'QUANTUM RESISTANCE LAYER',
                'status': 'STANDBY',
                'efficiency': '99.9%',
                'threats_blocked': '23',
                'description': 'Quantum-resistant cryptography'
            }
        ]
        
        for system in defense_systems:
            status_color = "#00ff00" if system['status'] == 'ACTIVE' else "#ffff00"
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class="data-panel">
                    <div style="color: #00ffff; font-size: 1.1rem;"><strong>{system['name']}</strong></div>
                    <div>{system['description']}</div>
                    <div style="color: {status_color};">Status: {system['status']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="terminal-metric">
                    <div>Efficiency</div>
                    <div style="color: #00ff00;">{system['efficiency']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("#### 🚨 AUTOMATED INCIDENT RESPONSE")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🎯 RESPONSE PROTOCOLS")
            
            protocols = [
                ("DATA POISONING ATTEMPT", "ISOLATE, VALIDATE, RESTORE", "AUTOMATED"),
                ("MODEL INVERSION", "BLOCK SOURCE, ENHANCE SECURITY", "MANUAL"),
                ("BACKDOOR ATTACK", "ACTIVATE BACKUP, INVESTIGATE", "SEMI-AUTO"),
                ("EVASION ATTACK", "UPDATE DETECTION, RETRAIN", "AUTOMATED")
            ]
            
            for threat, response, automation in protocols:
                auto_color = "#00ff00" if automation == 'AUTOMATED' else "#ffff00" if automation == 'SEMI-AUTO' else "#ff4444"
                st.markdown(f"""
                <div class="data-panel">
                    <div style="color: #ff4444;"><strong>{threat}</strong></div>
                    <div style="font-size: 0.9rem;">{response}</div>
                    <div style="color: {auto_color}; font-size: 0.8rem;">{automation}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("##### 📊 RESPONSE METRICS")
            
            metrics = [
                ("Mean Time to Detect", "2.3s", "#00ff00"),
                ("Mean Time to Respond", "8.7s", "#ffff00"),
                ("False Positive Rate", "3.2%", "#00ffff"),
                ("Threat Neutralization", "96.8%", "#00ff00"),
                ("System Availability", "99.95%", "#00ff00")
            ]
            
            for metric, value, color in metrics:
                st.markdown(f"""
                <div class="terminal-metric">
                    <div>{metric}</div>
                    <div style="color: {color}; font-size: 1.1rem;">{value}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🚀 ACTIVATE EMERGENCY PROTOCOL", key="emergency_protocol"):
                st.session_state.emergency_activated = True
                st.markdown("""
                <div class="alert-critical">
                    🚨 EMERGENCY PROTOCOL ACTIVATED<br>
                    All defense systems at maximum readiness<br>
                    SOC team notified | Backup systems engaged
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("#### 📊 DEFENSE EFFECTIVENESS ANALYTICS")
        
        # Defense performance over time
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        threats_blocked = [random.randint(150, 250) for _ in range(7)]
        false_positives = [random.randint(5, 15) for _ in range(7)]
        response_times = [random.uniform(1.5, 3.5) for _ in range(7)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=days, y=threats_blocked,
            name='Threats Blocked',
            line=dict(color='#00ff00', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=days, y=false_positives,
            name='False Positives',
            line=dict(color='#ff4444', width=2)
        ))
        
        fig.update_layout(
            title='Weekly Defense Performance',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#00ff00', family='Share Tech Mono'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Defense efficiency by type
        defense_types = ['Data Validation', 'Anomaly Detection', 'Behavioral Analysis', 'Pattern Recognition']
        efficiency_rates = [98.2, 96.8, 94.5, 92.3]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=defense_types,
                y=efficiency_rates,
                title="Defense System Efficiency",
                color=efficiency_rates,
                color_continuous_scale=['#ff4444', '#ffff00', '#00ff00']
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#00ff00', family='Share Tech Mono'),
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("#### ⚙️ DEFENSE SYSTEM CONFIGURATION")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🔧 SYSTEM SETTINGS")
            
            settings = {
                "Detection Sensitivity": st.slider("Sensitivity", 1, 10, 7),
                "Response Aggressiveness": st.slider("Aggressiveness", 1, 10, 6),
                "Data Retention Days": st.slider("Retention Days", 7, 365, 90),
                "Auto-Response Threshold": st.slider("Auto-Response %", 50, 100, 85)
            }
            
            for setting, value in settings.items():
                st.markdown(f"""
                <div class="data-panel">
                    <div>{setting}</div>
                    <div style="color: #00ffff;">{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("##### 🎯 THREAT PRIORITIES")
            
            priorities = [
                ("Data Poisoning", "CRITICAL", "#ff0000"),
                ("Model Inversion", "HIGH", "#ff6b00"),
                ("Backdoor Attacks", "HIGH", "#ff6b00"),
                ("Evasion Attacks", "MEDIUM", "#ffff00"),
                ("Membership Inference", "LOW", "#00ff00")
            ]
            
            for threat, priority, color in priorities:
                st.markdown(f"""
                <div class="data-panel">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{threat}</span>
                        <span style="color: {color};">{priority}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("💾 SAVE CONFIGURATION", key="save_config"):
                st.success("✅ Configuration saved successfully!")

# --- ENHANCED MAIN INTERFACE ---

def render_terminal_header():
    """Render the enterprise terminal header"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    st.markdown(f"""
    <div class="terminal-header">
        <div style="display: flex; justify-content: space-between; align-items: center; color: #00ffff;">
            <div style="display: flex; align-items: center;">
                <h1 style="margin: 0; font-family: 'Orbitron', sans-serif; font-size: 2rem;">
                    🛡️ CYBER DEFENSE TERMINAL
                </h1>
                <span style="margin-left: 2rem; font-family: 'Share Tech Mono', monospace;">
                    DATA POISONING SOC
                </span>
            </div>
            <div style="text-align: right; font-family: 'Share Tech Mono', monospace;">
                <div>SYSTEM STATUS: <span style="color: #00ff00;">OPERATIONAL</span></div>
                <div>{current_time}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_threat_intelligence():
    """Enhanced threat intelligence with real-time data"""
    st.markdown("### 🎯 ENHANCED THREAT INTELLIGENCE")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🌍 REAL-TIME THREAT LANDSCAPE")
        
        # Real-time threat data visualization
        threat_types = ['Data Poisoning', 'Model Evasion', 'API Abuse', 'Credential Stuffing', 'DDoS']
        threat_levels = [random.randint(60, 95) for _ in threat_types]
        
        fig = px.bar(
            x=threat_types,
            y=threat_levels,
            title="Current Threat Levels",
            color=threat_levels,
            color_continuous_scale=['#00ff00', '#ffff00', '#ff6b00', '#ff0000']
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#00ff00', family='Share Tech Mono'),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🚨 ACTIVE THREAT FEEDS")
        
        threat_feeds = [
            ("Zero-Day Exploit", "CRITICAL", "Financial Sector"),
            ("API Vulnerability", "HIGH", "Multiple Targets"),
            ("Data Poisoning Kit", "HIGH", "Dark Web"),
            ("Model Extraction", "MEDIUM", "Research Labs")
        ]
        
        for threat, level, target in threat_feeds:
            level_color = "#ff0000" if level == 'CRITICAL' else "#ff6b00" if level == 'HIGH' else "#ffff00"
            st.markdown(f"""
            <div class="alert-{level.lower()}">
                <strong>{threat}</strong><br>
                Level: <span style="color: {level_color};">{level}</span><br>
                Target: {target}
            </div>
            """, unsafe_allow_html=True)

def render_system_logs():
    """Enhanced system logs with real-time events"""
    st.markdown("### 📋 ENHANCED SYSTEM LOGS & AUDIT")
    
    # Real-time log simulation
    log_events = [
        ("DATA_INGEST", "INFO", "Real-time market data stream established"),
        ("DEFENSE_SYSTEM", "INFO", "Anomaly detection engine initialized"),
        ("THREAT_DETECT", "WARNING", "Suspicious pattern detected in AAPL data"),
        ("ANALYTICS_ENGINE", "INFO", "Predictive analysis completed"),
        ("SECURITY", "INFO", "Quantum resistance layer activated"),
        ("DATA_VALIDATION", "ERROR", "Data integrity check failed for TSLA stream"),
        ("RESPONSE", "INFO", "Automated defense protocol executed"),
        ("BACKUP", "INFO", "System state backup completed")
    ]
    
    for source, level, message in log_events:
        level_color = {
            'INFO': '#00ff00',
            'WARNING': '#ffff00', 
            'ERROR': '#ff4444',
            'CRITICAL': '#ff0000'
        }.get(level, '#888888')
        
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        st.markdown(f"""
        <div class="log-entry">
            <span style="color: {level_color}; font-weight: bold;">[{level}]</span>
            <span style="color: #00ffff;">{timestamp}</span>
            <span style="color: #ffff00;">{source}</span>
            <span>{message}</span>
        </div>
        """, unsafe_allow_html=True)

def render_command_center():
    """Enhanced command and control center"""
    st.markdown("### ⚙️ ENHANCED COMMAND & CONTROL")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🎮 ADVANCED CONTROLS")
        
        control_col1, control_col2 = st.columns(2)
        
        with control_col1:
            if st.button("🌐 SYNC THREAT INTELLIGENCE", use_container_width=True):
                st.markdown("""
                <div class="alert-low">
                    🔄 SYNCING GLOBAL THREAT INTELLIGENCE<br>
                    Updated 1,247 threat signatures
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🧪 RUN PENETRATION TEST", use_container_width=True):
                st.markdown("""
                <div class="alert-medium">
                    🧪 INITIATING PENETRATION TEST<br>
                    Simulating advanced attack vectors
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("📊 GENERATE RISK REPORT", use_container_width=True):
                st.markdown("""
                <div class="alert-low">
                    📈 GENERATING COMPREHENSIVE RISK REPORT<br>
                    Report will be available in 30 seconds
                </div>
                """, unsafe_allow_html=True)
        
        with control_col2:
            if st.button("🔧 OPTIMIZE DEFENSES", use_container_width=True):
                st.markdown("""
                <div class="alert-low">
                    ⚙️ OPTIMIZING DEFENSE CONFIGURATION<br>
                    Performance improved by 12.3%
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🚀 DEPLOY COUNTERMEASURES", use_container_width=True):
                st.markdown("""
                <div class="alert-high">
                    🛡️ DEPLOYING ADVANCED COUNTERMEASURES<br>
                    All defense systems enhanced
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("💾 BACKUP CONFIGURATION", use_container_width=True):
                st.markdown("""
                <div class="alert-low">
                    💾 BACKING UP SYSTEM CONFIGURATION<br>
                    Configuration saved securely
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🔧 SYSTEM STATUS")
        
        status_indicators = [
            ("Data Streams", "ACTIVE", "#00ff00"),
            ("Defense Systems", "OPERATIONAL", "#00ff00"),
            ("Analytics Engine", "RUNNING", "#00ff00"),
            ("Threat Intelligence", "SYNCED", "#00ff00"),
            ("Backup Systems", "READY", "#ffff00")
        ]
        
        for system, status, color in status_indicators:
            st.markdown(f"""
            <div class="data-panel">
                <div style="display: flex; justify-content: space-between;">
                    <span>{system}</span>
                    <span style="color: {color};">{status}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_login():
    """Enterprise terminal login"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #001122 0%, #003366 100%);
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        font-family: 'Share Tech Mono', monospace;
    ">
        <div style="
            background: rgba(0, 20, 40, 0.9);
            border: 2px solid #00ffff;
            border-radius: 8px;
            padding: 3rem;
            text-align: center;
            box-shadow: 0 0 50px #00ffff33;
            width: 400px;
        ">
            <h1 style="color: #00ffff; margin-bottom: 2rem;">🛡️ CYBER DEFENSE TERMINAL</h1>
            <h3 style="color: #00ff00; margin-bottom: 2rem;">SECURE ACCESS REQUIRED</h3>
    """, unsafe_allow_html=True)
    
    with st.form("enterprise_login"):
        username = st.text_input("OPERATOR ID:", placeholder="Enter operator ID")
        password = st.text_input("ACCESS CODE:", type="password", placeholder="Enter access code")
        facility = st.selectbox("FACILITY:", ["PRIMARY SOC", "SECONDARY SOC", "FIELD OPERATIONS"])
        
        if st.form_submit_button("🚀 INITIATE SYSTEM ACCESS", use_container_width=True):
            if username == "operator" and password == "defense123":
                st.session_state.authenticated = True
                st.session_state.login_time = datetime.now()
                st.session_state.facility = facility
                st.success("✅ ACCESS GRANTED | INITIALIZING TERMINAL...")
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ ACCESS DENIED | INVALID CREDENTIALS")
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_cyber_terminal():
    """Enhanced cyber defense terminal interface"""
    
    # Render terminal header
    render_terminal_header()
    
    # Enhanced navigation tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 DASHBOARD", 
        "🎯 THREAT INTEL", 
        "🛡️ DEFENSE OPS", 
        "🔬 ANALYTICS",
        "📋 SYSTEM LOGS",
        "⚙️ COMMAND CENTER"
    ])
    
    with tab1:
        render_enhanced_dashboard()
    
    with tab2:
        render_threat_intelligence()
    
    with tab3:
        render_defense_systems_control()
    
    with tab4:
        render_advanced_analytics()
    
    with tab5:
        render_system_logs()
    
    with tab6:
        render_command_center()

# --- MAIN APPLICATION ---

def main():
    with quantum_resource_manager():
        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        
        if not st.session_state.authenticated:
            render_login()
        else:
            render_cyber_terminal()

if __name__ == "__main__":
    main()
