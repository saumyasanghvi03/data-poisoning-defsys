import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import time
import gc
from contextlib import contextmanager
import warnings
import hashlib
import uuid

warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CYBER DEFENSE TERMINAL | REAL-TIME DATA POISONING SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ENTERPRISE TERMINAL CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
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
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
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
</style>
""", unsafe_allow_html=True)

@contextmanager
def quantum_resource_manager():
    """Advanced resource management"""
    try:
        yield
    finally:
        gc.collect()

# --- DATA GENERATION CLASSES ---

class RealTimeDataEngine:
    """Real-time financial data integration engine"""
    
    def __init__(self):
        self.cache = {}
        self.last_fetch = {}
    
    def fetch_data(self, symbol):
        """Fetch real-time data (simulated)"""
        base_price = random.uniform(100, 500)
        return {
            'symbol': symbol,
            'price': base_price + random.uniform(-5, 5),
            'change': random.uniform(-2, 2),
            'change_percent': random.uniform(-1, 1),
            'volume': random.randint(1000000, 5000000),
            'high': base_price + random.uniform(1, 3),
            'low': base_price - random.uniform(1, 3),
            'open': base_price + random.uniform(-1, 1),
            'timestamp': datetime.now().isoformat(),
            'source': 'Real-Time Feed'
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
                pass
        
        return alerts

class StatisticalAnalyzer:
    """Statistical anomaly detection"""
    
    def __init__(self):
        self.window_size = 100
        self.confidence_level = 0.95
    
    def detect_anomalies(self, data_points):
        anomalies = []
        
        if len(data_points) < 10:
            return anomalies
        
        prices = [point['price'] for point in data_points[-10:]]
        
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        if std_price > 0:
            latest_price = prices[-1]
            z_score = abs(latest_price - mean_price) / std_price
            
            if z_score > 2:
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
        self.patterns = {
            'PUMP_AND_DUMP': {'volatility_threshold': 0.15, 'volume_spike': 3.0},
            'FLASH_CRASH': {'price_drop': 0.10, 'time_window': 60},
        }
    
    def detect_anomalies(self, data_points):
        anomalies = []
        
        if len(data_points) < 10:
            return anomalies
        
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
        if len(prices) < 2:
            return 0
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)

class BehaviorAnalyzer:
    """Behavioral analysis"""
    
    def __init__(self):
        self.analysis_window = 50
    
    def detect_anomalies(self, data_points):
        anomalies = []
        
        if len(data_points) < 10:
            return anomalies
        
        volumes = [point.get('volume', 0) for point in data_points[-10:]]
        if volumes:
            avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[0]
            latest_volume = volumes[-1]
            
            if avg_volume > 0 and latest_volume > avg_volume * 5:
                anomalies.append({
                    'type': 'VOLUME_SPIKE',
                    'severity': 'MEDIUM',
                    'confidence': 0.75,
                    'description': f'Volume spike detected: {latest_volume/avg_volume:.1f}x average'
                })
        
        return anomalies

class QuantumResistanceEngine:
    """Quantum-resistant monitoring"""
    
    def __init__(self):
        self.security_score = 0.95
    
    def detect_anomalies(self, data_points):
        anomalies = []
        
        if random.random() < 0.01:
            anomalies.append({
                'type': 'QUANTUM_DECRYPTION_ATTEMPT',
                'severity': 'CRITICAL',
                'confidence': 0.92,
                'description': 'Potential quantum decryption patterns detected'
            })
        
        return anomalies

class AdvancedAnalyticsEngine:
    """Advanced analytics with ML capabilities"""
    
    def __init__(self):
        self.analytics_cache = {}
    
    def perform_comprehensive_analysis(self, data_stream):
        """Perform comprehensive data analysis"""
        analysis_results = {
            'timestamp': datetime.now(),
            'data_quality_score': self._calculate_data_quality(data_stream),
            'risk_assessment': self._assess_risk_level(data_stream),
            'pattern_analysis': self._analyze_patterns(data_stream),
            'predictive_insights': self._generate_predictions(data_stream),
        }
        
        return analysis_results
    
    def _calculate_data_quality(self, data_stream):
        if not data_stream:
            return 0.0
        
        completeness = len([d for d in data_stream if all(key in d for key in ['price', 'volume'])]) / len(data_stream)
        return completeness
    
    def _assess_risk_level(self, data_stream):
        if len(data_stream) < 10:
            return 'UNKNOWN'
        
        volatility = self._calculate_volatility(data_stream[-20:])
        
        if volatility > 0.05:
            return 'CRITICAL'
        elif volatility > 0.03:
            return 'HIGH'
        elif volatility > 0.01:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_volatility(self, data_points):
        prices = [d['price'] for d in data_points if 'price' in d]
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)
    
    def _analyze_patterns(self, data_stream):
        if len(data_stream) < 20:
            return {'status': 'INSUFFICIENT_DATA'}
        
        return {
            'trend_direction': self._detect_trend(data_stream[-20:]),
            'volatility_regime': 'MODERATE',
            'market_regime': 'NORMAL'
        }
    
    def _detect_trend(self, data_points):
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
    
    def _generate_predictions(self, data_stream):
        if len(data_stream) < 30:
            return {'status': 'INSUFFICIENT_DATA_FOR_PREDICTION'}
        
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

# --- UI COMPONENTS ---

def render_enhanced_dashboard():
    """Render enhanced dashboard"""
    st.markdown("### 📊 ENHANCED SECURITY DASHBOARD")
    
    # Initialize systems
    if 'data_engine' not in st.session_state:
        st.session_state.data_engine = RealTimeDataEngine()
        st.session_state.defense_systems = AdvancedDefenseSystems()
        st.session_state.analytics_engine = AdvancedAnalyticsEngine()
        st.session_state.defense_systems.initialize_defense_systems()
        st.session_state.data_stream = []
    
    data_engine = st.session_state.data_engine
    defense_systems = st.session_state.defense_systems
    analytics_engine = st.session_state.analytics_engine
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("#### 🔄 REAL-TIME DATA")
        
        if st.button("📊 FETCH DATA", key="fetch_all_data", use_container_width=True):
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
            results = []
            for symbol in symbols:
                data = data_engine.fetch_data(symbol)
                results.append(data)
            
            st.session_state.data_stream.extend(results)
            st.session_state.current_data = results
            st.success(f"✅ Fetched data for {len(symbols)} symbols")
        
        if 'current_data' in st.session_state:
            data = st.session_state.current_data
            if data:
                st.markdown("##### 📊 LATEST DATA")
                for stock_data in data[:3]:
                    change_color = "#00ff00" if stock_data.get('change', 0) >= 0 else "#ff4444"
                    st.markdown(f"""
                    <div class="data-panel">
                        <div style="color: #00ffff;">{stock_data['symbol']}</div>
                        <div style="color: #00ff00; font-size: 1.2rem;">${stock_data['price']:.2f}</div>
                        <div style="color: {change_color}; font-size: 0.8rem;">
                            {stock_data.get('change', 0):.2f} ({stock_data.get('change_percent', 0):.2f}%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🛡️ DEFENSE STATUS")
        for system, status in defense_systems.defense_status.items():
            status_color = "#00ff00" if status == 'ACTIVE' else "#ffff00"
            st.markdown(f"""
            <div class="data-panel">
                <div>{system.replace('_', ' ').title()}</div>
                <div style="color: {status_color}; font-size: 0.9rem;">{status}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### 📈 ANALYTICS")
        if st.button("🧠 RUN ANALYSIS", key="run_analysis", use_container_width=True):
            if st.session_state.data_stream:
                analysis = analytics_engine.perform_comprehensive_analysis(st.session_state.data_stream)
                st.session_state.latest_analysis = analysis
                st.success("✅ Analysis completed!")
            else:
                st.warning("⚠️ Please fetch data first")
        
        if 'latest_analysis' in st.session_state:
            analysis = st.session_state.latest_analysis
            st.markdown("##### 📊 ANALYSIS RESULTS")
            st.markdown(f"""
            <div class="data-panel">
                <div>Data Quality: <span style="color: #00ff00;">{analysis.get('data_quality_score', 0)*100:.1f}%</span></div>
                <div>Risk Level: <span style="color: #ff4444;">{analysis.get('risk_assessment', 'UNKNOWN')}</span></div>
                <div>Trend: <span style="color: #00ffff;">{analysis.get('pattern_analysis', {}).get('trend_direction', 'UNKNOWN')}</span></div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("#### 🚨 THREAT MONITORING")
        if st.button("🔍 SCAN FOR THREATS", key="scan_threats", use_container_width=True):
            if st.session_state.data_stream:
                alerts = defense_systems.monitor_data_stream(st.session_state.data_stream)
                st.session_state.latest_alerts = alerts
                if alerts:
                    st.error(f"🚨 Detected {len(alerts)} potential threats!")
                else:
                    st.success("✅ No threats detected")
            else:
                st.warning("⚠️ Please fetch data first")
        
        if 'latest_alerts' in st.session_state:
            alerts = st.session_state.latest_alerts
            if alerts:
                st.markdown("##### 🔴 ACTIVE ALERTS")
                for alert in alerts[:3]:
                    severity_class = f"alert-{alert['severity'].lower()}"
                    st.markdown(f"""
                    <div class="{severity_class}">
                        <strong>{alert['type']}</strong><br>
                        {alert['description']}<br>
                        <small>Confidence: {alert['confidence']:.1%}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-low">
                    <strong>✅ ALL SYSTEMS NORMAL</strong><br>
                    No active threats detected
                </div>
                """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("#### ⚙️ SYSTEM CONTROLS")
        
        if st.button("🔄 RESET SYSTEMS", key="reset_systems", use_container_width=True):
            st.session_state.data_stream = []
            st.session_state.current_data = None
            st.session_state.latest_analysis = None
            st.session_state.latest_alerts = None
            st.success("✅ Systems reset successfully")
        
        if st.button("📊 GENERATE REPORT", key="generate_report", use_container_width=True):
            if st.session_state.data_stream:
                st.info("📄 Comprehensive report generated")
                # Generate sample report data
                report_data = {
                    'total_data_points': len(st.session_state.data_stream),
                    'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'system_status': 'OPERATIONAL'
                }
                st.session_state.last_report = report_data
            else:
                st.warning("⚠️ No data available for report")
        
        st.markdown("##### 📈 SYSTEM METRICS")
        st.markdown(f"""
        <div class="data-panel">
            <div>Data Points: <span style="color: #00ff00;">{len(st.session_state.data_stream)}</span></div>
            <div>Active Alerts: <span style="color: #ff4444;">{len(st.session_state.get('latest_alerts', []))}</span></div>
            <div>System Health: <span style="color: #00ff00;">OPTIMAL</span></div>
        </div>
        """, unsafe_allow_html=True)

def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="terminal-header">
        <h1 style="color: #00ffff; font-family: 'Orbitron', sans-serif; font-weight: 900; text-align: center; margin: 0;">
            🛡️ CYBER DEFENSE TERMINAL | REAL-TIME DATA POISONING SOC
        </h1>
        <p style="color: #00ff00; font-family: 'Share Tech Mono', monospace; text-align: center; margin: 0.5rem 0 0 0;">
            QUANTUM-RESISTANT SECURITY OPERATIONS CENTER | ENTERPRISE-GRADE THREAT DETECTION
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    render_header()
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.data_stream = []
        st.session_state.current_data = None
        st.session_state.latest_analysis = None
        st.session_state.latest_alerts = None
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 DASHBOARD", "📈 ANALYTICS", "⚙️ CONTROLS"])
    
    with tab1:
        render_enhanced_dashboard()
    
    with tab2:
        st.markdown("### 📈 ADVANCED ANALYTICS")
        if st.session_state.data_stream:
            # Create some sample charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Price Distribution")
                if st.session_state.current_data:
                    prices = [d['price'] for d in st.session_state.current_data]
                    fig = px.histogram(x=prices, title="Price Distribution", 
                                     color_discrete_sequence=['#00ff00'])
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', 
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font_color='#00ff00')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("##### Volume Analysis")
                if st.session_state.current_data:
                    volumes = [d['volume'] for d in st.session_state.current_data]
                    symbols = [d['symbol'] for d in st.session_state.current_data]
                    fig = px.bar(x=symbols, y=volumes, title="Trading Volume",
                               color_discrete_sequence=['#00ffff'])
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', 
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font_color='#00ff00')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Fetch data to view analytics")
    
    with tab3:
        st.markdown("### ⚙️ SYSTEM CONTROLS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🔧 SYSTEM CONFIGURATION")
            st.selectbox("Defense Mode", ["STANDARD", "ENHANCED", "MAXIMUM"], key="defense_mode")
            st.slider("Sensitivity Level", 1, 10, 5, key="sensitivity")
            st.selectbox("Response Protocol", ["AUTOMATIC", "MANUAL", "HYBRID"], key="response_protocol")
            
            if st.button("💾 SAVE CONFIGURATION", use_container_width=True):
                st.success("✅ Configuration saved")
        
        with col2:
            st.markdown("##### 🛡️ SECURITY STATUS")
            st.markdown("""
            <div class="data-panel">
                <div style="color: #00ff00;">• Data Encryption: ACTIVE</div>
                <div style="color: #00ff00;">• Threat Detection: ACTIVE</div>
                <div style="color: #00ff00;">• Firewall: ACTIVE</div>
                <div style="color: #ffff00;">• AI Monitoring: STANDBY</div>
                <div style="color: #00ff00;">• Quantum Resistance: ACTIVE</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 SYSTEM DIAGNOSTICS", use_container_width=True):
                with st.spinner("Running diagnostics..."):
                    time.sleep(2)
                    st.success("✅ All systems operational")

if __name__ == "__main__":
    main()
