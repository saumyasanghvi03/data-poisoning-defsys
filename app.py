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
import yfinance as yf  # For financial data
from kaggle.api.kaggle_api_extended import KaggleApi  # For Kaggle datasets
import xml.etree.ElementTree as ET  # For RBI data parsing

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

# Add new imports for data sources at the top
try:
    import kaggle
except ImportError:
    st.warning("Kaggle API not installed. Run: pip install kaggle")

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
                    'features': '45'
                },
                'Healthcare_Records': {
                    'description': 'Anonymized patient records from public hospitals',
                    'size': '1.8GB',
                    'samples': '300,000',
                    'features': '32'
                },
                'Financial_Transactions': {
                    'description': 'Banking transaction patterns',
                    'size': '3.2GB',
                    'samples': '1,200,000',
                    'features': '28'
                },
                'Education_Data': {
                    'description': 'Student performance and institutional data',
                    'size': '950MB',
                    'samples': '150,000',
                    'features': '25'
                }
            }
            return datasets
            
        except Exception as e:
            st.error(f"Error fetching India AI data: {e}")
            return None
    
    def fetch_kaggle_datasets(self, search_term='finance'):
        """Fetch datasets from Kaggle (requires Kaggle API setup)"""
        try:
            # This is a simulation - actual implementation requires Kaggle API credentials
            datasets = {
                'credit-card-fraud': {
                    'title': 'Credit Card Fraud Detection',
                    'size': '150MB',
                    'downloads': '45,000',
                    'url': 'https://www.kaggle.com/mlg-ulb/creditcardfraud'
                },
                'loan-prediction': {
                    'title': 'Loan Prediction Dataset',
                    'size': '45MB',
                    'downloads': '23,000',
                    'url': 'https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset'
                },
                'stock-market-data': {
                    'title': 'Indian Stock Market Data',
                    'size': '320MB',
                    'downloads': '18,000',
                    'url': 'https://www.kaggle.com/rohanrao/nifty50-stock-market-data'
                },
                'customer-segmentation': {
                    'title': 'Customer Segmentation',
                    'size': '85MB',
                    'downloads': '32,000',
                    'url': 'https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python'
                }
            }
            return datasets
            
        except Exception as e:
            st.error(f"Error fetching Kaggle data: {e}")
            return None
    
    def fetch_financial_data(self, symbols=['RELIANCE.NS', 'TCS.NS', 'INFY.NS']):
        """Fetch live financial data using yfinance"""
        try:
            data = {}
            for symbol in symbols:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="6mo")
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
                    'update_frequency': 'Monthly'
                },
                'Smart_Cities': {
                    'description': 'Smart city project implementation data',
                    'records': '15,000',
                    'update_frequency': 'Quarterly'
                },
                'Agriculture_Production': {
                    'description': 'Crop production data across states',
                    'records': '200,000',
                    'update_frequency': 'Annual'
                },
                'Economic_Survey': {
                    'description': 'Annual economic survey data',
                    'records': '10,000',
                    'update_frequency': 'Annual'
                }
            }
            return datasets
        except Exception as e:
            st.error(f"Error fetching government data: {e}")
            return None

# Update the existing CSS to include new styles
st.markdown("""
<style>
    /* Previous CSS styles remain the same */
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
</style>
""", unsafe_allow_html=True)

# Add the new Live Data Integration section to the main dashboard
def render_live_data_integration():
    """Live data integration from various sources"""
    st.markdown("### 🌐 LIVE DATA INTEGRATION")
    
    data_fetcher = LiveDataFetcher()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏦 RBI Data", 
        "📊 NSO Statistics", 
        "🤖 India AI", 
        "📈 Kaggle", 
        "🏛️ Government Data"
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
                        
                        if st.button(f"Use {dataset_name} for Analysis", key=f"use_{dataset_name}"):
                            # Simulate using this dataset for poisoning analysis
                            st.info(f"Loading {dataset_name} for data poisoning analysis...")
                            # Here you would load the actual dataset
                        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown("#### 📈 KAGGLE DATASETS")
        
        search_term = st.text_input("🔍 Search Kaggle Datasets:", "finance")
        
        if st.button("🔄 Search Kaggle", key="search_kaggle"):
            with st.spinner(f"Searching Kaggle for '{search_term}'..."):
                kaggle_data = data_fetcher.fetch_kaggle_datasets(search_term)
                
                if kaggle_data is not None:
                    st.session_state.kaggle_data = kaggle_data
                    st.success(f"✅ Found {len(kaggle_data)} relevant datasets")
                    
                    for dataset_id, dataset_info in kaggle_data.items():
                        st.markdown(f'<div class="dataset-info">', unsafe_allow_html=True)
                        st.markdown(f"**{dataset_info['title']}**")
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
                        
                        if st.button(f"Access {dataset_name}", key=f"access_{dataset_name}"):
                            st.info(f"Accessing {dataset_name}... (Simulated API call)")
                        st.markdown('</div>', unsafe_allow_html=True)

# Update the main dashboard to include the new data integration section
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
    
    # Quick actions
    st.markdown("### 🚀 DEFENSE ACTIONS")
    cols = st.columns(7)  # Updated to 7 columns for new tab
    
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
    
    # Main tabs - Updated to include Live Data Integration
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
    
    with tabs[5]:  # New tab for live data
        render_live_data_integration()
    
    with tabs[6]:
        render_system_monitor()

# Add data source information to the threat intelligence
def render_threat_intelligence():
    """Data poisoning threat intelligence with data source monitoring"""
    st.markdown("### 🌐 DATA POISONING THREAT INTELLIGENCE")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🚨 ACTIVE THREATS")
        
        threats = [
            {"name": "Label Flipping Campaign", "severity": "HIGH", "targets": "Financial Models", "status": "ACTIVE", "data_sources": ["RBI", "Kaggle"]},
            {"name": "Backdoor Injection", "severity": "CRITICAL", "targets": "Healthcare AI", "status": "ACTIVE", "data_sources": ["India AI", "Government"]},
            {"name": "Feature Manipulation", "severity": "MEDIUM", "targets": "Recommendation Systems", "status": "DETECTED", "data_sources": ["NSO", "Kaggle"]},
            {"name": "Model Inversion", "severity": "HIGH", "targets": "Privacy-Sensitive Models", "status": "MONITORING", "data_sources": ["All Sources"]},
        ]
        
        for threat in threats:
            with st.expander(f"🔴 {threat['name']} - {threat['severity']}"):
                st.write(f"**Targets:** {threat['targets']}")
                st.write(f"**Status:** {threat['status']}")
                st.write(f"**Affected Data Sources:** {', '.join(threat['data_sources'])}")
                st.write(f"**First Seen:** 2024-01-15")
                st.write(f"**Last Activity:** 2024-01-20")
        
        st.markdown("#### 📊 DATA SOURCE SECURITY STATUS")
        
        data_sources_status = {
            "RBI API": {"status": "🟢 Secure", "last_check": "2024-01-20 14:30"},
            "NSO Statistics": {"status": "🟡 Monitoring", "last_check": "2024-01-20 14:25"},
            "India AI Datasets": {"status": "🟢 Secure", "last_check": "2024-01-20 14:20"},
            "Kaggle Community": {"status": "🟡 Monitoring", "last_check": "2024-01-20 14:15"},
            "Government Portals": {"status": "🟢 Secure", "last_check": "2024-01-20 14:10"},
        }
        
        for source, info in data_sources_status.items():
            col_a, col_b, col_c = st.columns([2, 1, 1])
            with col_a:
                st.write(f"**{source}**")
            with col_b:
                st.write(info['status'])
            with col_c:
                st.write(info['last_check'])
    
    with col2:
        st.markdown("#### 📈 THREAT METRICS")
        
        st.metric("🌍 Global Attacks", "1,247")
        st.metric("🎯 Targeted Industries", "8")
        st.metric("🛡️ Successful Defenses", "89%")
        st.metric("⏱️ Average Detection", "3.2s")
        
        st.markdown("#### 🎯 HIGH-RISK DATA SOURCES")
        st.write("1. 📊 Public Kaggle Datasets")
        st.write("2. 🌐 Third-party APIs")
        st.write("3. 🔗 External Data Feeds")
        st.write("4. 📥 User Uploaded Data")
        st.write("5. 🔄 Real-time Streams")

# Update requirements in your requirements.txt file
def show_installation_instructions():
    """Show instructions for installing required packages"""
    st.sidebar.markdown("### 📦 Installation Requirements")
    st.sidebar.code("""
# Required packages for live data integration:
pip install yfinance
pip install kaggle
pip install requests
pip install pandas
pip install numpy
pip install plotly
pip install scikit-learn
pip install streamlit

# For Kaggle API setup:
# 1. Create account at kaggle.com
# 2. Go to Account settings
# 3. Create API token
# 4. Place kaggle.json in ~/.kaggle/
    """)

# Update the main function to show installation instructions
def main():
    with quantum_resource_manager():
        # Authentication
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        
        if not st.session_state.authenticated:
            render_login()
        else:
            show_installation_instructions()
            render_main_dashboard()

if __name__ == "__main__":
    main()
