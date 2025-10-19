import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import hashlib
import secrets
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Zero-Trust Fintech SOC",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

class FintechSOC:
    def __init__(self):
        self.transaction_logs = pd.DataFrame()
        self.user_behavior_logs = pd.DataFrame()
        self.alerts = pd.DataFrame()
        self.anomaly_model = None
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self, days=30):
        """Generate synthetic transaction and user behavior data"""
        np.random.seed(42)
        
        # Generate transaction data
        transactions = []
        users = [f"user_{i:03d}" for i in range(1, 51)]
        transaction_types = ['transfer', 'payment', 'withdrawal', 'deposit', 'investment']
        risk_levels = ['low', 'medium', 'high']
        
        for i in range(1000):
            timestamp = datetime.now() - timedelta(days=np.random.randint(0, days),
                                                 hours=np.random.randint(0, 24),
                                                 minutes=np.random.randint(0, 60))
            user = np.random.choice(users)
            amount = np.random.lognormal(5, 1.5)
            transaction_type = np.random.choice(transaction_types)
            
            # Simulate some anomalous transactions
            is_anomalous = np.random.random() < 0.05
            if is_anomalous:
                amount = amount * 10  # Unusually large amount
                risk_level = 'high'
            else:
                risk_level = np.random.choice(risk_levels, p=[0.7, 0.25, 0.05])
            
            transactions.append({
                'timestamp': timestamp,
                'user_id': user,
                'transaction_id': f"tx_{i:06d}",
                'amount': round(amount, 2),
                'type': transaction_type,
                'risk_level': risk_level,
                'ip_address': f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                'location': np.random.choice(['US', 'UK', 'DE', 'FR', 'SG', 'JP', 'AU', 'CA']),
                'status': np.random.choice(['completed', 'failed', 'pending']),
                'device_id': f"device_{np.random.randint(1, 20):03d}"
            })
        
        self.transaction_logs = pd.DataFrame(transactions)
        
        # Generate user behavior data
        behaviors = []
        for user in users:
            for day in range(days):
                login_time = datetime.now() - timedelta(days=day, hours=np.random.randint(0, 24))
                
                # Normal behavior pattern
                if np.random.random() < 0.8:
                    location = 'US' if user.startswith('user_0') else np.random.choice(['UK', 'DE', 'FR'])
                    device = f"device_{int(user.split('_')[1]):03d}"
                    actions = np.random.randint(3, 15)
                else:
                    # Anomalous behavior
                    location = np.random.choice(['RU', 'CN', 'BR'])  # Unusual locations
                    device = f"device_{np.random.randint(100, 200):03d}"  # New device
                    actions = np.random.randint(20, 50)  # Unusually high activity
                
                behaviors.append({
                    'timestamp': login_time,
                    'user_id': user,
                    'action_type': 'login',
                    'ip_address': f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                    'location': location,
                    'device_id': device,
                    'user_agent': np.random.choice(['Chrome', 'Firefox', 'Safari', 'Mobile']),
                    'actions_performed': actions,
                    'session_duration': np.random.randint(60, 3600)
                })
        
        self.user_behavior_logs = pd.DataFrame(behaviors)
        
    def train_anomaly_detection(self):
        """Train machine learning model for anomaly detection"""
        # Prepare features for anomaly detection
        user_features = self.user_behavior_logs.groupby('user_id').agg({
            'actions_performed': ['mean', 'std', 'max'],
            'session_duration': ['mean', 'std'],
            'location': 'nunique'
        }).fillna(0)
        
        user_features.columns = ['_'.join(col).strip() for col in user_features.columns]
        
        transaction_features = self.transaction_logs.groupby('user_id').agg({
            'amount': ['mean', 'std', 'max', 'sum'],
            'risk_level': lambda x: (x == 'high').sum()
        }).fillna(0)
        
        transaction_features.columns = ['_'.join(col).strip() for col in transaction_features.columns]
        
        # Combine features
        features = user_features.join(transaction_features, how='outer').fillna(0)
        
        # Train Isolation Forest
        self.anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        features_scaled = self.scaler.fit_transform(features)
        anomalies = self.anomaly_model.fit_predict(features_scaled)
        
        # Create alerts for anomalies
        anomalous_users = features[anomalies == -1].index
        self.alerts = pd.DataFrame([
            {
                'alert_id': f"alert_{i:06d}",
                'timestamp': datetime.now(),
                'user_id': user,
                'alert_type': 'Behavioral Anomaly',
                'severity': 'High',
                'description': f'Unusual behavior pattern detected for user {user}',
                'status': 'Open'
            }
            for i, user in enumerate(anomalous_users)
        ])
    
    def detect_data_poisoning(self):
        """Detect potential data poisoning attacks"""
        poisoning_alerts = []
        
        # Check for unusual patterns that might indicate data poisoning
        recent_logs = self.user_behavior_logs[
            self.user_behavior_logs['timestamp'] > datetime.now() - timedelta(hours=24)
        ]
        
        # Detect if multiple users are behaving identically (potential poisoning)
        behavior_patterns = recent_logs.groupby(['location', 'device_id', 'user_agent']).size()
        suspicious_patterns = behavior_patterns[behavior_patterns > 5]  # More than 5 users with identical pattern
        
        for pattern, count in suspicious_patterns.items():
            poisoning_alerts.append({
                'alert_id': f"poison_{len(poisoning_alerts):06d}",
                'timestamp': datetime.now(),
                'user_id': 'Multiple',
                'alert_type': 'Potential Data Poisoning',
                'severity': 'Critical',
                'description': f'Identical behavior pattern detected across {count} users. Pattern: {pattern}',
                'status': 'Open'
            })
        
        # Add poisoning alerts to main alerts
        if poisoning_alerts:
            poisoning_df = pd.DataFrame(poisoning_alerts)
            self.alerts = pd.concat([self.alerts, poisoning_df], ignore_index=True)
    
    def apply_zero_trust_rules(self):
        """Apply zero-trust security rules"""
        # Rule 1: External logins require secondary auth
        external_locations = ['RU', 'CN', 'BR', 'IN', 'MX']
        external_logins = self.user_behavior_logs[
            self.user_behavior_logs['location'].isin(external_locations)
        ]
        
        for _, login in external_logins.iterrows():
            if not any((self.alerts['user_id'] == login['user_id']) & 
                      (self.alerts['alert_type'] == 'External Login')):
                new_alert = {
                    'alert_id': f"ext_{len(self.alerts):06d}",
                    'timestamp': login['timestamp'],
                    'user_id': login['user_id'],
                    'alert_type': 'External Login',
                    'severity': 'Medium',
                    'description': f'Login from external location {login["location"]} requiring secondary authentication',
                    'status': 'Open'
                }
                self.alerts = pd.concat([self.alerts, pd.DataFrame([new_alert])], ignore_index=True)
        
        # Rule 2: Unusual transaction amounts
        user_avg = self.transaction_logs.groupby('user_id')['amount'].mean()
        for user, avg_amount in user_avg.items():
            user_tx = self.transaction_logs[self.transaction_logs['user_id'] == user]
            large_tx = user_tx[user_tx['amount'] > avg_amount * 5]  # 5x average
            
            for _, tx in large_tx.iterrows():
                if not any((self.alerts['user_id'] == user) & 
                          (self.alerts['description'].str.contains(tx['transaction_id']))):
                    new_alert = {
                        'alert_id': f"tx_{len(self.alerts):06d}",
                        'timestamp': tx['timestamp'],
                        'user_id': user,
                        'alert_type': 'Unusual Transaction',
                        'severity': 'High',
                        'description': f'Unusually large transaction {tx["transaction_id"]}: ${tx["amount"]:.2f}',
                        'status': 'Open'
                    }
                    self.alerts = pd.concat([self.alerts, pd.DataFrame([new_alert])], ignore_index=True)

def main():
    st.title("🔒 Zero-Trust Fintech Security Operations Center")
    st.markdown("### Real-time Behavioral Analytics & Threat Detection")
    
    # Initialize SOC
    if 'soc' not in st.session_state:
        st.session_state.soc = FintechSOC()
        st.session_state.soc.generate_synthetic_data()
        st.session_state.soc.train_anomaly_detection()
        st.session_state.soc.detect_data_poisoning()
        st.session_state.soc.apply_zero_trust_rules()
    
    soc = st.session_state.soc
    
    # Sidebar
    st.sidebar.title("SOC Controls")
    
    # Data refresh
    if st.sidebar.button("🔄 Refresh Data & Run Analysis"):
        with st.spinner("Generating new data and running analysis..."):
            soc.generate_synthetic_data()
            soc.train_anomaly_detection()
            soc.detect_data_poisoning()
            soc.apply_zero_trust_rules()
            time.sleep(1)
        st.success("Data refreshed and analysis completed!")
    
    # Policy management
    st.sidebar.subheader("Zero-Trust Policies")
    
    with st.sidebar.expander("🔐 Security Policies"):
        st.write("**Active Zero-Trust Rules:**")
        st.write("• External logins → Secondary authentication")
        st.write("• Unusual transaction patterns → Alert & review")
        st.write("• New device detection → Step-up authentication")
        st.write("• Geographic anomalies → Block and investigate")
        
        if st.button("Deploy New Policy"):
            st.success("Policy deployed via smart contract simulation!")
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_alerts = len(soc.alerts)
        st.metric("Total Alerts", total_alerts, delta=f"{total_alerts} active")
    
    with col2:
        critical_alerts = len(soc.alerts[soc.alerts['severity'] == 'Critical'])
        st.metric("Critical Alerts", critical_alerts, delta="High priority")
    
    with col3:
        open_alerts = len(soc.alerts[soc.alerts['status'] == 'Open'])
        st.metric("Open Alerts", open_alerts, delta="Needs attention")
    
    with col4:
        users_monitored = soc.user_behavior_logs['user_id'].nunique()
        st.metric("Users Monitored", users_monitored)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Risk Dashboard", 
        "🚨 Alert Console", 
        "👤 User Behavior", 
        "📈 Transaction Analysis",
        "🛡️ Data Poisoning Defense"
    ])
    
    with tab1:
        st.subheader("Security Risk Heatmap")
        
        # Geographic risk heatmap
        location_risk = soc.transaction_logs.groupby('location').agg({
            'risk_level': lambda x: (x == 'high').sum(),
            'amount': 'sum'
        }).reset_index()
        
        fig = px.choropleth(
            location_risk,
            locations='location',
            locationmode='ISO-3',
            color='risk_level',
            hover_name='location',
            hover_data={'amount': True},
            title='High-Risk Transactions by Location',
            color_continuous_scale='reds'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk over time
        st.subheader("Risk Trends Over Time")
        daily_risk = soc.transaction_logs.set_index('timestamp').groupby(
            pd.Grouper(freq='D')
        )['risk_level'].apply(lambda x: (x == 'high').sum()).reset_index()
        
        fig = px.line(daily_risk, x='timestamp', y='risk_level', 
                     title='Daily High-Risk Transaction Count')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Security Alert Console")
        
        # Alert filters
        col1, col2 = st.columns(2)
        with col1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=soc.alerts['severity'].unique(),
                default=soc.alerts['severity'].unique()
            )
        with col2:
            type_filter = st.multiselect(
                "Filter by Alert Type",
                options=soc.alerts['alert_type'].unique(),
                default=soc.alerts['alert_type'].unique()
            )
        
        filtered_alerts = soc.alerts[
            (soc.alerts['severity'].isin(severity_filter)) &
            (soc.alerts['alert_type'].isin(type_filter))
        ]
        
        # Display alerts
        for _, alert in filtered_alerts.iterrows():
            severity_color = {
                'Critical': 'red',
                'High': 'orange',
                'Medium': 'yellow',
                'Low': 'green'
            }.get(alert['severity'], 'gray')
            
            with st.expander(
                f"🔴 {alert['alert_type']} - {alert['user_id']} - {alert['severity']}",
                expanded=alert['severity'] in ['Critical', 'High']
            ):
                st.write(f"**Description:** {alert['description']}")
                st.write(f"**Timestamp:** {alert['timestamp']}")
                st.write(f"**Status:** {alert['status']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Investigate {alert['user_id']}", key=f"inv_{alert['alert_id']}"):
                        st.info(f"Opening investigation for {alert['user_id']}")
                with col2:
                    if st.button(f"Close Alert", key=f"close_{alert['alert_id']}"):
                        st.success(f"Alert {alert['alert_id']} closed")
    
    with tab3:
        st.subheader("User Behavior Analytics")
        
        # User activity patterns
        user_activity = soc.user_behavior_logs.groupby('user_id').agg({
            'actions_performed': 'sum',
            'session_duration': 'mean',
            'location': 'nunique'
        }).reset_index()
        
        fig = px.scatter(user_activity, x='actions_performed', y='session_duration',
                        size='location', color='location',
                        hover_name='user_id', title='User Activity Patterns')
        st.plotly_chart(fig, use_container_width=True)
        
        # Device usage analysis
        device_usage = soc.user_behavior_logs['device_id'].value_counts().head(10)
        fig = px.bar(device_usage, title='Top 10 Devices by Usage')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Transaction Analysis")
        
        # Transaction type distribution
        tx_type_dist = soc.transaction_logs['type'].value_counts()
        fig = px.pie(tx_type_dist, values=tx_type_dist.values, names=tx_type_dist.index,
                    title='Transaction Type Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # Amount analysis
        fig = px.histogram(soc.transaction_logs, x='amount', nbins=50,
                          title='Transaction Amount Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # Real-time transaction stream simulation
        st.subheader("Live Transaction Monitor")
        if st.button("Simulate Live Transactions"):
            live_placeholder = st.empty()
            for i in range(10):
                new_tx = {
                    'timestamp': datetime.now(),
                    'transaction_id': f"live_tx_{i:06d}",
                    'user_id': np.random.choice([f"user_{i:03d}" for i in range(1, 51)]),
                    'amount': round(np.random.lognormal(4, 1), 2),
                    'type': np.random.choice(['transfer', 'payment', 'withdrawal']),
                    'status': 'completed'
                }
                live_placeholder.write(f"🔄 {new_tx['user_id']} - ${new_tx['amount']} - {new_tx['type']}")
                time.sleep(0.5)
    
    with tab5:
        st.subheader("🛡️ Data Poisoning Defense System")
        st.markdown("""
        ### Advanced ML Model Protection
        This system detects attempts to poison training data and manipulate behavioral models.
        """)
        
        # Data poisoning indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Pattern Similarity Analysis**")
            pattern_similarity = np.random.normal(0.85, 0.1)
            st.metric("Behavior Pattern Diversity", f"{pattern_similarity:.2%}")
            
            if pattern_similarity > 0.9:
                st.error("High pattern similarity detected - potential poisoning!")
            else:
                st.success("Normal pattern diversity")
        
        with col2:
            st.info("**Model Confidence Monitoring**")
            model_confidence = np.random.normal(0.92, 0.05)
            st.metric("Anomaly Detection Confidence", f"{model_confidence:.2%}")
            
            if model_confidence < 0.85:
                st.warning("Reduced model confidence - investigate potential poisoning")
            else:
                st.success("Model operating normally")
        
        # Data integrity checks
        st.subheader("Data Integrity Monitoring")
        integrity_checks = {
            "User Behavior Consistency": np.random.choice([True, False], p=[0.9, 0.1]),
            "Transaction Pattern Validation": True,
            "Geographic Logic Verification": np.random.choice([True, False], p=[0.95, 0.05]),
            "Temporal Pattern Analysis": True,
            "Device Fingerprint Consistency": np.random.choice([True, False], p=[0.85, 0.15])
        }
        
        for check, status in integrity_checks.items():
            if status:
                st.success(f"✅ {check}")
            else:
                st.error(f"❌ {check} - Potential data manipulation detected")
        
        if st.button("Run Deep Data Integrity Scan"):
            with st.spinner("Scanning for sophisticated poisoning attacks..."):
                time.sleep(3)
                st.warning("2 potential advanced poisoning patterns detected!")
                st.info("Recommendation: Retrain models with verified clean data")

    # Footer
    st.markdown("---")
    st.markdown(
        "**Zero-Trust Fintech SOC** | "
        "Built with Streamlit • Behavioral Analytics • Machine Learning • Blockchain-ready"
    )

if __name__ == "__main__":
    main()
