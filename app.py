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
import psutil
import platform
from collections import deque
import uuid

warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CYBER DEFENSE TERMINAL | REAL-TIME SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ULTRA REALISTIC TERMINAL CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=VT323&family=Courier+Prime&family=Orbitron:wght@400;700;900&display=swap');
    
    /* Complete page reset */
    .main {
        background: #0a0e1a !important;
        color: #00ff41 !important;
        font-family: 'Share Tech Mono', 'Courier Prime', monospace !important;
        padding: 0 !important;
    }
    
    /* Hide all Streamlit branding */
    #MainMenu, footer, header, .stDeployButton {visibility: hidden !important; display: none !important;}
    
    /* Terminal header with scanlines */
    .terminal-header {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-bottom: 3px solid #00ff41;
        padding: 1.5rem 2rem;
        margin: -1rem -1rem 0 -1rem;
        box-shadow: 0 0 40px #00ff4133, inset 0 0 30px #00000066;
        position: relative;
        overflow: hidden;
    }
    
    .terminal-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0, 255, 65, 0.03) 2px,
            rgba(0, 255, 65, 0.03) 4px
        );
        pointer-events: none;
    }
    
    /* Realistic terminal window */
    .terminal-window {
        background: #000000;
        border: 1px solid #00ff41;
        border-radius: 6px;
        padding: 0;
        margin: 1rem 0;
        box-shadow: 0 0 30px #00ff4133;
        font-family: 'VT323', 'Share Tech Mono', monospace;
        position: relative;
        overflow: hidden;
    }
    
    .terminal-window::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0, 255, 65, 0.05) 2px,
            rgba(0, 255, 65, 0.05) 4px
        );
        pointer-events: none;
        z-index: 1;
    }
    
    .terminal-titlebar {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1419 100%);
        border-bottom: 1px solid #00ff41;
        padding: 0.5rem 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
    }
    
    .terminal-buttons {
        display: flex;
        gap: 0.5rem;
    }
    
    .terminal-button {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
    }
    
    .btn-close { background: #ff5f56; box-shadow: 0 0 5px #ff5f56; }
    .btn-minimize { background: #ffbd2e; box-shadow: 0 0 5px #ffbd2e; }
    .btn-maximize { background: #27c93f; box-shadow: 0 0 5px #27c93f; }
    
    .terminal-content {
        padding: 1rem;
        min-height: 300px;
        max-height: 500px;
        overflow-y: auto;
        background: #000000;
        color: #00ff41;
        font-size: 0.95rem;
        line-height: 1.6;
        position: relative;
        z-index: 2;
    }
    
    /* Realistic log entries with timestamps */
    .log-line {
        margin: 0.3rem 0;
        font-family: 'VT323', monospace;
        font-size: 1.1rem;
        display: flex;
        gap: 1rem;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .log-timestamp {
        color: #666;
        min-width: 100px;
        font-family: 'Courier Prime', monospace;
    }
    
    .log-level-INFO { color: #00ff41; }
    .log-level-WARN { color: #ffbd2e; }
    .log-level-ERROR { color: #ff5f56; }
    .log-level-CRITICAL { color: #ff0000; animation: blink 1s infinite; }
    .log-level-SUCCESS { color: #27c93f; }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.5; }
    }
    
    .log-source {
        color: #58a6ff;
        min-width: 150px;
        font-weight: bold;
    }
    
    .log-message {
        color: #c9d1d9;
        flex: 1;
    }
    
    /* Command prompt */
    .command-prompt {
        background: #0d1117;
        border: 2px solid #00ff41;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'VT323', monospace;
        font-size: 1.2rem;
        box-shadow: 0 0 20px #00ff4133;
    }
    
    .prompt-line {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .prompt-user {
        color: #27c93f;
        font-weight: bold;
    }
    
    .prompt-separator {
        color: #666;
    }
    
    .prompt-path {
        color: #58a6ff;
    }
    
    .prompt-cursor {
        background: #00ff41;
        color: #000;
        padding: 0 0.2rem;
        animation: cursor-blink 1s infinite;
    }
    
    @keyframes cursor-blink {
        0%, 49% { opacity: 1; }
        50%, 100% { opacity: 0; }
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-left: 4px solid #00ff41;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-left-color: #58a6ff;
        box-shadow: 0 0 20px #00ff4133;
        transform: translateX(5px);
    }
    
    .metric-label {
        color: #8b949e;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        color: #00ff41;
        font-size: 1.8rem;
        font-weight: bold;
        font-family: 'Orbitron', monospace;
        margin: 0.5rem 0;
    }
    
    .metric-change {
        font-size: 0.9rem;
    }
    
    .metric-up { color: #27c93f; }
    .metric-down { color: #ff5f56; }
    
    /* Alert boxes */
    .alert-box {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        border-left: 5px solid;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .alert-critical {
        background: linear-gradient(90deg, #4a0000 0%, #2a0000 100%);
        border-left-color: #ff0000;
        color: #ff6b6b;
        animation: pulse-critical 2s infinite;
    }
    
    @keyframes pulse-critical {
        0%, 100% { box-shadow: 0 0 10px #ff000066; }
        50% { box-shadow: 0 0 30px #ff0000aa; }
    }
    
    .alert-high {
        background: linear-gradient(90deg, #4a2600 0%, #2a1600 100%);
        border-left-color: #ff6b00;
        color: #ffaa66;
    }
    
    .alert-medium {
        background: linear-gradient(90deg, #4a4a00 0%, #2a2a00 100%);
        border-left-color: #ffff00;
        color: #ffff99;
    }
    
    .alert-low {
        background: linear-gradient(90deg, #004a00 0%, #002a00 100%);
        border-left-color: #00ff41;
        color: #99ff99;
    }
    
    /* Status indicators with glow */
    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.5rem;
        animation: glow 2s infinite;
    }
    
    @keyframes glow {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .status-online {
        background: #00ff41;
        box-shadow: 0 0 10px #00ff41, 0 0 20px #00ff41;
    }
    
    .status-warning {
        background: #ffbd2e;
        box-shadow: 0 0 10px #ffbd2e, 0 0 20px #ffbd2e;
    }
    
    .status-offline {
        background: #ff5f56;
        box-shadow: 0 0 10px #ff5f56, 0 0 20px #ff5f56;
    }
    
    /* Progress bars */
    .progress-bar-container {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
        position: relative;
    }
    
    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #00ff41 0%, #27c93f 100%);
        border-radius: 10px;
        transition: width 0.5s ease;
        box-shadow: 0 0 10px #00ff41;
    }
    
    .progress-bar-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: #fff;
        font-size: 0.75rem;
        font-weight: bold;
        text-shadow: 0 0 5px #000;
    }
    
    /* Data table styling */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.9rem;
    }
    
    .data-table th {
        background: #161b22;
        color: #00ff41;
        padding: 0.75rem;
        text-align: left;
        border-bottom: 2px solid #00ff41;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .data-table td {
        padding: 0.75rem;
        border-bottom: 1px solid #30363d;
        color: #c9d1d9;
    }
    
    .data-table tr:hover {
        background: #0d111788;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0d1117;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00ff41;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #27c93f;
    }
    
    /* ASCII art container */
    .ascii-art {
        font-family: 'Courier Prime', monospace;
        color: #00ff41;
        font-size: 0.7rem;
        line-height: 1.2;
        white-space: pre;
        text-shadow: 0 0 5px #00ff41;
    }
    
    /* Network activity indicator */
    .network-pulse {
        width: 8px;
        height: 8px;
        background: #00ff41;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.5); opacity: 0.5; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #0d1117;
        border-bottom: 2px solid #00ff41;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #161b22;
        color: #8b949e;
        border: 1px solid #30363d;
        border-bottom: none;
        border-radius: 6px 6px 0 0;
        padding: 0.75rem 1.5rem;
        font-family: 'Share Tech Mono', monospace;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #1c2128;
        color: #00ff41;
    }
    
    .stTabs [aria-selected="true"] {
        background: #0d1117 !important;
        color: #00ff41 !important;
        border-color: #00ff41 !important;
        box-shadow: 0 -2px 10px #00ff4133;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 2px solid #00ff41;
        color: #00ff41;
        font-family: 'Share Tech Mono', monospace;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border-radius: 4px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        box-shadow: 0 0 20px #00ff41;
        transform: translateY(-2px);
    }
    
    /* Matrix rain background */
    #matrix-canvas {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        opacity: 0.15;
        pointer-events: none;
    }
</style>

<canvas id="matrix-canvas"></canvas>

<script>
// Enhanced Matrix Rain Effect
const canvas = document.getElementById('matrix-canvas');
const ctx = canvas.getContext('2d');

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const characters = 'ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜｦﾝ01';
const fontSize = 14;
const columns = canvas.width / fontSize;

const drops = Array(Math.floor(columns)).fill(1);

function draw() {
    ctx.fillStyle = 'rgba(10, 14, 26, 0.05)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.fillStyle = '#00ff41';
    ctx.font = fontSize + 'px monospace';
    
    for (let i = 0; i < drops.length; i++) {
        const text = characters[Math.floor(Math.random() * characters.length)];
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);
        
        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
            drops[i] = 0;
        }
        drops[i]++;
    }
}

setInterval(draw, 33);

window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});
</script>
""", unsafe_allow_html=True)

# --- SYSTEM MONITORING CLASS ---
class SystemMonitor:
    """Real-time system monitoring"""
    
    @staticmethod
    def get_cpu_usage():
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return random.uniform(45, 85)
    
    @staticmethod
    def get_memory_usage():
        try:
            return psutil.virtual_memory().percent
        except:
            return random.uniform(60, 90)
    
    @staticmethod
    def get_disk_usage():
        try:
            return psutil.disk_usage('/').percent
        except:
            return random.uniform(40, 70)
    
    @staticmethod
    def get_network_stats():
        try:
            net = psutil.net_io_counters()
            return {
                'bytes_sent': net.bytes_sent,
                'bytes_recv': net.bytes_recv,
                'packets_sent': net.packets_sent,
                'packets_recv': net.packets_recv
            }
        except:
            return {
                'bytes_sent': random.randint(1000000, 5000000),
                'bytes_recv': random.randint(5000000, 10000000),
                'packets_sent': random.randint(10000, 50000),
                'packets_recv': random.randint(50000, 100000)
            }
    
    @staticmethod
    def get_system_info():
        try:
            return {
                'platform': platform.system(),
                'release': platform.release(),
                'processor': platform.processor(),
                'architecture': platform.machine()
            }
        except:
            return {
                'platform': 'Linux',
                'release': '5.15.0',
                'processor': 'x86_64',
                'architecture': 'x86_64'
            }

# --- LOG STREAM CLASS ---
class LogStream:
    """Real-time log streaming with realistic entries"""
    
    def __init__(self, max_logs=100):
        self.logs = deque(maxlen=max_logs)
        self.log_sources = [
            'FIREWALL', 'IDS', 'THREAT-INTEL', 'DATA-VALIDATOR',
            'ML-ENGINE', 'NETWORK-MON', 'AUTH-SERVICE', 'BACKUP-SYS',
            'CRYPTO-MODULE', 'API-GATEWAY', 'DEFENSE-CORE', 'SCANNER'
        ]
        
    def generate_log(self):
        """Generate realistic log entry"""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        source = random.choice(self.log_sources)
        
        log_templates = {
            'INFO': [
                f"Connection established from {self._random_ip()}",
                f"Health check passed - System operational",
                f"Cache updated - {random.randint(100, 999)} entries",
                f"Backup completed - {random.randint(1, 99)}GB transferred",
                f"API request processed - Response time {random.randint(10, 200)}ms",
                f"Signature database updated - {random.randint(1000, 9999)} new entries"
            ],
            'WARN': [
                f"High CPU usage detected - {random.randint(80, 95)}%",
                f"Unusual traffic pattern from {self._random_ip()}",
                f"Rate limit approaching for API endpoint",
                f"Certificate expiring in {random.randint(7, 30)} days",
                f"Memory usage above threshold - {random.randint(85, 95)}%"
            ],
            'ERROR': [
                f"Connection timeout to {self._random_ip()}",
                f"Authentication failed for user {self._random_user()}",
                f"Data validation error - Checksum mismatch",
                f"Database query failed - Retrying...",
                f"Failed to parse response from external API"
            ],
            'CRITICAL': [
                f"🚨 Multiple failed login attempts from {self._random_ip()}",
                f"🚨 Data poisoning attempt detected and blocked",
                f"🚨 Anomalous behavior detected in model predictions",
                f"🚨 Potential DDoS attack - {random.randint(10000, 99999)} requests/sec"
            ],
            'SUCCESS': [
                f"✅ Threat neutralized - Source: {self._random_ip()}",
                f"✅ Security scan completed - 0 vulnerabilities",
                f"✅ Model validation passed - Accuracy {random.uniform(95, 99):.2f}%",
                f"✅ Encryption upgraded to quantum-resistant algorithm"
            ]
        }
        
        level = random.choices(
            ['INFO', 'WARN', 'ERROR', 'CRITICAL', 'SUCCESS'],
            weights=[60, 20, 10, 5, 5]
        )[0]
        
        message = random.choice(log_templates[level])
        
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'source': source,
            'message': message
        }
        
        self.logs.append(log_entry)
        return log_entry
    
    def _random_ip(self):
        return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    
    def _random_user(self):
        users = ['admin', 'operator', 'analyst', 'viewer', 'api_service']
        return random.choice(users)
    
    def get_recent_logs(self, count=20):
        return list(self.logs)[-count:]

# --- THREAT GENERATOR ---
class ThreatGenerator:
    """Generate realistic threat scenarios"""
    
    @staticmethod
    def generate_threats():
        threats = [
            {
                'id': str(uuid.uuid4())[:8].upper(),
                'type': 'DATA_POISONING',
                'severity': 'CRITICAL',
                'source': f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
                'target': random.choice(['ML_MODEL_A', 'TRAINING_PIPELINE', 'DATA_WAREHOUSE']),
                'status': random.choice(['DETECTED', 'MITIGATED', 'MONITORING']),
                'confidence': random.uniform(85, 99)
            },
            {
                'id': str(uuid.uuid4())[:8].upper(),
                'type': 'MODEL_EVASION',
                'severity': 'HIGH',
                'source': f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
                'target': random.choice(['FRAUD_DETECTOR', 'ANOMALY_SYSTEM', 'CLASSIFIER']),
                'status': random.choice(['DETECTED', 'ANALYZING', 'BLOCKED']),
                'confidence': random.uniform(75, 95)
            },
            {
                'id': str(uuid.uuid4())[:8].upper(),
                'type': 'API_ABUSE',
                'severity': random.choice(['MEDIUM', 'HIGH']),
                'source': f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
                'target': 'API_ENDPOINT_/v1/predict',
                'status': random.choice(['RATE_LIMITED', 'BLOCKED', 'MONITORING']),
                'confidence': random.uniform(80, 95)
            }
        ]
        return random.sample(threats, random.randint(1, 3))

# --- ASCII ART BANNER ---
ASCII_BANNER = """
╔═══════════════════════════════════════════════════════════════════════╗
║   ██████╗██╗   ██╗██████╗ ███████╗██████╗     ██████╗ ███████╗███████╗║
║  ██╔════╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗    ██╔══██╗██╔════╝██╔════╝║
║  ██║      ╚████╔╝ ██████╔╝█████╗  ██████╔╝    ██║  ██║█████╗  █████╗  ║
║  ██║       ╚██╔╝  ██╔══██╗██╔══╝  ██╔══██╗    ██║  ██║██╔══╝  ██╔══╝  ║
║  ╚██████╗   ██║   ██████╔╝███████╗██║  ██║    ██████╔╝███████╗██║     ║
║   ╚═════╝   ╚═╝   ╚═════╝ ╚══════╝╚═╝  ╚═╝    ╚═════╝ ╚══════╝╚═╝     ║
║                                                                         ║
║            REAL-TIME DATA POISONING DEFENSE TERMINAL v3.7.2            ║
║                    CLASSIFIED - AUTHORIZED PERSONNEL ONLY              ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

# --- INITIALIZE SESSION STATE ---
if 'log_stream' not in st.session_state:
    st.session_state.log_stream = LogStream()
    # Pre-populate with initial logs
    for _ in range(10):
        st.session_state.log_stream.generate_log()

if 'system_monitor' not in st.session_state:
    st.session_state.system_monitor = SystemMonitor()

if 'command_history' not in st.session_state:
    st.session_state.command_history = []

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# --- TERMINAL COMPONENTS ---

def render_terminal_header():
    """Render realistic terminal header"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    system_info = st.session_state.system_monitor.get_system_info()
    
    st.markdown(f"""
    <div class="terminal-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-family: 'Orbitron', sans-serif; font-size: 1.8rem; color: #00ff41;">
                    🛡️ CYBER DEFENSE TERMINAL
                </h1>
                <div style="font-size: 0.9rem; color: #666; margin-top: 0.3rem;">
                    SOC-PRIME-01 | {system_info['platform']} {system_info['architecture']} | CLEARANCE: TOP SECRET
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.1rem; color: #00ff41;">
                    <span class="status-dot status-online"></span> SYSTEMS OPERATIONAL
                </div>
                <div style="font-size: 0.9rem; color: #666; margin-top: 0.3rem;">
                    {current_time}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_ascii_banner():
    """Render ASCII art banner"""
    st.markdown(f"""
    <div class="ascii-art">
{ASCII_BANNER}
    </div>
    """, unsafe_allow_html=True)

def render_system_metrics():
    """Render real-time system metrics"""
    st.markdown("### 📊 SYSTEM PERFORMANCE MONITOR")
    
    col1, col2, col3, col4 = st.columns(4)
    
    cpu = st.session_state.system_monitor.get_cpu_usage()
    memory = st.session_state.system_monitor.get_memory_usage()
    disk = st.session_state.system_monitor.get_disk_usage()
    network = st.session_state.system_monitor.get_network_stats()
    
    with col1:
        cpu_color = "#00ff41" if cpu < 70 else "#ffbd2e" if cpu < 85 else "#ff5f56"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CPU USAGE</div>
            <div class="metric-value" style="color: {cpu_color};">{cpu:.1f}%</div>
            <div class="progress-bar-container">
                <div class="progress-bar-fill" style="width: {cpu}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        mem_color = "#00ff41" if memory < 70 else "#ffbd2e" if memory < 85 else "#ff5f56"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MEMORY</div>
            <div class="metric-value" style="color: {mem_color};">{memory:.1f}%</div>
            <div class="progress-bar-container">
                <div class="progress-bar-fill" style="width: {memory}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        disk_color = "#00ff41" if disk < 70 else "#ffbd2e" if disk < 85 else "#ff5f56"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">DISK I/O</div>
            <div class="metric-value" style="color: {disk_color};">{disk:.1f}%</div>
            <div class="progress-bar-container">
                <div class="progress-bar-fill" style="width: {disk}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        net_activity = min(100, (network['packets_recv'] % 1000) / 10)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">NETWORK <span class="network-pulse"></span></div>
            <div class="metric-value" style="color: #58a6ff;">{network['packets_recv']:,}</div>
            <div style="font-size: 0.8rem; color: #666;">packets received</div>
        </div>
        """, unsafe_allow_html=True)

def render_live_terminal_logs():
    """Render live streaming terminal logs"""
    st.markdown("### 📡 LIVE SYSTEM LOGS")
    
    # Generate new log entry
    new_log = st.session_state.log_stream.generate_log()
    
    # Get recent logs
    logs = st.session_state.log_stream.get_recent_logs(25)
    
    log_html = """
    <div class="terminal-window">
        <div class="terminal-titlebar">
            <div style="color: #00ff41; font-weight: bold;">⚡ REAL-TIME EVENT STREAM</div>
            <div class="terminal-buttons">
                <span class="terminal-button btn-close"></span>
                <span class="terminal-button btn-minimize"></span>
                <span class="terminal-button btn-maximize"></span>
            </div>
        </div>
        <div class="terminal-content">
    """
    
    for log in logs:
        log_html += f"""
        <div class="log-line">
            <span class="log-timestamp">{log['timestamp']}</span>
            <span class="log-level-{log['level']}">[{log['level']}]</span>
            <span class="log-source">{log['source']}</span>
            <span class="log-message">{log['message']}</span>
        </div>
        """
    
    log_html += """
        </div>
    </div>
    """
    
    st.markdown(log_html, unsafe_allow_html=True)

def render_command_interface():
    """Render interactive command-line interface"""
    st.markdown("### 💻 COMMAND INTERFACE")
    
    # Command input
    user = "operator"
    path = "~/defense-terminal"
    
    st.markdown(f"""
    <div class="command-prompt">
        <div class="prompt-line">
            <span class="prompt-user">{user}@soc-prime</span>
            <span class="prompt-separator">:</span>
            <span class="prompt-path">{path}</span>
            <span style="color: #00ff41;">$</span>
            <span class="prompt-cursor">_</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        command = st.text_input("", placeholder="Enter command (e.g., scan, status, threats, analyze)...", label_visibility="collapsed")
    
    with col2:
        if st.button("🚀 EXECUTE", use_container_width=True):
            if command:
                st.session_state.command_history.append({
                    'command': command,
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'output': execute_command(command)
                })
    
    # Display command output
    if st.session_state.command_history:
        latest = st.session_state.command_history[-1]
        st.markdown(f"""
        <div class="terminal-window">
            <div class="terminal-titlebar">
                <div style="color: #00ff41;">COMMAND OUTPUT: {latest['command']}</div>
            </div>
            <div class="terminal-content">
                {latest['output']}
            </div>
        </div>
        """, unsafe_allow_html=True)

def execute_command(command):
    """Execute terminal commands"""
    cmd = command.lower().strip()
    
    if cmd == 'scan':
        return """
        <div class="log-line">
            <span class="log-level-INFO">[INFO]</span>
            <span>Initiating comprehensive security scan...</span>
        </div>
        <div class="log-line">
            <span class="log-level-SUCCESS">[SUCCESS]</span>
            <span>Scanned 1,247 endpoints - 0 vulnerabilities detected</span>
        </div>
        <div class="log-line">
            <span class="log-level-SUCCESS">[SUCCESS]</span>
            <span>All defense systems operational</span>
        </div>
        """
    
    elif cmd == 'status':
        return """
        <div class="log-line">
            <span class="log-level-INFO">[SYSTEM]</span>
            <span>Defense Status: OPTIMAL</span>
        </div>
        <div class="log-line">
            <span class="log-level-INFO">[SYSTEM]</span>
            <span>Threat Level: LOW</span>
        </div>
        <div class="log-line">
            <span class="log-level-INFO">[SYSTEM]</span>
            <span>Active Defenses: 12/12</span>
        </div>
        """
    
    elif cmd == 'threats':
        threats = ThreatGenerator.generate_threats()
        output = '<div class="log-line"><span class="log-level-WARN">[THREAT-INTEL]</span><span>Active threats detected:</span></div>'
        for threat in threats:
            severity_class = f"log-level-{threat['severity']}" if threat['severity'] == 'CRITICAL' else "log-level-WARN"
            output += f'''
            <div class="log-line">
                <span class="{severity_class}">[{threat['severity']}]</span>
                <span>{threat['type']} | ID: {threat['id']} | Status: {threat['status']}</span>
            </div>
            '''
        return output
    
    elif cmd == 'analyze':
        return """
        <div class="log-line">
            <span class="log-level-INFO">[ANALYTICS]</span>
            <span>Running advanced ML analysis...</span>
        </div>
        <div class="log-line">
            <span class="log-level-SUCCESS">[ANALYTICS]</span>
            <span>Model accuracy: 98.7% | Anomaly score: 0.03</span>
        </div>
        <div class="log-line">
            <span class="log-level-SUCCESS">[ANALYTICS]</span>
            <span>Data quality: EXCELLENT | Confidence: 99.2%</span>
        </div>
        """
    
    elif cmd == 'help':
        return """
        <div class="log-line"><span style="color: #00ff41;">Available Commands:</span></div>
        <div class="log-line"><span>  scan     - Run security scan</span></div>
        <div class="log-line"><span>  status   - Check system status</span></div>
        <div class="log-line"><span>  threats  - List active threats</span></div>
        <div class="log-line"><span>  analyze  - Run ML analysis</span></div>
        <div class="log-line"><span>  clear    - Clear screen</span></div>
        """
    
    else:
        return f'''
        <div class="log-line">
            <span class="log-level-ERROR">[ERROR]</span>
            <span>Command not found: {command}</span>
        </div>
        <div class="log-line">
            <span>Type 'help' for available commands</span>
        </div>
        '''

def render_threat_dashboard():
    """Render active threats dashboard"""
    st.markdown("### 🎯 ACTIVE THREAT INTELLIGENCE")
    
    threats = ThreatGenerator.generate_threats()
    
    for threat in threats:
        severity_class = threat['severity'].lower()
        st.markdown(f"""
        <div class="alert-box alert-{severity_class}">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="flex: 1;">
                    <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">
                        🚨 {threat['type'].replace('_', ' ')}
                    </div>
                    <div style="font-size: 0.9rem; margin: 0.3rem 0;">
                        <strong>ID:</strong> {threat['id']} | 
                        <strong>Source:</strong> {threat['source']} | 
                        <strong>Target:</strong> {threat['target']}
                    </div>
                    <div style="font-size: 0.9rem; margin: 0.3rem 0;">
                        <strong>Status:</strong> {threat['status']} | 
                        <strong>Confidence:</strong> {threat['confidence']:.1f}%
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.5rem; font-weight: bold;">{threat['severity']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_network_map():
    """Render network activity map"""
    st.markdown("### 🌍 GLOBAL THREAT MAP")
    
    # Generate random threat locations
    threat_locations = []
    for _ in range(15):
        threat_locations.append({
            'lat': random.uniform(-60, 70),
            'lon': random.uniform(-180, 180),
            'severity': random.choice(['low', 'medium', 'high', 'critical']),
            'count': random.randint(1, 50)
        })
    
    df_threats = pd.DataFrame(threat_locations)
    
    # Create severity scores for coloring
    severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
    df_threats['severity_score'] = df_threats['severity'].map(severity_map)
    
    fig = px.scatter_geo(
        df_threats,
        lat='lat',
        lon='lon',
        size='count',
        color='severity_score',
        color_continuous_scale=['#00ff41', '#ffbd2e', '#ff6b00', '#ff0000'],
        size_max=30,
        title="Real-Time Threat Origins"
    )
    
    fig.update_layout(
        geo=dict(
            bgcolor='#0a0e1a',
            landcolor='#1c2128',
            coastlinecolor='#00ff41',
            showcountries=True,
            countrycolor='#30363d'
        ),
        paper_bgcolor='#0a0e1a',
        font=dict(color='#00ff41', family='Share Tech Mono'),
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_data_flow_visualization():
    """Render real-time data flow"""
    st.markdown("### 📊 REAL-TIME DATA FLOW ANALYTICS")
    
    # Generate time series data
    timestamps = pd.date_range(end=datetime.now(), periods=60, freq='1S')
    
    data_streams = {
        'Legitimate Traffic': [random.randint(800, 1200) for _ in range(60)],
        'Suspicious Activity': [random.randint(10, 100) for _ in range(60)],
        'Blocked Threats': [random.randint(5, 50) for _ in range(60)]
    }
    
    fig = go.Figure()
    
    colors = {'Legitimate Traffic': '#00ff41', 'Suspicious Activity': '#ffbd2e', 'Blocked Threats': '#ff5f56'}
    
    for stream, values in data_streams.items():
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            name=stream,
            line=dict(color=colors[stream], width=2),
            fill='tonexty' if stream != 'Legitimate Traffic' else None
        ))
    
    fig.update_layout(
        paper_bgcolor='#0a0e1a',
        plot_bgcolor='#0d1117',
        font=dict(color='#00ff41', family='Share Tech Mono'),
        xaxis=dict(gridcolor='#30363d'),
        yaxis=dict(gridcolor='#30363d'),
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_login():
    """Render login screen"""
    st.markdown("<br><br>", unsafe_allow_html=True)
    render_ascii_banner()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="terminal-window">
            <div class="terminal-titlebar">
                <div style="color: #00ff41; font-weight: bold;">🔐 SECURE AUTHENTICATION</div>
                <div class="terminal-buttons">
                    <span class="terminal-button btn-close"></span>
                    <span class="terminal-button btn-minimize"></span>
                    <span class="terminal-button btn-maximize"></span>
                </div>
            </div>
            <div class="terminal-content" style="padding: 2rem;">
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("##### OPERATOR CREDENTIALS")
            username = st.text_input("🔑 Username:", placeholder="operator")
            password = st.text_input("🔐 Password:", type="password", placeholder="Enter access code")
            clearance = st.selectbox("🎖️ Clearance Level:", ["TOP SECRET", "SECRET", "CONFIDENTIAL"])
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.form_submit_button("🚀 AUTHENTICATE", use_container_width=True):
                    if username == "operator" and password == "cyber2024":
                        st.session_state.authenticated = True
                        st.session_state.clearance = clearance
                        st.success("✅ ACCESS GRANTED")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ ACCESS DENIED - INVALID CREDENTIALS")
            
            with col_b:
                if st.form_submit_button("🔄 RESET", use_container_width=True):
                    st.info("Session cleared")
        
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem; color: #666; font-size: 0.85rem;">
            <p>⚠️ AUTHORIZED PERSONNEL ONLY</p>
            <p>All activity is monitored and logged</p>
            <p>Default credentials: operator / cyber2024</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application"""
    
    if not st.session_state.authenticated:
        render_login()
    else:
        # Render main terminal interface
        render_terminal_header()
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 DASHBOARD",
            "💻 TERMINAL",
            "🎯 THREAT INTEL",
            "🌍 NETWORK MAP"
        ])
        
        with tab1:
            render_system_metrics()
            st.markdown("---")
            render_data_flow_visualization()
            st.markdown("---")
            render_threat_dashboard()
        
        with tab2:
            render_live_terminal_logs()
            st.markdown("---")
            render_command_interface()
        
        with tab3:
            render_threat_dashboard()
            st.markdown("---")
            
            # Threat statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">THREATS DETECTED</div>
                    <div class="metric-value">1,247</div>
                    <div class="metric-change metric-up">↑ 12% from yesterday</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">THREATS BLOCKED</div>
                    <div class="metric-value">1,189</div>
                    <div class="metric-change metric-up">↑ 95.3% success rate</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">ACTIVE INVESTIGATIONS</div>
                    <div class="metric-value">23</div>
                    <div class="metric-change">→ 18 resolved today</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">MEAN TIME TO DETECT</div>
                    <div class="metric-value">2.3s</div>
                    <div class="metric-change metric-down">↓ 0.5s improvement</div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab4:
            render_network_map()
        
        # Auto-refresh indicator
        st.markdown("""
        <div style="position: fixed; bottom: 20px; right: 20px; background: #0d1117; border: 1px solid #00ff41; padding: 0.5rem 1rem; border-radius: 4px; font-size: 0.85rem;">
            <span class="network-pulse"></span> LIVE MONITORING ACTIVE
        </div>
        """, unsafe_allow_html=True)
        
        # Auto-refresh every 2 seconds
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()
