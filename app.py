import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import time
import gc
import warnings
import socket
import psutil
import platform
import subprocess
import asyncio
from contextlib import contextmanager

warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AION-7 | CYBER DEFENSE GRID TERMINAL",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ADVANCED CYBERPUNK TERMINAL CSS & JS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700&display=swap');

    :root {
        --primary-color: #00ff00;
        --secondary-color: #00ffff;
        --background-color: #000000;
        --dark-blue: #001122;
        --panel-bg: rgba(0, 20, 40, 0.85);
        --panel-border: 1px solid var(--secondary-color);
        --glow-color: rgba(0, 255, 0, 0.75);
        --red-glow: #ff0000;
    }

    body {
        background-color: var(--background-color);
    }

    .main {
        background: var(--background-color) !important;
        color: var(--primary-color) !important;
        font-family: 'Share Tech Mono', monospace !important;
        text-shadow: 0 0 3px var(--glow-color);
    }

    /* Scanline and Flicker Overlay */
    .main::before {
        content: " ";
        display: block;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: linear-gradient(0deg, rgba(0,0,0,0) 50%, rgba(0,255,0,0.1) 50%);
        background-size: 100% 4px;
        z-index: 9999;
        pointer-events: none;
        animation: scanlines 10s linear infinite;
    }

    .main::after {
        content: " ";
        display: block;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(18, 16, 16, 0.1);
        opacity: 0;
        z-index: 9998;
        pointer-events: none;
        animation: flicker 0.15s infinite;
    }

    @keyframes scanlines {
        0% { background-position: 0 0; }
        100% { background-position: 0 100%; }
    }

    @keyframes flicker {
        0% { opacity: 0.1; }
        50% { opacity: 0.2; }
        100% { opacity: 0.1; }
    }

    /* Hide Streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }

    /* Custom Terminal Header */
    .terminal-header {
        background: linear-gradient(90deg, #001a33 0%, #002b4d 50%, #001a33 100%);
        border: var(--panel-border);
        border-top: 0;
        border-left: 0;
        border-right: 0;
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 0 20px var(--secondary-color);
        font-family: 'Orbitron', sans-serif;
    }
    
    .terminal-title {
        font-size: 2.5rem;
        color: var(--secondary-color);
        text-shadow: 0 0 10px var(--secondary-color);
        letter-spacing: 4px;
    }

    /* Metric and Panel Styling */
    .terminal-metric {
        background: var(--panel-bg);
        border: var(--panel-border);
        border-radius: 4px;
        padding: 1rem;
        text-align: center;
        box-shadow: inset 0 0 10px rgba(0, 255, 255, 0.3);
    }
    .metric-title { color: var(--secondary-color); font-size: 0.9em; margin-bottom: 0.5rem; text-transform: uppercase; }
    .metric-value { font-size: 1.5em; font-weight: 700; letter-spacing: 2px; }

    /* Alert Styling */
    .alert-critical {
        border: 1px solid var(--red-glow);
        border-left: 5px solid var(--red-glow);
        background: rgba(255, 0, 0, 0.2);
        color: white; padding: 1rem; margin: 0.5rem 0;
        text-shadow: 0 0 5px var(--red-glow);
        animation: blink-critical 1.5s infinite;
    }
    @keyframes blink-critical { 0%, 50% { background: rgba(255, 0, 0, 0.3); } 51%, 100% { background: rgba(255, 0, 0, 0.1); } }

    /* Log and CLI Styling */
    .log-entry {
        background: rgba(0, 255, 255, 0.05);
        border-left: 3px solid var(--secondary-color);
        padding: 0.5rem; margin: 0.2rem 0;
        font-size: 0.9em; white-space: pre-wrap;
    }
    .command-line-container {
        background: var(--dark-blue); border: var(--panel-border); padding: 1rem;
    }
    .command-line-container .stTextInput > div > div > input {
        background: transparent; color: var(--primary-color); border: none; font-family: 'Share Tech Mono', monospace;
    }
    .command-output {
        height: 300px; overflow-y: scroll; border: 1px solid var(--dark-blue);
        padding: 1rem; background: rgba(0,0,0,0.5); margin-top: 1rem;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid var(--secondary-color); }
    .stTabs [data-baseweb="tab"] {
        background: var(--dark-blue); color: var(--secondary-color);
        border: 1px solid var(--secondary-color); border-bottom: none;
        font-family: 'Share Tech Mono', monospace;
    }
    .stTabs [aria-selected="true"] { background: var(--panel-bg) !important; font-weight: bold; }

    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--dark-blue); }
    ::-webkit-scrollbar-thumb { background: var(--secondary-color); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #008b8b; }
    
    /* Login Screen */
    .login-container {
        display: flex; justify-content: center; align-items: center; height: 100vh;
        background: radial-gradient(circle, #002244 0%, #000000 70%);
    }
    .login-box {
        background: var(--panel-bg); border: var(--panel-border); border-radius: 8px;
        padding: 3rem; text-align: center; box-shadow: 0 0 50px rgba(0, 255, 255, 0.5);
        width: 450px;
    }
    .login-box .stButton button {
        background: linear-gradient(90deg, var(--dark-blue) 0%, #003366 100%);
        border: 1px solid var(--secondary-color); color: var(--secondary-color);
        transition: all 0.3s ease;
    }
    .login-box .stButton button:hover {
        box-shadow: 0 0 15px var(--secondary-color); background: #003366;
    }

    /* Typewriter effect */
    .typewriter-text {
        overflow: hidden;
        white-space: nowrap;
        animation: typing 2s steps(40, end);
    }
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
</style>

<div id="audio-container">
  <audio id="keypress-sound" src="https://www.soundjay.com/buttons/sounds/button-16.mp3" preload="auto"></audio>
  <audio id="alert-sound" src="https://www.soundjay.com/buttons/sounds/beep-07.mp3" preload="auto"></audio>
</div>

<script>
    // Play sounds on interaction
    function playSound(id) {
        const sound = document.getElementById(id);
        if (sound) {
            sound.currentTime = 0;
            sound.play().catch(e => console.error("Sound play failed:", e));
        }
    }
    document.addEventListener('keydown', () => playSound('keypress-sound'));
</script>
""", unsafe_allow_html=True)


# --- STATE INITIALIZATION ---
def initialize_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'boot_sequence_complete' not in st.session_state:
        st.session_state.boot_sequence_complete = False
    if 'data_stream' not in st.session_state:
        st.session_state.data_stream = pd.DataFrame(columns=['time', 'price', 'volume', 'anomaly'])
    if 'log_entries' not in st.session_state:
        st.session_state.log_entries = []
    if 'active_alerts' not in st.session_state:
        st.session_state.active_alerts = []
    if 'command_history' not in st.session_state:
        st.session_state.command_history = []
    if 'system_status' not in st.session_state:
        st.session_state.system_status = "NOMINAL"
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()


# --- CORE SYSTEM FUNCTIONS ---
@contextmanager
def resource_manager():
    """A context manager for resource optimization."""
    try:
        yield
    finally:
        gc.collect()

def get_system_info():
    """Fetches real system metrics."""
    try:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net = psutil.net_io_counters()
        return {
            "CPU Usage": f"{cpu:.1f}%",
            "Memory Usage": f"{mem.percent:.1f}%",
            "Disk Space": f"{disk.percent:.1f}%",
            "Hostname": socket.gethostname(),
            "Platform": platform.system() + " " + platform.release(),
            "Network Out": f"{net.bytes_sent / (1024*1024):.2f} MB",
            "Network In": f"{net.bytes_recv / (1024*1024):.2f} MB"
        }
    except Exception:
        return {"Error": "Could not fetch system metrics."}

def add_log_entry(level, source, message):
    """Adds a new entry to the system log."""
    log = {
        "timestamp": datetime.now(),
        "level": level,
        "source": source,
        "message": message
    }
    st.session_state.log_entries.insert(0, log)
    if len(st.session_state.log_entries) > 100: # Limit log history
        st.session_state.log_entries.pop()

def update_live_data():
    """Simulates a live data stream, injects anomalies, and runs detection."""
    now = datetime.now()
    if (now - st.session_state.last_update).total_seconds() < 2:
        return # Update every 2 seconds

    # 1. Generate new data point
    last_price = st.session_state.data_stream['price'].iloc[-1] if not st.session_state.data_stream.empty else 150
    new_price = last_price + random.uniform(-0.5, 0.5) + np.sin(now.second / 10)
    new_volume = random.randint(10000, 50000)
    anomaly_score = 0

    # 2. Inject anomalies occasionally
    if random.random() < 0.05: # 5% chance of an anomaly
        anomaly_type = random.choice(['spike', 'drop', 'flatline'])
        if anomaly_type == 'spike':
            new_price *= random.uniform(1.05, 1.1)
            new_volume *= random.randint(5, 10)
            anomaly_score = random.uniform(0.8, 1.0)
            add_log_entry("CRITICAL", "DETECTOR_ALPHA", f"PRICE_SPIKE_ANOMALY DETECTED. SCORE: {anomaly_score:.2f}")
        elif anomaly_type == 'drop':
            new_price *= random.uniform(0.9, 0.95)
            anomaly_score = random.uniform(0.7, 0.9)
            add_log_entry("HIGH", "DETECTOR_BETA", f"SUDDEN_PRICE_DROP DETECTED. SCORE: {anomaly_score:.2f}")
        
        # Create alert
        alert = {
            "id": f"ALERT-{int(now.timestamp())}",
            "severity": "CRITICAL" if anomaly_score > 0.8 else "HIGH",
            "type": "DATA_POISONING_SUSPECTED",
            "timestamp": now,
            "details": f"Anomaly score {anomaly_score:.2f} with type '{anomaly_type}'."
        }
        st.session_state.active_alerts.insert(0, alert)
        st.session_state.system_status = "THREAT DETECTED"
        if len(st.session_state.active_alerts) > 10:
            st.session_state.active_alerts.pop()
        
        # Trigger sound for critical alerts
        if alert['severity'] == "CRITICAL":
            st.markdown('<script>playSound("alert-sound");</script>', unsafe_allow_html=True)
            
    else:
        st.session_state.system_status = "NOMINAL"


    # 3. Append to dataframe
    new_data = pd.DataFrame([{'time': now, 'price': new_price, 'volume': new_volume, 'anomaly': anomaly_score}])
    st.session_state.data_stream = pd.concat([st.session_state.data_stream, new_data], ignore_index=True)

    # 4. Keep dataframe size manageable
    if len(st.session_state.data_stream) > 300:
        st.session_state.data_stream = st.session_state.data_stream.iloc[-300:]
    
    st.session_state.last_update = now

# --- UI RENDERING FUNCTIONS ---

def render_login():
    """Renders the secure login screen."""
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.markdown('<h1 style="font-family: \'Orbitron\', sans-serif; color: #00ffff; text-shadow: 0 0 10px #00ffff;">AION-7 GRID</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #00ff00;">SECURE TERMINAL ACCESS</h3>', unsafe_allow_html=True)
    
    with st.form("enterprise_login"):
        username = st.text_input("OPERATOR ID:", value="soc_operator_01")
        password = st.text_input("ENCRYPTION KEY:", type="password")
        submitted = st.form_submit_button("== AUTHENTICATE ==", use_container_width=True)
        
        if submitted:
            if username == "soc_operator_01" and password == "aion7":
                st.session_state.authenticated = True
                add_log_entry("INFO", "AUTH_SYS", f"Operator '{username}' authenticated successfully.")
                st.rerun()
            else:
                st.error(":: ACCESS DENIED :: Invalid Credentials")
                add_log_entry("WARNING", "AUTH_SYS", f"Failed login attempt for user '{username}'.")
                
    st.markdown('</div></div>', unsafe_allow_html=True)

def render_boot_sequence():
    """Renders a cinematic boot-up sequence."""
    boot_logs = [
        ("INFO", "KERNEL", "Initializing AION-7 Core Kernel v3.4.1..."),
        ("OK", "MEMORY", "Calibrating quantum memory registers..."),
        ("OK", "NETWORK", "Establishing secure connection to Global Defense Grid..."),
        ("INFO", "MODULES", "Loading threat detection modules..."),
        ("OK", "MODULES", "[ALPHA] Statistical Anomaly Detector online."),
        ("OK", "MODULES", "[BETA] Behavioral Heuristics Engine online."),
        ("OK", "MODULES", "[GAMMA] Predictive Pattern Analyzer online."),
        ("INFO", "FIREWALL", "Activating Chameleon adaptive firewall..."),
        ("OK", "SENSORS", "All data stream sensors are active and nominal."),
        ("INFO", "UI", "Rendering primary operator interface..."),
        ("SUCCESS", "SYSTEM", "All systems operational. Welcome, Operator.")
    ]
    
    placeholder = st.empty()
    output = ""
    for level, source, msg in boot_logs:
        color = "#00ff00" if level in ["OK", "SUCCESS"] else "#ffff00" if level == "INFO" else "#ff0000"
        output += f'<span style="color: {color};">[{level}]</span> <span style="color: #00ffff;">::{source}::</span> {msg}\n'
        placeholder.markdown(f"```bash\n{output}\n```", unsafe_allow_html=True)
        time.sleep(random.uniform(0.2, 0.5))

    time.sleep(1)
    st.session_state.boot_sequence_complete = True
    st.rerun()
    
def render_terminal_header():
    """Renders the main terminal header with ASCII art and status."""
    status_color = "#00ff00" if st.session_state.system_status == "NOMINAL" else "#ff0000"
    
    header_html = f"""
    <div class="terminal-header">
        <pre style="color: {status_color}; font-family: 'Share Tech Mono', monospace; margin: 0; font-size: 12px; text-align: center;">
 █████╗  ██╗ ██████╗ ███╗   ██╗     ██████╗ ███████╗███████╗
██╔══██╗ ██║██╔═══██╗████╗  ██║    ██╔════╝ ██╔════╝██╔════╝
███████║ ██║██║   ██║██╔██╗ ██║    ██║  ███╗█████╗  ███████╗
██╔══██║ ██║██║   ██║██║╚██╗██║    ██║   ██║██╔══╝  ╚════██║
██║  ██║ ██║╚██████╔╝██║ ╚████║    ╚██████╔╝███████╗███████║
╚═╝  ╚═╝ ╚═╝ ╚═════╝ ╚═╝  ╚═══╝     ╚═════╝ ╚══════╝╚══════╝
        </pre>
        <div style="display: flex; justify-content: space-between; align-items: center; color: #00ffff; padding: 0 1rem;">
            <span style="font-family: 'Orbitron', sans-serif;">GRID STATUS: <span style="color: {status_color};">{st.session_state.system_status}</span></span>
            <span style="font-family: 'Orbitron', sans-serif;">AION-7 CYBER DEFENSE TERMINAL</span>
            <span style="font-family: 'Orbitron', sans-serif;">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_dashboard():
    """Renders the main live dashboard."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(":: REAL-TIME DATA STREAM ::")
        chart_placeholder = st.empty()
        
        df = st.session_state.data_stream
        if not df.empty:
            fig = go.Figure()
            # Price line
            fig.add_trace(go.Scatter(x=df['time'], y=df['price'], mode='lines', name='Price', line=dict(color=f'rgba(0, 255, 255, 1)')))
            # Anomaly markers
            anomalies = df[df['anomaly'] > 0]
            if not anomalies.empty:
                fig.add_trace(go.Scatter(x=anomalies['time'], y=anomalies['price'], mode='markers', name='Anomaly',
                                         marker=dict(color='red', size=10, symbol='x', line=dict(width=2))))

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0, 255, 255, 0.05)',
                font=dict(color='#00ff00'), height=400, showlegend=False,
                xaxis=dict(gridcolor='rgba(0, 255, 255, 0.2)'),
                yaxis=dict(gridcolor='rgba(0, 255, 255, 0.2)'),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(":: KEY METRICS ::")
        
        last_row = st.session_state.data_stream.iloc[-1] if not st.session_state.data_stream.empty else None
        
        price = last_row['price'] if last_row is not None else 0
        volume = last_row['volume'] if last_row is not None else 0
        threat_level = np.mean(st.session_state.data_stream['anomaly'][-10:]) * 100 if not df.empty else 0

        st.markdown(f"""
        <div class="terminal-metric"><div class="metric-title">CURRENT PRICE</div><div class="metric-value">${price:,.2f}</div></div>
        <div class="terminal-metric"><div class="metric-title">TRADING VOLUME</div><div class="metric-value">{volume:,}</div></div>
        <div class="terminal-metric">
            <div class="metric-title">THREAT LEVEL</div>
            <div class="metric-value" style="color: {'red' if threat_level > 50 else 'yellow' if threat_level > 20 else '#00ff00'};">{threat_level:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader(":: ACTIVE ALERTS ::")
    alert_placeholder = st.empty()
    alerts = st.session_state.active_alerts
    if not alerts:
        alert_placeholder.success("SYSTEM NOMINAL. NO ACTIVE THREATS.")
    else:
        for alert in alerts[:3]: # Show top 3
            alert_placeholder.markdown(f"""
            <div class="alert-critical">
                <strong>{alert['type']} [{alert['severity']}]</strong><br>
                <strong>TIMESTAMP:</strong> {alert['timestamp'].strftime('%H:%M:%S')} | <strong>ID:</strong> {alert['id']}<br>
                <strong>DETAILS:</strong> {alert['details']}
            </div>
            """, unsafe_allow_html=True)

def render_threat_intel():
    """Renders the threat intelligence tab with a globe."""
    st.subheader(":: GEOSPATIAL THREAT ANALYSIS ::")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Generate fake attack data
        num_attacks = random.randint(5, 15)
        attack_origins = {
            "Moscow": (55.7558, 37.6173), "Beijing": (39.9042, 116.4074),
            "Pyongyang": (39.0392, 125.7625), "Tehran": (35.6892, 51.3890)
        }
        target_loc = ("New York", 40.7128, -74.0060)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scattergeo(
            lon=[loc[1] for loc in attack_origins.values()] + [target_loc[2]],
            lat=[loc[0] for loc in attack_origins.values()] + [target_loc[1]],
            hoverinfo='text',
            text=list(attack_origins.keys()) + ["GRID HQ"],
            mode='markers',
            marker=dict(color=['red']*len(attack_origins) + ['#00ff00'], size=10, symbol=['cross']*len(attack_origins) + ['star'])
        ))

        for origin_name, origin_loc in random.sample(list(attack_origins.items()), k=min(len(attack_origins), 3)):
            fig.add_trace(go.Scattergeo(
                lon=[origin_loc[1], target_loc[2]],
                lat=[origin_loc[0], target_loc[1]],
                mode='lines',
                line=dict(width=1.5, color='red'),
            ))
            
        fig.update_layout(
            showlegend=False,
            geo=dict(
                projection_type='orthographic',
                showland=True, landcolor='rgba(0, 100, 0, 0.4)',
                showocean=True, oceancolor='rgba(0, 10, 40, 1)',
                bgcolor='rgba(0,0,0,0)',
                showcountries=True, countrycolor='rgba(0, 255, 0, 0.2)',
                lataxis_showgrid=True, lonaxis_showgrid=True
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(":: INTEL FEED ::")
        intel_items = [
            ("CRITICAL", "New zero-day exploit 'Cerberus' targeting financial APIs detected."),
            ("HIGH", "Coordinated DDoS activity originating from Eastern Europe."),
            ("HIGH", "Data poisoning toolkit 'VenomInject' shared on dark web forums."),
            ("MEDIUM", "Unusual reconnaissance scans on port 443 from multiple IPs."),
            ("LOW", "Increase in phishing campaigns impersonating major exchanges."),
        ]
        for level, detail in intel_items:
            color = {"CRITICAL": "#ff0000", "HIGH": "#ff8c00", "MEDIUM": "#ffff00"}.get(level, "#00ff00")
            st.markdown(f"""
            <div class="log-entry" style="border-left-color:{color};">
                <strong style="color:{color};">[{level}]</strong> {detail}
            </div>
            """, unsafe_allow_html=True)
            
def render_system_logs():
    """Renders the live system logs with a typewriter effect."""
    st.subheader(":: LIVE SYSTEM LOGS ::")
    log_container = st.empty()
    
    log_html = ""
    for log in st.session_state.log_entries[:15]: # Display last 15 logs
        level_color = {"CRITICAL": "#ff0000", "HIGH": "#ff8c00", "WARNING": "#ffff00"}.get(log['level'], "#00ff00")
        
        # Add typewriter effect to the latest log
        is_latest = log == st.session_state.log_entries[0]
        animation_class = "typewriter-text" if is_latest else ""
        
        log_html += f"""
        <div class="log-entry {animation_class}">
            <span style="color: #00ffff;">{log['timestamp'].strftime('%H:%M:%S.%f')[:-3]}</span>
            <span style="color: {level_color}; font-weight: bold;"> [{log['level']}]</span>
            <span style="color: #ffffff;">::{log['source']}::</span>
            <span>{log['message']}</span>
        </div>
        """
    log_container.markdown(log_html, unsafe_allow_html=True)

def process_command(command):
    """Processes user input from the CLI."""
    st.session_state.command_history.append(f"> {command}")
    cmd_parts = command.lower().split()
    cmd = cmd_parts[0]
    
    output = ""
    if cmd == "help":
        output = """
AVAILABLE COMMANDS:
  help              - Show this help message
  status            - Display current system and threat status
  scan <target>     - Simulate a security scan (e.g., 'scan api.grid.node')
  sysinfo           - Show detailed host system information
  clear             - Clear the command history
  alerts            - List active alerts
"""
    elif cmd == "status":
        output = f"System Status: {st.session_state.system_status}\n"
        output += f"Active Alerts: {len(st.session_state.active_alerts)}\n"
        output += f"Data Stream Integrity: {(1 - np.mean(st.session_state.data_stream['anomaly'].fillna(0))) * 100:.2f}%"
    elif cmd == "scan":
        if len(cmd_parts) > 1:
            target = cmd_parts[1]
            output = f"Initiating deep scan on {target}...\n"
            output += f"Scan complete. 0 vulnerabilities found. Target is secure."
        else:
            output = "Error: 'scan' requires a target."
    elif cmd == "sysinfo":
        info = get_system_info()
        output = "\n".join([f"{k}: {v}" for k, v in info.items()])
    elif cmd == "alerts":
        if not st.session_state.active_alerts:
            output = "No active alerts."
        else:
            output = "ACTIVE ALERTS:\n" + "\n".join([f"- [{a['id']}] {a['type']} ({a['severity']})" for a in st.session_state.active_alerts])
    elif cmd == "clear":
        st.session_state.command_history = []
    else:
        output = f"command not found: {cmd}"
        
    if output:
        st.session_state.command_history.append(output)

def render_command_center():
    """Renders the CLI and system health panels."""
    st.subheader(":: COMMAND & CONTROL ::")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="command-line-container">', unsafe_allow_html=True)
        
        # Display command history
        history_str = "\n".join(st.session_state.command_history[-20:])
        st.markdown(f'<div class="command-output"><pre>{history_str}</pre></div>', unsafe_allow_html=True)

        # Command input
        command = st.text_input("CMD>", key=f"cli_{len(st.session_state.command_history)}", on_change=lambda: process_command(st.session_state[f"cli_{len(st.session_state.command_history)}"]))
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader(":: HOST HEALTH ::")
        sys_info = get_system_info()
        if "Error" in sys_info:
            st.error(sys_info["Error"])
        else:
            for key, value in sys_info.items():
                if '%' in str(value):
                    val_str = str(value).replace('%','')
                    try:
                        val_num = float(val_str)
                        color = '#ff0000' if val_num > 90 else '#ffff00' if val_num > 70 else '#00ff00'
                        st.markdown(f"""<div class="log-entry">{key}: <strong style="color:{color}">{value}</strong></div>""", unsafe_allow_html=True)
                    except ValueError:
                         st.markdown(f"""<div class="log-entry">{key}: <strong>{value}</strong></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="log-entry">{key}: <strong>{value}</strong></div>""", unsafe_allow_html=True)

# --- MAIN APPLICATION LOGIC ---
def main():
    with resource_manager():
        initialize_state()

        if not st.session_state.authenticated:
            render_login()
            return
        
        if not st.session_state.boot_sequence_complete:
            render_boot_sequence()
            return

        # Main application loop for live updates
        render_terminal_header()
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 DASHBOARD", 
            "🎯 THREAT INTEL", 
            "📋 SYSTEM LOGS",
            "⚙️ COMMAND CENTER"
        ])
        
        with tab1:
            render_dashboard()
        with tab2:
            render_threat_intel()
        with tab3:
            render_system_logs()
        with tab4:
            render_command_center()
            
        # Core update loop
        update_live_data()
        time.sleep(2) # Refresh interval
        st.rerun()

if __name__ == "__main__":
    main()
