"""
AttentionNet Traffic Classifier - Streamlit Demo
=================================================
A simple, clean web interface for demonstrating the traffic classifier.

Run with:
    streamlit run demo_streamlit.py

The code is intentionally simple and well-commented so you can
easily understand and explain each part to your professor.
"""

import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# =============================================================================
# PAGE CONFIGURATION
# This must be the first Streamlit command
# =============================================================================
st.set_page_config(
    page_title="AttentionNet Traffic Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PATHS AND CONSTANTS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "model_output/memory_safe/hocaya_gosterilcek/p2p_change/2layer_cnn_hybrid_3fc/best_model.pth")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data/final/memory_safe/own_nonVPN_p2p_2/ratio_change")

MAX_LEN = 784  # 28x28 = 784 bytes per flow image
ROWS, COLS = 28, 28

# Class names for the 12 traffic categories
CLASS_NAMES = {
    0: "Chat (NonVPN)", 
    1: "Email (NonVPN)", 
    2: "File (NonVPN)", 
    3: "P2P (NonVPN)", 
    4: "Streaming (NonVPN)", 
    5: "VoIP (NonVPN)", 
    6: "Chat (VPN)", 
    7: "Email (VPN)", 
    8: "File (VPN)", 
    9: "P2P (VPN)", 
    10: "Streaming (VPN)", 
    11: "VoIP (VPN)"
}

# Colors for visualization
CLASS_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f1c40f", "#1abc9c",
    "#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#f39c12", "#16a085"
]

# =============================================================================
# MODEL DEFINITION
# This is the same hybrid CNN-Transformer architecture used for training
# =============================================================================
class TrafficCNN_TinyTransformer(nn.Module):
    """
    Hybrid CNN-Transformer model for traffic classification.
    
    Architecture:
    1. CNN backbone: Extracts spatial features from 28x28 images
       - Conv1: 1 -> 64 channels, 3x3 kernel
       - Conv2: 64 -> 128 channels, 3x3 kernel
       
    2. Transformer: Learns relationships between different parts of the flow
       - 2 encoder layers
       - 4 attention heads
       
    3. Classifier: Three fully-connected layers for final prediction
    """
    
    def __init__(self, num_classes=12):
        super().__init__()
        
        # CNN layers - extract features from the image
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        
        self.relu = nn.ReLU(inplace=True)
        
        # Transformer parameters
        self.d_model = 128  # Embedding dimension
        self.seq_len = 49   # 7x7 = 49 tokens
        
        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, self.d_model))
        
        # Layer normalization
        self.norm_in = nn.LayerNorm(self.d_model)
        self.norm_out = nn.LayerNorm(self.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,              # 4 attention heads
            dim_feedforward=256,  # FFN hidden dimension
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Classifier with dropout for regularization
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(self.d_model, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Reshape for transformer: (batch, 128, 7, 7) -> (batch, 49, 128)
        x = x.flatten(2).permute(0, 2, 1)
        
        # Add position embeddings
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Transformer processing
        x = self.norm_in(x)
        x = self.transformer(x)
        x = self.norm_out(x)
        
        # Global average pooling over sequence
        x = x.mean(dim=1)
        
        # Classification head
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        
        return x


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource  # Cache the model so it's only loaded once
def load_model():
    """Load the trained model from disk."""
    
    # Determine the best available device
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    else:
        device = torch.device("cpu")
    
    # Create model and load weights
    model = TrafficCNN_TinyTransformer(num_classes=12).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()  # Set to evaluation mode
    
    return model, device


def predict_batch(model, device, images):
    """
    Classify a batch of flow images.
    
    Args:
        model: The trained model
        device: torch device (cpu/cuda/mps)
        images: numpy array of shape (N, 28, 28)
    
    Returns:
        predictions: class indices
        probabilities: confidence scores for each class
    """
    # Convert to tensor and normalize to [0, 1]
    tensor = torch.FloatTensor(np.array(images)).unsqueeze(1) / 255.0
    tensor = tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
    
    probs = probs.cpu().numpy()
    preds = np.argmax(probs, axis=1)
    
    return preds, probs


def process_pcap_bytes(pcap_bytes, max_flows=50):
    """
    Process uploaded PCAP file and extract flow images.
    
    This function:
    1. Groups packets into flows by IP/port pairs
    2. Anonymizes IP and MAC addresses
    3. Converts first 784 bytes of each flow to a 28x28 image
    """
    try:
        from scapy.all import IP, TCP, UDP, PcapReader
    except ImportError:
        st.error("Scapy is required for PCAP processing. Install with: `pip install scapy`")
        return [], []
    
    # Save uploaded bytes to temporary file for scapy
    with tempfile.NamedTemporaryFile(suffix='.pcap', delete=False) as tmp:
        tmp.write(pcap_bytes)
        tmp_path = tmp.name
    
    flow_buffers = defaultdict(bytearray)
    seq_tracker = defaultdict(set)
    
    try:
        with PcapReader(tmp_path) as packets:
            for pkt in packets:
                # Only process IP packets with TCP or UDP
                if IP not in pkt or (TCP not in pkt and UDP not in pkt):
                    continue
                
                # Create flow key from IP addresses and ports
                ip = pkt[IP]
                proto = ip.proto
                sport = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport if UDP in pkt else 0
                dport = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport if UDP in pkt else 0
                
                # Sort to make flow bidirectional
                sorted_ips = tuple(sorted((ip.src, ip.dst)))
                sorted_ports = tuple(sorted((sport, dport)))
                key = (sorted_ips[0], sorted_ips[1], sorted_ports[0], sorted_ports[1], proto)
                
                # Skip TCP retransmissions
                if TCP in pkt:
                    conn_key = (ip.src, ip.dst, pkt[TCP].sport, pkt[TCP].dport)
                    seq = pkt[TCP].seq
                    if seq in seq_tracker[conn_key]:
                        continue
                    seq_tracker[conn_key].add(seq)
                
                # Only add bytes if we haven't reached the limit
                buffer = flow_buffers[key]
                if len(buffer) >= MAX_LEN:
                    continue
                
                # Anonymize packet (zero out IPs and MACs)
                pkt_copy = pkt.copy()
                pkt_copy[IP].src = "0.0.0.0"
                pkt_copy[IP].dst = "0.0.0.0"
                if "Ether" in pkt_copy:
                    pkt_copy["Ether"].src = "00:00:00:00:00:00"
                    pkt_copy["Ether"].dst = "00:00:00:00:00:00"
                
                pkt_bytes = bytes(pkt_copy)
                remaining = MAX_LEN - len(buffer)
                buffer.extend(pkt_bytes[:remaining])
    finally:
        os.unlink(tmp_path)  # Clean up temp file
    
    # Convert buffers to images
    images = []
    flow_info = []
    
    for key, buffer in flow_buffers.items():
        if max_flows and len(images) >= max_flows:
            break
        
        if len(buffer) == 0:
            continue
        
        # Convert bytes to numpy array
        arr = np.frombuffer(bytes(buffer), dtype=np.uint8)
        
        # Pad or truncate to exactly 784 bytes
        if len(arr) >= MAX_LEN:
            arr = arr[:MAX_LEN]
        else:
            arr = np.pad(arr, (0, MAX_LEN - len(arr)))
        
        # Reshape to 28x28 image
        img = arr.reshape(ROWS, COLS)
        images.append(img)
        
        flow_info.append({
            'src': f"{key[0]}:{key[2]}",
            'dst': f"{key[1]}:{key[3]}",
            'proto': 'TCP' if key[4] == 6 else 'UDP' if key[4] == 17 else str(key[4]),
            'bytes': len(buffer)
        })
    
    return images, flow_info


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    # Sidebar navigation
    st.sidebar.title("🔍 AttentionNet")
    st.sidebar.markdown("Network Traffic Classifier")
    
    page = st.sidebar.radio(
        "Select Mode",
        ["🔴 Live Capture", "📁 Process PCAP", "📊 Evaluate Dataset", "ℹ️ About Model"]
    )
    
    # Load model (cached)
    try:
        model, device = load_model()
        st.sidebar.success(f"✓ Model loaded ({device})")
    except Exception as e:
        st.sidebar.error(f"✗ Model error: {e}")
        st.stop()
    
    # Route to selected page
    if page == "🔴 Live Capture":
        capture_page(model, device)
    elif page == "📁 Process PCAP":
        pcap_page(model, device)
    elif page == "📊 Evaluate Dataset":
        dataset_page(model, device)
    else:
        about_page()


def capture_page(model, device):
    """Live network capture page."""
    
    st.title("🔴 Live Network Capture")
    st.markdown("Capture live network traffic and classify it in real-time.")
    
    # Check if scapy is available
    try:
        from scapy.all import sniff, IP, TCP, UDP
        scapy_available = True
    except ImportError:
        scapy_available = False
    
    if not scapy_available:
        st.error("⚠️ Scapy is required for live capture. Install with: `pip install scapy`")
        return
    
    # Warning about permissions
    st.warning("""
    **Note:** Live capture requires administrator/root privileges.
    - **macOS/Linux:** Run with `sudo streamlit run demo_streamlit.py`
    - **Windows:** Run as Administrator
    """)
    
    # Capture settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        interface = st.text_input(
            "Network Interface",
            value="en0",
            help="Network interface to capture from (e.g., en0, eth0, wlan0)"
        )
    
    with col2:
        duration = st.number_input(
            "Capture Duration (seconds)",
            min_value=5,
            max_value=60,
            value=10,
            help="How long to capture packets"
        )
    
    with col3:
        max_flows = st.number_input(
            "Max Flows",
            min_value=5,
            max_value=100,
            value=20,
            help="Maximum number of flows to process"
        )
    
    # Common interfaces help
    with st.expander("ℹ️ Common interface names"):
        st.markdown("""
        | OS | Interface | Description |
        |----|-----------|-------------|
        | **macOS** | `en0` | WiFi |
        | **macOS** | `en1` | Ethernet |
        | **Linux** | `eth0` | Ethernet |
        | **Linux** | `wlan0` | WiFi |
        | **Linux** | `enp3s0` | Ethernet (newer naming) |
        """)
    
    # Capture button
    if st.button("🔴 Start Capture", type="primary"):
        
        # Perform live capture
        with st.spinner(f"Capturing on {interface} for {duration} seconds..."):
            try:
                images, flow_info = capture_live_traffic(interface, duration, max_flows)
            except PermissionError:
                st.error("""
                **Permission denied!** Live capture requires root/admin privileges.
                
                Run with: `sudo streamlit run demo_streamlit.py`
                """)
                return
            except Exception as e:
                st.error(f"Capture error: {e}")
                return
        
        if not images:
            st.warning("No flows captured. Try a longer duration or check the interface name.")
            return
        
        # Classify
        with st.spinner("Classifying flows..."):
            preds, probs = predict_batch(model, device, images)
        
        # Store results
        st.session_state['capture_results'] = {
            'images': images,
            'flow_info': flow_info,
            'preds': preds,
            'probs': probs
        }
        
        st.success(f"✓ Captured and classified {len(images)} flows!")
    
    # Display results if available
    if 'capture_results' in st.session_state:
        display_capture_results(st.session_state['capture_results'])


def capture_live_traffic(interface, duration, max_flows):
    """
    Capture live network traffic and convert to flow images.
    
    Args:
        interface: Network interface name (e.g., 'en0')
        duration: Capture duration in seconds
        max_flows: Maximum number of flows to return
    
    Returns:
        images: List of 28x28 numpy arrays
        flow_info: List of flow metadata dicts
    """
    from scapy.all import sniff, IP, TCP, UDP
    
    flow_buffers = defaultdict(bytearray)
    seq_tracker = defaultdict(set)
    
    def packet_callback(pkt):
        """Process each captured packet."""
        if IP not in pkt or (TCP not in pkt and UDP not in pkt):
            return
        
        # Create flow key
        ip = pkt[IP]
        proto = ip.proto
        sport = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport if UDP in pkt else 0
        dport = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport if UDP in pkt else 0
        
        sorted_ips = tuple(sorted((ip.src, ip.dst)))
        sorted_ports = tuple(sorted((sport, dport)))
        key = (sorted_ips[0], sorted_ips[1], sorted_ports[0], sorted_ports[1], proto)
        
        # Skip retransmissions
        if TCP in pkt:
            conn_key = (ip.src, ip.dst, pkt[TCP].sport, pkt[TCP].dport)
            seq = pkt[TCP].seq
            if seq in seq_tracker[conn_key]:
                return
            seq_tracker[conn_key].add(seq)
        
        # Add bytes to flow buffer
        buffer = flow_buffers[key]
        if len(buffer) >= MAX_LEN:
            return
        
        # Anonymize packet
        pkt_copy = pkt.copy()
        pkt_copy[IP].src = "0.0.0.0"
        pkt_copy[IP].dst = "0.0.0.0"
        if "Ether" in pkt_copy:
            pkt_copy["Ether"].src = "00:00:00:00:00:00"
            pkt_copy["Ether"].dst = "00:00:00:00:00:00"
        
        pkt_bytes = bytes(pkt_copy)
        remaining = MAX_LEN - len(buffer)
        buffer.extend(pkt_bytes[:remaining])
    
    # Perform capture
    sniff(iface=interface, prn=packet_callback, timeout=duration, store=False)
    
    # Convert buffers to images
    images = []
    flow_info = []
    
    for key, buffer in flow_buffers.items():
        if len(images) >= max_flows:
            break
        
        if len(buffer) == 0:
            continue
        
        # Convert to numpy array
        arr = np.frombuffer(bytes(buffer), dtype=np.uint8)
        
        # Pad or truncate
        if len(arr) >= MAX_LEN:
            arr = arr[:MAX_LEN]
        else:
            arr = np.pad(arr, (0, MAX_LEN - len(arr)))
        
        img = arr.reshape(ROWS, COLS)
        images.append(img)
        
        flow_info.append({
            'src': f"{key[0]}:{key[2]}",
            'dst': f"{key[1]}:{key[3]}",
            'proto': 'TCP' if key[4] == 6 else 'UDP' if key[4] == 17 else str(key[4]),
            'bytes': len(buffer)
        })
    
    return images, flow_info


def display_capture_results(results):
    """Display live capture results."""
    
    images = results['images']
    flow_info = results['flow_info']
    preds = results['preds']
    probs = results['probs']
    
    # Summary statistics
    vpn_count = sum(1 for p in preds if p >= 6)
    nonvpn_count = len(preds) - vpn_count
    
    st.markdown("---")
    st.subheader("📊 Capture Results")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Flows", len(preds))
    col2.metric("Non-VPN Traffic", nonvpn_count)
    col3.metric("VPN Traffic", vpn_count)
    
    # Traffic type breakdown
    st.markdown("### Traffic Type Distribution")
    
    # Count each class
    from collections import Counter
    class_counts = Counter(preds)
    
    # Create bar chart
    import pandas as pd
    df_counts = pd.DataFrame([
        {"Class": CLASS_NAMES[cls], "Count": count, "VPN": "VPN" if cls >= 6 else "Non-VPN"}
        for cls, count in sorted(class_counts.items())
    ])
    
    if not df_counts.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = [CLASS_COLORS[cls] for cls in sorted(class_counts.keys())]
        ax.bar(range(len(df_counts)), df_counts['Count'], color=colors)
        ax.set_xticks(range(len(df_counts)))
        ax.set_xticklabels(df_counts['Class'], rotation=45, ha='right')
        ax.set_ylabel('Number of Flows')
        ax.set_title('Traffic Classification Distribution')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Two columns: images and table
    st.markdown("---")
    col_img, col_table = st.columns([1, 1])
    
    with col_img:
        st.subheader("Flow Visualizations")
        
        n_display = min(8, len(images))
        cols_per_row = 4
        
        for row_start in range(0, n_display, cols_per_row):
            cols = st.columns(cols_per_row)
            for i, col in enumerate(cols):
                idx = row_start + i
                if idx < n_display:
                    with col:
                        fig, ax = plt.subplots(figsize=(2, 2))
                        ax.imshow(images[idx], cmap='viridis', vmin=0, vmax=255)
                        ax.axis('off')
                        pred_name = CLASS_NAMES[preds[idx]]
                        if len(pred_name) > 12:
                            pred_name = pred_name[:10] + "..."
                        ax.set_title(f"Flow {idx+1}\n{pred_name}", fontsize=8)
                        st.pyplot(fig)
                        plt.close()
    
    with col_table:
        st.subheader("Classification Details")
        
        import pandas as pd
        
        data = []
        for i, (info, pred, prob) in enumerate(zip(flow_info, preds, probs)):
            conf = prob[pred] * 100
            is_vpn = "✓ VPN" if pred >= 6 else "✗ Non-VPN"
            data.append({
                "Flow": f"Flow {i+1}",
                "Protocol": info['proto'],
                "Bytes": info['bytes'],
                "Prediction": CLASS_NAMES[pred],
                "Confidence": f"{conf:.1f}%",
                "VPN": is_vpn
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Detailed probability view
    st.markdown("---")
    st.subheader("Detailed Probabilities")
    
    selected_flow = st.selectbox(
        "Select a flow to see detailed probabilities",
        range(len(images)),
        format_func=lambda x: f"Flow {x+1} - {CLASS_NAMES[preds[x]]} ({probs[x][preds[x]]*100:.1f}%)"
    )
    
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(12)
    bars = ax.barh(y_pos, probs[selected_flow], color=CLASS_COLORS)
    bars[preds[selected_flow]].set_edgecolor('black')
    bars[preds[selected_flow]].set_linewidth(2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([CLASS_NAMES[i] for i in range(12)])
    ax.set_xlabel('Probability')
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'Flow {selected_flow+1} - Classification Probabilities')
    st.pyplot(fig)
    plt.close()


def pcap_page(model, device):
    """PCAP file processing page."""
    
    st.title("📁 Process PCAP File")
    st.markdown("Upload a PCAP file to classify the network flows it contains.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PCAP file",
        type=['pcap', 'pcapng'],
        help="Upload a network capture file to analyze"
    )
    
    # Settings
    col1, col2 = st.columns(2)
    with col1:
        max_flows = st.slider("Maximum flows to process", 5, 100, 20)
    
    if uploaded_file is not None:
        # Process button
        if st.button("🔄 Process and Classify", type="primary"):
            with st.spinner("Processing PCAP file..."):
                # Process the PCAP
                images, flow_info = process_pcap_bytes(uploaded_file.read(), max_flows)
                
                if not images:
                    st.error("No valid flows found in the PCAP file.")
                    return
                
                # Classify flows
                preds, probs = predict_batch(model, device, images)
                
                # Store results in session state
                st.session_state['pcap_results'] = {
                    'images': images,
                    'flow_info': flow_info,
                    'preds': preds,
                    'probs': probs
                }
    
    # Display results if available
    if 'pcap_results' in st.session_state:
        results = st.session_state['pcap_results']
        display_pcap_results(results)


def display_pcap_results(results):
    """Display PCAP processing results."""
    
    images = results['images']
    flow_info = results['flow_info']
    preds = results['preds']
    probs = results['probs']
    
    st.success(f"✓ Processed {len(images)} flows")
    
    # Summary statistics
    vpn_count = sum(1 for p in preds if p >= 6)
    nonvpn_count = len(preds) - vpn_count
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Flows", len(preds))
    col2.metric("Non-VPN", nonvpn_count)
    col3.metric("VPN", vpn_count)
    
    st.markdown("---")
    
    # Two columns: images and results table
    col_img, col_table = st.columns([1, 1])
    
    with col_img:
        st.subheader("Flow Visualizations")
        
        # Show flow images in a grid
        n_display = min(12, len(images))
        cols_per_row = 4
        
        for row_start in range(0, n_display, cols_per_row):
            cols = st.columns(cols_per_row)
            for i, col in enumerate(cols):
                idx = row_start + i
                if idx < n_display:
                    with col:
                        fig, ax = plt.subplots(figsize=(2, 2))
                        ax.imshow(images[idx], cmap='viridis', vmin=0, vmax=255)
                        ax.axis('off')
                        pred_name = CLASS_NAMES[preds[idx]]
                        if len(pred_name) > 12:
                            pred_name = pred_name[:10] + "..."
                        ax.set_title(f"Flow {idx+1}\n{pred_name}", fontsize=8)
                        st.pyplot(fig)
                        plt.close()
    
    with col_table:
        st.subheader("Classification Results")
        
        # Create results dataframe
        import pandas as pd
        
        data = []
        for i, (info, pred, prob) in enumerate(zip(flow_info, preds, probs)):
            conf = prob[pred] * 100
            is_vpn = "✓ VPN" if pred >= 6 else "✗ Non-VPN"
            data.append({
                "Flow": f"Flow {i+1}",
                "Protocol": info['proto'],
                "Prediction": CLASS_NAMES[pred],
                "Confidence": f"{conf:.1f}%",
                "VPN": is_vpn
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Detailed probability view for selected flow
    st.markdown("---")
    st.subheader("Detailed Probabilities")
    
    selected_flow = st.selectbox(
        "Select a flow to see detailed probabilities",
        range(len(images)),
        format_func=lambda x: f"Flow {x+1} - {CLASS_NAMES[preds[x]]}"
    )
    
    # Show probability bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(12)
    bars = ax.barh(y_pos, probs[selected_flow], color=CLASS_COLORS)
    bars[preds[selected_flow]].set_edgecolor('black')
    bars[preds[selected_flow]].set_linewidth(2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([CLASS_NAMES[i] for i in range(12)])
    ax.set_xlabel('Probability')
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'Flow {selected_flow+1} Classification Probabilities')
    st.pyplot(fig)
    plt.close()


def dataset_page(model, device):
    """Dataset evaluation page."""
    
    st.title("📊 Evaluate Test Dataset")
    st.markdown("Load pre-processed test data and evaluate model performance.")
    
    # Dataset selection
    dataset_option = st.radio(
        "Select dataset",
        ["Test Set", "Training Set", "Upload Custom .npy"]
    )
    
    if dataset_option == "Test Set":
        data_path = os.path.join(TEST_DATA_DIR, "test_data_memory_safe_own_nonVPN_p2p.npy")
        labels_path = os.path.join(TEST_DATA_DIR, "test_labels_memory_safe_own_nonVPN_p2p.npy")
    elif dataset_option == "Training Set":
        data_path = os.path.join(TEST_DATA_DIR, "train_data_memory_safe_own_nonVPN_p2p.npy")
        labels_path = os.path.join(TEST_DATA_DIR, "train_labels_memory_safe_own_nonVPN_p2p.npy")
    else:
        col1, col2 = st.columns(2)
        with col1:
            data_file = st.file_uploader("Upload data .npy", type=['npy'])
        with col2:
            labels_file = st.file_uploader("Upload labels .npy", type=['npy'])
        
        if data_file and labels_file:
            data_path = data_file
            labels_path = labels_file
        else:
            st.info("Please upload both data and labels files.")
            return
    
    # Load and evaluate button
    if st.button("📈 Load and Evaluate", type="primary"):
        with st.spinner("Loading dataset..."):
            try:
                if isinstance(data_path, str):
                    images = np.load(data_path)
                    labels = np.load(labels_path)
                else:
                    images = np.load(data_path)
                    labels = np.load(labels_path)
                
                st.session_state['dataset'] = {
                    'images': images,
                    'labels': labels,
                    'name': dataset_option
                }
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                return
        
        with st.spinner("Running evaluation..."):
            # Predict in batches
            batch_size = 64
            all_preds = []
            
            progress_bar = st.progress(0)
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                preds, _ = predict_batch(model, device, batch)
                all_preds.extend(preds)
                progress_bar.progress(min(1.0, (i + batch_size) / len(images)))
            
            preds = np.array(all_preds)
            
            st.session_state['eval_results'] = {
                'preds': preds,
                'labels': labels
            }
    
    # Display results if available
    if 'eval_results' in st.session_state and 'dataset' in st.session_state:
        display_eval_results(
            st.session_state['dataset'],
            st.session_state['eval_results']
        )


def display_eval_results(dataset, results):
    """Display evaluation results."""
    
    images = dataset['images']
    labels = dataset['labels']
    preds = results['preds']
    
    # Calculate metrics
    accuracy = 100 * np.mean(preds == labels)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    st.success(f"✓ Evaluation complete!")
    
    # Metrics display
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(labels))
    col2.metric("Accuracy", f"{accuracy:.2f}%")
    col3.metric("Macro F1", f"{macro_f1:.4f}")
    
    st.markdown("---")
    
    # Two columns: confusion matrix and sample images
    col_cm, col_samples = st.columns([1, 1])
    
    with col_cm:
        st.subheader("Confusion Matrix")
        
        # Create confusion matrix
        all_labels = np.unique(np.concatenate([labels, preds]))
        cm = confusion_matrix(labels, preds, labels=all_labels)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        
        # Labels
        class_labels = [CLASS_NAMES.get(i, str(i))[:12] for i in all_labels]
        ax.set_xticks(range(len(all_labels)))
        ax.set_yticks(range(len(all_labels)))
        ax.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(class_labels, fontsize=8)
        
        # Add values
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", 
                       color=color, fontsize=7)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.1f}%)')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col_samples:
        st.subheader("Sample Images by Class")
        
        # Show a few samples per class
        unique_labels = np.unique(labels)
        samples_per_class = 3
        
        fig, axes = plt.subplots(len(unique_labels), samples_per_class, 
                                  figsize=(6, 2*len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            class_images = images[labels == label]
            for j in range(samples_per_class):
                ax = axes[i, j] if len(unique_labels) > 1 else axes[j]
                if j < len(class_images):
                    ax.imshow(class_images[j], cmap='viridis', vmin=0, vmax=255)
                ax.axis('off')
                if j == 0:
                    name = CLASS_NAMES.get(label, str(label))
                    if len(name) > 12:
                        name = name[:10] + "..."
                    ax.set_title(name, fontsize=8, loc='left')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Classification report
    st.markdown("---")
    st.subheader("Classification Report")
    
    all_labels_unique = np.unique(np.concatenate([labels, preds]))
    report = classification_report(
        labels, preds,
        labels=all_labels_unique,
        target_names=[CLASS_NAMES.get(i, str(i)) for i in all_labels_unique],
        digits=3,
        zero_division=0
    )
    
    st.code(report, language=None)


def about_page():
    """About page with model information."""
    
    st.title("ℹ️ About AttentionNet")
    
    st.markdown("""
    ## Model Architecture
    
    AttentionNet is a **hybrid CNN-Transformer** model for encrypted network traffic classification.
    
    ### Input Processing
    - Network packets are grouped into **flows** (by IP/port pairs)
    - First **784 bytes** of each flow are extracted
    - Bytes are reshaped into a **28×28 grayscale image**
    - IP and MAC addresses are **anonymized** (set to zeros)
    
    ### CNN Backbone
    The CNN extracts spatial features from the flow image:
    
    ```
    Input: 1×28×28
       ↓
    Conv2D (64 filters, 3×3) + BatchNorm + ReLU + MaxPool
       ↓
    Feature Map: 64×14×14
       ↓
    Conv2D (128 filters, 3×3) + BatchNorm + ReLU + MaxPool
       ↓
    Feature Map: 128×7×7
    ```
    
    ### Transformer Encoder
    The 7×7 feature map is flattened into **49 tokens** and processed by:
    - **2 encoder layers**
    - **4 attention heads**
    - **256 feedforward dimensions**
    
    This allows the model to learn relationships between different parts of the flow.
    
    ### Classification Head
    ```
    Global Average Pooling → 128D vector
       ↓
    FC (128→128) + Dropout(0.5) + ReLU
       ↓
    FC (128→128) + Dropout(0.3) + ReLU
       ↓
    FC (128→12) → 12 class probabilities
    ```
    
    ---
    
    ## Traffic Classes
    
    The model classifies traffic into **12 categories**:
    
    | Non-VPN | VPN |
    |---------|-----|
    | Chat (NonVPN) | Chat (VPN) |
    | Email (NonVPN) | Email (VPN) |
    | File Transfer (NonVPN) | File Transfer (VPN) |
    | P2P (NonVPN) | P2P (VPN) |
    | Streaming (NonVPN) | Streaming (VPN) |
    | VoIP (NonVPN) | VoIP (VPN) |
    
    ---
    
    ## Performance
    
    | Metric | Value |
    |--------|-------|
    | **Test Accuracy** | ~86% |
    | **Macro F1 Score** | ~0.88 |
    
    ---
    
    ## Processing Pipeline
    
    ```
    PCAP File
       ↓
    1. Parse packets (Scapy)
       ↓
    2. Group into flows (by IP+port)
       ↓
    3. Remove retransmissions
       ↓
    4. Anonymize (zero IPs/MACs)
       ↓
    5. Extract first 784 bytes
       ↓
    6. Reshape to 28×28 image
       ↓
    7. Normalize to [0, 1]
       ↓
    8. Classify with CNN-Transformer
       ↓
    Prediction + Confidence
    ```
    """)


if __name__ == "__main__":
    main()

