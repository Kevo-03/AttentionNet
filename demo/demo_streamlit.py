"""
AttentionNet Traffic Classifier - Streamlit Demo
=================================================
A web interface for demonstrating the network traffic classifier.

Run with:
    cd /path/to/AttentionNet
    streamlit run demo/demo_streamlit.py

Structure:
- Model architecture imported from src/model/hybrid_tiny.py
- Preprocessing helpers defined locally for modularity
"""

import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# =============================================================================
# PAGE CONFIGURATION
# This must be the first Streamlit command
# =============================================================================
st.set_page_config(
    page_title="AttentionNet Traffic Classifier",  # Network globe icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PATHS AND MODEL IMPORT
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add src/ to path for model import
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Import model architecture (shared with training code)
from model.hybrid_tiny import TrafficCNN_TinyTransformer

MODEL_PATH = os.path.join(PROJECT_ROOT, "model_output/memory_safe/hocaya_gosterilcek/p2p_change/2layer_cnn_hybrid_3fc/best_model.pth")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data/final/memory_safe/own_nonVPN_p2p_2/ratio_change")

# =============================================================================
# PREPROCESSING CONSTANTS
# =============================================================================
MAX_LEN = 784   # 28 x 28 = 784 bytes per flow image
ROWS = 28
COLS = 28

# =============================================================================
# PREPROCESSING HELPERS
# These functions handle packet processing for flow extraction
# =============================================================================

def flow_key(pkt):
    """
    Create a bidirectional flow identifier from a packet.
    
    Groups packets by IP addresses and ports, sorted to make the flow
    bidirectional (A->B and B->A are the same flow).
    
    Args:
        pkt: Scapy packet with IP layer
        
    Returns:
        tuple: (ip1, ip2, port1, port2, protocol)
    """
    from scapy.all import IP, TCP, UDP
    
    ip = pkt[IP]
    proto = ip.proto
    sport = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport if UDP in pkt else 0
    dport = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport if UDP in pkt else 0
    
    # Sort to make flow bidirectional
    sorted_ips = tuple(sorted((ip.src, ip.dst)))
    sorted_ports = tuple(sorted((sport, dport)))
    
    return (sorted_ips[0], sorted_ips[1], sorted_ports[0], sorted_ports[1], proto)


def anonymize_packet(pkt):
    """
    Anonymize a packet by zeroing out IP and MAC addresses.
    
    This ensures the model doesn't learn from specific IP addresses,
    making it generalize better to new networks.
    
    Args:
        pkt: Scapy packet
        
    Returns:
        bytes: Anonymized packet bytes
    """
    from scapy.all import IP
    
    pkt_copy = pkt.copy()
    pkt_copy[IP].src = "0.0.0.0"
    pkt_copy[IP].dst = "0.0.0.0"
    
    if "Ether" in pkt_copy:
        pkt_copy["Ether"].src = "00:00:00:00:00:00"
        pkt_copy["Ether"].dst = "00:00:00:00:00:00"
    
    return bytes(pkt_copy)


def is_corrupted(pkt):
    """
    Check if a packet is corrupted or invalid.
    
    Args:
        pkt: Scapy packet
        
    Returns:
        bool: True if packet is corrupted
    """
    from scapy.all import IP
    
    try:
        if IP not in pkt:
            return True
        if len(bytes(pkt)) == 0:
            return True
        return False
    except Exception:
        return True


def is_retransmission(pkt, seq_tracker):
    """
    Check if a TCP packet is a retransmission.
    
    Retransmissions have the same sequence number as a previous packet.
    We skip them to avoid duplicate data in the flow.
    
    Args:
        pkt: Scapy packet
        seq_tracker: dict tracking seen sequence numbers per connection
        
    Returns:
        bool: True if packet is a retransmission
    """
    from scapy.all import IP, TCP
    
    if TCP not in pkt:
        return False
    
    tcp = pkt[TCP]
    ip = pkt[IP]
    conn_key = (ip.src, ip.dst, tcp.sport, tcp.dport)
    seq = tcp.seq
    
    seen = seq_tracker[conn_key]
    if seq in seen:
        return True
    seen.add(seq)
    return False


def bytes_to_image(buffer):
    """
    Convert a byte buffer to a 28x28 grayscale image.
    
    Pads with zeros if buffer is shorter than 784 bytes,
    truncates if longer.
    
    Args:
        buffer: bytes or bytearray
        
    Returns:
        numpy.ndarray: 28x28 uint8 array, or None if empty
    """
    if len(buffer) == 0:
        return None
    
    arr = np.frombuffer(buffer if isinstance(buffer, bytes) else bytes(buffer), dtype=np.uint8)
    
    if len(arr) >= MAX_LEN:
        arr = arr[:MAX_LEN]
    else:
        arr = np.pad(arr, (0, MAX_LEN - len(arr)), "constant", constant_values=0)
    
    return arr.reshape(ROWS, COLS)

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
# HELPER FUNCTIONS
# Note: Model class is imported from src/model/hybrid_tiny.py
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
    
    # Try to use weights_only=True for security (PyTorch >= 1.13)
    # Fall back to default behavior for older versions
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch version that doesn't support weights_only parameter
        state_dict = torch.load(MODEL_PATH, map_location=device)
    
    model.load_state_dict(state_dict)
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


def process_pcap_bytes(pcap_bytes, max_flows=None):
    """
    Process uploaded PCAP file and extract flow images.
    
    Uses helper functions from src/preprocess/preprocess_memory_safe.py:
    - flow_key: Create bidirectional flow identifier
    - anonymize_packet: Zero out IPs and MACs
    - is_corrupted: Check for corrupted packets
    - is_retransmission: Skip TCP retransmissions
    - bytes_to_image: Convert buffer to 28x28 image
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
                # Skip corrupted or non-IP/TCP/UDP packets (using imported helper)
                if is_corrupted(pkt):
                    continue
                if IP not in pkt or (TCP not in pkt and UDP not in pkt):
                    continue
                
                # Skip retransmissions (using imported helper)
                if is_retransmission(pkt, seq_tracker):
                    continue
                
                # Get flow key (using imported helper)
                key = flow_key(pkt)
                buffer = flow_buffers[key]
                if len(buffer) >= MAX_LEN:
                    continue
                
                # Anonymize and get bytes (using imported helper)
                pkt_bytes = anonymize_packet(pkt)
                if not pkt_bytes:
                    continue
                
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
        
        # Convert to image (using imported helper)
        img = bytes_to_image(bytes(buffer))
        if img is None:
            continue
        
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
    st.sidebar.title("AttentionNet")
    st.sidebar.markdown("Network Traffic Classifier")
    
    page = st.sidebar.radio(
        "Select Mode",
        ["Live Capture & Classification", "Pre-captured Traffic (PCAP) Classification & Analysis", "Preprocessed Traffic Data Classification & Analysis", "About AttentionNet"]
    )
    
    # Load model (cached)
    try:
        model, device = load_model()
        st.sidebar.success(f"Model loaded ({device})")
    except Exception as e:
        st.sidebar.error(f"Model error: {e}")
        st.stop()
    
    # Route to selected page
    if page == "Live Capture & Classification":
        capture_page(model, device)
    elif page == "Pre-captured Traffic (PCAP) Classification & Analysis":
        pcap_page(model, device)
    elif page == "Preprocessed Traffic Data Classification & Analysis":
        dataset_page(model, device)
    else:
        about_page()


def capture_page(model, device):
    """Live network capture page."""
    
    st.title("Live Capture & Classification")
    st.markdown("Capture live network traffic and classify it in real-time.")
    
    # Check if scapy is available
    try:
        from scapy.all import sniff, IP, TCP, UDP
        scapy_available = True
    except ImportError:
        scapy_available = False
    
    if not scapy_available:
        st.error("Scapy is required for live capture. Install with: `pip install scapy`")
        return
    
    # Capture settings
    st.subheader("Capture Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        interface = st.text_input(
            "Network Interface",
            value="en0",
            help="Network interface to capture from (e.g., en0, eth0, wlan0)"
        )
    
    with col2:
        duration = st.slider(
            "Capture Duration (seconds)",
            min_value=5,
            max_value=120,
            value=15,
            help="How long to capture packets. Use longer duration for more traffic."
        )
    
    # Flow limit settings
    col3, col4 = st.columns(2)
    with col3:
        capture_all = st.checkbox("Capture all flows", value=True, key="capture_all_flows")
    with col4:
        if capture_all:
            max_flows = None
        else:
            max_flows = st.number_input(
                "Max Flows", min_value=5, max_value=500, value=100,
                key="capture_max_flows"
            )
    
    # Common interfaces help
    with st.expander("Common interface names"):
        st.markdown("""
        | OS | Interface | Description |
        |----|-----------|-------------|
        | **macOS** | `en0` | WiFi |
        | **macOS** | `bridge100` | Internet Sharing (Firewall mode) |
        | **Linux** | `eth0` | Ethernet |
        | **Linux** | `wlan0` | WiFi |
        """)
    
    # Capture button
    st.markdown("---")
    if st.button("Start Capture", type="primary", use_container_width=True):
        
        # Show simple status message
        with st.spinner(f"Capturing traffic on {interface} for {duration} seconds..."):
            try:
                images, flow_info = capture_live_traffic(
                    interface, duration, max_flows
                )
            except PermissionError:
                st.error("**Permission denied!** Live capture requires administrator privileges.")
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
        
        st.success(f"Captured and classified {len(images)} flows!")
    
    # Display results if available
    if 'capture_results' in st.session_state:
        display_capture_results(st.session_state['capture_results'])


def capture_live_traffic_with_progress(interface, duration, max_flows, progress_bar, status_text, stats_text):
    """
    Capture live network traffic with progress updates.
    
    Uses helper functions from src/preprocess/preprocess_memory_safe.py:
    - flow_key: Create bidirectional flow identifier
    - anonymize_packet: Zero out IPs and MACs
    - is_retransmission: Skip TCP retransmissions
    - bytes_to_image: Convert buffer to 28x28 image
    """
    import time
    from scapy.all import sniff, IP, TCP, UDP
    
    flow_buffers = defaultdict(bytearray)
    seq_tracker = defaultdict(set)
    packet_count = [0]  # Use list for mutable counter in nested function
    
    def packet_callback(pkt):
        """Process each captured packet using imported helpers."""
        if IP not in pkt or (TCP not in pkt and UDP not in pkt):
            return
        
        packet_count[0] += 1
        
        # Skip retransmissions (using imported helper)
        if is_retransmission(pkt, seq_tracker):
            return
        
        # Get flow key (using imported helper)
        key = flow_key(pkt)
        buffer = flow_buffers[key]
        if len(buffer) >= MAX_LEN:
            return
        
        # Anonymize and get bytes (using imported helper)
        pkt_bytes = anonymize_packet(pkt)
        if not pkt_bytes:
            return
        
        remaining = MAX_LEN - len(buffer)
        buffer.extend(pkt_bytes[:remaining])
    
    # Capture in short bursts to show progress
    start_time = time.time()
    burst_duration = 1  # Update progress every 1 second
    
    while True:
        elapsed = time.time() - start_time
        remaining_time = duration - elapsed
        
        if remaining_time <= 0:
            break
        
        # Update progress bar
        progress = min(elapsed / duration, 1.0)
        progress_bar.progress(progress)
        
        # Update status
        status_text.info(f"**Capturing on {interface}...** {int(remaining_time)}s remaining")
        stats_text.markdown(
            f"**Packets:** {packet_count[0]} | "
            f"**Flows:** {len(flow_buffers)}"
        )
        
        # Capture for a short burst
        capture_time = min(burst_duration, remaining_time)
        try:
            sniff(iface=interface, prn=packet_callback, timeout=capture_time, store=False)
        except Exception as e:
            if "Permission" in str(e) or "Operation not permitted" in str(e):
                raise PermissionError(str(e))
            raise
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.success(f"Capture complete!")
    
    # Convert buffers to images
    images = []
    flow_info = []
    
    for key, buffer in flow_buffers.items():
        if max_flows and len(images) >= max_flows:
            break
        
        # Convert to image (using imported helper)
        img = bytes_to_image(bytes(buffer))
        if img is None:
            continue
        
        images.append(img)
        
        flow_info.append({
            'src': f"{key[0]}:{key[2]}",
            'dst': f"{key[1]}:{key[3]}",
            'proto': 'TCP' if key[4] == 6 else 'UDP' if key[4] == 17 else str(key[4]),
            'bytes': len(buffer)
        })
    
    return images, flow_info


def capture_live_traffic(interface, duration, max_flows):
    """
    Capture live network traffic and convert to flow images.
    
    Uses helper functions from src/preprocess/preprocess_memory_safe.py
    """
    from scapy.all import sniff, IP, TCP, UDP
    
    flow_buffers = defaultdict(bytearray)
    seq_tracker = defaultdict(set)
    
    def packet_callback(pkt):
        """Process each captured packet using imported helpers."""
        if IP not in pkt or (TCP not in pkt and UDP not in pkt):
            return
        
        # Skip retransmissions (using imported helper)
        if is_retransmission(pkt, seq_tracker):
            return
        
        # Get flow key (using imported helper)
        key = flow_key(pkt)
        buffer = flow_buffers[key]
        if len(buffer) >= MAX_LEN:
            return
        
        # Anonymize and get bytes (using imported helper)
        pkt_bytes = anonymize_packet(pkt)
        if not pkt_bytes:
            return
        
        remaining = MAX_LEN - len(buffer)
        buffer.extend(pkt_bytes[:remaining])
    
    # Perform capture
    sniff(iface=interface, prn=packet_callback, timeout=duration, store=False)
    
    # Convert buffers to images
    images = []
    flow_info = []
    
    for key, buffer in flow_buffers.items():
        if max_flows and len(images) >= max_flows:
            break
        
        # Convert to image (using imported helper)
        img = bytes_to_image(bytes(buffer))
        if img is None:
            continue
        
        images.append(img)
        
        flow_info.append({
            'src': f"{key[0]}:{key[2]}",
            'dst': f"{key[1]}:{key[3]}",
            'proto': 'TCP' if key[4] == 6 else 'UDP' if key[4] == 17 else str(key[4]),
            'bytes': len(buffer)
        })
    
    return images, flow_info


def display_capture_results(results):
    """Display live capture results with enhanced visualizations."""
    
    images = results['images']
    flow_info = results['flow_info']
    preds = results['preds']
    probs = results['probs']
    
    # Summary statistics
    vpn_count = sum(1 for p in preds if p >= 6)
    nonvpn_count = len(preds) - vpn_count
    
    st.markdown("---")
    st.subheader("Capture Results")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Flows", len(preds))
    col2.metric("Non-VPN Traffic", nonvpn_count)
    col3.metric("VPN Traffic", vpn_count)
    
    # =========================================================================
    # CLASS DISTRIBUTION SECTION - Bar Chart (Flows per Class)
    # =========================================================================
    st.markdown("### Traffic Type Distribution")
    
    from collections import Counter
    import pandas as pd
    
    class_counts = Counter(preds)
    total = len(preds)
    
    col_chart, col_breakdown = st.columns([1.5, 1])
    
    with col_chart:
        # Bar chart - Flows per Class
        if class_counts:
            fig, ax = plt.subplots(figsize=(12, 5))
            sorted_classes = sorted(class_counts.keys(), key=lambda x: -class_counts[x])
            x_labels = [CLASS_NAMES[cls] for cls in sorted_classes]
            counts = [class_counts[cls] for cls in sorted_classes]
            colors = [CLASS_COLORS[cls] for cls in sorted_classes]
            
            bars = ax.bar(range(len(counts)), counts, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel('Number of Flows', fontsize=11)
            ax.set_xlabel('Traffic Class', fontsize=11)
            ax.set_title('Flows per Class', fontsize=14, fontweight='bold')
            
            # Add count and percentage labels on bars
            for bar, count in zip(bars, counts):
                pct = 100 * count / total
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                       f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
            
            ax.set_ylim(0, max(counts) * 1.2)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    with col_breakdown:
        # Detailed breakdown
        st.markdown("**Breakdown Table:**")
        breakdown_data = []
        for cls in sorted(class_counts.keys(), key=lambda x: -class_counts[x]):
            count = class_counts[cls]
            pct = 100 * count / total
            breakdown_data.append({
                "Class": CLASS_NAMES[cls],
                "Count": count,
                "Percentage": f"{pct:.1f}%",
                "Type": "VPN" if cls >= 6 else "Non-VPN"
            })
        
        df_breakdown = pd.DataFrame(breakdown_data)
        st.dataframe(df_breakdown, use_container_width=True, hide_index=True, height=250)
    
    # =========================================================================
    # FLOW VISUALIZATIONS (GRAYSCALE)
    # =========================================================================
    st.markdown("---")
    st.subheader("Flow Visualizations")
    
    # Controls
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        n_display = st.slider("Flows to display", 4, min(40, len(images)), 
                              min(16, len(images)), step=4, key="capture_n_display")
    with col_ctrl2:
        cols_per_row = st.selectbox("Columns", [4, 5, 6, 8], index=1, key="capture_cols")
    
    for row_start in range(0, n_display, cols_per_row):
        cols = st.columns(cols_per_row)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < n_display:
                with col:
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(images[idx], cmap='gray', vmin=0, vmax=255)  # GRAYSCALE
                    ax.axis('off')
                    pred_name = CLASS_NAMES[preds[idx]]
                    if len(pred_name) > 12:
                        pred_name = pred_name[:10] + "..."
                    conf = probs[idx][preds[idx]] * 100
                    ax.set_title(f"#{idx+1}\n{pred_name}\n{conf:.0f}%", fontsize=7)
                    st.pyplot(fig)
                    plt.close()
    
    # =========================================================================
    # CLASSIFICATION TABLE
    # =========================================================================
    st.markdown("---")
    st.subheader("📋 Classification Details")
    
    data = []
    for i, (info, pred, prob) in enumerate(zip(flow_info, preds, probs)):
        conf = prob[pred] * 100
        is_vpn = "VPN" if pred >= 6 else "Non-VPN"
        data.append({
            "Flow #": i+1,
            "Protocol": info['proto'],
            "Bytes": info['bytes'],
            "Prediction": CLASS_NAMES[pred],
            "Confidence": f"{conf:.1f}%",
            "VPN Status": is_vpn
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True, height=250)
    
    # =========================================================================
    # INDIVIDUAL FLOW INSPECTOR
    # =========================================================================
    st.markdown("---")
    st.subheader("Individual Flow Inspector")
    
    selected_flow = st.selectbox(
        "Select a flow to inspect",
        range(len(images)),
        format_func=lambda x: f"Flow {x+1} - {CLASS_NAMES[preds[x]]} ({probs[x][preds[x]]*100:.1f}%)",
        key="capture_flow_selector"
    )
    
    col_image, col_probs = st.columns([1, 2])
    
    with col_image:
        st.markdown("**Flow Image (28×28)**")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(images[selected_flow], cmap='gray', vmin=0, vmax=255)  # GRAYSCALE
        ax.axis('off')
        ax.set_title(f"Flow {selected_flow+1}", fontsize=12)
        st.pyplot(fig)
        plt.close()
        
        info = flow_info[selected_flow]
        st.markdown(f"""
        **Details:**
        - Source: `{info['src']}`
        - Dest: `{info['dst']}`
        - Protocol: {info['proto']}
        - Bytes: {info['bytes']}
        """)
    
    with col_probs:
        st.markdown("**Classification Probabilities**")
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = range(12)
        bars = ax.barh(y_pos, probs[selected_flow], color=CLASS_COLORS)
        bars[preds[selected_flow]].set_edgecolor('black')
        bars[preds[selected_flow]].set_linewidth(3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([CLASS_NAMES[i] for i in range(12)], fontsize=10)
        ax.set_xlabel('Probability')
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        for i, (bar, prob_val) in enumerate(zip(bars, probs[selected_flow])):
            if prob_val > 0.05:
                ax.text(prob_val + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{prob_val*100:.1f}%', va='center', fontsize=9)
        
        pred_class = CLASS_NAMES[preds[selected_flow]]
        confidence = probs[selected_flow][preds[selected_flow]] * 100
        ax.set_title(f'Prediction: {pred_class} ({confidence:.1f}%)', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def pcap_page(model, device):
    """PCAP file processing page."""
    
    st.title("Pre-captured Traffic (PCAP) Classification & Analysis")
    st.markdown("Upload a PCAP file to classify the network flows it contains.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PCAP file",
        type=['pcap', 'pcapng'],
        help="Upload a network capture file to analyze"
    )
    
    # Settings
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        process_all = st.checkbox("Process all flows", value=True, key="pcap_all_flows")
    with col2:
        if process_all:
            st.info("All flows will be processed")
            max_flows = None  # No limit
        else:
            max_flows = st.slider("Maximum flows", 5, 200, 50, key="pcap_max_flows")
    
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
    """Display PCAP processing results with enhanced visualizations."""
    
    images = results['images']
    flow_info = results['flow_info']
    preds = results['preds']
    probs = results['probs']
    
    st.success(f"Processed {len(images)} flows")
    
    # Summary statistics
    vpn_count = sum(1 for p in preds if p >= 6)
    nonvpn_count = len(preds) - vpn_count
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Flows", len(preds))
    col2.metric("Non-VPN", nonvpn_count)
    col3.metric("VPN", vpn_count)
    
    # =========================================================================
    # CLASS DISTRIBUTION SECTION - Bar Chart (Flows per Class)
    # =========================================================================
    st.markdown("---")
    st.subheader("Class Distribution")
    
    from collections import Counter
    import pandas as pd
    
    class_counts = Counter(preds)
    total = len(preds)
    
    col_chart, col_breakdown = st.columns([1.5, 1])
    
    with col_chart:
        # Bar chart - Flows per Class
        fig, ax = plt.subplots(figsize=(12, 5))
        sorted_classes = sorted(class_counts.keys(), key=lambda x: -class_counts[x])
        x_labels = [CLASS_NAMES[cls] for cls in sorted_classes]
        counts = [class_counts[cls] for cls in sorted_classes]
        colors = [CLASS_COLORS[cls] for cls in sorted_classes]
        
        bars = ax.bar(range(len(counts)), counts, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Number of Flows', fontsize=11)
        ax.set_xlabel('Traffic Class', fontsize=11)
        ax.set_title('Flows per Class', fontsize=14, fontweight='bold')
        
        # Add count and percentage labels on bars
        for bar, count in zip(bars, counts):
            pct = 100 * count / total
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylim(0, max(counts) * 1.2)  # Add space for labels
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col_breakdown:
        # Detailed breakdown table
        st.markdown("**Breakdown Table:**")
        
        breakdown_data = []
        for cls in sorted(class_counts.keys(), key=lambda x: -class_counts[x]):
            count = class_counts[cls]
            pct = 100 * count / total
            is_vpn = "VPN" if cls >= 6 else "Non-VPN"
            breakdown_data.append({
                "Class": CLASS_NAMES[cls],
                "Count": count,
                "Percentage": f"{pct:.1f}%",
                "Type": is_vpn
            })
        
        df_breakdown = pd.DataFrame(breakdown_data)
        st.dataframe(df_breakdown, use_container_width=True, hide_index=True, height=300)
    
    # =========================================================================
    # FLOW VISUALIZATIONS AND TABLE
    # =========================================================================
    st.markdown("---")
    st.subheader("Flow Visualizations")
    
    # Controls for visualization
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        n_display = st.slider("Number of flows to display", 
                              min_value=4, max_value=min(40, len(images)), 
                              value=min(20, len(images)), step=4)
    with col_ctrl2:
        cols_per_row = st.selectbox("Columns per row", [4, 5, 6, 8], index=1)
    
    # Show flow images in a grid (GRAYSCALE)
    for row_start in range(0, n_display, cols_per_row):
        cols = st.columns(cols_per_row)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < n_display:
                with col:
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(images[idx], cmap='gray', vmin=0, vmax=255)  # GRAYSCALE
                    ax.axis('off')
                    pred_name = CLASS_NAMES[preds[idx]]
                    if len(pred_name) > 12:
                        pred_name = pred_name[:10] + "..."
                    conf = probs[idx][preds[idx]] * 100
                    ax.set_title(f"#{idx+1}\n{pred_name}\n{conf:.0f}%", fontsize=7)
                    st.pyplot(fig)
                    plt.close()
    
    # =========================================================================
    # CLASSIFICATION TABLE
    # =========================================================================
    st.markdown("---")
    st.subheader("📋 Classification Results")
    
    import pandas as pd
    
    data = []
    for i, (info, pred, prob) in enumerate(zip(flow_info, preds, probs)):
        conf = prob[pred] * 100
        is_vpn = "VPN" if pred >= 6 else "Non-VPN"
        data.append({
            "Flow #": i+1,
            "Protocol": info['proto'],
            "Bytes": info['bytes'],
            "Prediction": CLASS_NAMES[pred],
            "Confidence": f"{conf:.1f}%",
            "VPN Status": is_vpn
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True, height=300)
    
    # =========================================================================
    # INDIVIDUAL FLOW VIEWER (NEW)
    # =========================================================================
    st.markdown("---")
    st.subheader("Individual Flow Inspector")
    
    selected_flow = st.selectbox(
        "Select a flow to inspect in detail",
        range(len(images)),
        format_func=lambda x: f"Flow {x+1} - {CLASS_NAMES[preds[x]]} ({probs[x][preds[x]]*100:.1f}%)",
        key="pcap_flow_selector"
    )
    
    col_image, col_probs = st.columns([1, 2])
    
    with col_image:
        # Large flow image (GRAYSCALE)
        st.markdown("**Flow Image (28×28 bytes)**")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(images[selected_flow], cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        ax.set_title(f"Flow {selected_flow+1}", fontsize=14)
        st.pyplot(fig)
        plt.close()
        
        # Flow info
        info = flow_info[selected_flow]
        st.markdown(f"""
        **Flow Details:**
        - **Source:** {info['src']}
        - **Destination:** {info['dst']}
        - **Protocol:** {info['proto']}
        - **Bytes captured:** {info['bytes']}
        """)
    
    with col_probs:
        # Probability bar chart
        st.markdown("**Classification Probabilities**")
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = range(12)
        bars = ax.barh(y_pos, probs[selected_flow], color=CLASS_COLORS)
        bars[preds[selected_flow]].set_edgecolor('black')
        bars[preds[selected_flow]].set_linewidth(3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([CLASS_NAMES[i] for i in range(12)], fontsize=10)
        ax.set_xlabel('Probability', fontsize=11)
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add probability labels
        for i, (bar, prob_val) in enumerate(zip(bars, probs[selected_flow])):
            if prob_val > 0.05:
                ax.text(prob_val + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{prob_val*100:.1f}%', va='center', fontsize=9)
        
        pred_class = CLASS_NAMES[preds[selected_flow]]
        confidence = probs[selected_flow][preds[selected_flow]] * 100
        ax.set_title(f'Prediction: {pred_class} ({confidence:.1f}%)', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def dataset_page(model, device):
    """Dataset evaluation page."""
    
    st.title("Preprocessed Traffic Data Classification & Analysis")
    st.markdown("Load pre-processed test data and evaluate model performance.")
    
    # Dataset selection
    dataset_option = st.radio(
        "Select dataset",
        ["Test Set", "Upload Custom .npy"]
    )
    
    if dataset_option == "Test Set":
        data_path = os.path.join(TEST_DATA_DIR, "test_data_memory_safe_own_nonVPN_p2p.npy")
        labels_path = os.path.join(TEST_DATA_DIR, "test_labels_memory_safe_own_nonVPN_p2p.npy")
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
    
    st.success(f"Evaluation complete!")
    
    # Metrics display
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(labels))
    col2.metric("Accuracy", f"{accuracy:.2f}%")
    col3.metric("Macro F1", f"{macro_f1:.4f}")
    
    st.markdown("---")
    
    # Two columns: confusion matrix and per-class accuracy
    col_cm, col_acc = st.columns([1.2, 1])
    
    with col_cm:
        st.subheader("Normalized Confusion Matrix (Row-wise)")
        
        # Create confusion matrix
        all_labels = np.unique(np.concatenate([labels, preds]))
        cm = confusion_matrix(labels, preds, labels=all_labels)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        
        # Labels
        class_labels = [CLASS_NAMES.get(i, str(i)) for i in all_labels]
        ax.set_xticks(range(len(all_labels)))
        ax.set_yticks(range(len(all_labels)))
        ax.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(class_labels, fontsize=9)
        
        # Add percentage values
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                # Show as percentage with 2 decimal places (like 0.65, 0.82, etc.)
                text = f'{cm_norm[i, j]:.2f}' if cm_norm[i, j] > 0 else '0.00'
                ax.text(j, i, text, ha="center", va="center", 
                       color=color, fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        ax.set_title(f'Normalized Confusion Matrix (Row-wise) - Test Acc: {accuracy:.2f}%', 
                    fontsize=13, fontweight='bold', pad=20)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col_acc:
        st.subheader("Per-Class Accuracy")
        
        # Calculate per-class accuracy
        per_class_acc = []
        for label in all_labels:
            mask = labels == label
            if mask.sum() > 0:
                acc = 100 * np.mean(preds[mask] == labels[mask])
            else:
                acc = 0
            per_class_acc.append(acc)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(8, 10))
        y_pos = range(len(all_labels))
        bars = ax.barh(y_pos, per_class_acc, color=[CLASS_COLORS[i] for i in all_labels],
                       edgecolor='black', linewidth=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_labels, fontsize=9)
        ax.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title('Per-Class Accuracy', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlim(0, 100)
        ax.axvline(x=accuracy, color='red', linestyle='--', linewidth=2, 
                  label=f'Overall Acc: {accuracy:.2f}%', alpha=0.7)
        ax.grid(axis='x', alpha=0.3)
        ax.legend(loc='lower right')
        
        # Add percentage labels on bars
        for i, (bar, acc_val) in enumerate(zip(bars, per_class_acc)):
            ax.text(acc_val + 1, bar.get_y() + bar.get_height()/2,
                   f'{acc_val:.1f}%', va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Sample images section
    st.markdown("---")
    st.subheader("Sample Images by Class")
    
    unique_labels = np.unique(labels)
    samples_per_class = 3
    
    # Calculate grid dimensions
    n_classes = len(unique_labels)
    fig, axes = plt.subplots(n_classes, samples_per_class, 
                              figsize=(8, 1.8*n_classes))
    
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    for i, label in enumerate(unique_labels):
        class_images = images[labels == label]
        for j in range(samples_per_class):
            ax = axes[i, j]
            if j < len(class_images):
                ax.imshow(class_images[j], cmap='gray', vmin=0, vmax=255)
            ax.axis('off')
            if j == 0:
                name = CLASS_NAMES.get(label, str(label))
                ax.set_title(name, fontsize=9, loc='left', fontweight='bold')
    
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
    
    st.title("About AttentionNet")
    
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
    Dropout(0.5) → FC (128→128) → ReLU
       ↓
    Dropout(0.3) → FC (128→128) → ReLU
       ↓
    Dropout(0.2) → FC (128→12) → 12 class probabilities
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

