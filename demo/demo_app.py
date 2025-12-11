#!/usr/bin/env python3
"""
AttentionNet Traffic Classifier - Demo Application
===================================================
A comprehensive demonstration system for network traffic classification.

Features:
1. Live traffic capture and classification
2. PCAP file upload and processing
3. Pre-processed dataset evaluation
4. Flow image visualization
5. Classification reports and confusion matrices

Usage:
    python demo_app.py
"""

import os
import sys
import threading
import queue
import tempfile
from datetime import datetime
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('TkAgg')

# Try to import scapy (needed for live capture and pcap processing)
try:
    from scapy.all import IP, TCP, UDP, PcapReader, sniff, conf
    conf.use_pcap = True  # Use pcap for better compatibility
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[Warning] Scapy not available - live capture and PCAP processing disabled")
    print("         Install with: pip install scapy")

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Paths - These are configured for your project structure
MODEL_PATH = os.path.join(PROJECT_ROOT, "model_output/memory_safe/hocaya_gosterilcek/p2p_change/2layer_cnn_hybrid_3fc/best_model.pth")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data/final/memory_safe/own_nonVPN_p2p_2/ratio_change")

MAX_LEN = 784
ROWS, COLS = 28, 28

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

CLASS_COLORS = {
    0: "#e74c3c", 1: "#3498db", 2: "#2ecc71", 
    3: "#9b59b6", 4: "#f1c40f", 5: "#1abc9c",
    6: "#c0392b", 7: "#2980b9", 8: "#27ae60", 
    9: "#8e44ad", 10: "#f39c12", 11: "#16a085"
}


# =============================================================================
# MODEL DEFINITION
# =============================================================================
class TrafficCNN_TinyTransformer(nn.Module):
    """Hybrid CNN-Transformer model for traffic classification"""
    
    def __init__(self, num_classes=12):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        
        self.d_model = 128
        self.seq_len = 49
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, self.d_model))
        
        self.norm_in = nn.LayerNorm(self.d_model)
        self.norm_out = nn.LayerNorm(self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dim_feedforward=256,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.d_model, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.norm_in(x)
        x = self.transformer(x)
        x = self.norm_out(x)
        x = x.mean(dim=1)
        
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.relu(self.fc2(x))
        x = self.dropout3(x)
        return self.fc3(x)


# =============================================================================
# PACKET PROCESSING UTILITIES
# =============================================================================
class PacketProcessor:
    """Handles conversion of packets/flows to images"""
    
    @staticmethod
    def flow_key(pkt):
        """Generate a unique key for a flow"""
        if IP not in pkt:
            return None
        ip = pkt[IP]
        proto = ip.proto
        sport = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport if UDP in pkt else 0
        dport = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport if UDP in pkt else 0
        sorted_ips = tuple(sorted((ip.src, ip.dst)))
        sorted_ports = tuple(sorted((sport, dport)))
        return (sorted_ips[0], sorted_ips[1], sorted_ports[0], sorted_ports[1], proto)
    
    @staticmethod
    def anonymize_packet(pkt):
        """Anonymize packet by zeroing IPs and MACs"""
        pkt_copy = pkt.copy()
        if IP in pkt_copy:
            pkt_copy[IP].src = "0.0.0.0"
            pkt_copy[IP].dst = "0.0.0.0"
        if "Ether" in pkt_copy:
            pkt_copy["Ether"].src = "00:00:00:00:00:00"
            pkt_copy["Ether"].dst = "00:00:00:00:00:00"
        return bytes(pkt_copy)
    
    @staticmethod
    def bytes_to_image(buffer):
        """Convert byte buffer to 28x28 image"""
        if len(buffer) == 0:
            return None
        arr = np.frombuffer(buffer if isinstance(buffer, bytes) else bytes(buffer), dtype=np.uint8)
        if len(arr) >= MAX_LEN:
            arr = arr[:MAX_LEN]
        else:
            arr = np.pad(arr, (0, MAX_LEN - len(arr)), 'constant', constant_values=0)
        return arr.reshape(ROWS, COLS)
    
    @staticmethod
    def process_pcap(pcap_path, max_flows=None, progress_callback=None):
        """Process PCAP file and extract flow images"""
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy is required for PCAP processing")
        
        flow_buffers = defaultdict(bytearray)
        seq_tracker = defaultdict(set)
        packet_count = 0
        
        try:
            with PcapReader(pcap_path) as packets:
                for pkt in packets:
                    packet_count += 1
                    if progress_callback and packet_count % 100 == 0:
                        progress_callback(f"Processing packet {packet_count}...")
                    
                    if IP not in pkt or (TCP not in pkt and UDP not in pkt):
                        continue
                    
                    key = PacketProcessor.flow_key(pkt)
                    if key is None:
                        continue
                    
                    # Skip retransmissions for TCP
                    if TCP in pkt:
                        tcp = pkt[TCP]
                        ip = pkt[IP]
                        conn_key = (ip.src, ip.dst, tcp.sport, tcp.dport)
                        seq = tcp.seq
                        if seq in seq_tracker[conn_key]:
                            continue
                        seq_tracker[conn_key].add(seq)
                    
                    buffer = flow_buffers[key]
                    if len(buffer) >= MAX_LEN:
                        continue
                    
                    pkt_bytes = PacketProcessor.anonymize_packet(pkt)
                    remaining = MAX_LEN - len(buffer)
                    buffer.extend(pkt_bytes[:remaining])
                    
        except Exception as e:
            print(f"Error processing PCAP: {e}")
            return [], []
        
        images = []
        flow_info = []
        for key, buffer in flow_buffers.items():
            if max_flows and len(images) >= max_flows:
                break
            img = PacketProcessor.bytes_to_image(buffer)
            if img is not None:
                images.append(img)
                flow_info.append({
                    'src': f"{key[0]}:{key[2]}",
                    'dst': f"{key[1]}:{key[3]}",
                    'proto': 'TCP' if key[4] == 6 else 'UDP' if key[4] == 17 else str(key[4]),
                    'bytes': len(buffer)
                })
        
        return images, flow_info


# =============================================================================
# CLASSIFIER
# =============================================================================
class TrafficClassifier:
    """Wrapper for the traffic classification model"""
    
    def __init__(self, model_path=MODEL_PATH):
        self.device = self._get_device()
        self.model = None
        self.model_path = model_path
        self.loaded = False
        
    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = TrafficCNN_TinyTransformer(num_classes=12).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.loaded = True
        return True
    
    def predict(self, image):
        """Predict class for a single 28x28 image"""
        if not self.loaded:
            self.load_model()
        
        # Prepare input
        tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
        
        probs = probs.cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        return pred_idx, probs
    
    def predict_batch(self, images):
        """Predict classes for multiple images"""
        if not self.loaded:
            self.load_model()
        
        tensor = torch.FloatTensor(np.array(images)).unsqueeze(1) / 255.0
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
        
        probs = probs.cpu().numpy()
        preds = np.argmax(probs, axis=1)
        return preds, probs


# =============================================================================
# MAIN APPLICATION
# =============================================================================
class AttentionNetDemo:
    """Main demo application with GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("AttentionNet Traffic Classifier - Demo")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Initialize components
        self.classifier = TrafficClassifier()
        self.current_images = []
        self.current_labels = []
        self.current_predictions = []
        self.capture_thread = None
        self.capturing = False
        
        # Store results for each mode
        self.capture_results = {}
        self.pcap_results = {}
        
        # Setup UI
        self._setup_styles()
        self._create_widgets()
        self._load_model()
    
    def _setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Helvetica', 24, 'bold'))
        style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('Status.TLabel', font=('Helvetica', 10))
        style.configure('Big.TButton', font=('Helvetica', 12), padding=10)
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
    
    def _create_widgets(self):
        """Create all UI components"""
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title bar
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(title_frame, text="🔍 AttentionNet Traffic Classifier", 
                  style='Title.TLabel').pack(side=tk.LEFT)
        
        # Status indicator
        self.status_var = tk.StringVar(value="Model: Loading...")
        self.status_label = ttk.Label(title_frame, textvariable=self.status_var, 
                                      style='Status.TLabel')
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Device info
        self.device_var = tk.StringVar(value="")
        ttk.Label(title_frame, textvariable=self.device_var, 
                  style='Status.TLabel').pack(side=tk.RIGHT, padx=10)
        
        # Notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self._create_capture_tab()
        self._create_pcap_tab()
        self._create_dataset_tab()
        self._create_about_tab()
    
    def _create_capture_tab(self):
        """Create live capture tab"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="🔴 Live Capture")
        
        # Controls frame
        controls = ttk.LabelFrame(tab, text="Capture Controls", padding=10)
        controls.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(controls, text="Interface:").pack(side=tk.LEFT)
        self.interface_var = tk.StringVar(value="en0")
        interface_entry = ttk.Entry(controls, textvariable=self.interface_var, width=15)
        interface_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(controls, text="Duration (s):").pack(side=tk.LEFT, padx=(20, 0))
        self.duration_var = tk.StringVar(value="10")
        duration_entry = ttk.Entry(controls, textvariable=self.duration_var, width=8)
        duration_entry.pack(side=tk.LEFT, padx=5)
        
        self.capture_btn = ttk.Button(controls, text="▶ Start Capture", 
                                       command=self._start_capture, style='Big.TButton')
        self.capture_btn.pack(side=tk.LEFT, padx=20)
        
        self.stop_btn = ttk.Button(controls, text="⏹ Stop", 
                                   command=self._stop_capture, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        
        # Note about permissions
        note_label = ttk.Label(controls, text="Note: May require sudo/admin for live capture", 
                               foreground='gray')
        note_label.pack(side=tk.RIGHT)
        
        # Results area (shared visualization)
        self._create_results_area(tab, "capture")
    
    def _create_pcap_tab(self):
        """Create PCAP file processing tab"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="📁 Load PCAP")
        
        # Controls
        controls = ttk.LabelFrame(tab, text="PCAP File Selection", padding=10)
        controls.pack(fill=tk.X, pady=(0, 10))
        
        self.pcap_path_var = tk.StringVar(value="No file selected")
        path_label = ttk.Label(controls, textvariable=self.pcap_path_var, width=70)
        path_label.pack(side=tk.LEFT)
        
        ttk.Button(controls, text="📂 Browse...", 
                   command=self._browse_pcap, style='Big.TButton').pack(side=tk.LEFT, padx=10)
        self.process_pcap_btn = ttk.Button(controls, text="🔄 Process & Classify", 
                   command=self._process_pcap, style='Big.TButton')
        self.process_pcap_btn.pack(side=tk.LEFT)
        
        # Max flows option
        ttk.Label(controls, text="Max flows:").pack(side=tk.LEFT, padx=(20, 5))
        self.max_flows_var = tk.StringVar(value="50")
        ttk.Entry(controls, textvariable=self.max_flows_var, width=6).pack(side=tk.LEFT)
        
        # Results area
        self._create_results_area(tab, "pcap")
    
    def _create_dataset_tab(self):
        """Create pre-processed dataset evaluation tab"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="📊 Test Dataset")
        
        # Controls
        controls = ttk.LabelFrame(tab, text="Dataset Selection", padding=10)
        controls.pack(fill=tk.X, pady=(0, 10))
        
        # Quick load buttons for existing datasets
        ttk.Button(controls, text="📥 Load Test Set", 
                   command=self._load_test_set, style='Big.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="📥 Load Train Set", 
                   command=self._load_train_set, style='Big.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="📂 Load Custom .npy...", 
                   command=self._browse_npy, style='Big.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(controls, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=15)
        
        self.evaluate_btn = ttk.Button(controls, text="📈 Run Evaluation", 
                   command=self._evaluate_dataset, style='Big.TButton')
        self.evaluate_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(controls, text="💾 Export Report", 
                   command=self._export_report).pack(side=tk.RIGHT, padx=5)
        
        # Dataset info label
        self.dataset_info_var = tk.StringVar(value="No dataset loaded")
        ttk.Label(controls, textvariable=self.dataset_info_var, 
                  foreground='gray').pack(side=tk.RIGHT, padx=20)
        
        # Split into left (images) and right (metrics) panels
        content = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Left: Sample images
        left_frame = ttk.LabelFrame(content, text="Sample Images by Class", padding=5)
        content.add(left_frame, weight=1)
        
        self.dataset_fig = Figure(figsize=(6, 8), dpi=100)
        self.dataset_canvas = FigureCanvasTkAgg(self.dataset_fig, master=left_frame)
        self.dataset_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Right: Metrics
        right_frame = ttk.Frame(content)
        content.add(right_frame, weight=1)
        
        # Confusion matrix
        cm_frame = ttk.LabelFrame(right_frame, text="Confusion Matrix", padding=5)
        cm_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.cm_fig = Figure(figsize=(6, 5), dpi=100)
        self.cm_canvas = FigureCanvasTkAgg(self.cm_fig, master=cm_frame)
        self.cm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Metrics text
        metrics_frame = ttk.LabelFrame(right_frame, text="Classification Report", padding=5)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        self.metrics_text = tk.Text(metrics_frame, height=12, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(metrics_frame, command=self.metrics_text.yview)
        self.metrics_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_about_tab(self):
        """Create about/info tab"""
        tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(tab, text="ℹ️ About")
        
        # Create scrollable frame
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        info_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    AttentionNet Traffic Classifier                            ║
║                         Demo Application v1.0                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

OVERVIEW
────────
A hybrid CNN-Transformer model for encrypted network traffic classification.
This demo allows you to classify network traffic from various sources.


MODEL ARCHITECTURE
──────────────────
• Input: 28×28 grayscale images (784 bytes from packet flows)

• CNN Backbone:
  - Conv2D (1→64 channels, 3×3 kernel) + BatchNorm + MaxPool
  - Conv2D (64→128 channels, 3×3 kernel) + BatchNorm + MaxPool
  - Output: 128 channels × 7×7 spatial = 49 tokens

• Transformer Encoder:
  - 2 encoder layers
  - 4 attention heads
  - 256 feedforward dimensions
  - GELU activation

• Classifier:
  - Global average pooling
  - FC (128→128) with dropout
  - FC (128→128) with dropout  
  - FC (128→12) output


CLASSIFICATION CLASSES
──────────────────────
The model classifies traffic into 12 categories:

  Non-VPN Traffic:                    VPN Traffic:
  ├── Chat (NonVPN)                   ├── Chat (VPN)
  ├── Email (NonVPN)                  ├── Email (VPN)
  ├── File Transfer (NonVPN)          ├── File Transfer (VPN)
  ├── P2P (NonVPN)                    ├── P2P (VPN)
  ├── Streaming (NonVPN)              ├── Streaming (VPN)
  └── VoIP (NonVPN)                   └── VoIP (VPN)


PROCESSING PIPELINE
───────────────────
1. Capture/Load network packets
2. Group packets into flows (by IP+port pairs)
3. Filter retransmissions (TCP)
4. Anonymize packets (zero out IPs/MACs)
5. Extract first 784 bytes from each flow
6. Reshape to 28×28 grayscale image
7. Normalize and classify using trained model


PERFORMANCE METRICS
───────────────────
• Test Accuracy: ~86%
• Macro F1 Score: ~0.88


USAGE INSTRUCTIONS
──────────────────
1. Live Capture Tab:
   - Enter network interface name (e.g., en0, eth0, wlan0)
   - Set capture duration in seconds
   - Click "Start Capture" (may require admin privileges)

2. Load PCAP Tab:
   - Click "Browse" to select a .pcap or .pcapng file
   - Set maximum flows to process
   - Click "Process & Classify"

3. Test Dataset Tab:
   - Click "Load Test Set" to load pre-processed test data
   - Click "Run Evaluation" to see metrics
   - View confusion matrix and classification report
   - Export results to text file


REQUIREMENTS
────────────
• Python 3.8+
• PyTorch
• NumPy
• scikit-learn
• matplotlib
• scapy (for live capture and PCAP processing)


        """
        
        text_widget = tk.Text(scrollable_frame, font=('Courier', 11), wrap=tk.NONE, 
                              width=90, height=40)
        text_widget.insert('1.0', info_text)
        text_widget.configure(state='disabled')
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_results_area(self, parent, prefix):
        """Create shared results visualization area"""
        # Split into left and right panels
        content = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Flow images
        left_frame = ttk.LabelFrame(content, text="Flow Visualizations (28×28 images)", padding=5)
        content.add(left_frame, weight=1)
        
        fig = Figure(figsize=(5, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=left_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        setattr(self, f'{prefix}_fig', fig)
        setattr(self, f'{prefix}_canvas', canvas)
        
        # Right panel - Classification results
        right_frame = ttk.Frame(content)
        content.add(right_frame, weight=1)
        
        # Summary stats
        stats_frame = ttk.LabelFrame(right_frame, text="Summary", padding=5)
        stats_frame.pack(fill=tk.X, pady=(0, 5))
        
        stats_var = tk.StringVar(value="No results yet")
        stats_label = ttk.Label(stats_frame, textvariable=stats_var, font=('Helvetica', 11))
        stats_label.pack(fill=tk.X)
        setattr(self, f'{prefix}_stats_var', stats_var)
        
        # Results list
        results_frame = ttk.LabelFrame(right_frame, text="Classification Results", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Treeview for results
        columns = ('flow', 'prediction', 'confidence', 'vpn')
        tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=10)
        tree.heading('flow', text='Flow')
        tree.heading('prediction', text='Prediction')
        tree.heading('confidence', text='Confidence')
        tree.heading('vpn', text='VPN?')
        tree.column('flow', width=180)
        tree.column('prediction', width=150)
        tree.column('confidence', width=100)
        tree.column('vpn', width=100)
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        setattr(self, f'{prefix}_tree', tree)
        
        # Probability chart
        prob_frame = ttk.LabelFrame(right_frame, text="Class Probabilities (Select a flow above)", padding=5)
        prob_frame.pack(fill=tk.BOTH, expand=True)
        
        prob_fig = Figure(figsize=(5, 3), dpi=100)
        prob_canvas = FigureCanvasTkAgg(prob_fig, master=prob_frame)
        prob_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        setattr(self, f'{prefix}_prob_fig', prob_fig)
        setattr(self, f'{prefix}_prob_canvas', prob_canvas)
        
        # Bind selection event
        tree.bind('<<TreeviewSelect>>', lambda e: self._on_flow_select(prefix))
    
    def _load_model(self):
        """Load the classification model"""
        def load_in_thread():
            try:
                self.classifier.load_model()
                self.root.after(0, lambda: self.status_var.set(f"Model: Loaded ✓"))
                self.root.after(0, lambda: self.device_var.set(f"Device: {self.classifier.device}"))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Model: Error"))
                self.root.after(0, lambda: messagebox.showerror("Model Error", f"Could not load model:\n{e}\n\nPath: {MODEL_PATH}"))
        
        threading.Thread(target=load_in_thread, daemon=True).start()
    
    # =========================================================================
    # LIVE CAPTURE
    # =========================================================================
    def _start_capture(self):
        """Start live packet capture"""
        if not SCAPY_AVAILABLE:
            messagebox.showerror("Error", 
                "Scapy is required for live capture.\n\n"
                "Install with: pip install scapy\n\n"
                "Note: Live capture may also require root/admin privileges.")
            return
        
        interface = self.interface_var.get()
        try:
            duration = int(self.duration_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid duration - please enter a number")
            return
        
        self.capturing = True
        self.capture_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set(f"Capturing on {interface}...")
        
        # Clear previous results
        tree = getattr(self, 'capture_tree')
        tree.delete(*tree.get_children())
        
        # Start capture in background thread
        self.capture_thread = threading.Thread(
            target=self._capture_packets, 
            args=(interface, duration),
            daemon=True
        )
        self.capture_thread.start()
    
    def _capture_packets(self, interface, duration):
        """Capture packets in background thread"""
        flow_buffers = defaultdict(bytearray)
        seq_tracker = defaultdict(set)
        
        def packet_callback(pkt):
            if not self.capturing:
                return
            if IP not in pkt or (TCP not in pkt and UDP not in pkt):
                return
            
            key = PacketProcessor.flow_key(pkt)
            if key is None:
                return
            
            # Skip retransmissions
            if TCP in pkt:
                tcp = pkt[TCP]
                ip = pkt[IP]
                conn_key = (ip.src, ip.dst, tcp.sport, tcp.dport)
                seq = tcp.seq
                if seq in seq_tracker[conn_key]:
                    return
                seq_tracker[conn_key].add(seq)
            
            buffer = flow_buffers[key]
            if len(buffer) < MAX_LEN:
                pkt_bytes = PacketProcessor.anonymize_packet(pkt)
                remaining = MAX_LEN - len(buffer)
                buffer.extend(pkt_bytes[:remaining])
        
        try:
            sniff(iface=interface, prn=packet_callback, timeout=duration, store=False)
        except PermissionError:
            self.root.after(0, lambda: messagebox.showerror("Permission Error", 
                "Live capture requires root/admin privileges.\n\n"
                "Try running with: sudo python demo_app.py"))
            self.root.after(0, self._capture_finished)
            return
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Capture Error", 
                f"Error during capture:\n{str(e)}\n\n"
                f"Make sure interface '{interface}' exists."))
            self.root.after(0, self._capture_finished)
            return
        
        # Process captured flows
        images = []
        flow_info = []
        for key, buffer in flow_buffers.items():
            img = PacketProcessor.bytes_to_image(buffer)
            if img is not None:
                images.append(img)
                flow_info.append({
                    'src': f"{key[0]}:{key[2]}",
                    'dst': f"{key[1]}:{key[3]}",
                    'proto': 'TCP' if key[4] == 6 else 'UDP' if key[4] == 17 else str(key[4]),
                    'bytes': len(buffer)
                })
        
        # Update UI on main thread
        self.root.after(0, lambda: self._display_results("capture", images, flow_info))
        self.root.after(0, self._capture_finished)
    
    def _stop_capture(self):
        """Stop live capture"""
        self.capturing = False
        self.status_var.set("Stopping capture...")
    
    def _capture_finished(self):
        """Called when capture is complete"""
        self.capturing = False
        self.capture_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set("Capture complete ✓")
    
    # =========================================================================
    # PCAP PROCESSING
    # =========================================================================
    def _browse_pcap(self):
        """Open file dialog for PCAP selection"""
        filepath = filedialog.askopenfilename(
            title="Select PCAP File",
            filetypes=[("PCAP files", "*.pcap *.pcapng"), ("All files", "*.*")]
        )
        if filepath:
            self.pcap_path_var.set(filepath)
    
    def _process_pcap(self):
        """Process selected PCAP file"""
        if not SCAPY_AVAILABLE:
            messagebox.showerror("Error", 
                "Scapy is required for PCAP processing.\n\n"
                "Install with: pip install scapy")
            return
        
        pcap_path = self.pcap_path_var.get()
        if not os.path.exists(pcap_path):
            messagebox.showerror("Error", "Please select a valid PCAP file")
            return
        
        try:
            max_flows = int(self.max_flows_var.get())
        except ValueError:
            max_flows = 50
        
        self.status_var.set("Processing PCAP...")
        self.process_pcap_btn.configure(state=tk.DISABLED)
        self.root.update()
        
        def process_in_thread():
            try:
                def update_status(msg):
                    self.root.after(0, lambda: self.status_var.set(msg))
                
                images, flow_info = PacketProcessor.process_pcap(pcap_path, 
                                                                  max_flows=max_flows,
                                                                  progress_callback=update_status)
                
                if not images:
                    self.root.after(0, lambda: messagebox.showwarning("Warning", 
                        "No valid flows found in PCAP.\n\n"
                        "Make sure the file contains IP/TCP/UDP packets."))
                    self.root.after(0, lambda: self.status_var.set("No flows found"))
                else:
                    self.root.after(0, lambda: self._display_results("pcap", images, flow_info))
                    self.root.after(0, lambda: self.status_var.set(f"Processed {len(images)} flows ✓"))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to process PCAP:\n{e}"))
                self.root.after(0, lambda: self.status_var.set("Processing failed"))
            finally:
                self.root.after(0, lambda: self.process_pcap_btn.configure(state=tk.NORMAL))
        
        threading.Thread(target=process_in_thread, daemon=True).start()
    
    # =========================================================================
    # DATASET EVALUATION
    # =========================================================================
    def _load_test_set(self):
        """Load the default test set"""
        data_path = os.path.join(TEST_DATA_DIR, "test_data_memory_safe_own_nonVPN_p2p.npy")
        labels_path = os.path.join(TEST_DATA_DIR, "test_labels_memory_safe_own_nonVPN_p2p.npy")
        self._load_dataset(data_path, labels_path, "Test")
    
    def _load_train_set(self):
        """Load the training set"""
        data_path = os.path.join(TEST_DATA_DIR, "train_data_memory_safe_own_nonVPN_p2p.npy")
        labels_path = os.path.join(TEST_DATA_DIR, "train_labels_memory_safe_own_nonVPN_p2p.npy")
        self._load_dataset(data_path, labels_path, "Train")
    
    def _load_dataset(self, data_path, labels_path, name):
        """Load a dataset from .npy files"""
        if not os.path.exists(data_path):
            messagebox.showerror("Error", f"Data file not found:\n{data_path}")
            return
        if not os.path.exists(labels_path):
            messagebox.showerror("Error", f"Labels file not found:\n{labels_path}")
            return
        
        self.status_var.set(f"Loading {name} dataset...")
        self.root.update()
        
        try:
            self.current_images = np.load(data_path)
            self.current_labels = np.load(labels_path)
            self.current_predictions = []
            
            # Calculate label distribution
            unique, counts = np.unique(self.current_labels, return_counts=True)
            dist_str = ", ".join([f"{CLASS_NAMES.get(l, str(l))[:10]}: {c}" 
                                  for l, c in zip(unique, counts)])
            
            self.dataset_info_var.set(f"{name}: {len(self.current_images)} samples | {len(unique)} classes")
            self.status_var.set(f"Loaded {len(self.current_images)} {name.lower()} samples ✓")
            
            # Display sample images
            self._display_dataset_samples()
            
            # Clear previous metrics
            self.metrics_text.configure(state='normal')
            self.metrics_text.delete('1.0', tk.END)
            self.metrics_text.insert('1.0', f"Dataset loaded: {len(self.current_images)} samples\n\n")
            self.metrics_text.insert(tk.END, "Label Distribution:\n" + "-"*40 + "\n")
            for l, c in zip(unique, counts):
                pct = 100 * c / len(self.current_labels)
                self.metrics_text.insert(tk.END, f"  {CLASS_NAMES.get(l, str(l)):20s}: {c:5d} ({pct:5.1f}%)\n")
            self.metrics_text.insert(tk.END, "\n→ Click 'Run Evaluation' to classify and see metrics")
            self.metrics_text.configure(state='disabled')
            
            # Clear confusion matrix
            self.cm_fig.clear()
            self.cm_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{e}")
            self.status_var.set("Load failed")
    
    def _browse_npy(self):
        """Browse for custom .npy data files"""
        data_path = filedialog.askopenfilename(
            title="Select Data File (*data*.npy or similar)",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        if not data_path:
            return
        
        # Try to find corresponding labels file
        labels_path = None
        possible_labels = [
            data_path.replace("data", "labels"),
            data_path.replace("_data", "_labels"),
            data_path.replace("Data", "Labels"),
        ]
        
        for path in possible_labels:
            if os.path.exists(path):
                labels_path = path
                break
        
        if not labels_path:
            labels_path = filedialog.askopenfilename(
                title="Select Labels File",
                initialdir=os.path.dirname(data_path),
                filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
            )
        
        if not labels_path:
            messagebox.showerror("Error", "Labels file is required")
            return
        
        self._load_dataset(data_path, labels_path, "Custom")
    
    def _display_dataset_samples(self):
        """Display sample images from dataset"""
        self.dataset_fig.clear()
        
        unique_labels = np.unique(self.current_labels)
        n_classes = len(unique_labels)
        n_samples = min(5, len(self.current_images) // n_classes) if n_classes > 0 else 0
        
        if n_samples == 0:
            return
        
        for i, label in enumerate(unique_labels):
            class_images = self.current_images[self.current_labels == label]
            for j in range(min(n_samples, len(class_images))):
                ax = self.dataset_fig.add_subplot(n_classes, n_samples, i * n_samples + j + 1)
                ax.imshow(class_images[j], cmap='viridis', vmin=0, vmax=255)
                ax.axis('off')
                if j == 0:
                    # Truncate long names
                    name = CLASS_NAMES.get(label, str(label))
                    if len(name) > 15:
                        name = name[:12] + "..."
                    ax.set_title(name, fontsize=8)
        
        self.dataset_fig.suptitle("Sample Images by Class", fontsize=10, fontweight='bold')
        self.dataset_fig.tight_layout()
        self.dataset_canvas.draw()
    
    def _evaluate_dataset(self):
        """Run evaluation on loaded dataset"""
        if len(self.current_images) == 0:
            messagebox.showerror("Error", "Please load a dataset first")
            return
        
        self.status_var.set("Evaluating...")
        self.evaluate_btn.configure(state=tk.DISABLED)
        self.root.update()
        
        def evaluate_in_thread():
            try:
                # Run predictions in batches
                batch_size = 64
                all_preds = []
                all_probs = []
                
                for i in range(0, len(self.current_images), batch_size):
                    batch = self.current_images[i:i+batch_size]
                    preds, probs = self.classifier.predict_batch(batch)
                    all_preds.extend(preds)
                    all_probs.extend(probs)
                    
                    progress = min(100, int(100 * (i + batch_size) / len(self.current_images)))
                    self.root.after(0, lambda p=progress: self.status_var.set(f"Evaluating... {p}%"))
                
                preds = np.array(all_preds)
                self.current_predictions = preds
                
                # Calculate metrics
                accuracy = 100 * np.mean(preds == self.current_labels)
                
                # Get unique labels that appear in either true or predicted
                all_labels = np.unique(np.concatenate([self.current_labels, preds]))
                
                # Classification report
                report = classification_report(
                    self.current_labels, preds,
                    labels=all_labels,
                    target_names=[CLASS_NAMES.get(i, str(i)) for i in all_labels],
                    digits=3,
                    zero_division=0
                )
                
                # Calculate macro F1
                macro_f1 = f1_score(self.current_labels, preds, average='macro', zero_division=0)
                
                # Update UI on main thread
                def update_ui():
                    self.metrics_text.configure(state='normal')
                    self.metrics_text.delete('1.0', tk.END)
                    self.metrics_text.insert('1.0', "="*60 + "\n")
                    self.metrics_text.insert(tk.END, "           EVALUATION RESULTS\n")
                    self.metrics_text.insert(tk.END, "="*60 + "\n\n")
                    self.metrics_text.insert(tk.END, f"Total Samples:    {len(self.current_labels)}\n")
                    self.metrics_text.insert(tk.END, f"Overall Accuracy: {accuracy:.2f}%\n")
                    self.metrics_text.insert(tk.END, f"Macro F1 Score:   {macro_f1:.4f}\n\n")
                    self.metrics_text.insert(tk.END, "-"*60 + "\n")
                    self.metrics_text.insert(tk.END, "Classification Report:\n")
                    self.metrics_text.insert(tk.END, "-"*60 + "\n\n")
                    self.metrics_text.insert(tk.END, report)
                    self.metrics_text.configure(state='disabled')
                    
                    self._plot_confusion_matrix(self.current_labels, preds)
                    self.status_var.set(f"Evaluation complete - Accuracy: {accuracy:.2f}%")
                    self.evaluate_btn.configure(state=tk.NORMAL)
                
                self.root.after(0, update_ui)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Evaluation failed:\n{e}"))
                self.root.after(0, lambda: self.status_var.set("Evaluation failed"))
                self.root.after(0, lambda: self.evaluate_btn.configure(state=tk.NORMAL))
        
        threading.Thread(target=evaluate_in_thread, daemon=True).start()
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        self.cm_fig.clear()
        ax = self.cm_fig.add_subplot(111)
        
        labels = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Normalize for color display
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
        
        # Labels
        class_labels = [CLASS_NAMES.get(i, str(i)) for i in labels]
        # Truncate labels for display
        class_labels_short = [l[:12] + "..." if len(l) > 15 else l for l in class_labels]
        
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(class_labels_short, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(class_labels_short, fontsize=7)
        
        # Add count values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text_color = "white" if cm_normalized[i, j] > 0.5 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", 
                       color=text_color, fontsize=7)
        
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('True', fontsize=9)
        
        # Calculate accuracy for title
        accuracy = 100 * np.trace(cm) / np.sum(cm)
        ax.set_title(f'Confusion Matrix (Acc: {accuracy:.1f}%)', fontsize=10)
        
        self.cm_fig.tight_layout()
        self.cm_canvas.draw()
    
    def _export_report(self):
        """Export evaluation report to file"""
        if len(self.current_predictions) == 0:
            messagebox.showerror("Error", "Please run evaluation first")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".txt",
            initialfile=f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        try:
            with open(filepath, 'w') as f:
                f.write("="*70 + "\n")
                f.write("         AttentionNet Classification Report\n")
                f.write("="*70 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {MODEL_PATH}\n")
                f.write("="*70 + "\n\n")
                f.write(self.metrics_text.get('1.0', tk.END))
            
            messagebox.showinfo("Success", f"Report saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report:\n{e}")
    
    # =========================================================================
    # RESULTS DISPLAY
    # =========================================================================
    def _display_results(self, prefix, images, flow_info):
        """Display classification results for captured/loaded flows"""
        if not images:
            return
        
        # Store for later reference
        results = {
            'images': images,
            'flow_info': flow_info
        }
        
        # Get predictions
        try:
            preds, probs = self.classifier.predict_batch(images)
            results['preds'] = preds
            results['probs'] = probs
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed:\n{e}")
            return
        
        setattr(self, f'{prefix}_results', results)
        
        # Update summary stats
        stats_var = getattr(self, f'{prefix}_stats_var')
        vpn_count = sum(1 for p in preds if p >= 6)
        nonvpn_count = len(preds) - vpn_count
        stats_var.set(f"Total: {len(images)} flows | Non-VPN: {nonvpn_count} | VPN: {vpn_count}")
        
        # Update flow images display
        fig = getattr(self, f'{prefix}_fig')
        fig.clear()
        
        n_display = min(12, len(images))
        cols = 4
        rows = (n_display + cols - 1) // cols
        
        for i in range(n_display):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(images[i], cmap='viridis', vmin=0, vmax=255)
            pred_name = CLASS_NAMES[preds[i]]
            # Truncate name for display
            if len(pred_name) > 15:
                pred_name = pred_name[:12] + "..."
            ax.set_title(f"Flow {i+1}\n{pred_name}", fontsize=8)
            ax.axis('off')
        
        fig.suptitle("Processed Flow Images", fontsize=10, fontweight='bold')
        fig.tight_layout()
        canvas = getattr(self, f'{prefix}_canvas')
        canvas.draw()
        
        # Update results tree
        tree = getattr(self, f'{prefix}_tree')
        tree.delete(*tree.get_children())
        
        for i, (info, pred, prob) in enumerate(zip(flow_info, preds, probs)):
            confidence = prob[pred] * 100
            is_vpn = "✓ VPN" if pred >= 6 else "✗ Non-VPN"
            tree.insert('', 'end', values=(
                f"Flow {i+1} ({info['proto']}, {info['bytes']}B)",
                CLASS_NAMES[pred],
                f"{confidence:.1f}%",
                is_vpn
            ), tags=(str(i),))
    
    def _on_flow_select(self, prefix):
        """Handle flow selection in treeview"""
        tree = getattr(self, f'{prefix}_tree')
        selection = tree.selection()
        if not selection:
            return
        
        item = selection[0]
        tags = tree.item(item, 'tags')
        if not tags:
            return
        
        idx = int(tags[0])
        results = getattr(self, f'{prefix}_results', None)
        if results is None or idx >= len(results.get('probs', [])):
            return
        
        probs = results['probs'][idx]
        
        # Update probability chart
        prob_fig = getattr(self, f'{prefix}_prob_fig')
        prob_fig.clear()
        ax = prob_fig.add_subplot(111)
        
        y_pos = range(12)
        colors = [CLASS_COLORS[i] for i in range(12)]
        bars = ax.barh(y_pos, probs, color=colors)
        
        # Highlight predicted class
        pred_idx = np.argmax(probs)
        bars[pred_idx].set_edgecolor('black')
        bars[pred_idx].set_linewidth(2)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([CLASS_NAMES[i][:15] for i in range(12)], fontsize=8)
        ax.set_xlabel('Probability')
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'Flow {idx+1} - Predicted: {CLASS_NAMES[pred_idx]}', fontsize=10)
        
        prob_fig.tight_layout()
        prob_canvas = getattr(self, f'{prefix}_prob_canvas')
        prob_canvas.draw()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    # Check for required packages
    missing_packages = []
    
    try:
        import torch
    except ImportError:
        missing_packages.append("torch (PyTorch)")
    
    try:
        import sklearn
    except ImportError:
        missing_packages.append("scikit-learn")
    
    try:
        import matplotlib
    except ImportError:
        missing_packages.append("matplotlib")
    
    if missing_packages:
        print("="*60)
        print("Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print("  pip install torch scikit-learn matplotlib numpy")
        print("="*60)
        sys.exit(1)
    
    if not SCAPY_AVAILABLE:
        print("="*60)
        print("Warning: Scapy not available")
        print("Live capture and PCAP processing will be disabled.")
        print("Install with: pip install scapy")
        print("="*60)
    
    # Start application
    root = tk.Tk()
    
    # Set app icon (if available)
    try:
        # You can add an icon file later if desired
        pass
    except:
        pass
    
    app = AttentionNetDemo(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()

