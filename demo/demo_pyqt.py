#!/usr/bin/env python3
"""
AttentionNet Traffic Classifier - PyQt6 Demo
=============================================
A desktop GUI application for demonstrating the traffic classifier.

Features:
1. Live traffic capture and classification
2. PCAP file processing
3. Pre-processed dataset evaluation
4. Flow image visualization

Run with:
    python demo_pyqt.py

Note: Live capture requires sudo/admin privileges.
"""

import os
import sys
import tempfile
import threading
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QLineEdit, QSpinBox, QComboBox,
    QFileDialog, QTextEdit, QTableWidget, QTableWidgetItem, QGroupBox,
    QProgressBar, QSplitter, QMessageBox, QHeaderView, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage

# Matplotlib for embedding plots
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# =============================================================================
# PATH SETUP - Import from existing project code
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Import model from existing code
from src.model.hybrid_tiny import TrafficCNN_TinyTransformer

# Try to import preprocessing functions
try:
    from src.preprocess.preprocess_memory_safe import (
        _flow_key, _anonymized_packet_bytes, _is_corrupted, 
        _is_retransmission, _bytes_to_image, MAX_LEN, ROWS, COLS
    )
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False
    MAX_LEN = 784
    ROWS, COLS = 28, 28

# Try to import scapy
try:
    from scapy.all import IP, TCP, UDP, PcapReader, sniff
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = os.path.join(PROJECT_ROOT, "model_output/memory_safe/hocaya_gosterilcek/p2p_change/2layer_cnn_hybrid_3fc/best_model.pth")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data/final/memory_safe/own_nonVPN_p2p_2/ratio_change")

CLASS_NAMES = {
    0: "Chat (NonVPN)", 1: "Email (NonVPN)", 2: "File (NonVPN)", 
    3: "P2P (NonVPN)", 4: "Streaming (NonVPN)", 5: "VoIP (NonVPN)", 
    6: "Chat (VPN)", 7: "Email (VPN)", 8: "File (VPN)", 
    9: "P2P (VPN)", 10: "Streaming (VPN)", 11: "VoIP (VPN)"
}

CLASS_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f1c40f", "#1abc9c",
    "#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#f39c12", "#16a085"
]

# =============================================================================
# HELPER FUNCTIONS (fallback if imports fail)
# =============================================================================
if not PREPROCESS_AVAILABLE:
    def _flow_key(pkt):
        ip = pkt[IP]
        proto = ip.proto
        sport = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport if UDP in pkt else 0
        dport = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport if UDP in pkt else 0
        sorted_ips = tuple(sorted((ip.src, ip.dst)))
        sorted_ports = tuple(sorted((sport, dport)))
        return (sorted_ips[0], sorted_ips[1], sorted_ports[0], sorted_ports[1], proto)
    
    def _anonymized_packet_bytes(pkt):
        pkt_copy = pkt.copy()
        pkt_copy[IP].src = "0.0.0.0"
        pkt_copy[IP].dst = "0.0.0.0"
        if "Ether" in pkt_copy:
            pkt_copy["Ether"].src = "00:00:00:00:00:00"
            pkt_copy["Ether"].dst = "00:00:00:00:00:00"
        return bytes(pkt_copy)
    
    def _is_corrupted(pkt):
        try:
            if IP not in pkt:
                return True
            if len(bytes(pkt)) == 0:
                return True
            return False
        except:
            return True
    
    def _is_retransmission(pkt, seq_tracker):
        if TCP not in pkt:
            return False
        tcp = pkt[TCP]
        ip = pkt[IP]
        conn_key = (ip.src, ip.dst, tcp.sport, tcp.dport)
        seq = tcp.seq
        if seq in seq_tracker[conn_key]:
            return True
        seq_tracker[conn_key].add(seq)
        return False
    
    def _bytes_to_image(buffer):
        if len(buffer) == 0:
            return None
        arr = np.frombuffer(buffer if isinstance(buffer, bytes) else bytes(buffer), dtype=np.uint8)
        if len(arr) >= MAX_LEN:
            arr = arr[:MAX_LEN]
        else:
            arr = np.pad(arr, (0, MAX_LEN - len(arr)))
        return arr.reshape(ROWS, COLS)


# =============================================================================
# CLASSIFIER CLASS
# =============================================================================
class TrafficClassifier:
    """Wrapper for the traffic classification model."""
    
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
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = TrafficCNN_TinyTransformer(num_classes=12).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.loaded = True
        return True
    
    def predict_batch(self, images):
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
# WORKER THREADS
# =============================================================================
class CaptureWorker(QThread):
    """Background thread for packet capture."""
    finished = pyqtSignal(list, list)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, interface, duration, max_flows):
        super().__init__()
        self.interface = interface
        self.duration = duration
        self.max_flows = max_flows
    
    def run(self):
        if not SCAPY_AVAILABLE:
            self.error.emit("Scapy not available. Install with: pip install scapy")
            return
        
        flow_buffers = defaultdict(bytearray)
        seq_tracker = defaultdict(set)
        
        def packet_callback(pkt):
            if IP not in pkt or (TCP not in pkt and UDP not in pkt):
                return
            if _is_corrupted(pkt):
                return
            if _is_retransmission(pkt, seq_tracker):
                return
            
            key = _flow_key(pkt)
            buffer = flow_buffers[key]
            if len(buffer) >= MAX_LEN:
                return
            
            pkt_bytes = _anonymized_packet_bytes(pkt)
            remaining = MAX_LEN - len(buffer)
            buffer.extend(pkt_bytes[:remaining])
        
        try:
            self.progress.emit(f"Capturing on {self.interface}...")
            sniff(iface=self.interface, prn=packet_callback, 
                  timeout=self.duration, store=False)
        except PermissionError:
            self.error.emit("Permission denied. Run with sudo for live capture.")
            return
        except Exception as e:
            self.error.emit(f"Capture error: {str(e)}")
            return
        
        # Convert to images
        images = []
        flow_info = []
        
        for key, buffer in flow_buffers.items():
            if len(images) >= self.max_flows:
                break
            img = _bytes_to_image(buffer)
            if img is not None:
                images.append(img)
                flow_info.append({
                    'src': f"{key[0]}:{key[2]}",
                    'dst': f"{key[1]}:{key[3]}",
                    'proto': 'TCP' if key[4] == 6 else 'UDP' if key[4] == 17 else str(key[4]),
                    'bytes': len(buffer)
                })
        
        self.finished.emit(images, flow_info)


class PcapWorker(QThread):
    """Background thread for PCAP processing."""
    finished = pyqtSignal(list, list)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, pcap_path, max_flows):
        super().__init__()
        self.pcap_path = pcap_path
        self.max_flows = max_flows
    
    def run(self):
        if not SCAPY_AVAILABLE:
            self.error.emit("Scapy not available")
            return
        
        flow_buffers = defaultdict(bytearray)
        seq_tracker = defaultdict(set)
        
        try:
            self.progress.emit("Reading PCAP file...")
            with PcapReader(self.pcap_path) as packets:
                for pkt in packets:
                    if IP not in pkt or (TCP not in pkt and UDP not in pkt):
                        continue
                    if _is_corrupted(pkt):
                        continue
                    if _is_retransmission(pkt, seq_tracker):
                        continue
                    
                    key = _flow_key(pkt)
                    buffer = flow_buffers[key]
                    if len(buffer) >= MAX_LEN:
                        continue
                    
                    pkt_bytes = _anonymized_packet_bytes(pkt)
                    remaining = MAX_LEN - len(buffer)
                    buffer.extend(pkt_bytes[:remaining])
        except Exception as e:
            self.error.emit(f"Error reading PCAP: {str(e)}")
            return
        
        # Convert to images
        images = []
        flow_info = []
        
        for key, buffer in flow_buffers.items():
            if len(images) >= self.max_flows:
                break
            img = _bytes_to_image(buffer)
            if img is not None:
                images.append(img)
                flow_info.append({
                    'src': f"{key[0]}:{key[2]}",
                    'dst': f"{key[1]}:{key[3]}",
                    'proto': 'TCP' if key[4] == 6 else 'UDP' if key[4] == 17 else str(key[4]),
                    'bytes': len(buffer)
                })
        
        self.finished.emit(images, flow_info)


class EvalWorker(QThread):
    """Background thread for dataset evaluation."""
    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, classifier, images, labels):
        super().__init__()
        self.classifier = classifier
        self.images = images
        self.labels = labels
    
    def run(self):
        batch_size = 64
        all_preds = []
        all_probs = []
        
        for i in range(0, len(self.images), batch_size):
            batch = self.images[i:i+batch_size]
            preds, probs = self.classifier.predict_batch(batch)
            all_preds.extend(preds)
            all_probs.extend(probs)
            
            progress = int(100 * min(i + batch_size, len(self.images)) / len(self.images))
            self.progress.emit(progress)
        
        self.finished.emit(
            np.array(all_preds),
            np.array(all_probs),
            self.labels
        )


# =============================================================================
# MATPLOTLIB CANVAS WIDGET
# =============================================================================
class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in PyQt."""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)


# =============================================================================
# MAIN APPLICATION WINDOW
# =============================================================================
class AttentionNetDemo(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AttentionNet Traffic Classifier")
        self.setMinimumSize(1200, 800)
        
        # Initialize classifier
        self.classifier = TrafficClassifier()
        
        # Results storage
        self.current_images = []
        self.current_labels = []
        self.current_preds = []
        self.current_probs = []
        
        # Setup UI
        self._setup_ui()
        self._load_model()
    
    def _setup_ui(self):
        """Create the user interface."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Title
        title = QLabel("🔍 AttentionNet Traffic Classifier")
        title.setFont(QFont("Helvetica", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Status bar
        self.status_label = QLabel("Loading model...")
        self.status_label.setStyleSheet("color: gray; padding: 5px;")
        layout.addWidget(self.status_label)
        
        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(self._create_capture_tab(), "🔴 Live Capture")
        tabs.addTab(self._create_pcap_tab(), "📁 Process PCAP")
        tabs.addTab(self._create_dataset_tab(), "📊 Evaluate Dataset")
        tabs.addTab(self._create_about_tab(), "ℹ️ About")
        layout.addWidget(tabs)
    
    def _create_capture_tab(self):
        """Create the live capture tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls group
        controls = QGroupBox("Capture Settings")
        controls_layout = QHBoxLayout(controls)
        
        # Mode selection
        controls_layout.addWidget(QLabel("Mode:"))
        self.capture_mode = QComboBox()
        self.capture_mode.addItems(["Local Traffic (en0)", "Firewall Mode (bridge100)"])
        self.capture_mode.currentIndexChanged.connect(self._on_capture_mode_changed)
        controls_layout.addWidget(self.capture_mode)
        
        # Interface
        controls_layout.addWidget(QLabel("Interface:"))
        self.interface_input = QLineEdit("en0")
        self.interface_input.setMaximumWidth(100)
        controls_layout.addWidget(self.interface_input)
        
        # Duration
        controls_layout.addWidget(QLabel("Duration (s):"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(5, 60)
        self.duration_spin.setValue(10)
        controls_layout.addWidget(self.duration_spin)
        
        # Max flows
        controls_layout.addWidget(QLabel("Max Flows:"))
        self.capture_max_flows = QSpinBox()
        self.capture_max_flows.setRange(5, 100)
        self.capture_max_flows.setValue(20)
        controls_layout.addWidget(self.capture_max_flows)
        
        controls_layout.addStretch()
        
        # Capture button
        self.capture_btn = QPushButton("🔴 Start Capture")
        self.capture_btn.clicked.connect(self._start_capture)
        self.capture_btn.setStyleSheet("background-color: #e74c3c; color: white; padding: 10px;")
        controls_layout.addWidget(self.capture_btn)
        
        layout.addWidget(controls)
        
        # Note about permissions
        note = QLabel("⚠️ Live capture requires sudo/admin privileges")
        note.setStyleSheet("color: orange; padding: 5px;")
        layout.addWidget(note)
        
        # Results area
        results_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Flow images
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("Flow Visualizations"))
        self.capture_canvas = MplCanvas(width=6, height=5)
        left_layout.addWidget(self.capture_canvas)
        results_splitter.addWidget(left_widget)
        
        # Right: Results table
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("Classification Results"))
        
        self.capture_table = QTableWidget()
        self.capture_table.setColumnCount(5)
        self.capture_table.setHorizontalHeaderLabels(["Flow", "Protocol", "Prediction", "Confidence", "VPN"])
        self.capture_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.capture_table.itemSelectionChanged.connect(lambda: self._on_flow_selected("capture"))
        right_layout.addWidget(self.capture_table)
        
        # Probability chart
        self.capture_prob_canvas = MplCanvas(width=6, height=3)
        right_layout.addWidget(self.capture_prob_canvas)
        
        results_splitter.addWidget(right_widget)
        layout.addWidget(results_splitter)
        
        return tab
    
    def _create_pcap_tab(self):
        """Create the PCAP processing tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls = QGroupBox("PCAP File Selection")
        controls_layout = QHBoxLayout(controls)
        
        self.pcap_path_label = QLabel("No file selected")
        self.pcap_path_label.setStyleSheet("padding: 5px; background: #f0f0f0; border-radius: 3px;")
        controls_layout.addWidget(self.pcap_path_label, stretch=1)
        
        browse_btn = QPushButton("📂 Browse...")
        browse_btn.clicked.connect(self._browse_pcap)
        controls_layout.addWidget(browse_btn)
        
        controls_layout.addWidget(QLabel("Max Flows:"))
        self.pcap_max_flows = QSpinBox()
        self.pcap_max_flows.setRange(5, 100)
        self.pcap_max_flows.setValue(50)
        controls_layout.addWidget(self.pcap_max_flows)
        
        self.process_btn = QPushButton("🔄 Process & Classify")
        self.process_btn.clicked.connect(self._process_pcap)
        self.process_btn.setStyleSheet("background-color: #3498db; color: white; padding: 10px;")
        controls_layout.addWidget(self.process_btn)
        
        layout.addWidget(controls)
        
        # Results area (similar to capture tab)
        results_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Flow images
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("Flow Visualizations"))
        self.pcap_canvas = MplCanvas(width=6, height=5)
        left_layout.addWidget(self.pcap_canvas)
        results_splitter.addWidget(left_widget)
        
        # Right: Results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("Classification Results"))
        
        self.pcap_table = QTableWidget()
        self.pcap_table.setColumnCount(5)
        self.pcap_table.setHorizontalHeaderLabels(["Flow", "Protocol", "Prediction", "Confidence", "VPN"])
        self.pcap_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.pcap_table.itemSelectionChanged.connect(lambda: self._on_flow_selected("pcap"))
        right_layout.addWidget(self.pcap_table)
        
        self.pcap_prob_canvas = MplCanvas(width=6, height=3)
        right_layout.addWidget(self.pcap_prob_canvas)
        
        results_splitter.addWidget(right_widget)
        layout.addWidget(results_splitter)
        
        return tab
    
    def _create_dataset_tab(self):
        """Create the dataset evaluation tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls = QGroupBox("Dataset Selection")
        controls_layout = QHBoxLayout(controls)
        
        load_test_btn = QPushButton("📥 Load Test Set")
        load_test_btn.clicked.connect(self._load_test_set)
        controls_layout.addWidget(load_test_btn)
        
        load_train_btn = QPushButton("📥 Load Train Set")
        load_train_btn.clicked.connect(self._load_train_set)
        controls_layout.addWidget(load_train_btn)
        
        load_custom_btn = QPushButton("📂 Load Custom .npy...")
        load_custom_btn.clicked.connect(self._browse_npy)
        controls_layout.addWidget(load_custom_btn)
        
        controls_layout.addStretch()
        
        self.eval_btn = QPushButton("📈 Run Evaluation")
        self.eval_btn.clicked.connect(self._evaluate_dataset)
        self.eval_btn.setStyleSheet("background-color: #2ecc71; color: white; padding: 10px;")
        controls_layout.addWidget(self.eval_btn)
        
        export_btn = QPushButton("💾 Export Report")
        export_btn.clicked.connect(self._export_report)
        controls_layout.addWidget(export_btn)
        
        layout.addWidget(controls)
        
        # Dataset info
        self.dataset_info = QLabel("No dataset loaded")
        self.dataset_info.setStyleSheet("padding: 5px; color: gray;")
        layout.addWidget(self.dataset_info)
        
        # Progress bar
        self.eval_progress = QProgressBar()
        self.eval_progress.setVisible(False)
        layout.addWidget(self.eval_progress)
        
        # Results splitter
        results_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Sample images and confusion matrix
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("Sample Images & Confusion Matrix"))
        self.dataset_canvas = MplCanvas(width=6, height=8)
        left_layout.addWidget(self.dataset_canvas)
        results_splitter.addWidget(left_widget)
        
        # Right: Classification report
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("Classification Report"))
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont("Courier", 10))
        right_layout.addWidget(self.report_text)
        results_splitter.addWidget(right_widget)
        
        layout.addWidget(results_splitter)
        
        return tab
    
    def _create_about_tab(self):
        """Create the about tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        about_text = QTextEdit()
        about_text.setReadOnly(True)
        about_text.setFont(QFont("Courier", 11))
        about_text.setHtml("""
        <h2>AttentionNet Traffic Classifier</h2>
        <p>A hybrid CNN-Transformer model for encrypted network traffic classification.</p>
        
        <h3>Model Architecture</h3>
        <ul>
            <li><b>Input:</b> 28×28 grayscale images (784 bytes from packet flows)</li>
            <li><b>CNN Backbone:</b> 2 convolutional layers (64 → 128 channels)</li>
            <li><b>Transformer:</b> 2 encoder layers, 4 attention heads</li>
            <li><b>Classifier:</b> 3 fully connected layers</li>
        </ul>
        
        <h3>Classification Classes (12 total)</h3>
        <table border="1" cellpadding="5">
            <tr><th>Non-VPN</th><th>VPN</th></tr>
            <tr><td>Chat (NonVPN)</td><td>Chat (VPN)</td></tr>
            <tr><td>Email (NonVPN)</td><td>Email (VPN)</td></tr>
            <tr><td>File (NonVPN)</td><td>File (VPN)</td></tr>
            <tr><td>P2P (NonVPN)</td><td>P2P (VPN)</td></tr>
            <tr><td>Streaming (NonVPN)</td><td>Streaming (VPN)</td></tr>
            <tr><td>VoIP (NonVPN)</td><td>VoIP (VPN)</td></tr>
        </table>
        
        <h3>Processing Pipeline</h3>
        <ol>
            <li>Capture/Load network packets</li>
            <li>Group packets into flows (by IP+port pairs)</li>
            <li>Filter retransmissions</li>
            <li>Anonymize (zero out IPs/MACs)</li>
            <li>Extract first 784 bytes</li>
            <li>Reshape to 28×28 image</li>
            <li>Classify with CNN-Transformer</li>
        </ol>
        
        <h3>Performance</h3>
        <ul>
            <li><b>Test Accuracy:</b> ~86%</li>
            <li><b>Macro F1 Score:</b> ~0.88</li>
        </ul>
        """)
        layout.addWidget(about_text)
        
        return tab
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def _load_model(self):
        """Load the classification model."""
        try:
            self.classifier.load_model()
            self.status_label.setText(f"✓ Model loaded ({self.classifier.device})")
            self.status_label.setStyleSheet("color: green; padding: 5px;")
        except Exception as e:
            self.status_label.setText(f"✗ Model error: {e}")
            self.status_label.setStyleSheet("color: red; padding: 5px;")
            QMessageBox.critical(self, "Model Error", f"Failed to load model:\n{e}")
    
    def _on_capture_mode_changed(self, index):
        """Handle capture mode change."""
        if index == 0:  # Local traffic
            self.interface_input.setText("en0")
        else:  # Firewall mode
            self.interface_input.setText("bridge100")
    
    def _start_capture(self):
        """Start live packet capture."""
        interface = self.interface_input.text()
        duration = self.duration_spin.value()
        max_flows = self.capture_max_flows.value()
        
        self.capture_btn.setEnabled(False)
        self.status_label.setText(f"Capturing on {interface}...")
        
        self.capture_worker = CaptureWorker(interface, duration, max_flows)
        self.capture_worker.finished.connect(self._on_capture_finished)
        self.capture_worker.error.connect(self._on_capture_error)
        self.capture_worker.start()
    
    def _on_capture_finished(self, images, flow_info):
        """Handle capture completion."""
        self.capture_btn.setEnabled(True)
        
        if not images:
            self.status_label.setText("No flows captured")
            QMessageBox.warning(self, "No Data", "No flows were captured. Try a longer duration.")
            return
        
        # Classify
        preds, probs = self.classifier.predict_batch(images)
        
        # Store results
        self.capture_images = images
        self.capture_flow_info = flow_info
        self.capture_preds = preds
        self.capture_probs = probs
        
        # Update UI
        self._display_flow_results(images, flow_info, preds, probs, 
                                   self.capture_canvas, self.capture_table)
        self.status_label.setText(f"✓ Captured {len(images)} flows")
    
    def _on_capture_error(self, error_msg):
        """Handle capture error."""
        self.capture_btn.setEnabled(True)
        self.status_label.setText("Capture failed")
        QMessageBox.critical(self, "Capture Error", error_msg)
    
    def _browse_pcap(self):
        """Browse for PCAP file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PCAP File", "",
            "PCAP Files (*.pcap *.pcapng);;All Files (*)"
        )
        if path:
            self.pcap_path_label.setText(path)
    
    def _process_pcap(self):
        """Process selected PCAP file."""
        path = self.pcap_path_label.text()
        if not os.path.exists(path):
            QMessageBox.warning(self, "No File", "Please select a PCAP file first")
            return
        
        max_flows = self.pcap_max_flows.value()
        
        self.process_btn.setEnabled(False)
        self.status_label.setText("Processing PCAP...")
        
        self.pcap_worker = PcapWorker(path, max_flows)
        self.pcap_worker.finished.connect(self._on_pcap_finished)
        self.pcap_worker.error.connect(self._on_pcap_error)
        self.pcap_worker.start()
    
    def _on_pcap_finished(self, images, flow_info):
        """Handle PCAP processing completion."""
        self.process_btn.setEnabled(True)
        
        if not images:
            self.status_label.setText("No flows found")
            QMessageBox.warning(self, "No Data", "No valid flows found in the PCAP file.")
            return
        
        # Classify
        preds, probs = self.classifier.predict_batch(images)
        
        # Store results
        self.pcap_images = images
        self.pcap_flow_info = flow_info
        self.pcap_preds = preds
        self.pcap_probs = probs
        
        # Update UI
        self._display_flow_results(images, flow_info, preds, probs,
                                   self.pcap_canvas, self.pcap_table)
        self.status_label.setText(f"✓ Processed {len(images)} flows")
    
    def _on_pcap_error(self, error_msg):
        """Handle PCAP processing error."""
        self.process_btn.setEnabled(True)
        self.status_label.setText("Processing failed")
        QMessageBox.critical(self, "Processing Error", error_msg)
    
    def _load_test_set(self):
        """Load test dataset."""
        data_path = os.path.join(TEST_DATA_DIR, "test_data_memory_safe_own_nonVPN_p2p.npy")
        labels_path = os.path.join(TEST_DATA_DIR, "test_labels_memory_safe_own_nonVPN_p2p.npy")
        self._load_dataset(data_path, labels_path, "Test")
    
    def _load_train_set(self):
        """Load training dataset."""
        data_path = os.path.join(TEST_DATA_DIR, "train_data_memory_safe_own_nonVPN_p2p.npy")
        labels_path = os.path.join(TEST_DATA_DIR, "train_labels_memory_safe_own_nonVPN_p2p.npy")
        self._load_dataset(data_path, labels_path, "Train")
    
    def _browse_npy(self):
        """Browse for custom .npy files."""
        data_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "",
            "NumPy Files (*data*.npy);;All Files (*)"
        )
        if not data_path:
            return
        
        # Try to find labels
        labels_path = data_path.replace("data", "labels")
        if not os.path.exists(labels_path):
            labels_path, _ = QFileDialog.getOpenFileName(
                self, "Select Labels File", os.path.dirname(data_path),
                "NumPy Files (*labels*.npy);;All Files (*)"
            )
        
        if labels_path and os.path.exists(labels_path):
            self._load_dataset(data_path, labels_path, "Custom")
    
    def _load_dataset(self, data_path, labels_path, name):
        """Load dataset from .npy files."""
        if not os.path.exists(data_path) or not os.path.exists(labels_path):
            QMessageBox.critical(self, "File Not Found", f"Could not find dataset files")
            return
        
        try:
            self.current_images = np.load(data_path)
            self.current_labels = np.load(labels_path)
            self.current_preds = []
            
            unique, counts = np.unique(self.current_labels, return_counts=True)
            self.dataset_info.setText(
                f"{name} Set: {len(self.current_images)} samples, {len(unique)} classes"
            )
            self.status_label.setText(f"✓ Loaded {name.lower()} set")
            
            # Show sample images
            self._display_sample_images()
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load dataset:\n{e}")
    
    def _evaluate_dataset(self):
        """Evaluate loaded dataset."""
        if len(self.current_images) == 0:
            QMessageBox.warning(self, "No Data", "Please load a dataset first")
            return
        
        self.eval_btn.setEnabled(False)
        self.eval_progress.setVisible(True)
        self.eval_progress.setValue(0)
        self.status_label.setText("Evaluating...")
        
        self.eval_worker = EvalWorker(
            self.classifier, self.current_images, self.current_labels
        )
        self.eval_worker.finished.connect(self._on_eval_finished)
        self.eval_worker.progress.connect(self.eval_progress.setValue)
        self.eval_worker.start()
    
    def _on_eval_finished(self, preds, probs, labels):
        """Handle evaluation completion."""
        self.eval_btn.setEnabled(True)
        self.eval_progress.setVisible(False)
        
        self.current_preds = preds
        
        # Calculate metrics
        accuracy = 100 * np.mean(preds == labels)
        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        
        # Generate report
        all_labels = np.unique(np.concatenate([labels, preds]))
        report = classification_report(
            labels, preds,
            labels=all_labels,
            target_names=[CLASS_NAMES.get(i, str(i)) for i in all_labels],
            digits=3,
            zero_division=0
        )
        
        # Update report text
        self.report_text.setPlainText(
            f"{'='*60}\n"
            f"EVALUATION RESULTS\n"
            f"{'='*60}\n\n"
            f"Total Samples:    {len(labels)}\n"
            f"Overall Accuracy: {accuracy:.2f}%\n"
            f"Macro F1 Score:   {macro_f1:.4f}\n\n"
            f"{'='*60}\n"
            f"Classification Report:\n"
            f"{'='*60}\n\n"
            f"{report}"
        )
        
        # Display confusion matrix
        self._display_confusion_matrix(labels, preds)
        
        self.status_label.setText(f"✓ Evaluation complete - Accuracy: {accuracy:.2f}%")
    
    def _export_report(self):
        """Export classification report."""
        if len(self.current_preds) == 0:
            QMessageBox.warning(self, "No Results", "Please run evaluation first")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", 
            f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if path:
            with open(path, 'w') as f:
                f.write(self.report_text.toPlainText())
            QMessageBox.information(self, "Saved", f"Report saved to:\n{path}")
    
    # =========================================================================
    # VISUALIZATION HELPERS
    # =========================================================================
    
    def _display_flow_results(self, images, flow_info, preds, probs, canvas, table):
        """Display flow classification results."""
        # Clear and populate table
        table.setRowCount(len(images))
        
        for i, (info, pred, prob) in enumerate(zip(flow_info, preds, probs)):
            conf = prob[pred] * 100
            is_vpn = "✓ VPN" if pred >= 6 else "✗ Non-VPN"
            
            table.setItem(i, 0, QTableWidgetItem(f"Flow {i+1}"))
            table.setItem(i, 1, QTableWidgetItem(info['proto']))
            table.setItem(i, 2, QTableWidgetItem(CLASS_NAMES[pred]))
            table.setItem(i, 3, QTableWidgetItem(f"{conf:.1f}%"))
            table.setItem(i, 4, QTableWidgetItem(is_vpn))
        
        # Display flow images
        canvas.fig.clear()
        n_display = min(12, len(images))
        cols = 4
        rows = (n_display + cols - 1) // cols
        
        for i in range(n_display):
            ax = canvas.fig.add_subplot(rows, cols, i + 1)
            ax.imshow(images[i], cmap='viridis', vmin=0, vmax=255)
            ax.axis('off')
            name = CLASS_NAMES[preds[i]]
            if len(name) > 12:
                name = name[:10] + "..."
            ax.set_title(f"Flow {i+1}\n{name}", fontsize=8)
        
        canvas.fig.tight_layout()
        canvas.draw()
    
    def _on_flow_selected(self, mode):
        """Handle flow selection in table."""
        if mode == "capture":
            table = self.capture_table
            probs = getattr(self, 'capture_probs', None)
            preds = getattr(self, 'capture_preds', None)
            canvas = self.capture_prob_canvas
        else:
            table = self.pcap_table
            probs = getattr(self, 'pcap_probs', None)
            preds = getattr(self, 'pcap_preds', None)
            canvas = self.pcap_prob_canvas
        
        if probs is None or len(table.selectedItems()) == 0:
            return
        
        row = table.currentRow()
        if row < 0 or row >= len(probs):
            return
        
        # Draw probability chart
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        
        y_pos = range(12)
        bars = ax.barh(y_pos, probs[row], color=CLASS_COLORS)
        bars[preds[row]].set_edgecolor('black')
        bars[preds[row]].set_linewidth(2)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([CLASS_NAMES[i][:15] for i in range(12)], fontsize=8)
        ax.set_xlabel('Probability')
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'Flow {row+1} - {CLASS_NAMES[preds[row]]}', fontsize=10)
        
        canvas.fig.tight_layout()
        canvas.draw()
    
    def _display_sample_images(self):
        """Display sample images from dataset."""
        self.dataset_canvas.fig.clear()
        
        unique_labels = np.unique(self.current_labels)
        n_classes = min(6, len(unique_labels))  # Show max 6 classes
        n_samples = 3
        
        for i, label in enumerate(unique_labels[:n_classes]):
            class_images = self.current_images[self.current_labels == label]
            for j in range(min(n_samples, len(class_images))):
                ax = self.dataset_canvas.fig.add_subplot(n_classes, n_samples, i * n_samples + j + 1)
                ax.imshow(class_images[j], cmap='viridis', vmin=0, vmax=255)
                ax.axis('off')
                if j == 0:
                    name = CLASS_NAMES.get(label, str(label))
                    if len(name) > 12:
                        name = name[:10] + "..."
                    ax.set_title(name, fontsize=8)
        
        self.dataset_canvas.fig.suptitle("Sample Images by Class", fontsize=10)
        self.dataset_canvas.fig.tight_layout()
        self.dataset_canvas.draw()
    
    def _display_confusion_matrix(self, y_true, y_pred):
        """Display confusion matrix."""
        self.dataset_canvas.fig.clear()
        
        # Create confusion matrix
        labels = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        ax = self.dataset_canvas.fig.add_subplot(111)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        
        # Labels
        class_labels = [CLASS_NAMES.get(i, str(i))[:10] for i in labels]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(class_labels, fontsize=8)
        
        # Add values
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                       color=color, fontsize=7)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        accuracy = 100 * np.trace(cm) / np.sum(cm)
        ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.1f}%)')
        
        self.dataset_canvas.fig.colorbar(im, ax=ax)
        self.dataset_canvas.fig.tight_layout()
        self.dataset_canvas.draw()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = AttentionNetDemo()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()







