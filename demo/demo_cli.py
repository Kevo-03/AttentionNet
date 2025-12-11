#!/usr/bin/env python3
"""
AttentionNet Traffic Classifier - CLI Demo
===========================================
A command-line demonstration system for network traffic classification.
Use this if Tkinter is not available on your system.

Features:
1. Process PCAP files
2. Evaluate pre-processed test datasets
3. Visualize flow images (saves to files)
4. Interactive menu-based interface

Usage:
    python demo_cli.py
"""

import os
import sys
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - saves to files
import matplotlib.pyplot as plt

# Try to import scapy
try:
    from scapy.all import IP, TCP, UDP, PcapReader
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# Try to import rich for better terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "model_output/memory_safe/hocaya_gosterilcek/p2p_change/2layer_cnn_hybrid_3fc/best_model.pth")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data/final/memory_safe/own_nonVPN_p2p_2/ratio_change")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

MAX_LEN = 784
ROWS, COLS = 28, 28

CLASS_NAMES = {
    0: "Chat (NonVPN)", 1: "Email (NonVPN)", 2: "File (NonVPN)", 
    3: "P2P (NonVPN)", 4: "Streaming (NonVPN)", 5: "VoIP (NonVPN)", 
    6: "Chat (VPN)", 7: "Email (VPN)", 8: "File (VPN)", 
    9: "P2P (VPN)", 10: "Streaming (VPN)", 11: "VoIP (VPN)"
}

# =============================================================================
# UTILITIES
# =============================================================================
def print_header(text):
    """Print a styled header"""
    if RICH_AVAILABLE:
        console.print(Panel(text, style="bold cyan"))
    else:
        print("\n" + "="*60)
        print(f"  {text}")
        print("="*60)

def print_success(text):
    """Print success message"""
    if RICH_AVAILABLE:
        console.print(f"[green]✓[/green] {text}")
    else:
        print(f"✓ {text}")

def print_error(text):
    """Print error message"""
    if RICH_AVAILABLE:
        console.print(f"[red]✗[/red] {text}")
    else:
        print(f"✗ {text}")

def print_info(text):
    """Print info message"""
    if RICH_AVAILABLE:
        console.print(f"[blue]ℹ[/blue] {text}")
    else:
        print(f"ℹ {text}")

# =============================================================================
# MODEL
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
# CLASSIFIER
# =============================================================================
class TrafficClassifier:
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
# PACKET PROCESSING
# =============================================================================
def process_pcap(pcap_path, max_flows=50):
    """Process PCAP file and extract flow images"""
    if not SCAPY_AVAILABLE:
        raise RuntimeError("Scapy is required for PCAP processing. Install with: pip install scapy")
    
    flow_buffers = defaultdict(bytearray)
    seq_tracker = defaultdict(set)
    
    print_info(f"Processing: {pcap_path}")
    
    try:
        with PcapReader(pcap_path) as packets:
            packet_count = 0
            for pkt in packets:
                packet_count += 1
                if packet_count % 1000 == 0:
                    print(f"  Processed {packet_count} packets...", end='\r')
                
                if IP not in pkt or (TCP not in pkt and UDP not in pkt):
                    continue
                
                # Flow key
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
                        continue
                    seq_tracker[conn_key].add(seq)
                
                buffer = flow_buffers[key]
                if len(buffer) >= MAX_LEN:
                    continue
                
                # Anonymize and add bytes
                pkt_copy = pkt.copy()
                pkt_copy[IP].src = "0.0.0.0"
                pkt_copy[IP].dst = "0.0.0.0"
                if "Ether" in pkt_copy:
                    pkt_copy["Ether"].src = "00:00:00:00:00:00"
                    pkt_copy["Ether"].dst = "00:00:00:00:00:00"
                
                pkt_bytes = bytes(pkt_copy)
                remaining = MAX_LEN - len(buffer)
                buffer.extend(pkt_bytes[:remaining])
            
            print(f"  Processed {packet_count} packets total")
    except Exception as e:
        print_error(f"Error reading PCAP: {e}")
        return [], []
    
    # Convert to images
    images = []
    flow_info = []
    
    for key, buffer in flow_buffers.items():
        if max_flows and len(images) >= max_flows:
            break
        
        if len(buffer) == 0:
            continue
        
        arr = np.frombuffer(bytes(buffer), dtype=np.uint8)
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
    
    print_success(f"Extracted {len(images)} flows")
    return images, flow_info


# =============================================================================
# VISUALIZATION
# =============================================================================
def save_flow_visualizations(images, preds, probs, flow_info, output_path):
    """Save flow visualizations to file"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    n_flows = min(12, len(images))
    cols = 4
    rows = (n_flows + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = axes.flatten() if n_flows > 1 else [axes]
    
    for i in range(n_flows):
        ax = axes[i]
        ax.imshow(images[i], cmap='viridis', vmin=0, vmax=255)
        
        pred = preds[i]
        conf = probs[i][pred] * 100
        is_vpn = "VPN" if pred >= 6 else "NonVPN"
        
        ax.set_title(f"Flow {i+1}\n{CLASS_NAMES[pred]}\n{conf:.1f}% ({is_vpn})", fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_flows, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Flow Image Visualizations (28×28)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print_success(f"Saved visualization to: {output_path}")


def save_confusion_matrix(y_true, y_pred, output_path):
    """Save confusion matrix to file"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Normalize for coloring
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    
    # Labels
    class_labels = [CLASS_NAMES.get(i, str(i))[:15] for i in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_labels, fontsize=9)
    
    # Add values
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    
    accuracy = 100 * np.trace(cm) / np.sum(cm)
    ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Normalized frequency')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print_success(f"Saved confusion matrix to: {output_path}")


def save_sample_images(images, labels, output_path, samples_per_class=5):
    """Save sample images for each class"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(2*samples_per_class, 2*n_classes))
    
    for i, label in enumerate(unique_labels):
        class_images = images[labels == label]
        
        for j in range(samples_per_class):
            ax = axes[i, j] if n_classes > 1 else axes[j]
            
            if j < len(class_images):
                ax.imshow(class_images[j], cmap='viridis', vmin=0, vmax=255)
            ax.axis('off')
            
            if j == 0:
                name = CLASS_NAMES.get(label, str(label))
                if len(name) > 15:
                    name = name[:12] + "..."
                ax.set_title(name, fontsize=9, loc='left')
    
    plt.suptitle('Sample Images by Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print_success(f"Saved sample images to: {output_path}")


# =============================================================================
# MENU FUNCTIONS
# =============================================================================
def process_pcap_menu(classifier):
    """PCAP processing menu"""
    print_header("Process PCAP File")
    
    if not SCAPY_AVAILABLE:
        print_error("Scapy is not available. Install with: pip install scapy")
        return
    
    pcap_path = input("Enter PCAP file path: ").strip()
    if not os.path.exists(pcap_path):
        print_error("File not found")
        return
    
    max_flows = input("Max flows to process [50]: ").strip()
    max_flows = int(max_flows) if max_flows else 50
    
    # Process
    images, flow_info = process_pcap(pcap_path, max_flows)
    if not images:
        print_error("No flows extracted")
        return
    
    # Classify
    print_info("Classifying flows...")
    preds, probs = classifier.predict_batch(images)
    
    # Display results
    print("\n" + "-"*60)
    print("Classification Results:")
    print("-"*60)
    
    if RICH_AVAILABLE:
        table = Table(title="Flow Classification Results")
        table.add_column("Flow", style="cyan")
        table.add_column("Prediction", style="green")
        table.add_column("Confidence", style="yellow")
        table.add_column("VPN?", style="magenta")
        
        for i, (info, pred, prob) in enumerate(zip(flow_info, preds, probs)):
            conf = prob[pred] * 100
            is_vpn = "✓ VPN" if pred >= 6 else "✗ Non-VPN"
            table.add_row(
                f"Flow {i+1} ({info['proto']})",
                CLASS_NAMES[pred],
                f"{conf:.1f}%",
                is_vpn
            )
        
        console.print(table)
    else:
        for i, (info, pred, prob) in enumerate(zip(flow_info, preds, probs)):
            conf = prob[pred] * 100
            is_vpn = "VPN" if pred >= 6 else "Non-VPN"
            print(f"  Flow {i+1:2d} ({info['proto']:3s}): {CLASS_NAMES[pred]:20s} ({conf:5.1f}%) [{is_vpn}]")
    
    # Summary
    vpn_count = sum(1 for p in preds if p >= 6)
    print(f"\nSummary: {len(preds)} flows | Non-VPN: {len(preds)-vpn_count} | VPN: {vpn_count}")
    
    # Save visualizations
    save_viz = input("\nSave visualizations? [y/N]: ").strip().lower()
    if save_viz == 'y':
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_path = os.path.join(OUTPUT_DIR, f"pcap_flows_{timestamp}.png")
        save_flow_visualizations(images, preds, probs, flow_info, viz_path)


def evaluate_dataset_menu(classifier):
    """Dataset evaluation menu"""
    print_header("Evaluate Dataset")
    
    print("\nAvailable datasets:")
    print("  1. Test set (default)")
    print("  2. Train set")
    print("  3. Custom .npy files")
    
    choice = input("\nSelect [1]: ").strip() or "1"
    
    if choice == "1":
        data_path = os.path.join(TEST_DATA_DIR, "test_data_memory_safe_own_nonVPN_p2p.npy")
        labels_path = os.path.join(TEST_DATA_DIR, "test_labels_memory_safe_own_nonVPN_p2p.npy")
        dataset_name = "Test"
    elif choice == "2":
        data_path = os.path.join(TEST_DATA_DIR, "train_data_memory_safe_own_nonVPN_p2p.npy")
        labels_path = os.path.join(TEST_DATA_DIR, "train_labels_memory_safe_own_nonVPN_p2p.npy")
        dataset_name = "Train"
    else:
        data_path = input("Enter data .npy path: ").strip()
        labels_path = input("Enter labels .npy path: ").strip()
        dataset_name = "Custom"
    
    # Load data
    if not os.path.exists(data_path):
        print_error(f"Data file not found: {data_path}")
        return
    if not os.path.exists(labels_path):
        print_error(f"Labels file not found: {labels_path}")
        return
    
    print_info("Loading dataset...")
    images = np.load(data_path)
    labels = np.load(labels_path)
    print_success(f"Loaded {len(images)} samples")
    
    # Show distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nLabel Distribution:")
    for l, c in zip(unique, counts):
        pct = 100 * c / len(labels)
        print(f"  {CLASS_NAMES.get(l, str(l)):20s}: {c:5d} ({pct:5.1f}%)")
    
    # Evaluate
    print_info("\nRunning evaluation...")
    
    batch_size = 64
    all_preds = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        preds, _ = classifier.predict_batch(batch)
        all_preds.extend(preds)
        print(f"  Progress: {min(i+batch_size, len(images))}/{len(images)}", end='\r')
    
    preds = np.array(all_preds)
    print()
    
    # Metrics
    accuracy = 100 * np.mean(preds == labels)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset:          {dataset_name}")
    print(f"Total Samples:    {len(labels)}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Macro F1 Score:   {macro_f1:.4f}")
    print("="*60)
    
    # Classification report
    all_labels = np.unique(np.concatenate([labels, preds]))
    report = classification_report(
        labels, preds,
        labels=all_labels,
        target_names=[CLASS_NAMES.get(i, str(i)) for i in all_labels],
        digits=3,
        zero_division=0
    )
    print("\nClassification Report:")
    print("-"*60)
    print(report)
    
    # Save outputs
    save_output = input("\nSave results? [y/N]: ").strip().lower()
    if save_output == 'y':
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save confusion matrix
        cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{timestamp}.png")
        save_confusion_matrix(labels, preds, cm_path)
        
        # Save sample images
        samples_path = os.path.join(OUTPUT_DIR, f"sample_images_{timestamp}.png")
        save_sample_images(images, labels, samples_path)
        
        # Save report
        report_path = os.path.join(OUTPUT_DIR, f"classification_report_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("AttentionNet Classification Report\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Total Samples: {len(labels)}\n")
            f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
            f.write(f"Macro F1 Score: {macro_f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write("-"*60 + "\n")
            f.write(report)
        print_success(f"Saved report to: {report_path}")


def show_model_info():
    """Show model information"""
    print_header("Model Information")
    
    info = """
AttentionNet Traffic Classifier
═══════════════════════════════

Model Architecture:
  • Input: 28×28 grayscale images (784 bytes from packet flows)
  
  • CNN Backbone:
    - Conv2D (1→64 channels) + BatchNorm + MaxPool
    - Conv2D (64→128 channels) + BatchNorm + MaxPool
    - Output: 128×7×7 = 49 tokens
  
  • Transformer Encoder:
    - 2 encoder layers
    - 4 attention heads
    - 256 feedforward dimensions
  
  • Classifier:
    - FC (128→128) + Dropout
    - FC (128→128) + Dropout
    - FC (128→12) output

Classification Classes (12 total):
  Non-VPN:                    VPN:
  ├── Chat (NonVPN)           ├── Chat (VPN)
  ├── Email (NonVPN)          ├── Email (VPN)
  ├── File (NonVPN)           ├── File (VPN)
  ├── P2P (NonVPN)            ├── P2P (VPN)
  ├── Streaming (NonVPN)      ├── Streaming (VPN)
  └── VoIP (NonVPN)           └── VoIP (VPN)

Processing Pipeline:
  1. Capture/Load network packets
  2. Group packets into flows (by IP+port pairs)
  3. Filter retransmissions
  4. Anonymize (zero out IPs/MACs)
  5. Extract first 784 bytes
  6. Reshape to 28×28 image
  7. Classify with trained model

Performance:
  • Test Accuracy: ~86%
  • Macro F1 Score: ~0.88
"""
    print(info)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print_header("AttentionNet Traffic Classifier - CLI Demo")
    
    # Load model
    print_info("Loading model...")
    classifier = TrafficClassifier()
    
    try:
        classifier.load_model()
        print_success(f"Model loaded on {classifier.device}")
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        return
    
    # Main menu loop
    while True:
        print("\n" + "-"*40)
        print("Main Menu:")
        print("-"*40)
        print("  1. Process PCAP file")
        print("  2. Evaluate test dataset")
        print("  3. Model information")
        print("  4. Exit")
        print("-"*40)
        
        choice = input("Select option [1-4]: ").strip()
        
        if choice == "1":
            process_pcap_menu(classifier)
        elif choice == "2":
            evaluate_dataset_menu(classifier)
        elif choice == "3":
            show_model_info()
        elif choice == "4":
            print_info("Goodbye!")
            break
        else:
            print_error("Invalid option")


if __name__ == "__main__":
    main()

