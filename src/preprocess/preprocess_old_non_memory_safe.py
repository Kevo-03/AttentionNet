import os
import shutil
from scapy.all import IP, TCP, UDP, PcapReader
from collections import defaultdict
import numpy as np
import hashlib
from tqdm import tqdm

# --- CONFIG ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
RAW_DIR = os.path.join(PROJECT_ROOT, "categorized_pcaps")
IDX_DIR = os.path.join(PROJECT_ROOT, "processed_data/idx")
MAX_LEN = 784  # 28x28
ROWS, COLS = 28, 28

os.makedirs(IDX_DIR, exist_ok=True)


# ================================
# Single-pass: PCAP → flows → images
# ================================
def process_pcap_to_images(pcap_path, label, seen_hashes, max_flows=None):
    """
    Process a PCAP file directly into images without writing intermediate flow files.
    
    Args:
        pcap_path: Path to the PCAP file
        label: The label for this PCAP's category
        seen_hashes: Set of SHA1 hashes for deduplication
        max_flows: Maximum number of flows to process from this PCAP (None = all)
    
    Returns:
        List of (image, label) tuples for valid, unique flows
    """
    flows = defaultdict(list)
    results = []
    
    # Step 1: Split into flows (in memory)
    try:
        with PcapReader(pcap_path) as packets:
            for pkt in packets:
                if IP in pkt and (TCP in pkt or UDP in pkt):
                    ip = pkt[IP]
                    proto = ip.proto
                    sorted_ips = tuple(sorted((ip.src, ip.dst)))
                    sport = getattr(pkt, 'sport', 0)
                    dport = getattr(pkt, 'dport', 0)
                    sorted_ports = tuple(sorted((sport, dport)))
                    key = (sorted_ips[0], sorted_ips[1], sorted_ports[0], sorted_ports[1], proto)
                    flows[key].append(pkt)
    except Exception as e:
        print(f"[!] Could not read or process {pcap_path}: {e}")
        return results
    
    # Step 2: Process each flow into an image
    flow_count = 0
    for flow_packets in flows.values():
        if max_flows and flow_count >= max_flows:
            break
            
        if len(flow_packets) == 0:
            continue
            
        img = extract_flow_image(flow_packets)
        if img is None:
            continue
        
        # Deduplication using hash
        digest = hashlib.sha1(img.tobytes()).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        
        results.append((img, label))
        flow_count += 1
    
    # Clear flows from memory
    flows.clear()
    
    return results

def is_retransmission(pkt, seen_seq_nums):
    """
    Detect TCP retransmissions by tracking sequence numbers.
    A retransmission occurs when we see the same seq number twice.
    """
    if TCP not in pkt:
        return False
    
    tcp = pkt[TCP]
    ip = pkt[IP]
    
    # Create a unique key for this connection direction
    key = (ip.src, ip.dst, tcp.sport, tcp.dport, tcp.seq)
    
    if key in seen_seq_nums:
        return True
    
    seen_seq_nums.add(key)
    return False

def is_corrupted(pkt):
    """
    Basic corruption detection - check if packet has obvious issues.
    More advanced checks could validate checksums.
    """
    try:
        # Check if packet has minimum required fields
        if IP not in pkt:
            return True
        
        # Check for zero-length packets with no payload
        if len(bytes(pkt)) == 0:
            return True
            
        return False
    except:
        return True
# ================================
# Extract payload from packet list → image
# ================================
def extract_flow_image(packets):
    """
    Extract image from a list of packets (representing a flow).
    Returns a 28x28 numpy array or None if the flow is invalid.
    """
    all_packet_bytes = b""
    seen_seq_nums = set()
    
    for pkt in packets:
        if len(all_packet_bytes) >= MAX_LEN:
            break

        if is_corrupted(pkt):
            continue
        if is_retransmission(pkt, seen_seq_nums):
            continue

        # We only care about IP packets
        if IP in pkt:
            # Create a copy to avoid changing the original packet list
            pkt_copy = pkt.copy()
            
            # Set IPs to 0.0.0.0 (or any constant)
            pkt_copy[IP].src = "0.0.0.0"
            pkt_copy[IP].dst = "0.0.0.0"
            
            # Remove MAC addresses if they exist
            if "Ether" in pkt_copy:
                pkt_copy["Ether"].src = "00:00:00:00:00:00"
                pkt_copy["Ether"].dst = "00:00:00:00:00:00"
            # --- END ANONYMIZATION ---

            # Now, add the bytes of the *entire* anonymized packet
            all_packet_bytes += bytes(pkt_copy)
    
    # Remove empty flows
    if len(all_packet_bytes) == 0:
        return None

    # Pad or truncate the full packet data
    arr = np.frombuffer(all_packet_bytes, dtype=np.uint8)
    if len(arr) >= MAX_LEN:
        arr = arr[:MAX_LEN]
    else:
        arr = np.pad(arr, (0, MAX_LEN - len(arr)), 'constant', constant_values=0)
    
    return arr.reshape(ROWS, COLS)


def process_all_pcaps(max_files_per_category=None, max_flows_per_file=None):
    """
    Process all PCAPs directly into images with batch saving to avoid memory overflow.
    
    Args:
        max_files_per_category: Limit number of PCAP files per category (None = all)
        max_flows_per_file: Limit number of flows per PCAP file (None = all)
    """
    label_map = {
        "NonVPN": {"Chat": 0, "Email": 1, "File": 2, "Streaming": 3, "VoIP": 4, "P2P": 5},
        "VPN":    {"Chat": 6, "Email": 7, "File": 8, "P2P": 9, "Streaming": 10, "VoIP": 11}
    }

    seen_hashes = set()
    batch_images, batch_labels = [], []
    batch_size = 1000  # Save every 1000 samples to prevent memory overflow
    batch_count = 0
    total_samples = 0
    
    if max_files_per_category:
        print(f"[TEST MODE] Processing max {max_files_per_category} files per category")
    if max_flows_per_file:
        print(f"[TEST MODE] Processing max {max_flows_per_file} flows per file")

    for vpn_type in ["VPN", "NonVPN"]:
        for category in label_map[vpn_type]:
            category_dir = os.path.join(RAW_DIR, vpn_type, category)
            if not os.path.isdir(category_dir):
                continue

            label = label_map[vpn_type][category]
            print(f"[Processing] {vpn_type}/{category} → Label {label}")

            pcap_files = [f for f in os.listdir(category_dir) if f.endswith((".pcap", ".pcapng"))]
            
            # Limit files if in test mode
            if max_files_per_category:
                pcap_files = pcap_files[:max_files_per_category]
            
            for fname in tqdm(pcap_files):
                pcap_path = os.path.join(category_dir, fname)
                
                # Process this PCAP and get all its flow images
                flow_images = process_pcap_to_images(pcap_path, label, seen_hashes, max_flows_per_file)
                
                for img, lbl in flow_images:
                    batch_images.append(img)
                    batch_labels.append(lbl)
                    total_samples += 1
                    
                    # Save batch if we've accumulated enough samples
                    if len(batch_images) >= batch_size:
                        _save_batch(batch_images, batch_labels, batch_count)
                        batch_count += 1
                        batch_images.clear()
                        batch_labels.clear()

    # Save any remaining samples
    if len(batch_images) > 0:
        _save_batch(batch_images, batch_labels, batch_count)
        batch_count += 1

    # Merge all batches into final files
    print(f"[+] Merging {batch_count} batches...")
    _merge_batches(batch_count)
    print(f"[+] Complete! Total samples: {total_samples}")


def _save_batch(images, labels, batch_idx):
    """Save a batch of images and labels to temporary files."""
    batch_dir = os.path.join(IDX_DIR, "batches")
    os.makedirs(batch_dir, exist_ok=True)
    
    np.save(os.path.join(batch_dir, f"data_batch_{batch_idx}.npy"), np.array(images))
    np.save(os.path.join(batch_dir, f"labels_batch_{batch_idx}.npy"), np.array(labels))


def _merge_batches(num_batches):
    """Merge all batch files into final data.npy and labels.npy files."""
    batch_dir = os.path.join(IDX_DIR, "batches")
    
    all_images = []
    all_labels = []
    
    for i in range(num_batches):
        data_file = os.path.join(batch_dir, f"data_batch_{i}.npy")
        labels_file = os.path.join(batch_dir, f"labels_batch_{i}.npy")
        
        if os.path.exists(data_file) and os.path.exists(labels_file):
            all_images.append(np.load(data_file))
            all_labels.append(np.load(labels_file))
    
    # Concatenate all batches
    final_images = np.concatenate(all_images, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    
    # Save final files
    np.save(os.path.join(IDX_DIR, "data_fixed.npy"), final_images)
    np.save(os.path.join(IDX_DIR, "labels_fixed.npy"), final_labels)
    
    # Clean up batch files
    shutil.rmtree(batch_dir)
    
    print(f"[+] Saved {len(final_labels)} samples to {IDX_DIR}")


def visualize_samples(samples_per_label=16):
    """
    Visualize sample images from the processed dataset to verify output.
    Creates a separate 4x4 grid for each label.
    
    Args:
        samples_per_label: Number of samples to show per label (default: 16 for 4x4 grid)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not installed. Run: pip install matplotlib")
        return
    
    data_file = os.path.join(IDX_DIR, "data_fixed.npy")
    labels_file = os.path.join(IDX_DIR, "labels_fixed.npy")
    
    if not os.path.exists(data_file) or not os.path.exists(labels_file):
        print("[!] No processed data found. Run processing first.")
        return
    
    images = np.load(data_file)
    labels = np.load(labels_file)
    
    label_names = {
        0: "NonVPN-Chat", 1: "NonVPN-Email", 2: "NonVPN-File", 
        3: "NonVPN-Streaming", 4: "NonVPN-VoIP", 5: "NonVPN-P2P",
        6: "VPN-Chat", 7: "VPN-Email", 8: "VPN-File", 
        9: "VPN-P2P", 10: "VPN-Streaming", 11: "VPN-VoIP"
    }
    
    print(f"\n[+] Dataset Statistics:")
    print(f"    Total images: {len(images)}")
    print(f"    Image shape: {images[0].shape}")
    print(f"    Label distribution:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for lbl, count in zip(unique_labels, counts):
        print(f"      {label_names.get(lbl, f'Unknown-{lbl}')}: {count} samples")
    
    # Create visualization for each label
    rows = 4
    cols = 4
    
    for label_idx in unique_labels:
        # Get all images for this label
        label_mask = labels == label_idx
        label_images = images[label_mask]
        
        if len(label_images) == 0:
            continue
        
        print(f"\n[+] Generating visualization for {label_names.get(label_idx, f'Unknown-{label_idx}')}...")
        
        # Take up to samples_per_label images
        num_to_show = min(samples_per_label, len(label_images))
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx in range(rows * cols):
            if idx < num_to_show:
                axes[idx].imshow(label_images[idx], cmap='gray')
                axes[idx].set_title(f"Sample {idx}", fontsize=8)
                axes[idx].axis('off')
            else:
                axes[idx].axis('off')
        
        # Add super title for the whole figure
        fig.suptitle(f"{label_names.get(label_idx, f'Unknown-{label_idx}')} - Label {label_idx}", 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save with label-specific filename
        viz_path = os.path.join(IDX_DIR, f"visualization_label_{label_idx}_{label_names.get(label_idx, 'Unknown').replace('-', '_')}.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"    Saved: {viz_path}")
        plt.close()  # Close to free memory
    
    print(f"\n[+] All visualizations saved to: {IDX_DIR}")


# ======================================================
# Run Processing
# ======================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode: process 4 files per category, max 10 flows per file
        print("\n" + "="*60)
        print("TEST MODE: Processing 4 files per category, 10 flows max")
        print("="*60 + "\n")
        process_all_pcaps(max_files_per_category=4, max_flows_per_file=10)
        print("\n" + "="*60)
        print("Visualizing results...")
        print("="*60 + "\n")
        visualize_samples()
    else:
        # Production mode: process all data
        process_all_pcaps()
        print("\n" + "="*60)
        print("Visualizing results...")
        print("="*60 + "\n")
        visualize_samples()
