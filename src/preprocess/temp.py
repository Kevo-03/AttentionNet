import os
from scapy.all import rdpcap, wrpcap, IP, TCP, UDP, Raw
from collections import defaultdict
import numpy as np
import hashlib
from tqdm import tqdm
import struct
import random

# --- CONFIG ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
RAW_DIR = os.path.join(PROJECT_ROOT, "categorized_pcaps")
FLOW_DIR = os.path.join(PROJECT_ROOT,"processed_test/flows")
IDX_DIR = os.path.join(PROJECT_ROOT,"processed_test/idx")
MAX_LEN = 784  # 28x28
MAX_FILES_PER_CLASS = 4   # only for testing small previews
MAX_FLOWS_PER_PCAP = 10   # reduce splitting to speed up
ROWS, COLS = 28, 28

# Processing options
USE_L7_ONLY = False  # Extract Layer 7 (application layer) payload only
FILTER_RETRANSMISSIONS = True  # Remove TCP retransmissions
ANONYMIZE_IPS = True  # Randomize IP addresses
SAVE_IDX_FORMAT = False  # Save in IDX format (in addition to .npy)

os.makedirs(FLOW_DIR, exist_ok=True)
os.makedirs(IDX_DIR, exist_ok=True)


# ================================
# Helper: Anonymization mapping
# ================================
class IPAnonymizer:
    """Consistent IP anonymization - same input IP always maps to same random IP"""
    def __init__(self, seed=42):
        self.mapping = {}
        self.counter = 1
        random.seed(seed)
    
    def anonymize(self, ip):
        if ip not in self.mapping:
            # Generate random IP address
            self.mapping[ip] = f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        return self.mapping[ip]


# ================================
# Step 1: Split PCAPs into flows
# ================================
def split_to_flows(pcap_path, output_dir):
    """
    Split PCAP into flows using 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol).
    Bidirectional flows are grouped together.
    """
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"[!] Could not read {pcap_path}: {e}")
        return

    flows = defaultdict(list)

    for pkt in packets:
        if IP in pkt and (TCP in pkt or UDP in pkt):
            ip = pkt[IP]
            proto = ip.proto
            
            # Get source and destination ports
            if TCP in pkt:
                sport = pkt[TCP].sport
                dport = pkt[TCP].dport
            elif UDP in pkt:
                sport = pkt[UDP].sport
                dport = pkt[UDP].dport
            else:
                continue
            
            # Create bidirectional flow key (sorted to group both directions)
            sorted_ips = tuple(sorted((ip.src, ip.dst)))
            sorted_ports = tuple(sorted((sport, dport)))
            key = (sorted_ips[0], sorted_ips[1], sorted_ports[0], sorted_ports[1], proto)
            flows[key].append(pkt)

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(pcap_path).rsplit(".", 1)[0]

    # Limit number of flows for testing (fixed bug: limit flows not packets)
    flow_count = 0
    for idx, pkts in enumerate(flows.values()):
        if flow_count >= MAX_FLOWS_PER_PCAP:
            break
        if len(pkts) > 0:  # Only save non-empty flows
            wrpcap(os.path.join(output_dir, f"{base}_flow{idx}.pcap"), pkts)
            flow_count += 1


def run_step1_split_all():
    for vpn_type in ["VPN", "NonVPN"]:
        for category in os.listdir(os.path.join(RAW_DIR, vpn_type)):
            category_dir = os.path.join(RAW_DIR, vpn_type, category)
            if not os.path.isdir(category_dir):
                continue

            output_dir = os.path.join(FLOW_DIR, vpn_type, category)
            os.makedirs(output_dir, exist_ok=True)

            print(f"[Step1] Splitting {vpn_type}/{category}")
            for i, fname in enumerate(tqdm(os.listdir(category_dir))):
                if i >= MAX_FILES_PER_CLASS:
                    break
                if fname.endswith((".pcap", ".pcapng")):
                    split_to_flows(os.path.join(category_dir, fname), output_dir)


# ================================
# Step 2: Traffic Cleaning & Extraction
# ================================
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


def extract_payload_array(pcap_path):
    """
    Extract Layer 7 (application layer) payload from packets.
    Implements traffic cleaning: removes retransmissions, corrupted packets.
    """
    all_payload_bytes = b""
    seen_seq_nums = set()
    anonymizer = IPAnonymizer() if ANONYMIZE_IPS else None

    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"[!] Could not read {pcap_path}: {e}")
        return None

    for pkt in packets:
        if len(all_payload_bytes) >= MAX_LEN:
            break

        # Skip corrupted packets
        if is_corrupted(pkt):
            continue

        # Skip retransmissions
        if FILTER_RETRANSMISSIONS and is_retransmission(pkt, seen_seq_nums):
            continue

        if IP in pkt:
            if USE_L7_ONLY:
                # Extract Layer 7 (application layer) payload only
                if Raw in pkt:
                    payload = bytes(pkt[Raw].load)
                    all_payload_bytes += payload
            else:
                # Use entire packet (with anonymization if enabled)
                if ANONYMIZE_IPS:
                    pkt_copy = pkt.copy()
                    pkt_copy[IP].src = anonymizer.anonymize(pkt[IP].src)
                    pkt_copy[IP].dst = anonymizer.anonymize(pkt[IP].dst)
                    
                    if "Ether" in pkt_copy:
                        pkt_copy["Ether"].src = "00:00:00:00:00:00"
                        pkt_copy["Ether"].dst = "00:00:00:00:00:00"
                    
                    all_payload_bytes += bytes(pkt_copy)
                else:
                    all_payload_bytes += bytes(pkt)

    # Remove empty flows
    if len(all_payload_bytes) == 0:
        return None

    # Pad or truncate to MAX_LEN
    arr = np.frombuffer(all_payload_bytes, dtype=np.uint8)
    if len(arr) >= MAX_LEN:
        arr = arr[:MAX_LEN]
    else:
        arr = np.pad(arr, (0, MAX_LEN - len(arr)), 'constant', constant_values=0)
    
    return arr.reshape(ROWS, COLS)


def save_idx_format(images, labels, idx_dir):
    """
    Save data in IDX format (used by MNIST and compatible with PyTorch/TensorFlow).
    IDX3: images (magic number 0x00000803 for 3D array)
    IDX1: labels (magic number 0x00000801 for 1D array)
    """
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.uint8)
    
    # Save images as IDX3 format
    with open(os.path.join(idx_dir, "train-images.idx3-ubyte"), 'wb') as f:
        # Magic number for IDX3
        f.write(struct.pack('>I', 0x00000803))
        # Number of images
        f.write(struct.pack('>I', len(images)))
        # Number of rows
        f.write(struct.pack('>I', ROWS))
        # Number of columns
        f.write(struct.pack('>I', COLS))
        # Write image data
        f.write(images.tobytes())
    
    # Save labels as IDX1 format
    with open(os.path.join(idx_dir, "train-labels.idx1-ubyte"), 'wb') as f:
        # Magic number for IDX1
        f.write(struct.pack('>I', 0x00000801))
        # Number of labels
        f.write(struct.pack('>I', len(labels)))
        # Write label data
        f.write(labels.tobytes())
    
    print(f"[+] IDX format files saved: train-images.idx3-ubyte, train-labels.idx1-ubyte")


def run_step2_extract_and_clean():
    """
    Step 2: Extract Layer 7 payload, clean traffic, and generate images.
    - Removes retransmissions
    - Removes corrupted packets
    - Removes duplicates
    - Converts to 28x28 images
    """
    label_map = {
        "NonVPN": {"Chat": 0, "Email": 1, "File": 2, "P2P": 3, "Streaming": 4, "VoIP": 5},
        "VPN":    {"Chat": 6, "Email": 7, "File": 8, "P2P": 9, "Streaming": 10, "VoIP": 11}
    }

    all_images, all_labels = [], []
    seen_hashes = set()
    stats = {"total": 0, "empty": 0, "duplicate": 0, "valid": 0}

    for vpn_type in ["VPN", "NonVPN"]:
        for category in label_map[vpn_type]:
            input_dir = os.path.join(FLOW_DIR, vpn_type, category)
            if not os.path.isdir(input_dir):
                continue

            label = label_map[vpn_type][category]
            print(f"[Step2] {vpn_type}/{category} → Label {label}")

            for fname in tqdm(os.listdir(input_dir)):
                if not fname.endswith(".pcap"):
                    continue

                stats["total"] += 1
                img = extract_payload_array(os.path.join(input_dir, fname))
                
                if img is None:
                    stats["empty"] += 1
                    continue

                # Remove duplicates using hash
                digest = hashlib.sha1(img.tobytes()).hexdigest()
                if digest in seen_hashes:
                    stats["duplicate"] += 1
                    continue
                seen_hashes.add(digest)

                all_images.append(img)
                all_labels.append(label)
                stats["valid"] += 1

    # Save in NumPy format
    np.save(os.path.join(IDX_DIR, "data_images.npy"), np.array(all_images))
    np.save(os.path.join(IDX_DIR, "data_labels.npy"), np.array(all_labels))
    
    # Also save in IDX format if enabled
    if SAVE_IDX_FORMAT and len(all_images) > 0:
        save_idx_format(all_images, all_labels, IDX_DIR)
    
    print(f"\n[+] Dataset processing complete!")
    print(f"    Total flows processed: {stats['total']}")
    print(f"    Empty flows removed: {stats['empty']}")
    print(f"    Duplicate flows removed: {stats['duplicate']}")
    print(f"    Valid samples saved: {stats['valid']}")
    print(f"    Output directory: {IDX_DIR}")


# ======================================================
# Run Steps
# ======================================================
if __name__ == "__main__":
    print("="*70)
    print("PCAP PREPROCESSING PIPELINE")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Layer 7 Payload Only: {USE_L7_ONLY}")
    print(f"  - Filter Retransmissions: {FILTER_RETRANSMISSIONS}")
    print(f"  - Anonymize IPs: {ANONYMIZE_IPS}")
    print(f"  - Save IDX Format: {SAVE_IDX_FORMAT}")
    print(f"  - Max Length: {MAX_LEN} bytes ({ROWS}x{COLS})")
    print(f"  - Max Files per Class: {MAX_FILES_PER_CLASS} (testing mode)")
    print(f"  - Max Flows per PCAP: {MAX_FLOWS_PER_PCAP} (testing mode)")
    print(f"\nDirectories:")
    print(f"  - Input: {RAW_DIR}")
    print(f"  - Flows: {FLOW_DIR}")
    print(f"  - Output: {IDX_DIR}")
    print("="*70)
    print()
    
    run_step1_split_all()
    print()
    run_step2_extract_and_clean()