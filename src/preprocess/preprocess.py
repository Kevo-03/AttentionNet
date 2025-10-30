import os
from scapy.all import rdpcap, wrpcap, IP, TCP, UDP
from collections import defaultdict
import numpy as np
import hashlib
from tqdm import tqdm

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

os.makedirs(FLOW_DIR, exist_ok=True)
os.makedirs(IDX_DIR, exist_ok=True)


# ================================
# Step 1: Split PCAPs into flows
# ================================
def split_to_flows(pcap_path, output_dir):
    try:
        packets = rdpcap(pcap_path)
    except:
        print(f"[!] Could not read {pcap_path}")
        return

    flows = defaultdict(list)
    count = 0

    #Limit splitted flows for testing
    for pkt in packets:
        if IP in pkt and (TCP in pkt or UDP in pkt):
            ip = pkt[IP]
            proto = ip.proto
            sorted_ips = tuple(sorted((ip.src,ip.dst)))
            sport = getattr(pkt, 'sport', 0)
            dport = getattr(pkt, 'dport', 0)
            sorted_ports = tuple(sorted((sport,dport)))
            key = (sorted_ips[0], sorted_ips[1], sorted_ports[0], sorted_ports[1], proto)
            flows[key].append(pkt)

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(pcap_path).rsplit(".", 1)[0]

    flow_count = 0
    for idx, pkts in enumerate(flows.values()):
        if flow_count >= MAX_FLOWS_PER_PCAP:
            break
        if len(pkts) > 0:
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
# Step 2: Extract payload → image
# ================================
def extract_payload_array(pcap_path):
    all_packet_bytes = b"" # This will now hold full packet bytes
    seen_seq_nums = set()
    try:
        packets = rdpcap(pcap_path)
    except:
        print(f"[!] Could not read {pcap_path}")
        return None

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


def run_step2_extract_and_clean():
    label_map = {
        "NonVPN": {"Chat": 0, "Email": 1, "File": 2, "P2P": 3, "Streaming": 4, "VoIP": 5},
        "VPN":    {"Chat": 6, "Email": 7, "File": 8, "P2P": 9, "Streaming": 10, "VoIP": 11}
    }

    all_images, all_labels = [], []
    seen_hashes = set()

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

                img = extract_payload_array(os.path.join(input_dir, fname))
                if img is None:
                    continue

                # Remove duplicates using hash
                digest = hashlib.sha1(img.tobytes()).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)

                all_images.append(img)
                all_labels.append(label)

    np.save(os.path.join(IDX_DIR, "data.npy"), np.array(all_images))
    np.save(os.path.join(IDX_DIR, "labels.npy"), np.array(all_labels))
    print(f"[+] Cleaned dataset saved. Total samples: {len(all_images)}")


# ======================================================
# Run Steps
# ======================================================
if __name__ == "__main__":
    run_step1_split_all()
    run_step2_extract_and_clean()