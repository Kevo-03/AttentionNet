import os
from scapy.all import rdpcap, wrpcap, IP, TCP, UDP
from collections import defaultdict
import numpy as np
import struct
from tqdm import tqdm


# --- CONFIG ---
RAW_DIR = "categorized_pcaps"          
FLOW_DIR = "processed_test/flows"
IDX_DIR = "processed_test/idx"
MAX_LEN = 784  # bytes per flow
MAX_FILES = 2
MAX_FLOWS = 9
ROWS, COLS, CHANNELS = 28, 28, 1
os.makedirs(FLOW_DIR, exist_ok=True)
os.makedirs(IDX_DIR, exist_ok=True)

# ======================================================
# STEP 1: Split PCAPs into flows
# ======================================================
def split_to_flows(pcap_path, output_dir):
    from scapy.layers.inet import IP
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"[!] Could not read {pcap_path}: {e}")
        return

    flow_count = 0
    flows = defaultdict(list)
    for pkt in packets:
        if flow_count > MAX_FLOWS:
            break
        if IP in pkt:
            proto = pkt[IP].proto
            if proto not in (6, 17):  # TCP or UDP
                continue
            src = pkt[IP].src
            dst = pkt[IP].dst
            sport = getattr(pkt, 'sport', 0)
            dport = getattr(pkt, 'dport', 0)
            key = (src, dst, sport, dport, proto)
            flows[key].append(pkt)
        flow_count = flow_count + 1

    os.makedirs(output_dir, exist_ok=True)  # ✅ Ensure target folder exists

    base = os.path.basename(pcap_path).replace('.pcap', '').replace('.pcapng', '')
    for i, (key, pkts) in enumerate(flows.items()):
        flow_name = f"{base}_flow{i}.pcap"
        wrpcap(os.path.join(output_dir, flow_name), pkts)

    print(f"[+] Split {pcap_path} into {len(flows)} flows.")


def run_step1_split_all():
    """Process all PCAPs in your categorized folders."""
    for vpn_type in ["VPN", "NonVPN"]:
        input_dir = os.path.join(RAW_DIR, vpn_type)
        output_dir = os.path.join(FLOW_DIR, vpn_type)
        os.makedirs(output_dir, exist_ok=True)
        for category in os.listdir(input_dir):
            category_dir = os.path.join(input_dir, category)
            if not os.path.isdir(category_dir):
                continue
            print(f"[Step1] Splitting {vpn_type}/{category}")
            for i, fname in enumerate(tqdm(os.listdir(category_dir))):
                if i >= MAX_FILES:
                    break
                if fname.endswith((".pcap", ".pcapng")):
                    split_to_flows(os.path.join(category_dir, fname),
                                os.path.join(output_dir, category))


# ======================================================
# STEP 2: Convert flows to byte arrays
# ======================================================
def pcap_to_bytes(pcap_path):
    payload_bytes = b""
    try:
        packets = rdpcap(pcap_path)

        for pkt in packets:
            if len(payload_bytes) >= MAX_LEN:
                break

            if IP in pkt:
                # Use IP-layer bytes (includes TCP/UDP header + payload)
                payload_bytes += bytes(pkt[IP])

        arr = np.frombuffer(payload_bytes, dtype=np.uint8)

        if len(arr) < MAX_LEN:
            arr = np.pad(arr, (0, MAX_LEN - len(arr)), 'constant', constant_values=0)
        else:
            arr = arr[:MAX_LEN]

        return arr.reshape(ROWS, COLS)

    except Exception as e:
        print(f"[!] Error processing {pcap_path}: {e}")
        return None

# --- STEP 2: Convert flows to byte arrays with detailed labels ---
def run_step2_convert_bytes():
    """Convert all flows into fixed-length byte sequences with 12-class labels."""
    all_data, all_labels = [], []

    # Define labels for both VPN and NonVPN categories
    label_map = {
        "NonVPN": {
            "Chat": 0,
            "Email": 1,
            "File": 2,
            "P2P": 3,
            "Streaming": 4,
            "VoIP": 5,
        },
        "VPN": {
            "Chat": 6,
            "Email": 7,
            "File": 8,
            "P2P": 9,
            "Streaming": 10,
            "VoIP": 11,
        }
    }

    for vpn_type in ["VPN", "NonVPN"]:
        input_dir = os.path.join(FLOW_DIR, vpn_type)
        for category in os.listdir(input_dir):
            category_dir = os.path.join(input_dir, category)
            if not os.path.isdir(category_dir):
                continue
            if category not in label_map[vpn_type]:
                print(f"[!] Skipping {category_dir} (uncategorized)")
                continue

            label_value = label_map[vpn_type][category]
            print(f"[Step2] Processing {vpn_type}/{category} → Label {label_value}")

            for fname in tqdm(os.listdir(category_dir)):
                if not fname.endswith(".pcap"):
                    continue
                arr = pcap_to_bytes(os.path.join(category_dir, fname))
                if arr is not None:
                    all_data.append(arr)
                    all_labels.append(label_value)

    np.save(os.path.join(IDX_DIR, "data.npy"), np.array(all_data))
    np.save(os.path.join(IDX_DIR, "labels.npy"), np.array(all_labels))
    print("[+] Saved NumPy data and label arrays with 12-class labeling.")

# ======================================================
# STEP 3: Write IDX files
# ======================================================
def write_idx(data_array, label_array):
    """Write IDX3 (images) and IDX1 (labels) compatible files."""
    data_path = os.path.join(IDX_DIR, "train-images.idx3-ubyte")
    label_path = os.path.join(IDX_DIR, "train-labels.idx1-ubyte")

    with open(data_path, "wb") as f:
        f.write(struct.pack('>IIII', 0x00000803, len(data_array), ROWS, COLS))
        for x in data_array:
            f.write(x.tobytes())

    with open(label_path, "wb") as f:
        f.write(struct.pack('>II', 0x00000801, len(label_array)))
        for y in label_array:
            f.write(struct.pack('>B', y))

    print(f"[+] IDX files written to {IDX_DIR}")


def run_step3_write_idx():
    """Read from saved .npy files and export to IDX format."""
    data = np.load(os.path.join(IDX_DIR, "data.npy"))
    labels = np.load(os.path.join(IDX_DIR, "labels.npy"))
    write_idx(data, labels)


# ======================================================
# Optional: run each step
# ======================================================
if __name__ == "__main__":
    # Uncomment only one at a time when testing!
    #run_step1_split_all()
    run_step2_convert_bytes()
    #run_step3_write_idx()
    pass