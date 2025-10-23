import os
import numpy as np
from scapy.all import rdpcap, wrpcap
from collections import defaultdict
import struct
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- CONFIG ---
RAW_DIR = "categorized_pcaps"
FLOW_DIR = "processed_data/flows"
IDX_DIR = "processed_data/idx"
MAX_LEN = 784
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

    flows = defaultdict(list)
    for pkt in packets:
        if IP in pkt:
            proto = pkt[IP].proto
            if proto not in (6, 17):  # TCP or UDP only
                continue
            src, dst = pkt[IP].src, pkt[IP].dst
            sport, dport = getattr(pkt, 'sport', 0), getattr(pkt, 'dport', 0)
            key = (src, dst, sport, dport, proto)
            flows[key].append(pkt)

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(pcap_path).replace('.pcap', '').replace('.pcapng', '')
    for i, (key, pkts) in enumerate(flows.items()):
        flow_name = f"{base}_flow{i}.pcap"
        wrpcap(os.path.join(output_dir, flow_name), pkts)

    print(f"[+] Split {pcap_path} into {len(flows)} flows.")


def run_step1_split_all():
    """Process all PCAPs in categorized structure."""
    for vpn_type in ["VPN", "NonVPN"]:
        input_dir = os.path.join(RAW_DIR, vpn_type)
        output_dir = os.path.join(FLOW_DIR, vpn_type)
        os.makedirs(output_dir, exist_ok=True)

        for category in os.listdir(input_dir):
            category_dir = os.path.join(input_dir, category)
            if not os.path.isdir(category_dir):
                continue
            print(f"[Step1] Splitting {vpn_type}/{category}")
            for fname in tqdm(os.listdir(category_dir)):
                if fname.endswith((".pcap", ".pcapng")):
                    split_to_flows(
                        os.path.join(category_dir, fname),
                        os.path.join(output_dir, category)
                    )


# ======================================================
# STEP 2: Convert flows to bytes with 12-class labels
# ======================================================
def pcap_to_bytes(pcap_path):
    """Read one pcap and return a padded/truncated byte array."""
    try:
        with open(pcap_path, 'rb') as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        if len(arr) > MAX_LEN:
            arr = arr[:MAX_LEN]
        elif len(arr) < MAX_LEN:
            arr = np.pad(arr, (0, MAX_LEN - len(arr)), 'constant')
        return arr
    except Exception as e:
        print(f"[!] Error in {pcap_path}: {e}")
        return None


def run_step2_convert_bytes():
    """Convert all flows into fixed-length byte sequences."""
    all_data, all_labels = [], []

    label_map = {
        "NonVPN": {"Chat": 0, "Email": 1, "File": 2, "P2P": 3, "Streaming": 4, "VoIP": 5},
        "VPN": {"Chat": 6, "Email": 7, "File": 8, "P2P": 9, "Streaming": 10, "VoIP": 11},
    }

    for vpn_type in ["VPN", "NonVPN"]:
        vpn_dir = os.path.join(FLOW_DIR, vpn_type)
        for category in os.listdir(vpn_dir):
            category_dir = os.path.join(vpn_dir, category)
            if not os.path.isdir(category_dir):
                continue
            if category not in label_map[vpn_type]:
                print(f"[!] Skipping {category_dir}")
                continue

            label = label_map[vpn_type][category]
            print(f"[Step2] {vpn_type}/{category} → Label {label}")

            for fname in tqdm(os.listdir(category_dir)):
                if not fname.endswith(".pcap"):
                    continue
                arr = pcap_to_bytes(os.path.join(category_dir, fname))
                if arr is not None:
                    all_data.append(arr)
                    all_labels.append(label)

    np.save(os.path.join(IDX_DIR, "data.npy"), np.array(all_data))
    np.save(os.path.join(IDX_DIR, "labels.npy"), np.array(all_labels))
    print(f"[+] Saved all data/labels arrays ({len(all_data)} samples).")


# ======================================================
# STEP 3: Split into train/test sets
# ======================================================
def run_step3_split_data():
    data = np.load(os.path.join(IDX_DIR, "data.npy"))
    labels = np.load(os.path.join(IDX_DIR, "labels.npy"))

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=42
    )

    np.save(os.path.join(IDX_DIR, "train_data.npy"), X_train)
    np.save(os.path.join(IDX_DIR, "train_labels.npy"), y_train)
    np.save(os.path.join(IDX_DIR, "test_data.npy"), X_test)
    np.save(os.path.join(IDX_DIR, "test_labels.npy"), y_test)

    print(f"[+] Train/Test split complete → Train: {len(y_train)}, Test: {len(y_test)}")


# ======================================================
# STEP 4: Write IDX files (train/test)
# ======================================================
def write_idx(data_array, label_array, prefix):
    data_path = os.path.join(IDX_DIR, f"{prefix}-images.idx3-ubyte")
    label_path = os.path.join(IDX_DIR, f"{prefix}-labels.idx1-ubyte")

    with open(data_path, "wb") as f:
        f.write(struct.pack('>IIIII', 0x00000803, len(data_array), ROWS, COLS, CHANNELS))
        for x in data_array:
            f.write(x.tobytes())

    with open(label_path, "wb") as f:
        f.write(struct.pack('>II', 0x00000801, len(label_array)))
        for y in label_array:
            f.write(struct.pack('>B', y))

    print(f"[+] IDX files written for {prefix} set.")


def run_step4_write_idx():
    train_data = np.load(os.path.join(IDX_DIR, "train_data.npy"))
    train_labels = np.load(os.path.join(IDX_DIR, "train_labels.npy"))
    test_data = np.load(os.path.join(IDX_DIR, "test_data.npy"))
    test_labels = np.load(os.path.join(IDX_DIR, "test_labels.npy"))

    write_idx(train_data, train_labels, "train")
    write_idx(test_data, test_labels, "t10k")


# ======================================================
# RUN SELECTIVELY
# ======================================================
if __name__ == "__main__":
    # Uncomment step by step as you progress
    # run_step1_split_all()
    # run_step2_convert_bytes()
    # run_step3_split_data()
    # run_step4_write_idx()
    pass