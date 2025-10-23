import os
from scapy.all import rdpcap, wrpcap
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import struct

# --- CONFIGURATION ---
RAW_DIR = "categorized_pcaps"  # your organized dataset
FLOW_DIR = "processed_data/flows"
IDX_DIR = "processed_data/idx"
MAX_LEN = 784  # TranSECANet’s fixed input length

os.makedirs(FLOW_DIR, exist_ok=True)
os.makedirs(IDX_DIR, exist_ok=True)

# --- STEP 1: Split PCAPs into flows ---
def split_to_flows(pcap_path, output_dir):
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"[!] Could not read {pcap_path}: {e}")
        return

    flows = defaultdict(list)
    for pkt in packets:
        if 'IP' in pkt:
            ip = pkt['IP']
            proto = ip.proto
            src = ip.src
            dst = ip.dst
            sport = getattr(pkt, 'sport', 0)
            dport = getattr(pkt, 'dport', 0)
            key = (src, dst, sport, dport, proto)
            flows[key].append(pkt)

    base = os.path.basename(pcap_path).replace('.pcap', '').replace('.pcapng', '')
    for i, (key, pkts) in enumerate(flows.items()):
        flow_name = f"{base}_flow{i}.pcap"
        wrpcap(os.path.join(output_dir, flow_name), pkts)

# --- STEP 2: Convert flows to byte sequences ---
def pcap_to_bytes(pcap_path):
    try:
        with open(pcap_path, 'rb') as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        if len(arr) > MAX_LEN:
            arr = arr[:MAX_LEN]
        elif len(arr) < MAX_LEN:
            arr = np.pad(arr, (0, MAX_LEN - len(arr)), 'constant', constant_values=0)
        return arr
    except:
        return None

# --- STEP 3: IDX writing ---
def write_idx(data_array, label_array, data_path, label_path):
    with open(data_path, "wb") as f:
        f.write(struct.pack('>IIII', 0x00000803, len(data_array), MAX_LEN, 1))
        for x in data_array:
            f.write(x.tobytes())

    with open(label_path, "wb") as f:
        f.write(struct.pack('>II', 0x00000801, len(label_array)))
        for y in label_array:
            f.write(struct.pack('>B', y))

# --- STEP 4: Label Mapping ---
def get_label_map(base_dir):
    labels = {}
    label_id = 0
    for vpn_type in ["VPN", "NonVPN"]:
        vpn_path = os.path.join(base_dir, vpn_type)
        for category in os.listdir(vpn_path):
            label_name = f"{vpn_type}-{category}"
            labels[label_name] = label_id
            label_id += 1
    return labels

# --- STEP 5: Pipeline Execution ---
label_map = get_label_map(RAW_DIR)
print("Detected label mapping:")
for k, v in label_map.items():
    print(f"{v}: {k}")

data_samples, label_samples = [], []

for label_name, label_id in label_map.items():
    vpn_type, category = label_name.split("-")
    folder = os.path.join(RAW_DIR, vpn_type, category)
    output_flows = os.path.join(FLOW_DIR, vpn_type, category)
    os.makedirs(output_flows, exist_ok=True)

    print(f"\n[+] Processing {label_name}...")
    for fname in tqdm(os.listdir(folder)):
        if not (fname.endswith(".pcap") or fname.endswith(".pcapng")):
            continue
        pcap_path = os.path.join(folder, fname)
        split_to_flows(pcap_path, output_flows)

    print(f"   -> Converting flows to byte arrays...")
    for f in tqdm(os.listdir(output_flows)):
        if not f.endswith(".pcap"):
            continue
        arr = pcap_to_bytes(os.path.join(output_flows, f))
        if arr is not None:
            data_samples.append(arr)
            label_samples.append(label_id)

print("\n[+] Saving IDX files...")
data_samples = np.array(data_samples, dtype=np.uint8)
label_samples = np.array(label_samples, dtype=np.uint8)

write_idx(data_samples, label_samples,
           os.path.join(IDX_DIR, "train-images.idx3-ubyte"),
           os.path.join(IDX_DIR, "train-labels.idx1-ubyte"))

print("✅ Preprocessing complete! Data saved in processed_data/idx/")