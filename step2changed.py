import os
from scapy.all import rdpcap, wrpcap, IP, TCP, UDP
from collections import defaultdict
import numpy as np
import hashlib
from tqdm import tqdm

# --- CONFIG ---
RAW_DIR = "categorized_pcaps"
FLOW_DIR = "processed_test/flows"
IDX_DIR = "processed_test/idx"
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
        if count >= MAX_FLOWS_PER_PCAP:
            break

        if IP in pkt and (TCP in pkt or UDP in pkt):
            ip = pkt[IP]
            proto = ip.proto
            sorted_ips = tuple(sorted((ip.src,ip.dst)))
            sport = getattr(pkt, 'sport', 0)
            dport = getattr(pkt, 'dport', 0)
            sorted_ports = tuple(sorted((sport,dport)))
            key = (sorted_ips[0], sorted_ips[1], sorted_ports[0], sorted_ports[1], proto)
            flows[key].append(pkt)
            count += 1

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(pcap_path).rsplit(".", 1)[0]

    for idx, pkts in enumerate(flows.values()):
        wrpcap(os.path.join(output_dir, f"{base}_flow{idx}.pcap"), pkts)


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
# Step 2: Extract payload → image
# ================================
def extract_payload_array(pcap_path):
    payload = b""

    packets = rdpcap(pcap_path)
    for pkt in packets:
        if len(payload) >= MAX_LEN:
            break

        #Get only the TCP and UDP payload
        if TCP in pkt:
            payload += bytes(pkt[TCP].payload)
        elif UDP in pkt:
            payload += bytes(pkt[UDP].payload)

    # Remove empty flows
    if len(payload) == 0:
        return None

    arr = np.frombuffer(payload, dtype=np.uint8)
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