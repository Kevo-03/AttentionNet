import os
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
# We no longer need a separate FLOW_DIR
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data/idx")
MAX_LEN = 784  # 28x28
ROWS, COLS = 28, 28
MAX_FILES_PER_CLASS = 4
MAX_SESSIONS_PER_CLASS = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================
# Helper Functions
# ================================

def is_retransmission(pkt, seen_seq_nums):
    if TCP not in pkt:
        return False
    tcp = pkt[TCP]
    ip = pkt[IP]
    key = (ip.src, ip.dst, tcp.sport, tcp.dport, tcp.seq)
    if key in seen_seq_nums:
        return True
    seen_seq_nums.add(key)
    return False

def is_corrupted(pkt):
    try:
        if IP not in pkt:
            return True
        if len(bytes(pkt)) == 0:
            return True
        return False
    except:
        return True

# ================================
# New Single-Pass Processing
# ================================

def process_pcap_file(pcap_path, max_flows=None):
    """
    Reads a single large pcap file, processes all its flows in a memory-efficient
    way, and returns a list of processed 28x28 images.
    
    Args:
        max_flows: Maximum number of flows to process (None = all)
    """
    flows = defaultdict(list)  # Use list instead of bytes for O(1) appends
    seen_seq_nums = set()
    
    try:
        with PcapReader(pcap_path) as pcap_reader:
            for pkt in pcap_reader:
                
                if is_corrupted(pkt):
                    continue
                if is_retransmission(pkt, seen_seq_nums):
                    continue
                
                if IP in pkt and (TCP in pkt or UDP in pkt):
                    ip = pkt[IP]
                    proto = ip.proto
                    sorted_ips = tuple(sorted((ip.src, ip.dst)))
                    sport = getattr(pkt, 'sport', 0)
                    dport = getattr(pkt, 'dport', 0)
                    sorted_ports = tuple(sorted((sport, dport)))
                    key = (sorted_ips[0], sorted_ips[1], sorted_ports[0], sorted_ports[1], proto)
                    
                    # Anonymize and store as list of bytes
                    pkt_copy = pkt.copy()
                    pkt_copy[IP].src = "0.0.0.0"
                    pkt_copy[IP].dst = "0.0.0.0"
                    if "Ether" in pkt_copy:
                        pkt_copy["Ether"].src = "00:00:00:00:00:00"
                        pkt_copy["Ether"].dst = "00:00:00:00:00:00"
                    
                    flows[key].append(bytes(pkt_copy))  # Append to list - O(1)

    except Exception as e:
        print(f"[!] Error processing {pcap_path}: {e}")
        return [] # Return an empty list on error

    # --- Convert flow byte-lists to images ---
    images = []
    flow_count = 0
    
    for flow_packet_bytes_list in flows.values():
        if max_flows and flow_count >= max_flows:
            break
            
        # Concatenate bytes ONCE at the end - O(n) instead of O(n²)
        flow_bytes = b''.join(flow_packet_bytes_list)
        
        if len(flow_bytes) == 0:
            continue
        
        # Truncate to MAX_LEN if needed
        if len(flow_bytes) > MAX_LEN:
            flow_bytes = flow_bytes[:MAX_LEN]
            
        arr = np.frombuffer(flow_bytes, dtype=np.uint8)
        
        if len(arr) >= MAX_LEN:
            arr = arr[:MAX_LEN]
        else:
            arr = np.pad(arr, (0, MAX_LEN - len(arr)), 'constant', constant_values=0)
        
        images.append(arr.reshape(ROWS, COLS))
        flow_count += 1
        
    return images


def run_preprocessing_pipeline():
    """
    Replaces both Step 1 and Step 2.
    Processes all pcaps and saves the final .npy files.
    """
    label_map = {
        "NonVPN": {"Chat": 0, "Email": 1, "File": 2, "Streaming": 3, "VoIP": 4},
        "VPN":    {"Chat": 5, "Email": 6, "File": 7, "P2P": 8, "Streaming": 9, "VoIP": 10}
    }

    all_images, all_labels = [], []
    seen_hashes = set()

    for vpn_type in ["VPN", "NonVPN"]:
        category_list = os.listdir(os.path.join(RAW_DIR, vpn_type))
        category_list.sort()
        
        for category in category_list:
            category_dir = os.path.join(RAW_DIR, vpn_type, category)
            if not os.path.isdir(category_dir):
                continue
            
            if category not in label_map[vpn_type]:
                continue # Skip unmapped categories like P2P/NonVPN

            label = label_map[vpn_type][category]
            print(f"[+] Processing {vpn_type}/{category} → Label {label}")

            session_count_for_this_class = 0
            
            file_list = os.listdir(category_dir)
            file_list.sort()

            for i,fname in enumerate(tqdm(file_list, desc=f"  Files in {category}")):
                
                if i >= MAX_FILES_PER_CLASS:
                    print(f"    (Reached test limit of {MAX_FILES_PER_CLASS} files)")
                    break

                if session_count_for_this_class >= MAX_SESSIONS_PER_CLASS:
                    print(f"    (Reached session limit of {MAX_SESSIONS_PER_CLASS})")
                    break # Stop processing files for this category

                if fname.endswith((".pcap", ".pcapng")):
                    
                    # Process one large pcap file
                    pcap_path = os.path.join(category_dir, fname)
                    remaining_sessions = MAX_SESSIONS_PER_CLASS - session_count_for_this_class
                    images_from_file = process_pcap_file(pcap_path, max_flows=remaining_sessions)
                    
                    # Add the results to our main dataset
                    for img in images_from_file:
                        digest = hashlib.sha1(img.tobytes()).hexdigest()
                        if digest not in seen_hashes:
                            seen_hashes.add(digest)
                            all_images.append(img)
                            all_labels.append(label)
                            session_count_for_this_class += 1  # INCREMENT HERE!
                            
                            if session_count_for_this_class >= MAX_SESSIONS_PER_CLASS:
                                break

    np.save(os.path.join(OUTPUT_DIR, "data_gemini.npy"), np.array(all_images))
    np.save(os.path.join(OUTPUT_DIR, "labels_gemini.npy"), np.array(all_labels))
    print(f"\n[+] Preprocessing complete!")
    print(f"[+] Cleaned dataset saved. Total samples: {len(all_images)}")
    print(f"[+] Output directory: {OUTPUT_DIR}")


# ======================================================
# Run Steps
# ======================================================
if __name__ == "__main__":
    # Make sure your project structure is correct (script in src/preprocessing)
    run_preprocessing_pipeline()