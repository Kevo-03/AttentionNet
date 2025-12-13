import os
import shutil
import hashlib
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from scapy.all import IP, TCP, UDP, PcapReader  # type: ignore

# --- CONFIG ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
RAW_DIR = os.path.join(PROJECT_ROOT, "own_captures")
IDX_DIR = os.path.join(PROJECT_ROOT, "processed_data/own_captures_test")
MAX_LEN = 784  # 28x28
ROWS, COLS = 28, 28

os.makedirs(IDX_DIR, exist_ok=True)


# ================================
# Helpers
# ================================
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
    except Exception:
        return True


def _is_retransmission(pkt, seq_tracker):
    if TCP not in pkt:
        return False

    tcp = pkt[TCP]
    ip = pkt[IP]
    conn_key = (ip.src, ip.dst, tcp.sport, tcp.dport)
    seq = tcp.seq

    seen = seq_tracker[conn_key]
    if seq in seen:
        return True
    seen.add(seq)
    return False


def _bytes_to_image(buffer):
    if len(buffer) == 0:
        return None

    arr = np.frombuffer(buffer, dtype=np.uint8)
    if len(arr) >= MAX_LEN:
        arr = arr[:MAX_LEN]
    else:
        arr = np.pad(arr, (0, MAX_LEN - len(arr)), "constant", constant_values=0)
    return arr.reshape(ROWS, COLS)


# ================================
# Single-pass: PCAP → flows → images
# ================================
def iter_flow_images(pcap_path, label, seen_hashes, max_flows=None):
    """
    Stream images from a PCAP file without holding Scapy packets in memory.

    Yields:
        (image, label) tuples for deduplicated flows.
    """
    flow_buffers = defaultdict(bytearray)
    seq_tracker = defaultdict(set)
    produced = 0

    try:
        with PcapReader(pcap_path) as packets:
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
                if not pkt_bytes:
                    continue

                remaining = MAX_LEN - len(buffer)
                buffer.extend(pkt_bytes[:remaining])
    except Exception as exc:
        print(f"[!] Could not read or process {pcap_path}: {exc}")
        flow_buffers.clear()
        seq_tracker.clear()
        return

    for buffer in flow_buffers.values():
        if max_flows and produced >= max_flows:
            break
        img = _bytes_to_image(buffer)
        if img is None:
            continue
        digest = hashlib.sha1(img.tobytes()).digest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        yield img, label
        produced += 1

    flow_buffers.clear()
    seq_tracker.clear()


def process_all_pcaps(max_files_per_category=None, max_flows_per_file=None):
    """
    Process all PCAPs from own_captures directory directly into images with memory-safe batching.
    """
    # Label mapping for own_captures (NonVPN only)
    # Maps category folder names to labels matching the training data
    label_map = {
        "Chat_test": 0,      # NonVPN-Chat
        "Email_test": 1,     # NonVPN-Email
        "File_test": 2,      # NonVPN-File
        "P2P_test": 3,       # NonVPN-P2P
        "Streaming_test": 4, # NonVPN-Streaming
        "VoIP_test": 5,      # NonVPN-VoIP
    }

    seen_hashes = set()
    batch_size = 1000
    batch_images = np.empty((batch_size, ROWS, COLS), dtype=np.uint8)
    batch_labels = np.empty((batch_size,), dtype=np.int16)
    batch_pos = 0
    batch_count = 0
    total_samples = 0

    if max_files_per_category:
        print(f"[TEST MODE] Processing max {max_files_per_category} files per category")
    if max_flows_per_file:
        print(f"[TEST MODE] Processing max {max_flows_per_file} flows per file")

    for category in label_map:
        category_dir = os.path.join(RAW_DIR, category)
        if not os.path.isdir(category_dir):
            print(f"[!] Directory not found: {category_dir}")
            continue

        label = label_map[category]
        print(f"[Processing] {category} → Label {label}")
        pcap_files = [
            f for f in os.listdir(category_dir) if f.endswith((".pcap", ".pcapng"))
        ]

        if max_files_per_category:
            pcap_files = pcap_files[:max_files_per_category]

        for fname in tqdm(pcap_files):
            pcap_path = os.path.join(category_dir, fname)
            for img, lbl in iter_flow_images(
                pcap_path, label, seen_hashes, max_flows_per_file
            ):
                batch_images[batch_pos] = img
                batch_labels[batch_pos] = lbl
                batch_pos += 1
                total_samples += 1

                if batch_pos >= batch_size:
                    _save_batch(batch_images[:batch_pos], batch_labels[:batch_pos], batch_count)
                    batch_count += 1
                    batch_pos = 0

    if batch_pos > 0:
        _save_batch(batch_images[:batch_pos], batch_labels[:batch_pos], batch_count)
        batch_count += 1

    print(f"[+] Merging {batch_count} batches...")
    _merge_batches(batch_count)
    print(f"[+] Complete! Total samples: {total_samples}")


def _save_batch(images, labels, batch_idx):
    """Save a batch of images and labels to temporary files."""
    batch_dir = os.path.join(IDX_DIR, "batches")
    os.makedirs(batch_dir, exist_ok=True)
    np.save(os.path.join(batch_dir, f"data_batch_{batch_idx}.npy"), images)
    np.save(os.path.join(batch_dir, f"labels_batch_{batch_idx}.npy"), labels)


def _merge_batches(num_batches):
    """Merge all batch files into final data.npy and labels.npy files without large allocations."""
    batch_dir = os.path.join(IDX_DIR, "batches")
    if not os.path.isdir(batch_dir):
        print("[!] No batches to merge.")
        return

    batch_sizes = []
    data_dtype = None
    labels_dtype = None

    for i in range(num_batches):
        labels_file = os.path.join(batch_dir, f"labels_batch_{i}.npy")
        data_file = os.path.join(batch_dir, f"data_batch_{i}.npy")
        if not (os.path.exists(labels_file) and os.path.exists(data_file)):
            continue

        labels = np.load(labels_file, mmap_mode="r")
        data = np.load(data_file, mmap_mode="r")

        batch_sizes.append(labels.shape[0])
        if data_dtype is None:
            data_dtype = data.dtype
        if labels_dtype is None:
            labels_dtype = labels.dtype

    total = sum(batch_sizes)
    if total == 0:
        print("[!] No data found in batches.")
        shutil.rmtree(batch_dir)
        return

    data_path = os.path.join(IDX_DIR, "data_memory_safe.npy")
    labels_path = os.path.join(IDX_DIR, "labels_memory_safe.npy")
    final_images = np.lib.format.open_memmap(
        data_path, mode="w+", dtype=data_dtype or np.uint8, shape=(total, ROWS, COLS)
    )
    final_labels = np.lib.format.open_memmap(
        labels_path, mode="w+", dtype=labels_dtype or np.int16, shape=(total,)
    )

    offset = 0
    for i in range(num_batches):
        labels_file = os.path.join(batch_dir, f"labels_batch_{i}.npy")
        data_file = os.path.join(batch_dir, f"data_batch_{i}.npy")
        if not (os.path.exists(labels_file) and os.path.exists(data_file)):
            continue

        labels = np.load(labels_file, mmap_mode="r")
        data = np.load(data_file, mmap_mode="r")
        count = labels.shape[0]

        final_labels[offset : offset + count] = labels[:count]
        final_images[offset : offset + count] = data[:count]
        offset += count

    del final_images
    del final_labels
    shutil.rmtree(batch_dir)
    print(f"[+] Saved {total} samples to {IDX_DIR}")


def visualize_samples(samples_per_label=16):
    """
    Visualize sample images from the processed dataset to verify output.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not installed. Run: pip install matplotlib")
        return

    data_file = os.path.join(IDX_DIR, "data_memory_safe.npy")
    labels_file = os.path.join(IDX_DIR, "labels_memory_safe.npy")

    if not os.path.exists(data_file) or not os.path.exists(labels_file):
        print("[!] No processed data found. Run processing first.")
        return

    images = np.load(data_file)
    labels = np.load(labels_file)

    label_names = {
        0: "NonVPN-Chat",
        1: "NonVPN-Email",
        2: "NonVPN-File",
        3: "NonVPN-P2P",
        4: "NonVPN-Streaming",
        5: "NonVPN-VoIP",
    }

    print(f"\n[+] Dataset Statistics:")
    print(f"    Total images: {len(images)}")
    print(f"    Image shape: {images[0].shape}")
    print(f"    Label distribution:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for lbl, count in zip(unique_labels, counts):
        print(f"      {label_names.get(lbl, f'Unknown-{lbl}')}: {count} samples")

    rows = 4
    cols = 4

    for label_idx in unique_labels:
        label_mask = labels == label_idx
        label_images = images[label_mask]

        if len(label_images) == 0:
            continue

        print(
            f"\n[+] Generating visualization for {label_names.get(label_idx, f'Unknown-{label_idx}')}..."
        )

        num_to_show = min(samples_per_label, len(label_images))

        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        axes = axes.flatten()

        for idx in range(rows * cols):
            if idx < num_to_show:
                axes[idx].imshow(label_images[idx], cmap="gray")
                axes[idx].set_title(f"Sample {idx}", fontsize=8)
                axes[idx].axis("off")
            else:
                axes[idx].axis("off")

        fig.suptitle(
            f"{label_names.get(label_idx, f'Unknown-{label_idx}')} - Label {label_idx}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        viz_path = os.path.join(
            IDX_DIR,
            f"visualization_label_{label_idx}_{label_names.get(label_idx, 'Unknown').replace('-', '_')}.png",
        )
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        print(f"    Saved: {viz_path}")
        plt.close()

    print(f"\n[+] All visualizations saved to: {IDX_DIR}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("\n" + "=" * 60)
        print("TEST MODE: Processing 4 files per category, 10 flows max")
        print("=" * 60 + "\n")
        process_all_pcaps(max_files_per_category=4, max_flows_per_file=10)
        print("\n" + "=" * 60)
        print("Visualizing results...")
        print("=" * 60 + "\n")
        visualize_samples()
    else:
        process_all_pcaps()
        print("\n" + "=" * 60)
        print("Visualizing results...")
        print("=" * 60 + "\n")
        visualize_samples()


