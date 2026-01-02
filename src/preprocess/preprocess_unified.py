"""
Unified PCAP Preprocessing Script
==================================
This script consolidates all preprocessing variants (unidirectional/bidirectional, L7/full)
into a single configurable script.

Usage:
    python preprocess_unified.py --mode bidirectional-full
    python preprocess_unified.py --mode unidirectional-l7 --output custom_dir
    python preprocess_unified.py --mode bidirectional-l7 --test
    python preprocess_unified.py --help

Modes:
    - bidirectional-full: Bidirectional flows with full packet data (anonymized)
    - unidirectional-full: Unidirectional flows with full packet data (anonymized)
    - bidirectional-l7: Bidirectional flows with L7 payload only
    - unidirectional-l7: Unidirectional flows with L7 payload only
"""

import os
import sys
import shutil
import hashlib
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from scapy.all import IP, TCP, UDP, PcapReader  # type: ignore


class PreprocessConfig:
    """Configuration for preprocessing modes."""
    
    def __init__(self, mode, output_dir=None, test_mode=False):
        self.mode = mode
        self.test_mode = test_mode
        
        # Parse mode
        if mode not in ["bidirectional-full", "unidirectional-full", 
                       "bidirectional-l7", "unidirectional-l7"]:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: "
                           "bidirectional-full, unidirectional-full, "
                           "bidirectional-l7, unidirectional-l7")
        
        # Set direction and data extraction flags
        self.unidirectional = "unidirectional" in mode
        self.use_l7_only = "l7" in mode
        
        # Set paths
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        self.project_root = os.path.dirname(os.path.dirname(script_dir))
        self.raw_dir = os.path.join(self.project_root, "categorized_pcaps")
        
        # Set output directory
        if output_dir:
            self.output_dir = os.path.join(self.project_root, output_dir)
        else:
            default_dirs = {
                "bidirectional-full": "processed_data/memory_safe",
                "unidirectional-full": "processed_data/memory_safe_unidirectional_full",
                "bidirectional-l7": "processed_data/memory_safe_bidirectional_l7",
                "unidirectional-l7": "processed_data/memory_safe_unidirectional_l7"
            }
            self.output_dir = os.path.join(self.project_root, default_dirs[mode])
        
        # Set filenames
        mode_suffix = mode.replace("-", "_")
        self.data_filename = f"data_{mode_suffix}.npy"
        self.labels_filename = f"labels_{mode_suffix}.npy"
        
        # Image dimensions
        self.max_len = 784  # 28x28
        self.rows = 28
        self.cols = 28
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def print_config(self):
        """Print current configuration."""
        print("\n" + "=" * 70)
        print("PREPROCESSING CONFIGURATION")
        print("=" * 70)
        print(f"Mode:              {self.mode}")
        print(f"Direction:         {'Unidirectional' if self.unidirectional else 'Bidirectional'}")
        print(f"Data Type:         {'L7 Payload Only' if self.use_l7_only else 'Full Packet (Anonymized)'}")
        print(f"Output Directory:  {self.output_dir}")
        print(f"Data File:         {self.data_filename}")
        print(f"Labels File:       {self.labels_filename}")
        print(f"Image Size:        {self.rows}x{self.cols} ({self.max_len} bytes)")
        if self.test_mode:
            print(f"Test Mode:         ENABLED (4 files/category, 10 flows/file)")
        print("=" * 70 + "\n")


class PcapProcessor:
    """Main processor for PCAP files."""
    
    def __init__(self, config):
        self.config = config
        self.label_map = {
            "NonVPN": {"Chat": 0, "Email": 1, "File": 2, "Streaming": 3, "VoIP": 4, "P2P": 5},
            "VPN": {"Chat": 6, "Email": 7, "File": 8, "P2P": 9, "Streaming": 10, "VoIP": 11},
        }
    
    def _flow_key(self, pkt):
        """Generate flow key based on direction mode."""
        ip = pkt[IP]
        proto = ip.proto
        sport = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport if UDP in pkt else 0
        dport = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport if UDP in pkt else 0
        
        if self.config.unidirectional:
            return (ip.src, ip.dst, sport, dport, proto)
        
        # Bidirectional: sort IPs and ports
        sorted_ips = tuple(sorted((ip.src, ip.dst)))
        sorted_ports = tuple(sorted((sport, dport)))
        return (sorted_ips[0], sorted_ips[1], sorted_ports[0], sorted_ports[1], proto)
    
    def _application_payload_bytes(self, pkt):
        """Extract L7 payload from packet."""
        if TCP in pkt:
            payload_layer = pkt[TCP].payload
        elif UDP in pkt:
            payload_layer = pkt[UDP].payload
        else:
            payload_layer = pkt[IP].payload
        
        if not payload_layer:
            return b""
        
        try:
            return bytes(payload_layer)
        except Exception:
            return b""
    
    def _anonymized_packet_bytes(self, pkt):
        """Get full packet bytes with anonymized IPs and MACs."""
        pkt_copy = pkt.copy()
        pkt_copy[IP].src = "0.0.0.0"
        pkt_copy[IP].dst = "0.0.0.0"
        
        if "Ether" in pkt_copy:
            pkt_copy["Ether"].src = "00:00:00:00:00:00"
            pkt_copy["Ether"].dst = "00:00:00:00:00:00"
        
        return bytes(pkt_copy)
    
    def _packet_bytes(self, pkt):
        """Extract bytes from packet based on data mode."""
        if self.config.use_l7_only:
            return self._application_payload_bytes(pkt)
        return self._anonymized_packet_bytes(pkt)
    
    def _is_corrupted(self, pkt):
        """Check if packet is corrupted."""
        try:
            if IP not in pkt:
                return True
            if len(bytes(pkt)) == 0:
                return True
            return False
        except Exception:
            return True
    
    def _is_retransmission(self, pkt, seq_tracker):
        """Check if TCP packet is a retransmission."""
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
    
    def _bytes_to_image(self, buffer):
        """Convert byte buffer to image array."""
        if len(buffer) == 0:
            return None
        
        arr = np.frombuffer(buffer, dtype=np.uint8)
        if len(arr) >= self.config.max_len:
            arr = arr[:self.config.max_len]
        else:
            arr = np.pad(arr, (0, self.config.max_len - len(arr)), "constant", constant_values=0)
        return arr.reshape(self.config.rows, self.config.cols)
    
    def iter_flow_images(self, pcap_path, label, seen_hashes, max_flows=None):
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
                    if self._is_corrupted(pkt):
                        continue
                    if self._is_retransmission(pkt, seq_tracker):
                        continue
                    
                    key = self._flow_key(pkt)
                    buffer = flow_buffers[key]
                    if len(buffer) >= self.config.max_len:
                        continue
                    
                    pkt_bytes = self._packet_bytes(pkt)
                    if not pkt_bytes:
                        continue
                    
                    remaining = self.config.max_len - len(buffer)
                    buffer.extend(pkt_bytes[:remaining])
        except Exception as exc:
            print(f"[!] Could not read or process {pcap_path}: {exc}")
            flow_buffers.clear()
            seq_tracker.clear()
            return
        
        for buffer in flow_buffers.values():
            if max_flows and produced >= max_flows:
                break
            img = self._bytes_to_image(buffer)
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
    
    def process_all_pcaps(self, max_files_per_category=None, max_flows_per_file=None):
        """Process all PCAPs directly into images with memory-safe batching."""
        seen_hashes = set()
        batch_size = 1000
        batch_images = np.empty((batch_size, self.config.rows, self.config.cols), dtype=np.uint8)
        batch_labels = np.empty((batch_size,), dtype=np.int16)
        batch_pos = 0
        batch_count = 0
        total_samples = 0
        
        if max_files_per_category:
            print(f"[TEST MODE] Processing max {max_files_per_category} files per category")
        if max_flows_per_file:
            print(f"[TEST MODE] Processing max {max_flows_per_file} flows per file")
        
        for vpn_type in ["NonVPN", "VPN"]:
            for category in self.label_map[vpn_type]:
                category_dir = os.path.join(self.config.raw_dir, vpn_type, category)
                if not os.path.isdir(category_dir):
                    continue
                
                label = self.label_map[vpn_type][category]
                print(f"[Processing] {vpn_type}/{category} → Label {label}")
                pcap_files = [
                    f for f in os.listdir(category_dir) if f.endswith((".pcap", ".pcapng"))
                ]
                
                if max_files_per_category:
                    pcap_files = pcap_files[:max_files_per_category]
                
                for fname in tqdm(pcap_files):
                    pcap_path = os.path.join(category_dir, fname)
                    for img, lbl in self.iter_flow_images(
                        pcap_path, label, seen_hashes, max_flows_per_file
                    ):
                        batch_images[batch_pos] = img
                        batch_labels[batch_pos] = lbl
                        batch_pos += 1
                        total_samples += 1
                        
                        if batch_pos >= batch_size:
                            self._save_batch(batch_images[:batch_pos], batch_labels[:batch_pos], batch_count)
                            batch_count += 1
                            batch_pos = 0
        
        if batch_pos > 0:
            self._save_batch(batch_images[:batch_pos], batch_labels[:batch_pos], batch_count)
            batch_count += 1
        
        print(f"[+] Merging {batch_count} batches...")
        self._merge_batches(batch_count)
        print(f"[+] Complete! Total samples: {total_samples}")
    
    def _save_batch(self, images, labels, batch_idx):
        """Save a batch of images and labels to temporary files."""
        batch_dir = os.path.join(self.config.output_dir, "batches")
        os.makedirs(batch_dir, exist_ok=True)
        np.save(os.path.join(batch_dir, f"data_batch_{batch_idx}.npy"), images)
        np.save(os.path.join(batch_dir, f"labels_batch_{batch_idx}.npy"), labels)
    
    def _merge_batches(self, num_batches):
        """Merge all batch files into final data/label files without large allocations."""
        batch_dir = os.path.join(self.config.output_dir, "batches")
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
        
        data_path = os.path.join(self.config.output_dir, self.config.data_filename)
        labels_path = os.path.join(self.config.output_dir, self.config.labels_filename)
        final_images = np.lib.format.open_memmap(
            data_path, mode="w+", dtype=data_dtype or np.uint8, 
            shape=(total, self.config.rows, self.config.cols)
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
        print(f"[+] Saved {total} samples to {self.config.output_dir}")


class Visualizer:
    """Visualization helper for processed datasets."""
    
    def __init__(self, config):
        self.config = config
        self.label_names = {
            0: "NonVPN-Chat",
            1: "NonVPN-Email",
            2: "NonVPN-File",
            3: "NonVPN-Streaming",
            4: "NonVPN-VoIP",
            5: "NonVPN-P2P",
            6: "VPN-Chat",
            7: "VPN-Email",
            8: "VPN-File",
            9: "VPN-P2P",
            10: "VPN-Streaming",
            11: "VPN-VoIP",
        }
    
    def visualize_samples(self, samples_per_label=16):
        """Visualize sample images from the processed dataset to verify output."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[!] matplotlib not installed. Run: pip install matplotlib")
            return
        
        data_file = os.path.join(self.config.output_dir, self.config.data_filename)
        labels_file = os.path.join(self.config.output_dir, self.config.labels_filename)
        
        if not os.path.exists(data_file) or not os.path.exists(labels_file):
            print("[!] No processed data found. Run processing first.")
            return
        
        images = np.load(data_file)
        labels = np.load(labels_file)
        
        print(f"\n[+] Dataset Statistics:")
        print(f"    Total images: {len(images)}")
        print(f"    Image shape: {images[0].shape}")
        print(f"    Label distribution:")
        unique_labels, counts = np.unique(labels, return_counts=True)
        for lbl, count in zip(unique_labels, counts):
            print(f"      {self.label_names.get(lbl, f'Unknown-{lbl}')}: {count} samples")
        
        rows = 4
        cols = 4
        
        for label_idx in unique_labels:
            label_mask = labels == label_idx
            label_images = images[label_mask]
            
            if len(label_images) == 0:
                continue
            
            print(
                f"\n[+] Generating visualization for {self.label_names.get(label_idx, f'Unknown-{label_idx}')}..."
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
                f"{self.label_names.get(label_idx, f'Unknown-{label_idx}')} - Label {label_idx}",
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout()
            
            viz_path = os.path.join(
                self.config.output_dir,
                f"visualization_label_{label_idx}_{self.label_names.get(label_idx, 'Unknown').replace('-', '_')}.png",
            )
            plt.savefig(viz_path, dpi=150, bbox_inches="tight")
            print(f"    Saved: {viz_path}")
            plt.close()
        
        print(f"\n[+] All visualizations saved to: {self.config.output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified PCAP Preprocessing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bidirectional flows with full packet data (default preprocessing)
  python preprocess_unified.py --mode bidirectional-full
  
  # Unidirectional flows with L7 payload only
  python preprocess_unified.py --mode unidirectional-l7
  
  # Custom output directory
  python preprocess_unified.py --mode bidirectional-l7 --output processed_data/custom
  
  # Test mode (4 files per category, 10 flows per file)
  python preprocess_unified.py --mode unidirectional-full --test
  
  # Skip visualization
  python preprocess_unified.py --mode bidirectional-full --no-visualize
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["bidirectional-full", "unidirectional-full", 
                "bidirectional-l7", "unidirectional-l7"],
        help="Preprocessing mode"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output directory (relative to project root)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 4 files per category with 10 flows each"
    )
    
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization step"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = PreprocessConfig(args.mode, args.output, args.test)
    config.print_config()
    
    # Process PCAPs
    processor = PcapProcessor(config)
    if args.test:
        processor.process_all_pcaps(max_files_per_category=4, max_flows_per_file=10)
    else:
        processor.process_all_pcaps()
    
    # Visualize results
    if not args.no_visualize:
        print("\n" + "=" * 70)
        print("Visualizing results...")
        print("=" * 70 + "\n")
        visualizer = Visualizer(config)
        visualizer.visualize_samples()


if __name__ == "__main__":
    main()

