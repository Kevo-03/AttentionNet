from scapy.all import rdpcap
from scapy.layers.inet import IP, IP_PROTOS
from collections import Counter

# --- CONFIG ---
PCAP_FILE = "aim_chat_3acopy.pcap"  # <- change this to your file name

def analyze_protocols(pcap_path):
    packets = rdpcap(pcap_path)
    protocol_counter = Counter()
    total_packets = len(packets)

    for pkt in packets:
        if IP in pkt:  # Check if the packet contains an IP layer
            proto_num = pkt[IP].proto  # Get protocol number
            proto_name = IP_PROTOS[proto_num] if proto_num in IP_PROTOS else f"Unknown({proto_num})"
            protocol_counter[proto_name] += 1
        else:
            protocol_counter["Non-IP"] += 1  # Like ARP, LLDP, etc.

    print(f"\n📊 Protocol analysis for: {pcap_path}")
    print(f"Total packets: {total_packets}\n")
    for proto, count in protocol_counter.most_common():
        print(f"{proto:<15} {count:>10} packets  ({count / total_packets * 100:.2f}%)")

if __name__ == "__main__":
    analyze_protocols(PCAP_FILE)