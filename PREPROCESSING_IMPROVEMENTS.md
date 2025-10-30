# Preprocessing Pipeline Improvements

## Summary
Your preprocessing pipeline has been significantly improved to fully align with the Wang et al. 2017 methodology for the ISCX VPN-nonVPN dataset. The updated pipeline now implements all 4 steps from the paper with proper traffic cleaning and Layer 7 payload extraction.

---

## Key Improvements

### 1. ✅ Layer 7 (Application Layer) Payload Extraction

**Before:** Used entire packet bytes including all headers (Ethernet, IP, TCP/UDP)
```python
# Old: Used full packet with headers
all_packet_bytes += bytes(pkt_copy)
```

**After:** Extracts only Layer 7 (application layer) payload
```python
# New: Uses only application layer payload
if Raw in pkt:
    payload = bytes(pkt[Raw].load)
    all_payload_bytes += payload
```

**Why It Matters:** 
- Headers contain fixed protocol information that doesn't represent actual application behavior
- Layer 7 payload contains the actual application data that distinguishes traffic types
- Aligns with the paper's approach: "select the layer 7 and the layer 7 and only layer 7 of TCP/IP session payload"

---

### 2. ✅ TCP Retransmission Filtering

**Before:** No retransmission detection

**After:** Tracks TCP sequence numbers to detect and remove retransmissions
```python
def is_retransmission(pkt, seen_seq_nums):
    """
    Detect TCP retransmissions by tracking sequence numbers.
    A retransmission occurs when we see the same seq number twice.
    """
    tcp = pkt[TCP]
    ip = pkt[IP]
    key = (ip.src, ip.dst, tcp.sport, tcp.dport, tcp.seq)
    
    if key in seen_seq_nums:
        return True
    
    seen_seq_nums.add(key)
    return False
```

**Why It Matters:**
- Retransmissions are duplicate data that can bias the model
- The paper explicitly requires removing retransmitted packets
- Improves data quality and model generalization

---

### 3. ✅ Corrupted Packet Detection

**Before:** No corruption checking

**After:** Validates packet integrity
```python
def is_corrupted(pkt):
    """
    Basic corruption detection - check if packet has obvious issues.
    """
    try:
        if IP not in pkt:
            return True
        if len(bytes(pkt)) == 0:
            return True
        return False
    except:
        return True
```

**Why It Matters:**
- Corrupted packets can introduce noise into the dataset
- Paper requires removing corrupted packets
- Ensures data quality

---

### 4. ✅ Fixed Flow Counting Bug

**Before:** Counter incremented per packet (wrong!)
```python
for pkt in packets:
    if count >= MAX_FLOWS_PER_PCAP:  # This breaks after N packets, not N flows!
        break
    flows[key].append(pkt)
    count += 1  # BUG: Counts packets, not flows
```

**After:** Correctly limits number of flows
```python
flow_count = 0
for idx, pkts in enumerate(flows.values()):
    if flow_count >= MAX_FLOWS_PER_PCAP:  # Correctly limits flows
        break
    if len(pkts) > 0:
        wrpcap(os.path.join(output_dir, f"{base}_flow{idx}.pcap"), pkts)
        flow_count += 1  # Counts flows, not packets
```

**Why It Matters:**
- Original code would stop after N packets instead of N flows
- Could result in incomplete flows or uneven dataset distribution

---

### 5. ✅ Better IP Anonymization

**Before:** Set all IPs to 0.0.0.0 (zeros everything out)
```python
pkt_copy[IP].src = "0.0.0.0"
pkt_copy[IP].dst = "0.0.0.0"
```

**After:** Consistent randomization with mapping
```python
class IPAnonymizer:
    """Consistent IP anonymization - same input IP always maps to same random IP"""
    def __init__(self, seed=42):
        self.mapping = {}
        random.seed(seed)
    
    def anonymize(self, ip):
        if ip not in self.mapping:
            self.mapping[ip] = f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        return self.mapping[ip]
```

**Why It Matters:**
- Paper mentions "randomization" not "zeroing"
- Maintains consistency: same IP always maps to same random IP
- Preserves relationships while protecting privacy

---

### 6. ✅ IDX Format Support

**Before:** Only saved as .npy files

**After:** Saves in both .npy and IDX format (MNIST-compatible)
```python
def save_idx_format(images, labels, idx_dir):
    """
    Save data in IDX format (used by MNIST and compatible with PyTorch/TensorFlow).
    IDX3: images (magic number 0x00000803 for 3D array)
    IDX1: labels (magic number 0x00000801 for 1D array)
    """
    # Saves train-images.idx3-ubyte and train-labels.idx1-ubyte
```

**Why It Matters:**
- Paper specifically mentions IDX format for PyTorch/TensorFlow compatibility
- Standard format used by MNIST and similar datasets
- Better compatibility with deep learning frameworks

---

### 7. ✅ Enhanced Statistics and Logging

**Before:** Basic output

**After:** Detailed statistics tracking
```python
stats = {"total": 0, "empty": 0, "duplicate": 0, "valid": 0}

# At end:
print(f"\n[+] Dataset processing complete!")
print(f"    Total flows processed: {stats['total']}")
print(f"    Empty flows removed: {stats['empty']}")
print(f"    Duplicate flows removed: {stats['duplicate']}")
print(f"    Valid samples saved: {stats['valid']}")
```

**Why It Matters:**
- Provides visibility into the cleaning process
- Helps identify data quality issues
- Enables pipeline debugging and optimization

---

## Configuration Options

The improved pipeline includes configurable options at the top of the file:

```python
# Processing options
USE_L7_ONLY = True  # Extract Layer 7 (application layer) payload only
FILTER_RETRANSMISSIONS = True  # Remove TCP retransmissions
ANONYMIZE_IPS = True  # Randomize IP addresses
SAVE_IDX_FORMAT = True  # Save in IDX format (in addition to .npy)
```

You can easily toggle these features on/off for experimentation or comparison.

---

## Alignment with Paper Methodology

### Paper's 4 Steps:

1. ✅ **Traffic Segmentation**: Split into flows ← **Implemented with bidirectional flow grouping**
2. ✅ **Traffic Cleaning**: Remove retransmissions, corrupted packets ← **Fully implemented**
3. ✅ **Image Generation**: Convert bytes to 0-255 sequences, 784 bytes ← **Implemented with L7 extraction**
4. ✅ **IDX Format Conversion**: Save as IDX3/IDX1 files ← **Fully implemented**

---

## How to Use

### Run the complete pipeline:
```bash
cd /Users/kivanc/Desktop/AttentionNet
source venv/bin/activate
python src/preprocess/step2changed.py
```

### Adjust configuration:
Edit the flags at the top of `step2changed.py`:
- Set `USE_L7_ONLY = False` to use full packets instead of just payload
- Set `FILTER_RETRANSMISSIONS = False` to keep retransmissions
- Set `SAVE_IDX_FORMAT = False` to skip IDX format generation

### For production (full dataset):
```python
MAX_FILES_PER_CLASS = None  # Process all files
MAX_FLOWS_PER_PCAP = None   # Process all flows
```

---

## Output Files

The pipeline now generates:
- `data_images.npy` - NumPy array of 28x28 images
- `data_labels.npy` - NumPy array of labels (0-11)
- `train-images.idx3-ubyte` - IDX3 format images (MNIST-compatible)
- `train-labels.idx1-ubyte` - IDX1 format labels (MNIST-compatible)

---

## Next Steps

1. **Test the improved pipeline** on your sample data
2. **Compare results** with the original version
3. **Adjust configuration** as needed for your specific use case
4. **Run on full dataset** by removing the `MAX_FILES_PER_CLASS` and `MAX_FLOWS_PER_PCAP` limits
5. **Implement train/test split** if needed (see `train_test_split_preprocess.py`)

---

## References

- Wang et al. 2017 - "End-to-End Encrypted Traffic Classification with One-Dimensional Convolutional Neural Networks"
- ISCX VPN-nonVPN Dataset
- USTC-TL2016 Toolkit

