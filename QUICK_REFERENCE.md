# Quick Reference: Preprocessing Pipeline

## What Changed?

Your preprocessing pipeline has been upgraded from a **basic implementation** to a **paper-compliant implementation** that fully follows the Wang et al. 2017 methodology.

---

## Before vs After Comparison

| Feature | Before ❌ | After ✅ |
|---------|----------|---------|
| **Payload Extraction** | Full packet with all headers | Layer 7 (application layer) only |
| **Retransmission Handling** | No filtering | TCP sequence number tracking |
| **Corrupted Packets** | No checking | Validates packet integrity |
| **Flow Counting** | Bug: counted packets not flows | Fixed: correctly counts flows |
| **IP Anonymization** | Zeros (0.0.0.0) | Consistent randomization |
| **Output Format** | Only .npy files | .npy + IDX format (MNIST-style) |
| **Statistics** | Basic logging | Detailed metrics tracking |
| **Documentation** | Minimal | Comprehensive docstrings |

---

## Critical Fixes

### 🐛 Bug Fix #1: Flow Counting
**Impact: HIGH** - This was causing incorrect dataset generation

**Before:**
```python
count = 0
for pkt in packets:
    if count >= MAX_FLOWS_PER_PCAP:  # ❌ Breaks after N packets!
        break
    flows[key].append(pkt)
    count += 1  # ❌ Counting packets instead of flows
```

**After:**
```python
flow_count = 0
for idx, pkts in enumerate(flows.values()):
    if flow_count >= MAX_FLOWS_PER_PCAP:  # ✅ Correctly limits flows
        break
    wrpcap(...)
    flow_count += 1  # ✅ Counting flows
```

### 🎯 Improvement #1: Layer 7 Extraction
**Impact: HIGH** - This is the most important change for model accuracy

**Before:**
```python
all_packet_bytes += bytes(pkt_copy)  # ❌ Includes headers (Eth, IP, TCP/UDP)
```

**After:**
```python
if Raw in pkt:
    payload = bytes(pkt[Raw].load)  # ✅ Only application data
    all_payload_bytes += payload
```

**Why it matters:** Headers are protocol overhead. Application payload is what distinguishes Chat from Streaming, etc.

### 🧹 Improvement #2: Traffic Cleaning
**Impact: MEDIUM** - Improves data quality

**Added:**
- Retransmission detection (TCP seq numbers)
- Corrupted packet filtering
- Duplicate removal (already existed, kept)

---

## Configuration Guide

### Location
All settings are at the top of `step2changed.py`:

```python
# Processing options
USE_L7_ONLY = True               # ← Layer 7 only (recommended)
FILTER_RETRANSMISSIONS = True    # ← Remove retrans (recommended)
ANONYMIZE_IPS = True              # ← Randomize IPs (recommended)
SAVE_IDX_FORMAT = True            # ← IDX output (recommended)
```

### Testing vs Production

**Testing Mode (current):**
```python
MAX_FILES_PER_CLASS = 4   # Process only 4 files per category
MAX_FLOWS_PER_PCAP = 10   # Split only 10 flows per file
```

**Production Mode:**
```python
MAX_FILES_PER_CLASS = None   # Process ALL files
MAX_FLOWS_PER_PCAP = None    # Split ALL flows
```

---

## Running the Pipeline

### 1. Activate virtual environment:
```bash
cd /Users/kivanc/Desktop/AttentionNet
source venv/bin/activate
```

### 2. Run preprocessing:
```bash
python src/preprocess/step2changed.py
```

### 3. Check output:
```bash
ls -lh processed_test/idx/
# Should see:
# - data_images.npy
# - data_labels.npy
# - train-images.idx3-ubyte
# - train-labels.idx1-ubyte
```

---

## Output Format

### NumPy Files (.npy)
- `data_images.npy`: Shape (N, 28, 28), dtype=uint8
- `data_labels.npy`: Shape (N,), dtype=uint8, values 0-11

### IDX Files (MNIST format)
- `train-images.idx3-ubyte`: 3D image array
- `train-labels.idx1-ubyte`: 1D label array

### Label Mapping
```
NonVPN:
  0 = Chat
  1 = Email
  2 = File
  3 = P2P
  4 = Streaming
  5 = VoIP

VPN:
  6 = Chat
  7 = Email
  8 = File
  9 = P2P
  10 = Streaming
  11 = VoIP
```

---

## Expected Output (Testing Mode)

```
======================================================================
PCAP PREPROCESSING PIPELINE
======================================================================
Configuration:
  - Layer 7 Payload Only: True
  - Filter Retransmissions: True
  - Anonymize IPs: True
  - Save IDX Format: True
  - Max Length: 784 bytes (28x28)
  - Max Files per Class: 4 (testing mode)
  - Max Flows per PCAP: 10 (testing mode)
...
[+] Dataset processing complete!
    Total flows processed: 80
    Empty flows removed: 12
    Duplicate flows removed: 3
    Valid samples saved: 65
```

---

## Troubleshooting

### "Could not read *.pcap"
- Check that PCAP files exist in `categorized_pcaps/`
- Verify Scapy can read the files: `scapy` → `rdpcap('path/to/file.pcap')`

### "No module named scapy"
- Activate venv: `source venv/bin/activate`
- Install: `pip install scapy`

### Empty flows / Low valid samples
- This is normal! Many flows have no Layer 7 payload (e.g., TCP handshakes)
- If you want more samples, increase `MAX_FILES_PER_CLASS` and `MAX_FLOWS_PER_PCAP`
- Or set `USE_L7_ONLY = False` to use full packets (less accurate but more data)

---

## Performance Tips

### Speed up processing:
1. Use fewer files: `MAX_FILES_PER_CLASS = 2`
2. Use fewer flows: `MAX_FLOWS_PER_PCAP = 5`
3. Skip IDX: `SAVE_IDX_FORMAT = False`

### Get more samples:
1. Process more files: `MAX_FILES_PER_CLASS = None`
2. Process more flows: `MAX_FLOWS_PER_PCAP = None`
3. Use full packets: `USE_L7_ONLY = False` (trades accuracy for quantity)

---

## Next Steps

1. ✅ **Test with sample data** (current settings are good for this)
2. ⬜ **Review statistics** to understand data quality
3. ⬜ **Compare L7 vs full packet** performance (experiment!)
4. ⬜ **Run on full dataset** for training (remove limits)
5. ⬜ **Implement train/test split** (80/20 or 70/30)
6. ⬜ **Build your CNN model** using the processed data

---

## Questions?

- Check `PREPROCESSING_IMPROVEMENTS.md` for detailed explanations
- Review code comments in `step2changed.py`
- Compare with the paper methodology (images provided)

