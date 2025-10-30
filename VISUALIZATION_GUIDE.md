# Visualization Script Guide

## Overview

Your visualization script has been **significantly enhanced** from a basic image grid to a comprehensive data exploration tool.

---

## What Was Improved

### ✅ Before vs After

| Feature | Before ❌ | After ✅ |
|---------|----------|---------|
| **File paths** | Wrong (data.npy, labels.npy) | Fixed (data_images.npy, data_labels.npy) |
| **Error handling** | None | File existence checks |
| **Statistics** | None | Comprehensive metrics |
| **Visualizations** | 1 grid | 2 figures (6 plots total) |
| **Class imbalance** | Not shown | Detected and warned |
| **Sample counts** | Not shown | Shown per class |
| **Average images** | Not shown | NonVPN vs VPN comparison |
| **Intensity distribution** | Not shown | Histogram |
| **Documentation** | None | Full docstrings |

---

## What It Shows Now

### Figure 1: Sample Images (15x20)
- **12 rows** (one per class: Chat, Email, File, P2P, Streaming, VoIP × 2 for VPN/NonVPN)
- **6 samples per row** (random samples from each class)
- **Labels with counts** - Shows class name and number of samples

### Figure 2: Statistics Dashboard (14x10)

#### Top Left: Class Distribution Bar Chart
- Shows number of samples per class
- **Blue bars** = NonVPN classes (0-5)
- **Red bars** = VPN classes (6-11)
- Helps identify class imbalance

#### Top Right: Pixel Intensity Histogram
- Distribution of byte values (0-255) across all images
- Sampled from 1000 images for efficiency
- Helps understand data characteristics

#### Bottom Left: Average NonVPN Images
- 3×2 grid showing average image per NonVPN category
- Reveals typical patterns in non-encrypted traffic

#### Bottom Right: Average VPN Images
- 3×2 grid showing average image per VPN category
- Reveals typical patterns in encrypted traffic

---

## Console Output

The script also prints detailed statistics to the terminal:

```
======================================================================
DATASET STATISTICS
======================================================================

Total samples: 65
Image shape: (65, 28, 28)
Data type: uint8
Value range: [0, 255]

Class distribution:
   0 - Chat (NonVPN)       :    8 samples ( 12.3%)
   1 - Email (NonVPN)      :    6 samples (  9.2%)
   2 - File (NonVPN)       :   12 samples ( 18.5%)
   3 - P2P (NonVPN)        :    0 samples (  0.0%)
   4 - Streaming (NonVPN)  :    7 samples ( 10.8%)
   5 - VoIP (NonVPN)       :    0 samples (  0.0%)
   6 - Chat (VPN)          :    9 samples ( 13.8%)
   7 - Email (VPN)         :    5 samples (  7.7%)
   8 - File (VPN)          :   10 samples ( 15.4%)
   9 - P2P (VPN)           :    3 samples (  4.6%)
  10 - Streaming (VPN)     :    4 samples (  6.2%)
  11 - VoIP (VPN)          :    1 samples (  1.5%)

Class imbalance ratio: inf
  ⚠️  Warning: Significant class imbalance detected!
======================================================================
```

---

## How to Use

### 1. First, run preprocessing:
```bash
cd /Users/kivanc/Desktop/AttentionNet
source venv/bin/activate
python src/preprocess/step2changed.py
```

### 2. Then run visualization:
```bash
python src/preprocess/visualize.py
```

### 3. Review the output:
- Check console for statistics
- Two plot windows will open
- Close both windows to exit

---

## What to Look For

### 📊 Data Quality Indicators

#### 1. **Class Balance**
- **Good:** All classes have similar counts (within 2-3x)
- **Bad:** Some classes have 0 samples or 10x+ more than others
- **Fix:** Adjust `MAX_FILES_PER_CLASS` and `MAX_FLOWS_PER_PCAP` in preprocessing

#### 2. **Pixel Intensity Distribution**
- **Good:** Values spread across 0-255 range
- **Bad:** Most values near 0 (too much zero-padding)
- **Fix:** If using `USE_L7_ONLY=True`, many flows may be empty. Try `USE_L7_ONLY=False` or process more flows

#### 3. **Visual Patterns**
- **Good:** Different classes show distinct visual patterns
- **Bad:** All images look very similar or mostly black
- **Why:** With `USE_L7_ONLY=True`, many TCP/UDP packets have no application payload (e.g., handshakes, ACKs)

#### 4. **Sample Counts**
- **Too few samples (<10 per class):** Increase `MAX_FILES_PER_CLASS` and `MAX_FLOWS_PER_PCAP`
- **Too many samples (>10,000 per class):** Consider if you need this much for testing

---

## Interpreting Average Images

### NonVPN vs VPN Comparison
The average images (bottom row of Figure 2) show typical byte patterns:

- **NonVPN (left):** May show more structured patterns (protocol headers visible)
- **VPN (right):** Should look more random/noisy (encryption effect)

**Note:** With Layer 7 extraction (`USE_L7_ONLY=True`):
- NonVPN: Application data visible (e.g., HTTP headers, plaintext)
- VPN: Encrypted application data (should be random)

**With full packets** (`USE_L7_ONLY=False`):
- More visible structure in both (protocol headers)
- VPN still has encrypted payload section

---

## Common Issues & Solutions

### ❌ "Image file not found"
**Cause:** Preprocessing hasn't been run yet
**Fix:**
```bash
python src/preprocess/step2changed.py
```

### ❌ "No samples for class X"
**Cause:** 
- Not enough files processed
- Class has no data in sample set
- Empty flows due to `USE_L7_ONLY=True`

**Fix:**
- Increase `MAX_FILES_PER_CLASS` in `step2changed.py`
- Check if category folder exists and has .pcap files
- Try `USE_L7_ONLY=False` to include packets without payload

### ❌ "Class imbalance ratio: inf"
**Cause:** Some classes have 0 samples

**Fix:**
- Process more files
- Check data directory structure
- Verify all categories have PCAP files

### ❌ Most images are black/empty
**Cause:** Using Layer 7 only, but most packets are TCP handshakes/ACKs with no payload

**Fix:** Two options:
1. **Keep L7, process more flows:** Increase `MAX_FLOWS_PER_PCAP` (more flows = more with payload)
2. **Use full packets:** Set `USE_L7_ONLY=False` in `step2changed.py`

---

## Configuration Changes Detected

I noticed you changed these settings:
```python
USE_L7_ONLY = False     # Using full packets instead of L7 only
SAVE_IDX_FORMAT = False  # Not saving IDX format
```

### Impact on Visualization:

**`USE_L7_ONLY = False`:**
- ✅ **More data** - Will include all packets (headers + payload)
- ✅ **Fewer empty flows** - TCP handshakes will contribute data
- ⚠️ **Less paper-aligned** - Paper uses Layer 7 only
- 🔍 **Different patterns** - Images will show protocol headers

**Expected results:**
- More samples per class
- Less class imbalance
- Images will have more visible structure (headers)
- May have slightly lower accuracy (headers add noise)

---

## Recommended Experiments

### Experiment 1: Compare L7 vs Full Packet
1. Run with `USE_L7_ONLY = True`, visualize
2. Run with `USE_L7_ONLY = False`, visualize
3. Compare: Which shows clearer class distinctions?

### Experiment 2: Sample Size Impact
1. Process with `MAX_FLOWS_PER_PCAP = 10` (current)
2. Process with `MAX_FLOWS_PER_PCAP = 50`
3. Process with `MAX_FLOWS_PER_PCAP = 100`
4. Check how class balance improves

### Experiment 3: Average Image Patterns
- Look at average NonVPN vs VPN images
- Can you visually distinguish encrypted vs non-encrypted?
- Do different traffic types (Chat vs Streaming) show distinct patterns?

---

## Next Steps

1. ✅ **Run visualization** to see current data quality
2. ⬜ **Check class balance** - adjust preprocessing if needed
3. ⬜ **Verify patterns** - ensure classes are visually distinct
4. ⬜ **Document findings** - note which settings work best
5. ⬜ **Prepare for training** - once data looks good, build your model

---

## Additional Features You Could Add

If you want to extend the visualization further:

1. **Save plots to file** instead of just displaying
2. **Confusion matrix** of visually similar classes
3. **t-SNE visualization** of image embeddings
4. **Per-class pixel intensity distributions**
5. **Entropy analysis** (VPN should have higher entropy)
6. **Packet size distributions per class**

Let me know if you'd like any of these additions!

