# Detailed Labeling - Quick Start Guide

This is a quick reference to get started with the detailed labeling system.

## 🚀 Quick Start (3 Steps)

### Step 1: Activate Environment
```bash
cd /Users/kivanc/Desktop/AttentionNet
source venv/bin/activate
```

### Step 2: Run Preprocessing with Detailed Labels
```bash
python src/preprocess/detailed_preprocess.py
```

**What this does:**
- Processes PCAP files from `categorized_pcaps/`
- Splits them into flows
- Extracts 28x28 packet images
- Creates detailed labels: `"VPN_AppType_AppName"` (e.g., `"NonVPN_Streaming_netflix"`)
- Saves to `processed_detailed/idx/`

**Time:** ~5-10 minutes (with test settings)

### Step 3: Generate Visualizations
```bash
python src/preprocess/detailed_visualize.py
```

**What this does:**
- Creates 7 visualization images + 1 summary report
- Saves to `processed_detailed/visualizations/`

**Time:** ~1-2 minutes

## 📊 View Results

After running, open these files:

```bash
# Open visualization folder
open processed_detailed/visualizations/

# View summary report
cat processed_detailed/visualizations/detailed_summary_report.txt
```

## 📁 Generated Files

### Data Files (`processed_detailed/idx/`)
```
data_images.npy           # 28x28 images (N, 28, 28)
data_labels_detailed.npy  # Full labels: "VPN_Chat_facebook"
data_app_types.npy        # Just app types: "Chat", "Streaming"
data_apps.npy             # Just apps: "facebook", "netflix"
data_vpn_types.npy        # Just VPN: "VPN", "NonVPN"
```

### Visualizations (`processed_detailed/visualizations/`)
```
sample_images_detailed.png              # Sample images per label
distribution_by_app_type.png            # Bar chart by app type
distribution_by_app.png                 # Bar chart by specific app
vpn_vs_nonvpn_comparison.png            # VPN comparison
heatmap_app_vs_apptype.png              # Relationship heatmap
average_patterns_by_app_type.png        # Traffic patterns by type
average_patterns_by_app.png             # Traffic patterns by app
detailed_summary_report.txt             # Statistical summary
```

## 🔬 Explore the Data

### Example 1: Load and Explore
```bash
python src/preprocess/detailed_analysis_example.py
```

This script demonstrates 7 different use cases for the detailed data.

### Example 2: Use in Python
```python
import numpy as np

# Load data
images = np.load("processed_detailed/idx/data_images.npy")
labels = np.load("processed_detailed/idx/data_labels_detailed.npy")
app_types = np.load("processed_detailed/idx/data_app_types.npy")
apps = np.load("processed_detailed/idx/data_apps.npy")
vpn_types = np.load("processed_detailed/idx/data_vpn_types.npy")

# Explore
print(f"Total samples: {len(images)}")
print(f"Unique labels: {len(np.unique(labels))}")
print(f"Sample label: {labels[0]}")

# Filter by app type (e.g., Streaming)
streaming_mask = app_types == "Streaming"
streaming_images = images[streaming_mask]
streaming_apps = apps[streaming_mask]

print(f"Streaming samples: {len(streaming_images)}")
print(f"Streaming apps: {np.unique(streaming_apps)}")
```

## ⚙️ Configuration

Edit `src/preprocess/detailed_preprocess.py` to change settings:

```python
MAX_FILES_PER_CLASS = 4    # Process more files (or set to None for all)
MAX_FLOWS_PER_PCAP = 10    # Process more flows per file
MAX_LEN = 784              # Image size (28x28 = 784)
```

**For full dataset:**
```python
MAX_FILES_PER_CLASS = None  # Process all files
MAX_FLOWS_PER_PCAP = 100    # More flows
```

## 📊 Understanding Labels

### Label Format
```
<VPNType>_<AppType>_<AppName>
```

### Examples
```python
"NonVPN_Chat_facebook"     # Facebook chat without VPN
"VPN_Streaming_netflix"    # Netflix with VPN
"NonVPN_File_skype"        # Skype file transfer without VPN
"VPN_VoIP_skype"           # Skype voice call with VPN
```

### App Name Extraction

The system automatically extracts app names from filenames:

| Filename | → | App Name |
|----------|---|----------|
| `facebook_chat_4a.pcap` | → | `facebook` |
| `vpn_netflix_A.pcap` | → | `netflix` |
| `skype_file1.pcap` | → | `skype` |
| `youtube1.pcap` | → | `youtube` |

## 🎯 Use Cases

### 1. Binary VPN Detection
```python
from sklearn.preprocessing import LabelEncoder

vpn_types = np.load("processed_detailed/idx/data_vpn_types.npy")
y = LabelEncoder().fit_transform(vpn_types)
# Classes: VPN (1) vs NonVPN (0)
```

### 2. App Type Classification
```python
app_types = np.load("processed_detailed/idx/data_app_types.npy")
y = LabelEncoder().fit_transform(app_types)
# Classes: Chat, Email, File, P2P, Streaming, VoIP
```

### 3. Specific App Classification
```python
apps = np.load("processed_detailed/idx/data_apps.npy")
y = LabelEncoder().fit_transform(apps)
# Classes: facebook, netflix, skype, youtube, etc.
```

### 4. Fine-grained (All Details)
```python
labels = np.load("processed_detailed/idx/data_labels_detailed.npy")
y = LabelEncoder().fit_transform(labels)
# Classes: NonVPN_Chat_facebook, VPN_Streaming_netflix, etc.
```

## 📈 Next Steps

1. **View visualizations** to understand your data
2. **Run the example script** to see different use cases
3. **Choose your task** (binary, multi-class, or hierarchical)
4. **Build your model** (CNN, Random Forest, etc.)
5. **Compare results** at different granularity levels

## 📚 More Information

- `DETAILED_LABELING_README.md` - Full documentation
- `LABELING_COMPARISON.md` - Compare with original system
- `src/preprocess/detailed_analysis_example.py` - Code examples

## 🐛 Troubleshooting

### Issue: "No such file or directory"
**Solution:** Run preprocessing first:
```bash
python src/preprocess/detailed_preprocess.py
```

### Issue: "Not enough samples"
**Solution:** Increase `MAX_FILES_PER_CLASS` in `detailed_preprocess.py`:
```python
MAX_FILES_PER_CLASS = 10  # or None for all files
```

### Issue: Visualization script fails
**Solution:** Make sure preprocessing completed successfully and generated the `.npy` files in `processed_detailed/idx/`

## ✅ Checklist

- [ ] Activated virtual environment
- [ ] Ran `detailed_preprocess.py`
- [ ] Checked that `processed_detailed/idx/` contains 5 `.npy` files
- [ ] Ran `detailed_visualize.py`
- [ ] Checked that `processed_detailed/visualizations/` contains 8 files
- [ ] Viewed the visualizations
- [ ] Ran `detailed_analysis_example.py` to see usage examples
- [ ] Ready to train models!

## 🎉 You're Ready!

You now have:
- ✅ Processed data with detailed labels
- ✅ Multiple label granularities (VPN, app type, specific app)
- ✅ Comprehensive visualizations
- ✅ Example code for different use cases

Happy analyzing! 🚀

