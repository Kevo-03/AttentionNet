# Labeling Systems Comparison

This document compares the different labeling approaches used in the AttentionNet project.

## Overview of Labeling Systems

### 1. Original System (`step2changed.py`)

**Approach:** Numeric labels based on VPN type and application category

**Label Format:** Integer (0-11)

**Label Mapping:**
```
NonVPN:
  - 0: Chat
  - 1: Email  
  - 2: File
  - 3: P2P
  - 4: Streaming
  - 5: VoIP

VPN:
  - 6: Chat
  - 7: Email
  - 8: File
  - 9: P2P
  - 10: Streaming
  - 11: VoIP
```

**Output Files:**
- `processed_test/idx/data.npy` - Images
- `processed_test/idx/labels.npy` - Numeric labels

**Use Cases:**
- Simple classification tasks
- VPN detection combined with app type classification
- Lightweight storage and processing

---

### 2. Detailed System (`detailed_preprocess.py`)

**Approach:** String labels combining VPN type, application type, and specific application

**Label Format:** String `"VPNType_AppType_AppName"`

**Example Labels:**
```
NonVPN_Chat_facebook
NonVPN_Chat_aim
NonVPN_Chat_skype
NonVPN_Streaming_netflix
NonVPN_Streaming_youtube
NonVPN_File_skype
VPN_Chat_facebook
VPN_Streaming_netflix
... etc.
```

**Output Files:**
- `processed_detailed/idx/data_images.npy` - Images
- `processed_detailed/idx/data_labels_detailed.npy` - Full detailed labels
- `processed_detailed/idx/data_app_types.npy` - Just app types
- `processed_detailed/idx/data_apps.npy` - Just app names
- `processed_detailed/idx/data_vpn_types.npy` - Just VPN types

**Use Cases:**
- Fine-grained classification of specific applications
- Multi-level hierarchical classification
- Pattern analysis at different granularity levels
- Transfer learning experiments
- Feature importance analysis per application

---

## Comparison Table

| Feature | Original System | Detailed System |
|---------|----------------|-----------------|
| **Label Type** | Integer (0-11) | String (e.g., "VPN_Chat_facebook") |
| **Granularity** | VPN + App Type | VPN + App Type + Specific App |
| **Total Classes** | 12 | ~30-50 (depends on data) |
| **Storage Size** | Smaller | Larger (multiple files) |
| **Flexibility** | Fixed categories | Multi-level analysis possible |
| **Visualization** | Basic by category | Detailed by app and type |
| **Best For** | Simple classification | Research & detailed analysis |

---

## When to Use Each System

### Use Original System When:
- ✅ You need simple VPN detection + app type classification
- ✅ You want faster training and inference
- ✅ Storage and memory are limited
- ✅ You only care about general traffic categories
- ✅ You're building production systems with 12 classes

### Use Detailed System When:
- ✅ You need to identify specific applications (facebook vs skype)
- ✅ You want to analyze patterns within app types
- ✅ You're doing research on traffic fingerprinting
- ✅ You need multi-level classification (hierarchical)
- ✅ You want to study VPN effects on specific apps
- ✅ You're building explainable AI systems

---

## Example Workflows

### Workflow 1: Simple VPN + App Type Classification
```bash
# Use original system
python src/preprocess/step2changed.py
python src/preprocess/visualize.py

# Train model with 12 classes (0-11)
# Fast, simple, effective
```

### Workflow 2: Detailed Application Fingerprinting
```bash
# Use detailed system
python src/preprocess/detailed_preprocess.py
python src/preprocess/detailed_visualize.py

# Train models at different levels:
# - Level 1: Binary (VPN vs NonVPN)
# - Level 2: App Type (Chat, Streaming, etc.)
# - Level 3: Specific App (facebook, netflix, etc.)
```

### Workflow 3: Hierarchical Classification
```bash
# Use detailed system
python src/preprocess/detailed_preprocess.py

# Train hierarchical classifier:
# Step 1: Classify VPN vs NonVPN
# Step 2: If NonVPN, classify app type
# Step 3: If Chat, classify specific chat app
```

---

## Data Structure Comparison

### Original System
```python
import numpy as np

# Load data
images = np.load("processed_test/idx/data.npy")
labels = np.load("processed_test/idx/labels.npy")

# Shape: images.shape = (N, 28, 28)
# Shape: labels.shape = (N,)
# Labels: integers 0-11
```

### Detailed System
```python
import numpy as np

# Load data
images = np.load("processed_detailed/idx/data_images.npy")
detailed_labels = np.load("processed_detailed/idx/data_labels_detailed.npy")
app_types = np.load("processed_detailed/idx/data_app_types.npy")
apps = np.load("processed_detailed/idx/data_apps.npy")
vpn_types = np.load("processed_detailed/idx/data_vpn_types.npy")

# Shape: images.shape = (N, 28, 28)
# Shape: all labels = (N,)
# detailed_labels: strings like "NonVPN_Chat_facebook"
# app_types: strings like "Chat", "Streaming"
# apps: strings like "facebook", "netflix"
# vpn_types: strings like "VPN", "NonVPN"

# You can use any level independently!
```

---

## Performance Considerations

### Original System
- **Training Speed:** ⚡⚡⚡ Fast (12 classes)
- **Inference Speed:** ⚡⚡⚡ Fast
- **Memory Usage:** 💾 Low
- **Accuracy:** Good for general categories

### Detailed System
- **Training Speed:** ⚡⚡ Slower (30-50 classes)
- **Inference Speed:** ⚡⚡ Slower
- **Memory Usage:** 💾💾 Higher (multiple label files)
- **Accuracy:** Better for specific apps, can also do general

---

## Converting Between Systems

### Detailed → Original (Simple)
```python
import numpy as np

# Load detailed labels
detailed = np.load("processed_detailed/idx/data_labels_detailed.npy")
app_types = np.load("processed_detailed/idx/data_app_types.npy")
vpn_types = np.load("processed_detailed/idx/data_vpn_types.npy")

# Map to original numeric labels
label_map = {
    ("NonVPN", "Chat"): 0, ("NonVPN", "Email"): 1,
    ("NonVPN", "File"): 2, ("NonVPN", "P2P"): 3,
    ("NonVPN", "Streaming"): 4, ("NonVPN", "VoIP"): 5,
    ("VPN", "Chat"): 6, ("VPN", "Email"): 7,
    ("VPN", "File"): 8, ("VPN", "P2P"): 9,
    ("VPN", "Streaming"): 10, ("VPN", "VoIP"): 11
}

original_labels = np.array([
    label_map[(vpn, app_type)] 
    for vpn, app_type in zip(vpn_types, app_types)
])
```

---

## Visualization Capabilities

### Original System (`visualize.py`)
- Sample images per label (12 categories)
- Distribution chart
- Average patterns per category

### Detailed System (`detailed_visualize.py`)
- Sample images per detailed label (30-50 combinations)
- Distribution by app type
- Distribution by specific app
- VPN vs NonVPN comparison
- Heatmap of app vs app type relationships
- Average patterns by app type
- Average patterns by specific app
- Detailed summary report

---

## Recommendation

**For Production/Deployment:** Use the **Original System**
- Simpler, faster, more efficient
- Sufficient for most use cases

**For Research/Analysis:** Use the **Detailed System**
- More insights and flexibility
- Better for understanding traffic patterns
- Essential for application-specific fingerprinting

**Best Approach:** Generate both!
- Keep original system for baseline comparisons
- Use detailed system for in-depth analysis
- They don't interfere with each other (different output folders)

