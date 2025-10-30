# Detailed Labeling System

This document explains the new detailed labeling system that combines VPN type, application type, and specific application names.

## Overview

The new preprocessing and visualization system creates labels in the format:
- `NonVPN_Chat_facebook`
- `VPN_Streaming_netflix`
- `NonVPN_File_skype`
- etc.

This allows for more granular analysis of traffic patterns both at the application type level (Chat, Streaming, File, etc.) and at the specific application level (facebook, netflix, skype, etc.).

## Files Created

### 1. `src/preprocess/detailed_preprocess.py`
Main preprocessing script that:
- Uses the same preprocessing logic as `step2changed.py`
- Extracts application names from filenames automatically
- Creates detailed labels combining VPN type, app type, and specific app
- Saves multiple data files for different analysis levels

### 2. `src/preprocess/detailed_visualize.py`
Comprehensive visualization script that generates:
- Sample images for each unique label
- Distribution charts by application type
- Distribution charts by specific application
- VPN vs NonVPN comparison
- Heatmap showing app vs app type relationships
- Average traffic patterns by app type
- Average traffic patterns by specific app
- Detailed summary report

## Usage

### Step 1: Run Preprocessing

```bash
# Activate virtual environment
source venv/bin/activate

# Run detailed preprocessing
python src/preprocess/detailed_preprocess.py
```

This will create:
- `processed_detailed/flows/` - Flow files organized by VPN/app type
- `processed_detailed/idx/` - Numpy data files with detailed labels

### Step 2: Generate Visualizations

```bash
python src/preprocess/detailed_visualize.py
```

This will create visualizations in `processed_detailed/visualizations/`:
1. `sample_images_detailed.png` - Grid of sample images per label
2. `distribution_by_app_type.png` - Bar chart of samples by app type
3. `distribution_by_app.png` - Bar chart of samples by specific app
4. `vpn_vs_nonvpn_comparison.png` - Grouped bar chart comparing VPN/NonVPN
5. `heatmap_app_vs_apptype.png` - Heatmap showing relationships
6. `average_patterns_by_app_type.png` - Average traffic patterns per app type
7. `average_patterns_by_app.png` - Average traffic patterns per app (top 12)
8. `detailed_summary_report.txt` - Text summary with statistics

## Output Data Files

The preprocessing creates the following numpy files in `processed_detailed/idx/`:

- `data_images.npy` - 28x28 grayscale images (shape: [N, 28, 28])
- `data_labels_detailed.npy` - Full detailed labels (e.g., "NonVPN_Chat_facebook")
- `data_app_types.npy` - Just app types (e.g., "Chat", "Streaming")
- `data_apps.npy` - Just app names (e.g., "facebook", "netflix")
- `data_vpn_types.npy` - Just VPN types (e.g., "VPN", "NonVPN")

## Application Name Extraction

The system automatically extracts application names from filenames:

| Filename | Extracted App |
|----------|---------------|
| `facebook_chat_4a.pcap` | facebook |
| `vpn_netflix_A.pcap` | netflix |
| `aim_chat_3a.pcap` | aim |
| `skype_file1.pcap` | skype |
| `youtube1.pcap` | youtube |
| `spotify2.pcap` | spotify |

## Configuration

You can adjust these settings in `detailed_preprocess.py`:

```python
MAX_FILES_PER_CLASS = 4    # Limit files per class (for testing)
MAX_FLOWS_PER_PCAP = 10    # Limit flows per pcap (for speed)
MAX_LEN = 784              # Image size (28x28)
```

For full dataset processing, set `MAX_FILES_PER_CLASS = None` or increase significantly.

## Analysis Capabilities

The detailed labeling system enables:

1. **Multi-level Analysis**
   - VPN vs NonVPN patterns
   - Application type patterns (Chat, Streaming, File, etc.)
   - Specific application patterns (facebook, netflix, etc.)

2. **Pattern Recognition**
   - Compare traffic patterns within same app type but different apps
   - Identify VPN encryption effects on different app types
   - Discover unique signatures of specific applications

3. **Statistical Insights**
   - Distribution of samples across categories
   - Correlation between app types and specific apps
   - VPN usage patterns across different applications

## Example Visualizations

After running the visualization script, you'll see:

- **Distribution charts** showing which app types and apps have the most samples
- **Heatmaps** revealing which apps belong to which categories
- **Average patterns** showing the "fingerprint" of each app type and specific app
- **VPN comparisons** highlighting how VPN affects different traffic types

## Next Steps

After generating these detailed labels and visualizations, you can:

1. Train classifiers at different granularity levels:
   - Binary: VPN vs NonVPN
   - Multi-class: App types (Chat, Streaming, etc.)
   - Fine-grained: Specific applications (facebook, netflix, etc.)

2. Perform transfer learning experiments
3. Analyze feature importance at different levels
4. Build hierarchical classifiers

## Notes

- The preprocessing uses the same anonymization and cleaning logic as `step2changed.py`
- Duplicate flows are removed using SHA-1 hashing
- TCP retransmissions and corrupted packets are filtered out
- IP addresses and MAC addresses are anonymized

