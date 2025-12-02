import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
IMAGE_DIR = os.path.join(PROJECT_ROOT, "processed_data/memory_safe/own_nonVPN_p2p_2/data_memory_safe.npy")
LABELS_DIR = os.path.join(PROJECT_ROOT, "processed_data/memory_safe/own_nonVPN_p2p_2/labels_memory_safe.npy")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data/memory_safe/own_nonVPN_p2p_2/visualization")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("[+] Loading data...")
images = np.load(IMAGE_DIR)   # shape (N, 28, 28)
labels = np.load(LABELS_DIR)  # shape (N,)

class_names = {
    0: "NonVPN-Chat",
    1: "NonVPN-Email",
    2: "NonVPN-File",
    3: "NonVPN-P2P",
    4: "NonVPN-Streaming",
    5: "NonVPN-VoIP",
    6: "VPN-Chat",
    7: "VPN-Email",
    8: "VPN-File",
    9: "VPN-P2P",
    10: "VPN-Streaming",
    11: "VPN-VoIP",
}

# =====================================
# 1. Dataset Statistics Report
# =====================================
print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)
print(f"Total images: {len(images)}")
print(f"Image shape: {images[0].shape}")
print(f"\nLabel distribution:")

unique_labels, counts = np.unique(labels, return_counts=True)
for lbl, count in zip(unique_labels, counts):
    percentage = (count / len(labels)) * 100
    print(f"  Label {lbl:2d} - {class_names.get(lbl, 'Unknown'):20s}: {count:6d} samples ({percentage:5.2f}%)")

# Calculate data density statistics
print("\nData Density Analysis:")
for lbl in unique_labels:
    class_imgs = images[labels == lbl]
    non_zero_percentages = np.mean(class_imgs > 0, axis=(1, 2)) * 100
    avg_density = np.mean(non_zero_percentages)
    print(f"  {class_names.get(lbl, 'Unknown'):20s}: {avg_density:5.2f}% avg non-zero pixels")

# =====================================
# 2. Large Grid Visualization (50 samples per class)
# =====================================
print("\n[+] Creating comprehensive grid visualization...")

SAMPLES_PER_CLASS = 50
COLS = 10
ROWS_PER_CLASS = (SAMPLES_PER_CLASS + COLS - 1) // COLS  # Ceiling division

# Create one large figure
total_classes = len(unique_labels)
fig = plt.figure(figsize=(20, total_classes * ROWS_PER_CLASS * 2.2))
gs = GridSpec(total_classes * ROWS_PER_CLASS, COLS, figure=fig, hspace=0.3, wspace=0.05)

for class_idx, label in enumerate(sorted(unique_labels)):
    class_imgs = images[labels == label]
    num_to_show = min(SAMPLES_PER_CLASS, len(class_imgs))
    
    # Calculate starting row for this class
    start_row = class_idx * ROWS_PER_CLASS
    
    for i in range(num_to_show):
        row = start_row + (i // COLS)
        col = i % COLS
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(class_imgs[i], cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        
        # Add sample number on first of each row
        if col == 0:
            ax.text(-0.1, 0.5, f"#{i}-{i+COLS-1}", transform=ax.transAxes,
                   fontsize=8, va='center', ha='right')
    
    # Add class label on the left of first row
    ax = fig.add_subplot(gs[start_row, 0])
    ax.text(-0.15, 1.2, f"Label {label}: {class_names.get(label, 'Unknown')}",
           transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom')

plt.savefig(os.path.join(OUTPUT_DIR, "comprehensive_grid_50_per_class.png"), 
           dpi=150, bbox_inches='tight')
print(f"  Saved: comprehensive_grid_50_per_class.png")
plt.close()

# =====================================
# 3. Density-Filtered Visualization (only rich samples)
# =====================================
print("\n[+] Creating density-filtered visualization (rich samples only)...")

DENSITY_THRESHOLD = 10  # At least 10% non-zero pixels
SAMPLES_PER_CLASS = 30
COLS = 10

fig = plt.figure(figsize=(20, total_classes * 3.5))
gs = GridSpec(total_classes * 3, COLS, figure=fig, hspace=0.3, wspace=0.05)

for class_idx, label in enumerate(sorted(unique_labels)):
    class_imgs = images[labels == label]
    
    # Filter for images with sufficient data density
    densities = np.mean(class_imgs > 0, axis=(1, 2)) * 100
    rich_indices = np.where(densities >= DENSITY_THRESHOLD)[0]
    
    if len(rich_indices) == 0:
        print(f"  Warning: No rich samples found for label {label}")
        continue
    
    # Take first N rich samples
    rich_imgs = class_imgs[rich_indices[:SAMPLES_PER_CLASS]]
    num_to_show = len(rich_imgs)
    
    start_row = class_idx * 3
    
    for i in range(min(num_to_show, SAMPLES_PER_CLASS)):
        row = start_row + (i // COLS)
        col = i % COLS
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(rich_imgs[i], cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
    
    # Add class label
    ax = fig.add_subplot(gs[start_row, 0])
    ax.text(-0.15, 1.2, f"Label {label}: {class_names.get(label, 'Unknown')} ({len(rich_indices)} rich samples)",
           transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom')

plt.savefig(os.path.join(OUTPUT_DIR, "rich_samples_only.png"), 
           dpi=150, bbox_inches='tight')
print(f"  Saved: rich_samples_only.png")
plt.close()

# =====================================
# 4. Random Sample Visualization
# =====================================
print("\n[+] Creating random sample visualization...")

SAMPLES_PER_CLASS = 30
COLS = 10

fig = plt.figure(figsize=(20, total_classes * 3.5))
gs = GridSpec(total_classes * 3, COLS, figure=fig, hspace=0.3, wspace=0.05)

np.random.seed(42)  # For reproducibility

for class_idx, label in enumerate(sorted(unique_labels)):
    class_imgs = images[labels == label]
    num_available = len(class_imgs)
    num_to_show = min(SAMPLES_PER_CLASS, num_available)
    
    # Random sampling
    random_indices = np.random.choice(num_available, size=num_to_show, replace=False)
    random_imgs = class_imgs[random_indices]
    
    start_row = class_idx * 3
    
    for i in range(num_to_show):
        row = start_row + (i // COLS)
        col = i % COLS
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(random_imgs[i], cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
    
    # Add class label
    ax = fig.add_subplot(gs[start_row, 0])
    ax.text(-0.15, 1.2, f"Label {label}: {class_names.get(label, 'Unknown')}",
           transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom')

plt.savefig(os.path.join(OUTPUT_DIR, "random_samples_30_per_class.png"), 
           dpi=150, bbox_inches='tight')
print(f"  Saved: random_samples_30_per_class.png")
plt.close()

# =====================================
# 5. Density Distribution Histograms
# =====================================
print("\n[+] Creating density distribution analysis...")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, label in enumerate(sorted(unique_labels)):
    if idx >= len(axes):
        break
    
    class_imgs = images[labels == label]
    densities = np.mean(class_imgs > 0, axis=(1, 2)) * 100
    
    ax = axes[idx]
    ax.hist(densities, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(densities), color='red', linestyle='--', 
              label=f'Mean: {np.mean(densities):.1f}%')
    ax.axvline(np.median(densities), color='green', linestyle='--',
              label=f'Median: {np.median(densities):.1f}%')
    ax.set_title(f"Label {label}: {class_names.get(label, 'Unknown')}", fontsize=10)
    ax.set_xlabel('Data Density (%)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "density_distributions.png"), 
           dpi=150, bbox_inches='tight')
print(f"  Saved: density_distributions.png")
plt.close()

# =====================================
# 6. Summary Report
# =====================================
report_path = os.path.join(OUTPUT_DIR, "visualization_report.txt")
with open(report_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("COMPREHENSIVE DATASET VISUALIZATION REPORT\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Total samples: {len(images)}\n")
    f.write(f"Image shape: {images[0].shape}\n\n")
    
    f.write("Label Distribution:\n")
    f.write("-" * 60 + "\n")
    for lbl, count in zip(unique_labels, counts):
        percentage = (count / len(labels)) * 100
        f.write(f"  Label {lbl:2d} - {class_names.get(lbl, 'Unknown'):20s}: "
               f"{count:6d} ({percentage:5.2f}%)\n")
    
    f.write("\n" + "="*60 + "\n")
    f.write("Data Density Analysis:\n")
    f.write("="*60 + "\n\n")
    
    for lbl in unique_labels:
        class_imgs = images[labels == lbl]
        densities = np.mean(class_imgs > 0, axis=(1, 2)) * 100
        
        f.write(f"{class_names.get(lbl, 'Unknown')}:\n")
        f.write(f"  Average density: {np.mean(densities):6.2f}%\n")
        f.write(f"  Median density:  {np.median(densities):6.2f}%\n")
        f.write(f"  Min density:     {np.min(densities):6.2f}%\n")
        f.write(f"  Max density:     {np.max(densities):6.2f}%\n")
        f.write(f"  Std deviation:   {np.std(densities):6.2f}%\n")
        
        # Count samples by density ranges
        sparse = np.sum(densities < 10)
        medium = np.sum((densities >= 10) & (densities < 50))
        dense = np.sum(densities >= 50)
        
        f.write(f"  Sparse (<10%):   {sparse:6d} ({sparse/len(densities)*100:5.2f}%)\n")
        f.write(f"  Medium (10-50%): {medium:6d} ({medium/len(densities)*100:5.2f}%)\n")
        f.write(f"  Dense (>50%):    {dense:6d} ({dense/len(densities)*100:5.2f}%)\n")
        f.write("\n")

print(f"  Saved: visualization_report.txt")

print("\n" + "="*60)
print("VISUALIZATION COMPLETE!")
print("="*60)
print(f"All visualizations saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  1. comprehensive_grid_50_per_class.png - First 50 samples per class")
print("  2. rich_samples_only.png - Only samples with >10% data density")
print("  3. random_samples_30_per_class.png - 30 random samples per class")
print("  4. density_distributions.png - Histogram of data densities")
print("  5. visualization_report.txt - Detailed statistics")