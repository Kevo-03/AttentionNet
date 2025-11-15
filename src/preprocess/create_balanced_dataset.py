"""
NEW BALANCED DATASET PREPARATION
=================================

Strategy:
1. Keep ALL data (minimal filtering: density >= 0.01)
2. Split into train/val/test BEFORE any balancing
3. NO augmentation in preprocessing (will be done in PyTorch)
4. Calculate class weights for training
5. Save to new directory: processed_data/balanced/

This approach maximizes data utilization and lets PyTorch handle augmentation.
"""

import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ============================================================================
# CONFIGURATION
# ============================================================================
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))

INPUT_DATA = os.path.join(PROJECT_ROOT, "processed_data/idx/data_fixed.npy")
INPUT_LABELS = os.path.join(PROJECT_ROOT, "processed_data/idx/labels_fixed.npy")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data/balanced")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
MIN_DENSITY = 0.01  # Keep 99.9% of data (only remove truly empty images)
MAX_SAMPLES_PER_CLASS = 35000  # Cap very large classes (VoIP NonVPN)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASS_NAMES = {
    0: "Chat (NonVPN)", 1: "Email (NonVPN)", 2: "File (NonVPN)", 
    3: "Streaming (NonVPN)", 4: "VoIP (NonVPN)",
    5: "Chat (VPN)", 6: "Email (VPN)", 7: "File (VPN)", 
    8: "P2P (VPN)", 9: "Streaming (VPN)", 10: "VoIP (VPN)"
}

print("="*80)
print("CREATING BALANCED DATASET WITH MINIMAL FILTERING")
print("="*80)
print(f"Strategy:")
print(f"  1. Minimal filtering (density >= {MIN_DENSITY*100:.1f}%)")
print(f"  2. Cap majority classes at {MAX_SAMPLES_PER_CLASS:,}")
print(f"  3. Split: {TRAIN_RATIO*100:.0f}% train, {VAL_RATIO*100:.0f}% val, {TEST_RATIO*100:.0f}% test")
print(f"  4. NO augmentation here (will be done in PyTorch)")
print(f"  5. Calculate class weights for training")
print("="*80)

# ============================================================================
# STEP 1: Load and Apply Minimal Filtering
# ============================================================================
print("\n[Step 1/5] Loading data with minimal filtering...")
images = np.load(INPUT_DATA)
labels = np.load(INPUT_LABELS)
print(f"  Loaded: {len(images):,} samples")

# Apply minimal filtering - only remove truly empty images
densities = np.mean(images > 0, axis=(1, 2))
density_mask = densities >= MIN_DENSITY

filtered_images = images[density_mask]
filtered_labels = labels[density_mask]
filtered_densities = densities[density_mask]

removed_count = len(images) - len(filtered_images)
print(f"  After filtering (density >= {MIN_DENSITY*100:.1f}%): {len(filtered_images):,} samples")
print(f"  Removed: {removed_count} samples ({removed_count/len(images)*100:.2f}%)")

# ============================================================================
# STEP 2: Analyze Class Distribution
# ============================================================================
print("\n[Step 2/5] Analyzing class distribution...")
class_counts = Counter(filtered_labels)

print(f"\nOriginal distribution:")
for cls in sorted(class_counts.keys()):
    count = class_counts[cls]
    pct = count / len(filtered_labels) * 100
    print(f"  {CLASS_NAMES[cls]:<25} {count:>8,} ({pct:>5.2f}%)")

# ============================================================================
# STEP 3: Cap Majority Classes (Optional)
# ============================================================================
print(f"\n[Step 3/5] Capping majority classes at {MAX_SAMPLES_PER_CLASS:,}...")

balanced_images = []
balanced_labels = []

for cls in range(11):
    cls_mask = filtered_labels == cls
    cls_images = filtered_images[cls_mask]
    cls_labels = filtered_labels[cls_mask]
    
    if len(cls_images) > MAX_SAMPLES_PER_CLASS:
        # Randomly sample MAX_SAMPLES_PER_CLASS
        indices = np.random.choice(len(cls_images), MAX_SAMPLES_PER_CLASS, replace=False)
        cls_images = cls_images[indices]
        cls_labels = cls_labels[indices]
        print(f"  {CLASS_NAMES[cls]:<25} Capped: {class_counts[cls]:,} → {MAX_SAMPLES_PER_CLASS:,}")
    else:
        print(f"  {CLASS_NAMES[cls]:<25} Kept all: {len(cls_images):,}")
    
    balanced_images.append(cls_images)
    balanced_labels.append(cls_labels)

# Combine all classes
balanced_images = np.vstack(balanced_images)
balanced_labels = np.concatenate(balanced_labels)

print(f"\nAfter capping: {len(balanced_images):,} total samples")

# ============================================================================
# STEP 4: Split into Train/Val/Test
# ============================================================================
print(f"\n[Step 4/5] Splitting into train/val/test...")

# First split: train vs (val+test)
train_images, temp_images, train_labels, temp_labels = train_test_split(
    balanced_images, balanced_labels,
    test_size=(VAL_RATIO + TEST_RATIO),
    stratify=balanced_labels,
    random_state=42
)

# Second split: val vs test
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    stratify=temp_labels,
    random_state=42
)

print(f"  Training set:   {len(train_images):,} samples ({len(train_images)/len(balanced_images)*100:.1f}%)")
print(f"  Validation set: {len(val_images):,} samples ({len(val_images)/len(balanced_images)*100:.1f}%)")
print(f"  Test set:       {len(test_images):,} samples ({len(test_images)/len(balanced_images)*100:.1f}%)")

# Show per-class distribution for ALL sets
train_class_counts = Counter(train_labels)
val_class_counts = Counter(val_labels)
test_class_counts = Counter(test_labels)

print(f"\n{'='*80}")
print(f"DISTRIBUTION BY CLASS (TRAIN / VAL / TEST)")
print(f"{'='*80}")
print(f"{'Class':<27} {'Train':>12} {'Val':>12} {'Test':>12} {'Total':>12}")
print(f"{'-'*80}")

for cls in range(11):
    train_count = train_class_counts[cls]
    val_count = val_class_counts[cls]
    test_count = test_class_counts[cls]
    total_count = train_count + val_count + test_count
    
    print(f"{CLASS_NAMES[cls]:<27} {train_count:>6,} ({train_count/len(train_labels)*100:>4.1f}%) "
          f"{val_count:>6,} ({val_count/len(val_labels)*100:>4.1f}%) "
          f"{test_count:>6,} ({test_count/len(test_labels)*100:>4.1f}%) "
          f"{total_count:>6,}")

print(f"{'-'*80}")
print(f"{'TOTAL':<27} {len(train_labels):>12,} {len(val_labels):>12,} {len(test_labels):>12,} {len(balanced_labels):>12,}")
print(f"{'='*80}")

# ============================================================================
# STEP 5: Calculate Class Weights and Augmentation Targets
# ============================================================================
print(f"\n[Step 5/5] Calculating class weights and augmentation targets...")

# Calculate class weights for weighted loss
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

print(f"\nClass weights for training (to handle imbalance):")
for cls, weight in enumerate(class_weights):
    print(f"  {CLASS_NAMES[cls]:<25} {weight:.4f}")

# Calculate augmentation multipliers for PyTorch
# Target: bring minority classes up to ~median size through resampling
median_count = np.median(list(train_class_counts.values()))
print(f"\nMedian class size: {int(median_count):,}")

augmentation_multipliers = {}
print(f"\nAugmentation multipliers for PyTorch (oversample minority classes):")
for cls in range(11):
    count = train_class_counts[cls]
    # Calculate how many times to repeat this class
    multiplier = max(1.0, median_count / count)
    augmentation_multipliers[cls] = multiplier
    print(f"  {CLASS_NAMES[cls]:<25} x{multiplier:.2f} "
          f"({count:>6,} → {int(count * multiplier):>6,} effective samples)")

# ============================================================================
# SAVE EVERYTHING
# ============================================================================
print(f"\n[Saving] Writing datasets to {OUTPUT_DIR}...")

# Save train/val/test sets
np.save(os.path.join(OUTPUT_DIR, "train_data_weighted.npy"), train_images)
np.save(os.path.join(OUTPUT_DIR, "train_labels_weighted.npy"), train_labels)
np.save(os.path.join(OUTPUT_DIR, "val_data_weighted.npy"), val_images)
np.save(os.path.join(OUTPUT_DIR, "val_labels_weighted.npy"), val_labels)
np.save(os.path.join(OUTPUT_DIR, "test_data_weighted.npy"), test_images)
np.save(os.path.join(OUTPUT_DIR, "test_labels_weighted.npy"), test_labels)

# Save metadata
np.save(os.path.join(OUTPUT_DIR, "class_weights.npy"), class_weights)
np.save(os.path.join(OUTPUT_DIR, "augmentation_multipliers.npy"), 
        np.array([augmentation_multipliers[i] for i in range(11)]))

print(f"  ✓ train_data_weighted.npy / train_labels_weighted.npy")
print(f"  ✓ val_data_weighted.npy / val_labels_weighted.npy")
print(f"  ✓ test_data_weighted.npy / test_labels_weighted.npy")
print(f"  ✓ class_weights.npy")
print(f"  ✓ augmentation_multipliers.npy")

# ============================================================================
# GENERATE REPORT
# ============================================================================
report_path = os.path.join(OUTPUT_DIR, "dataset_report.txt")
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("BALANCED DATASET WITH MINIMAL FILTERING\n")
    f.write("="*80 + "\n\n")
    
    f.write("METHODOLOGY:\n")
    f.write(f"  1. Minimal filtering (density >= {MIN_DENSITY*100:.1f}%)\n")
    f.write(f"  2. Cap majority classes at {MAX_SAMPLES_PER_CLASS:,}\n")
    f.write(f"  3. Split: {TRAIN_RATIO*100:.0f}% train / {VAL_RATIO*100:.0f}% val / {TEST_RATIO*100:.0f}% test\n")
    f.write(f"  4. NO preprocessing augmentation\n")
    f.write(f"  5. Class weights + PyTorch oversampling for balance\n\n")
    
    f.write(f"Original dataset:        {len(images):>8,} samples\n")
    f.write(f"After filtering:         {len(filtered_images):>8,} samples\n")
    f.write(f"After capping:           {len(balanced_images):>8,} samples\n\n")
    
    f.write(f"TRAIN SET:               {len(train_images):>8,} samples (all real)\n")
    f.write(f"VAL SET:                 {len(val_images):>8,} samples (all real)\n")
    f.write(f"TEST SET:                {len(test_images):>8,} samples (all real)\n\n")
    
    f.write("-"*80 + "\n")
    f.write("DISTRIBUTION BY CLASS (TRAIN / VAL / TEST):\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Class':<27} {'Train':>12} {'Val':>12} {'Test':>12} {'Total':>12}\n")
    f.write("-"*80 + "\n")
    
    for cls in range(11):
        train_count = train_class_counts[cls]
        val_count = val_class_counts[cls]
        test_count = test_class_counts[cls]
        total_count = train_count + val_count + test_count
        
        f.write(f"{CLASS_NAMES[cls]:<27} {train_count:>6,} ({train_count/len(train_labels)*100:>4.1f}%) "
                f"{val_count:>6,} ({val_count/len(val_labels)*100:>4.1f}%) "
                f"{test_count:>6,} ({test_count/len(test_labels)*100:>4.1f}%) "
                f"{total_count:>6,}\n")
    
    f.write("-"*80 + "\n")
    f.write(f"{'TOTAL':<27} {len(train_labels):>12,} {len(val_labels):>12,} {len(test_labels):>12,} {len(balanced_labels):>12,}\n")
    f.write("-"*80 + "\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("CLASS WEIGHTS (for weighted loss):\n")
    f.write("-"*80 + "\n")
    for cls in range(11):
        f.write(f"Label {cls:>2} - {CLASS_NAMES[cls]:<25} : {class_weights[cls]:>7.4f}\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("AUGMENTATION MULTIPLIERS (for PyTorch oversampling):\n")
    f.write("-"*80 + "\n")
    for cls in range(11):
        mult = augmentation_multipliers[cls]
        count = train_class_counts[cls]
        effective = int(count * mult)
        f.write(f"Label {cls:>2} - {CLASS_NAMES[cls]:<25} : x{mult:>5.2f} "
                f"({count:>6,} → {effective:>6,})\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("NEXT STEP: Use train_balanced.py for training with PyTorch augmentation\n")
    f.write("="*80 + "\n")

print(f"  ✓ {report_path}")

# ============================================================================
# VISUALIZE DISTRIBUTION
# ============================================================================
print(f"\n[Visualizing] Creating distribution plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Original vs After Capping
ax = axes[0, 0]
original_counts = [class_counts[i] for i in range(11)]
after_cap_counts = [len(balanced_labels[balanced_labels == i]) for i in range(11)]
x = np.arange(11)
width = 0.35

ax.bar(x - width/2, original_counts, width, label='Original', alpha=0.8)
ax.bar(x + width/2, after_cap_counts, width, label='After Capping', alpha=0.8)
ax.set_xlabel('Class')
ax.set_ylabel('Sample Count')
ax.set_title('Original vs After Capping Distribution')
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in range(11)])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: Train/Val/Test Distribution
ax = axes[0, 1]
train_counts = [train_class_counts[i] for i in range(11)]
val_counts = [len(val_labels[val_labels == i]) for i in range(11)]
test_counts = [len(test_labels[test_labels == i]) for i in range(11)]

x = np.arange(11)
width = 0.25

ax.bar(x - width, train_counts, width, label='Train', alpha=0.8)
ax.bar(x, val_counts, width, label='Val', alpha=0.8)
ax.bar(x + width, test_counts, width, label='Test', alpha=0.8)
ax.set_xlabel('Class')
ax.set_ylabel('Sample Count')
ax.set_title('Train/Val/Test Split Distribution')
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in range(11)])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 3: Class Weights
ax = axes[1, 0]
ax.bar(range(11), class_weights, alpha=0.8, color='orange')
ax.set_xlabel('Class')
ax.set_ylabel('Weight')
ax.set_title('Class Weights for Weighted Loss')
ax.set_xticks(range(11))
ax.set_xticklabels([str(i) for i in range(11)])
ax.grid(axis='y', alpha=0.3)

# Plot 4: Augmentation Multipliers
ax = axes[1, 1]
multipliers = [augmentation_multipliers[i] for i in range(11)]
colors = ['green' if m == 1.0 else 'blue' if m < 3 else 'red' for m in multipliers]
ax.bar(range(11), multipliers, alpha=0.8, color=colors)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No augmentation')
ax.set_xlabel('Class')
ax.set_ylabel('Multiplier')
ax.set_title('Augmentation Multipliers for PyTorch')
ax.set_xticks(range(11))
ax.set_xticklabels([str(i) for i in range(11)])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "distribution_analysis.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  ✓ {plot_path}")

print("\n" + "="*80)
print("DATASET PREPARATION COMPLETE!")
print("="*80)
print(f"\nDataset saved to: {OUTPUT_DIR}")
print(f"\nSummary:")
print(f"  - Training:   {len(train_images):>8,} samples (all real)")
print(f"  - Validation: {len(val_images):>8,} samples (all real)")
print(f"  - Test:       {len(test_images):>8,} samples (all real)")
print(f"  - Total:      {len(balanced_images):>8,} samples")
print(f"\nNext step: Run src/model/train_balanced.py for training")
print("="*80)

