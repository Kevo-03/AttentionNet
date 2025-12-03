import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))

INPUT_DATA = os.path.join(PROJECT_ROOT, "processed_data/memory_safe/own_nonVPN_p2p_2/data_memory_safe.npy")
INPUT_LABELS = os.path.join(PROJECT_ROOT, "processed_data/memory_safe/own_nonVPN_p2p_2/labels_memory_safe.npy")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data/final/memory_safe/own_nonVPN_p2p_2_balanced")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
MIN_DENSITY = 0.01  # Keep flows with at least 1% non-zero pixels
MIN_SAMPLES_PER_CLASS = 0  # Drop classes with fewer samples
TRAIN_RATIO = 0.8
VAL_RATIO = 0.05
TEST_RATIO = 0.15

CLASS_NAMES = {
    0: "Chat (NonVPN)", 1: "Email (NonVPN)", 2: "File (NonVPN)", 
    3: "P2P (NonVPN)", 4: "Streaming (NonVPN)", 5: "VoIP (NonVPN)",
    6: "Chat (VPN)", 7: "Email (VPN)", 8: "File (VPN)", 
    9: "P2P (VPN)", 10: "Streaming (VPN)", 11: "VoIP (VPN)"
}

print("="*80)
print("CREATING BALANCED TRAINING SET (CAPPED TO SMALLEST CLASS)")
print("="*80)
print(f"Strategy:")
print(f"  1. Filter sparse samples (< {MIN_DENSITY*100:.0f}% density)")
print(f"  2. Split: {TRAIN_RATIO*100:.0f}% train, {VAL_RATIO*100:.0f}% val, {TEST_RATIO*100:.0f}% test")
print(f"  3. Cap ONLY training set to smallest class size (no augmentation)")
print(f"  4. Val/Test keep full distribution - 100% real samples")
print("="*80)

# ============================================================================
# STEP 1: Load and Filter
# ============================================================================
print("\n[Step 1/5] Loading and filtering data...")
images = np.load(INPUT_DATA)
labels = np.load(INPUT_LABELS)
print(f"  Loaded: {len(images)} samples")

# Filter sparse samples
densities = np.mean(images > 0, axis=(1, 2))
density_mask = densities >= MIN_DENSITY

filtered_images = images[density_mask]
filtered_labels = labels[density_mask]
filtered_densities = densities[density_mask]

print(f"  After filtering: {len(filtered_images)} samples")

# Remove classes with too few samples
valid_classes = []
for label in np.unique(filtered_labels):
    count = np.sum(filtered_labels == label)
    if count >= MIN_SAMPLES_PER_CLASS:
        valid_classes.append(label)
    else:
        print(f"  Dropping Label {label} ({CLASS_NAMES.get(label)}): only {count} samples")

# Keep only valid classes
valid_mask = np.isin(filtered_labels, valid_classes)
filtered_images = filtered_images[valid_mask]
filtered_labels = filtered_labels[valid_mask]
filtered_densities = filtered_densities[valid_mask]

print(f"  Valid classes: {len(valid_classes)}")
print(f"  Final after filtering: {len(filtered_images)} samples")

# ============================================================================
# STEP 2: Split into Train/Val/Test (BEFORE balancing)
# ============================================================================
print(f"\n[Step 2/5] Splitting into train/val/test...")

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    filtered_images, filtered_labels, 
    test_size=TEST_RATIO, 
    stratify=filtered_labels, 
    random_state=42
)

# Second split: separate train and val from temp
val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=val_ratio_adjusted,
    stratify=y_temp,
    random_state=42
)

print(f"  Train set: {len(X_train)} samples ({len(X_train)/len(filtered_images)*100:.1f}%)")
print(f"  Val set:   {len(X_val)} samples ({len(X_val)/len(filtered_images)*100:.1f}%)")
print(f"  Test set:  {len(X_test)} samples ({len(X_test)/len(filtered_images)*100:.1f}%)")

print(f"\n  Train distribution (before balancing): {dict(Counter(y_train))}")
print(f"  Val distribution:   {dict(Counter(y_val))}")
print(f"  Test distribution:  {dict(Counter(y_test))}")

# ============================================================================
# STEP 3: Cap ONLY Training Set to Smallest Class
# ============================================================================
print(f"\n[Step 3/5] Capping ONLY training set to smallest class size...")

# Count samples per class in training set
train_class_counts = Counter(y_train)
min_train_class_size = min(train_class_counts.values())
min_train_class_label = min(train_class_counts, key=train_class_counts.get)

print(f"  Smallest class in train: Label {min_train_class_label} ({CLASS_NAMES.get(min_train_class_label)}) with {min_train_class_size} samples")
print(f"  Capping training classes to {min_train_class_size} samples each")

# Calculate densities for training data
train_densities = np.mean(X_train > 0, axis=(1, 2))

balanced_train_indices = []

for label in valid_classes:
    label_mask = y_train == label
    label_indices = np.where(label_mask)[0]
    n_available = len(label_indices)
    label_densities = train_densities[label_mask]
    
    class_name = CLASS_NAMES.get(label, f"Unknown-{label}")
    
    if n_available > min_train_class_size:
        # Undersample: keep the densest samples
        sorted_idx = np.argsort(label_densities)[::-1]
        top_indices = sorted_idx[:min_train_class_size]
        selected = label_indices[top_indices]
        print(f"  Label {label:2d} ({class_name:25s}): {n_available:5d} → {min_train_class_size:5d} (kept densest)")
    else:
        # Keep all (this should only happen for the smallest class)
        selected = label_indices
        print(f"  Label {label:2d} ({class_name:25s}): {n_available:5d} → {n_available:5d} (smallest class)")
    
    balanced_train_indices.extend(selected)

X_train_balanced = X_train[balanced_train_indices]
y_train_balanced = y_train[balanced_train_indices]

print(f"\n  Balanced training set: {len(X_train_balanced)} samples")
print(f"  Train distribution (after balancing): {dict(Counter(y_train_balanced))}")

# Shuffle training set
shuffle_idx = np.random.permutation(len(X_train_balanced))
X_train_final = X_train_balanced[shuffle_idx]
y_train_final = y_train_balanced[shuffle_idx]

# ============================================================================
# STEP 4: Save All Splits
# ============================================================================
print(f"\n[Step 4/5] Saving datasets...")

np.save(os.path.join(OUTPUT_DIR, "train_data_memory_safe_own_nonVPN_p2p.npy"), X_train_final)
np.save(os.path.join(OUTPUT_DIR, "train_labels_memory_safe_own_nonVPN_p2p.npy"), y_train_final)
np.save(os.path.join(OUTPUT_DIR, "val_data_memory_safe_own_nonVPN_p2p.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "val_labels_memory_safe_own_nonVPN_p2p.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "test_data_memory_safe_own_nonVPN_p2p.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "test_labels_memory_safe_own_nonVPN_p2p.npy"), y_test)

print(f"  ✓ Saved training set: {len(X_train_final)} samples")
print(f"  ✓ Saved validation set: {len(X_val)} samples")
print(f"  ✓ Saved test set: {len(X_test)} samples")

# ============================================================================
# STEP 5: Generate Report and Visualizations
# ============================================================================
print(f"\n[Step 5/5] Generating visualizations and report...")

# Calculate balance metrics
train_counts = Counter(y_train_final)
max_count = max(train_counts.values())
min_count = min(train_counts.values())
balance_ratio = min_count / max_count

print(f"\n  Training set balance ratio: {balance_ratio:.2f}")
print(f"  (1.0 = perfect balance)")

# Visualization 1: Distribution comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (data, title) in enumerate([
    (y_train_final, 'Training Set'),
    (y_val, 'Validation Set'),
    (y_test, 'Test Set')
]):
    ax = axes[idx]
    counter = Counter(data)
    labels_list = sorted(valid_classes)
    counts = [counter[l] for l in labels_list]
    
    bars = ax.bar(range(len(labels_list)), counts, color='steelblue', edgecolor='black')
    ax.set_xlabel('Class Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(labels_list)))
    ax.set_xticklabels(labels_list, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
               f'{count}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "split_distribution.png"), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: split_distribution.png")
plt.close()

# Visualization 2: Sample images from test set
n_samples = 10
fig, axes = plt.subplots(len(valid_classes), n_samples, figsize=(n_samples*1.5, len(valid_classes)*1.5))

if len(valid_classes) == 1:
    axes = axes.reshape(1, -1)

for label_idx, label in enumerate(sorted(valid_classes)):
    class_imgs = X_test[y_test == label]
    
    for i in range(n_samples):
        ax = axes[label_idx, i] if len(valid_classes) > 1 else axes[i]
        
        if i < len(class_imgs):
            ax.imshow(class_imgs[i], cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        
        if i == 0:
            class_name = CLASS_NAMES.get(label, f"L{label}")
            ax.text(-0.3, 0.5, f"{label}\n{class_name}", 
                   transform=ax.transAxes, fontsize=8, va='center', ha='right',
                   fontweight='bold')

plt.suptitle('Test Set Samples (100% Real Data - No Augmentation)', 
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "test_samples.png"), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: test_samples.png")
plt.close()

# Report
report_path = os.path.join(OUTPUT_DIR, "dataset_report.txt")
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("BALANCED TRAINING SET (CAPPED TO SMALLEST CLASS)\n")
    f.write("="*80 + "\n\n")
    
    f.write("METHODOLOGY:\n")
    f.write("  1. Filtered sparse samples (< 1% density)\n")
    f.write("  2. Split into train/val/test with stratification\n")
    f.write(f"  3. Capped ONLY training set to smallest class ({min_train_class_size} samples/class)\n")
    f.write("  4. Val/Test keep original distribution - NO augmentation - 100% real samples\n\n")
    
    f.write(f"Original dataset:     {len(images):,} samples\n")
    f.write(f"After filtering:      {len(filtered_images):,} samples\n\n")
    
    f.write(f"Smallest training class: Label {min_train_class_label} ({CLASS_NAMES.get(min_train_class_label)})\n")
    f.write(f"Training samples per class: {min_train_class_size}\n\n")
    
    f.write(f"TRAIN SET:            {len(X_train_final):,} samples (balanced, 100% real)\n")
    f.write(f"VAL SET:              {len(X_val):,} samples (original distribution, 100% real)\n")
    f.write(f"TEST SET:             {len(X_test):,} samples (original distribution, 100% real)\n\n")
    
    f.write(f"Training balance ratio: {balance_ratio:.2f} (1.0 = perfect)\n\n")
    
    f.write("-"*80 + "\n")
    f.write("TRAINING SET DISTRIBUTION (BALANCED):\n")
    f.write("-"*80 + "\n")
    for label in sorted(valid_classes):
        count = train_counts[label]
        percentage = count / len(X_train_final) * 100
        class_name = CLASS_NAMES.get(label, f"Unknown-{label}")
        f.write(f"Label {label:2d} - {class_name:25s}: {count:5d} ({percentage:5.2f}%)\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("VAL SET DISTRIBUTION (ORIGINAL):\n")
    f.write("-"*80 + "\n")
    val_counts = Counter(y_val)
    for label in sorted(valid_classes):
        count = val_counts[label]
        percentage = count / len(y_val) * 100
        class_name = CLASS_NAMES.get(label, f"Unknown-{label}")
        f.write(f"Label {label:2d} - {class_name:25s}: {count:5d} ({percentage:5.2f}%)\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("TEST SET DISTRIBUTION (ORIGINAL):\n")
    f.write("-"*80 + "\n")
    test_counts = Counter(y_test)
    for label in sorted(valid_classes):
        count = test_counts[label]
        percentage = count / len(y_test) * 100
        class_name = CLASS_NAMES.get(label, f"Unknown-{label}")
        f.write(f"Label {label:2d} - {class_name:25s}: {count:5d} ({percentage:5.2f}%)\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("TRAINING SET BALANCED - NO CLASS WEIGHTS NEEDED FOR TRAINING!\n")
    f.write("="*80 + "\n")

print(f"  ✓ Saved: dataset_report.txt")

print("\n" + "="*80)
print("✅ COMPLETE! Balanced training set prepared")
print("="*80)
print(f"\nFiles in {OUTPUT_DIR}:")
print(f"  • train_data_memory_safe_own_nonVPN_p2p.npy / train_labels_memory_safe_own_nonVPN_p2p.npy ({len(X_train_final):,} samples)")
print(f"  • val_data_memory_safe_own_nonVPN_p2p.npy / val_labels_memory_safe_own_nonVPN_p2p.npy ({len(X_val):,} samples)")  
print(f"  • test_data_memory_safe_own_nonVPN_p2p.npy / test_labels_memory_safe_own_nonVPN_p2p.npy ({len(X_test):,} samples)")
print(f"  • split_distribution.png")
print(f"  • test_samples.png")
print(f"  • dataset_report.txt")
print(f"\nKey points:")
print(f"  ✓ ALL data is 100% real (no augmentation)")
print(f"  ✓ ONLY training set capped to {min_train_class_size} samples/class")
print(f"  ✓ Val/Test keep original distribution")
print(f"  ✓ Training set balance ratio: {balance_ratio:.2f}")
print(f"  ✓ No class weights needed for training")
print(f"  ✓ Ready to train!")
print("="*80)

