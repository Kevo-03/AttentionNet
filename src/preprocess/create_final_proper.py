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

INPUT_DATA = os.path.join(PROJECT_ROOT, "processed_data/memory_safe/own_nonVPN_p2p/data_memory_safe.npy")
INPUT_LABELS = os.path.join(PROJECT_ROOT, "processed_data/memory_safe/own_nonVPN_p2p/labels_memory_safe.npy")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data/final/memory_safe/own_nonVPN_p2p/more_data_basic_aug")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
MIN_DENSITY = 0.01  # Keep flows with at least 1% non-zero pixels
MIN_SAMPLES_PER_CLASS = 0  # Drop classes with fewer samples
TARGET_BALANCE = 8000  # Target samples per class (balance training set)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASS_NAMES = {
    0: "Chat (NonVPN)", 1: "Email (NonVPN)", 2: "File (NonVPN)", 
    3: "P2P (NonVPN)",4: "Streaming (NonVPN)", 5: "VoIP (NonVPN)",
    6: "Chat (VPN)", 7: "Email (VPN)", 8: "File (VPN)", 
    9: "P2P (VPN)",10: "Streaming (VPN)", 11: "VoIP (VPN)"
}

print("="*80)
print("CREATING PROPERLY SPLIT AND BALANCED DATASET")
print("="*80)
print(f"Strategy:")
print(f"  1. Filter sparse samples (< {MIN_DENSITY*100:.0f}% density)")
print(f"  2. Balance via undersampling (target: {TARGET_BALANCE} per class)")
print(f"  3. Split: {TRAIN_RATIO*100:.0f}% train, {VAL_RATIO*100:.0f}% val, {TEST_RATIO*100:.0f}% test")
print(f"  4. Augment ONLY training data")
print("="*80)

# ============================================================================
# STEP 1: Load and Filter
# ============================================================================
print("\n[Step 1/6] Loading and filtering data...")
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

unique_labels = np.unique(filtered_labels)

# ============================================================================
# STEP 2: Balance via Smart Undersampling
# ============================================================================
print(f"\n[Step 2/6] Balancing dataset...")

balanced_indices = []

for label in unique_labels:
    label_mask = filtered_labels == label
    label_indices = np.where(label_mask)[0]
    n_available = len(label_indices)
    label_densities = filtered_densities[label_mask]
    
    class_name = CLASS_NAMES.get(label, f"Unknown-{label}")
    
    if n_available > TARGET_BALANCE:
        # Undersample: keep the densest samples
        sorted_idx = np.argsort(label_densities)[::-1]
        top_indices = sorted_idx[:TARGET_BALANCE]
        selected = label_indices[top_indices]
        print(f"  Label {label:2d} ({class_name:25s}): {n_available:5d} → {TARGET_BALANCE:5d} (kept densest)")
    else:
        # Keep all
        selected = label_indices
        print(f"  Label {label:2d} ({class_name:25s}): {n_available:5d} → {n_available:5d} (kept all)")
    
    balanced_indices.extend(selected)

balanced_images = filtered_images[balanced_indices]
balanced_labels = filtered_labels[balanced_indices]

print(f"\n  Balanced dataset: {len(balanced_images)} samples")
print(f"  Distribution: {dict(Counter(balanced_labels))}")

# ============================================================================
# STEP 3: Split into Train/Val/Test (BEFORE augmentation!)
# ============================================================================
print(f"\n[Step 3/6] Splitting into train/val/test...")

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    balanced_images, balanced_labels, 
    test_size=TEST_RATIO, 
    stratify=balanced_labels, 
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

print(f"  Train set: {len(X_train)} samples ({len(X_train)/len(balanced_images)*100:.1f}%)")
print(f"  Val set:   {len(X_val)} samples ({len(X_val)/len(balanced_images)*100:.1f}%)")
print(f"  Test set:  {len(X_test)} samples ({len(X_test)/len(balanced_images)*100:.1f}%)")

print(f"\n  Train distribution: {dict(Counter(y_train))}")
print(f"  Val distribution:   {dict(Counter(y_val))}")
print(f"  Test distribution:  {dict(Counter(y_test))}")

# ============================================================================
# STEP 4: Augment ONLY Training Data for Minority Classes
# ============================================================================
print(f"\n[Step 4/6] Augmenting minority classes in TRAINING set only...")

def augment_image(img):
    """Minimal augmentation for network traffic"""
    augmented = img.copy().astype(np.float32)
    
    # Only add very small noise (simulates transmission errors)
    noise = np.random.normal(0, 2, img.shape)
    augmented = augmented + noise
    
    # Clip to valid range
    augmented = np.clip(augmented, 0, 255)
    return augmented.astype(np.uint8)

# Find target for augmentation (use median class size in training set)
train_counter = Counter(y_train)
class_sizes = sorted(train_counter.values())
max_real = max(train_counter.values())
aug_target = max_real

print(f"  Augmentation target: {aug_target} samples per minority class")

augmented_train_images = []
augmented_train_labels = []

for label in unique_labels:
    n_in_train = train_counter[label]
    
    if n_in_train < aug_target:
        # Need to augment
        n_to_add = aug_target - n_in_train
        
        # Get all training samples for this class
        class_mask = y_train == label
        class_images = X_train[class_mask]
        
        # Generate augmented samples
        for _ in range(n_to_add):
            idx = np.random.randint(0, len(class_images))
            augmented = augment_image(class_images[idx])
            augmented_train_images.append(augmented)
            augmented_train_labels.append(label)
        
        class_name = CLASS_NAMES.get(label, f"L{label}")
        print(f"  Label {label:2d} ({class_name:25s}): {n_in_train} → {aug_target} (+{n_to_add} augmented)")

# Combine original training + augmented
if augmented_train_images:
    X_train_aug = np.concatenate([X_train, np.array(augmented_train_images)], axis=0)
    y_train_aug = np.concatenate([y_train, np.array(augmented_train_labels)], axis=0)
else:
    X_train_aug = X_train
    y_train_aug = y_train

# Shuffle training set
shuffle_idx = np.random.permutation(len(X_train_aug))
X_train_final = X_train_aug[shuffle_idx]
y_train_final = y_train_aug[shuffle_idx]

print(f"\n  Final training set: {len(X_train_final)} samples (includes augmentation)")
print(f"  Final distribution: {dict(Counter(y_train_final))}")

# ============================================================================
# STEP 5: Save All Splits
# ============================================================================
print(f"\n[Step 5/6] Saving datasets...")

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
# STEP 6: Generate Report and Visualizations
# ============================================================================
print(f"\n[Step 6/6] Generating visualizations and report...")

# Calculate balance metrics
train_counts = Counter(y_train_final)
max_count = max(train_counts.values())
min_count = min(train_counts.values())
balance_ratio = min_count / max_count

print(f"\n  Training set balance ratio: {balance_ratio:.2f}")
print(f"  (1.0 = perfect balance, >0.5 = good)")

# Visualization 1: Distribution comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (data, title) in enumerate([
    (y_train_final, 'Training Set'),
    (y_val, 'Validation Set'),
    (y_test, 'Test Set')
]):
    ax = axes[idx]
    counter = Counter(data)
    labels_list = sorted(unique_labels)
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

# Visualization 2: Sample images from test set (real data only!)
n_samples = 10
fig, axes = plt.subplots(len(unique_labels), n_samples, figsize=(n_samples*1.5, len(unique_labels)*1.5))

if len(unique_labels) == 1:
    axes = axes.reshape(1, -1)

for label_idx, label in enumerate(sorted(unique_labels)):
    class_imgs = X_test[y_test == label]
    
    for i in range(n_samples):
        ax = axes[label_idx, i] if len(unique_labels) > 1 else axes[i]
        
        if i < len(class_imgs):
            ax.imshow(class_imgs[i], cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        
        if i == 0:
            class_name = CLASS_NAMES.get(label, f"L{label}")
            ax.text(-0.3, 0.5, f"{label}\n{class_name}", 
                   transform=ax.transAxes, fontsize=8, va='center', ha='right',
                   fontweight='bold')

plt.suptitle('Test Set Samples (Real Data Only - No Augmentation)', 
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "test_samples.png"), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: test_samples.png")
plt.close()

# Report
report_path = os.path.join(OUTPUT_DIR, "dataset_report.txt")
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("PROPER TRAIN/VAL/TEST SPLIT WITH BALANCED TRAINING\n")
    f.write("="*80 + "\n\n")
    
    f.write("METHODOLOGY:\n")
    f.write("  1. Filtered sparse samples (< 8% density)\n")
    f.write("  2. Balanced via undersampling majority classes\n")
    f.write("  3. Split into train/val/test BEFORE augmentation\n")
    f.write("  4. Augmented ONLY minority classes in training set\n")
    f.write("  5. Validation and test sets contain 100% real data\n\n")
    
    f.write(f"Original dataset:     {len(images):,} samples\n")
    f.write(f"After filtering:      {len(filtered_images):,} samples\n")
    f.write(f"After balancing:      {len(balanced_images):,} samples\n\n")
    
    f.write(f"TRAIN SET:            {len(X_train_final):,} samples\n")
    f.write(f"  - Original:         {len(X_train):,}\n")
    f.write(f"  - Augmented:        {len(augmented_train_images):,}\n")
    f.write(f"  - Balance ratio:    {balance_ratio:.2f}\n\n")
    
    f.write(f"VAL SET:              {len(X_val):,} samples (100% real)\n")
    f.write(f"TEST SET:             {len(X_test):,} samples (100% real)\n\n")
    
    f.write("-"*80 + "\n")
    f.write("TRAINING SET DISTRIBUTION:\n")
    f.write("-"*80 + "\n")
    for label in sorted(unique_labels):
        count = train_counts[label]
        percentage = count / len(X_train_final) * 100
        class_name = CLASS_NAMES.get(label, f"Unknown-{label}")
        f.write(f"Label {label:2d} - {class_name:25s}: {count:5d} ({percentage:5.2f}%)\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("NO CLASS WEIGHTS NEEDED - Dataset is balanced!\n")
    f.write("="*80 + "\n")

print(f"  ✓ Saved: dataset_report.txt")

print("\n" + "="*80)
print("✅ COMPLETE! Dataset properly prepared for training")
print("="*80)
print(f"\nFiles in {OUTPUT_DIR}:")
print(f"  • train_data.npy / train_labels.npy ({len(X_train_final):,} samples)")
print(f"  • val_data.npy / val_labels.npy ({len(X_val):,} samples)")  
print(f"  • test_data.npy / test_labels.npy ({len(X_test):,} samples)")
print(f"  • split_distribution.png")
print(f"  • test_samples.png")
print(f"  • dataset_report.txt")
print(f"\nKey points:")
print(f"  ✓ Test/Val sets are 100% real data (no augmentation)")
print(f"  ✓ Training set is balanced (ratio: {balance_ratio:.2f})")
print(f"  ✓ No class weights needed")
print(f"  ✓ Ready to train!")
print("="*80)