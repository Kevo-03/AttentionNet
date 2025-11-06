import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))

INPUT_DATA = os.path.join(PROJECT_ROOT, "processed_data/idx/data_without_mem.npy")
INPUT_LABELS = os.path.join(PROJECT_ROOT, "processed_data/idx/labels_without_mem.npy")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data/final")
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# Strategy parameters
MIN_DENSITY = 0.08  # Keep flows with at least 8% non-zero pixels
TARGET_SAMPLES_PER_CLASS = 3000  # Target for balanced dataset
MIN_SAMPLES_TO_KEEP = 200  # Minimum samples to keep per class

CLASS_NAMES = {
    0: "Chat (NonVPN)", 1: "Email (NonVPN)", 2: "File (NonVPN)", 
    3: "Streaming (NonVPN)", 4: "VoIP (NonVPN)",
    5: "Chat (VPN)", 6: "Email (VPN)", 7: "File (VPN)", 
    8: "P2P (VPN)", 9: "Streaming (VPN)", 10: "VoIP (VPN)"
}

print("="*80)
print("CREATING FINAL TRAINING DATASET")
print("="*80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[Step 1/5] Loading data...")
images = np.load(INPUT_DATA)
labels = np.load(INPUT_LABELS)
print(f"  Loaded: {len(images)} samples")
print(f"  Original distribution: {dict(Counter(labels))}")

# ============================================================================
# STEP 2: Filter Sparse Samples
# ============================================================================
print(f"\n[Step 2/5] Filtering sparse samples (density < {MIN_DENSITY*100:.0f}%)...")
densities = np.mean(images > 0, axis=(1, 2))
density_mask = densities >= MIN_DENSITY

filtered_images = images[density_mask]
filtered_labels = labels[density_mask]
filtered_densities = densities[density_mask]

print(f"  Removed: {len(images) - len(filtered_images)} sparse samples")
print(f"  Remaining: {len(filtered_images)} samples")
print(f"  Filtered distribution: {dict(Counter(filtered_labels))}")

# ============================================================================
# STEP 3: Balance Classes with Smart Strategy
# ============================================================================
print(f"\n[Step 3/5] Balancing classes...")
print(f"  Target: {TARGET_SAMPLES_PER_CLASS} samples per class")
print(f"  Minimum: {MIN_SAMPLES_TO_KEEP} samples per class")

balanced_indices = []
augmentation_plan = {}

for label in sorted(np.unique(filtered_labels)):
    label_mask = filtered_labels == label
    label_indices = np.where(label_mask)[0]
    n_available = len(label_indices)
    
    class_name = CLASS_NAMES.get(label, f"Unknown-{label}")
    
    if n_available == 0:
        print(f"  Label {label:2d} ({class_name}): SKIPPED (no samples)")
        continue
    
    # Determine how many to keep/augment
    if n_available >= TARGET_SAMPLES_PER_CLASS:
        # Undersample: randomly select TARGET samples
        selected = np.random.choice(label_indices, size=TARGET_SAMPLES_PER_CLASS, replace=False)
        balanced_indices.extend(selected)
        print(f"  Label {label:2d} ({class_name:25s}): {n_available:5d} → {TARGET_SAMPLES_PER_CLASS:5d} (undersampled)")
    
    elif n_available >= MIN_SAMPLES_TO_KEEP:
        # Keep all and plan augmentation
        balanced_indices.extend(label_indices)
        n_to_augment = TARGET_SAMPLES_PER_CLASS - n_available
        augmentation_plan[label] = {
            'indices': label_indices,
            'n_augment': n_to_augment
        }
        print(f"  Label {label:2d} ({class_name:25s}): {n_available:5d} → {TARGET_SAMPLES_PER_CLASS:5d} (kept all + augment {n_to_augment})")
    
    else:
        # Too few samples, keep all but don't augment to target
        balanced_indices.extend(label_indices)
        print(f"  Label {label:2d} ({class_name:25s}): {n_available:5d} → {n_available:5d} (kept all, below minimum)")

# Extract balanced data
balanced_images = filtered_images[balanced_indices]
balanced_labels = filtered_labels[balanced_indices]

print(f"\n  Before augmentation: {len(balanced_images)} samples")

# ============================================================================
# STEP 4: Augment Minority Classes
# ============================================================================
print(f"\n[Step 4/5] Augmenting minority classes...")

def augment_image(img):
    """Simple augmentation: add noise, shift pixels"""
    augmented = img.copy().astype(np.float32)
    
    # Add small random noise
    noise = np.random.normal(0, 3, img.shape)
    augmented = augmented + noise
    
    # Random horizontal flip
    if np.random.random() > 0.5:
        augmented = np.fliplr(augmented)
    
    # Random vertical shift (±2 pixels)
    shift = np.random.randint(-2, 3)
    if shift != 0:
        augmented = np.roll(augmented, shift, axis=0)
        if shift > 0:
            augmented[:shift, :] = 0
        else:
            augmented[shift:, :] = 0
    
    # Clip to valid range
    augmented = np.clip(augmented, 0, 255)
    return augmented.astype(np.uint8)

augmented_images = []
augmented_labels = []

for label, plan in augmentation_plan.items():
    indices = plan['indices']
    n_to_augment = plan['n_augment']
    
    # Get original images for this class
    original_images = filtered_images[indices]
    
    # Generate augmented samples
    for _ in range(n_to_augment):
        # Randomly pick an original image
        idx = np.random.randint(0, len(original_images))
        original = original_images[idx]
        
        # Augment it
        augmented = augment_image(original)
        
        augmented_images.append(augmented)
        augmented_labels.append(label)
    
    print(f"  Label {label:2d}: Generated {n_to_augment} augmented samples")

# Combine original balanced + augmented
if augmented_images:
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    
    final_images = np.concatenate([balanced_images, augmented_images], axis=0)
    final_labels = np.concatenate([balanced_labels, augmented_labels], axis=0)
else:
    final_images = balanced_images
    final_labels = balanced_labels

# Shuffle to mix augmented samples
shuffle_idx = np.random.permutation(len(final_images))
final_images = final_images[shuffle_idx]
final_labels = final_labels[shuffle_idx]

print(f"\n  After augmentation: {len(final_images)} samples")

# ============================================================================
# STEP 5: Save Final Dataset
# ============================================================================
print(f"\n[Step 5/5] Saving final dataset...")

np.save(os.path.join(OUTPUT_DIR, "data.npy"), final_images)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), final_labels)

print(f"  ✓ Saved: {os.path.join(OUTPUT_DIR, 'data.npy')}")
print(f"  ✓ Saved: {os.path.join(OUTPUT_DIR, 'labels.npy')}")

# ============================================================================
# FINAL STATISTICS
# ============================================================================
print("\n" + "="*80)
print("FINAL DATASET STATISTICS")
print("="*80)

final_counter = Counter(final_labels)
print(f"\nTotal samples: {len(final_images)}")
print(f"\nClass distribution:")
print("-"*80)

for label in sorted(np.unique(final_labels)):
    count = final_counter[label]
    percentage = (count / len(final_labels)) * 100
    class_name = CLASS_NAMES.get(label, f"Unknown-{label}")
    
    # Calculate density for this class
    class_mask = final_labels == label
    class_densities = np.mean(final_images[class_mask] > 0, axis=(1, 2))
    avg_density = np.mean(class_densities) * 100
    
    print(f"Label {label:2d} - {class_name:25s}: {count:5d} ({percentage:5.2f}%) | "
          f"Avg density: {avg_density:5.2f}%")

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================
print(f"\n[Bonus] Generating visualizations...")

# 1. Class distribution bar chart
plt.figure(figsize=(12, 6))
labels_list = sorted(np.unique(final_labels))
counts = [final_counter[l] for l in labels_list]
names = [CLASS_NAMES.get(l, f"Label {l}") for l in labels_list]

bars = plt.bar(range(len(labels_list)), counts, color='steelblue', edgecolor='black')
plt.xlabel('Class', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.title('Final Dataset - Class Distribution', fontsize=14, fontweight='bold')
plt.xticks(range(len(labels_list)), [f"{l}" for l in labels_list], fontsize=10)

# Add count labels on bars
for i, (bar, count) in enumerate(zip(bars, counts)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'{count}', ha='center', va='bottom', fontsize=9)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "class_distribution.png"), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: class_distribution.png")
plt.close()

# 2. Sample images grid (10 per class)
fig, axes = plt.subplots(11, 10, figsize=(20, 22))
for label_idx, label in enumerate(sorted(np.unique(final_labels))):
    class_imgs = final_images[final_labels == label]
    
    # Show first 10 samples
    for i in range(10):
        ax = axes[label_idx, i]
        if i < len(class_imgs):
            ax.imshow(class_imgs[i], cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        
        if i == 0:
            ax.text(-0.5, 0.5, f"L{label}", transform=ax.transAxes,
                   fontsize=10, fontweight='bold', va='center', ha='right')

plt.suptitle('Final Dataset - Sample Images (10 per class)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "sample_images.png"), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: sample_images.png")
plt.close()

# 3. Comparison report
report_path = os.path.join(OUTPUT_DIR, "dataset_report.txt")
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("FINAL DATASET CREATION REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Original dataset: {len(images)} samples\n")
    f.write(f"After filtering:  {len(filtered_images)} samples ({len(filtered_images)/len(images)*100:.1f}%)\n")
    f.write(f"Final dataset:    {len(final_images)} samples\n\n")
    
    f.write("Processing steps:\n")
    f.write(f"  1. Filtered samples with density < {MIN_DENSITY*100:.0f}%\n")
    f.write(f"  2. Balanced classes to ~{TARGET_SAMPLES_PER_CLASS} samples each\n")
    f.write(f"  3. Augmented minority classes\n\n")
    
    f.write("Final class distribution:\n")
    f.write("-"*80 + "\n")
    for label in sorted(np.unique(final_labels)):
        count = final_counter[label]
        percentage = (count / len(final_labels)) * 100
        class_name = CLASS_NAMES.get(label, f"Unknown-{label}")
        f.write(f"Label {label:2d} - {class_name:25s}: {count:5d} ({percentage:5.2f}%)\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("Dataset is ready for training!\n")
    f.write("="*80 + "\n")

print(f"  ✓ Saved: dataset_report.txt")

print("\n" + "="*80)
print("✅ COMPLETE! Your dataset is ready for training.")
print("="*80)
print(f"\nFiles created in: {OUTPUT_DIR}")
print(f"  • data.npy - Training images ({len(final_images)} samples)")
print(f"  • labels.npy - Training labels")
print(f"  • dataset_report.txt - Detailed report")
print(f"  • visualizations/ - Plots and sample images")
print("\nNext steps:")
print("  1. Review the visualizations to verify quality")
print("  2. Split into train/val/test sets")
print("  3. Train your model!")
print("="*80)