"""
Simple visualization for detailed labels - exactly like visualize.py format
Shows multiple samples per detailed label in a grid
"""
import os
import numpy as np
import matplotlib.pyplot as plt

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
IMAGE_DIR = os.path.join(PROJECT_ROOT, "processed_data/memory_safe/data_memory_safe.npy")
LABELS_DIR = os.path.join(PROJECT_ROOT, "processed_data/memory_safe/labels_memory_safe.npy")

# Load processed data
images = np.load(IMAGE_DIR)   # shape (N, 28, 28)
labels = np.load(LABELS_DIR)  # shape (N,) - detailed string labels

# Get unique labels and sort them
unique_labels = sorted(np.unique(labels))
print(f"[+] Found {len(unique_labels)} unique detailed labels:")
for i, label in enumerate(unique_labels, 1):
    count = np.sum(labels == label)
    print(f"    {i}. {label}: {count} samples")

# Configuration
SAMPLES = 6  # How many samples to show per class
LABELS_PER_FIGURE = 10  # How many labels to show per figure/GUI

# Split labels into chunks for multiple figures
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

label_chunks = list(chunks(unique_labels, LABELS_PER_FIGURE))
print(f"\n[+] Creating {len(label_chunks)} separate figures ({LABELS_PER_FIGURE} labels each)")

# Create output directory
output_dir = os.path.join(PROJECT_ROOT, "processed_test/visualizations")
os.makedirs(output_dir, exist_ok=True)

# Create a figure for each chunk
for fig_num, label_chunk in enumerate(label_chunks, 1):
    n_classes_in_chunk = len(label_chunk)
    fig_height = max(n_classes_in_chunk * 1.5, 8)
    
    plt.figure(figsize=(18, fig_height))
    
    for row_idx, label in enumerate(label_chunk):
        class_imgs = images[labels == label]

        if len(class_imgs) == 0:
            print(f"[!] No samples for label: {label}")
            continue

        sample_idxs = np.random.choice(len(class_imgs), 
                                       size=min(SAMPLES, len(class_imgs)),
                                       replace=False)

        # Create a subplot for this class
        for col_idx, idx in enumerate(sample_idxs):
            plt.subplot(n_classes_in_chunk, SAMPLES, row_idx * SAMPLES + col_idx + 1)
            plt.imshow(class_imgs[idx], cmap="gray", interpolation='nearest')
            plt.axis("off")

            # Only label first image in the row
            if col_idx == 0:
                plt.title(label, fontsize=9, pad=3, loc='left', fontweight='bold')

    plt.suptitle(f'Detailed Labels Visualization - Part {fig_num}/{len(label_chunks)}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save to file
    output_file = os.path.join(output_dir, f"detailed_labels_grid_part{fig_num}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"    [✓] Saved part {fig_num}: {output_file}")

print(f"\n[✓] All {len(label_chunks)} visualizations saved to: {output_dir}")
print(f"\n[+] Opening all figures...")
plt.show()

