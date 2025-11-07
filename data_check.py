import numpy as np
import matplotlib.pyplot as plt

# Load test data
test_data = np.load("processed_data/final/test_data.npy")
test_labels = np.load("processed_data/final/test_labels.npy")

# Classes that confuse each other
confused_pairs = [
    (0, 1),   # Chat vs Email (NonVPN)
    (0, 4),   # Chat vs VoIP (NonVPN)
    (1, 4),   # Email vs VoIP (NonVPN)
]

fig, axes = plt.subplots(len(confused_pairs), 6, figsize=(18, 9))

for pair_idx, (label1, label2) in enumerate(confused_pairs):
    # Get samples from each class
    class1_samples = test_data[test_labels == label1][:3]
    class2_samples = test_data[test_labels == label2][:3]
    
    for i in range(3):
        axes[pair_idx, i].imshow(class1_samples[i], cmap='gray')
        axes[pair_idx, i].set_title(f"Label {label1}", fontsize=9)
        axes[pair_idx, i].axis('off')
        
        axes[pair_idx, i+3].imshow(class2_samples[i], cmap='gray')
        axes[pair_idx, i+3].set_title(f"Label {label2}", fontsize=9)
        axes[pair_idx, i+3].axis('off')

plt.tight_layout()
plt.savefig("confused_classes_comparison.png", dpi=150)
plt.show()

print("Look at the images - can YOU tell them apart?")
print("If not, the model has no chance!")

# Check density distribution for problematic classes
import numpy as np

test_data = np.load("processed_data/final/test_data.npy")
test_labels = np.load("processed_data/final/test_labels.npy")

problematic_classes = [0, 1, 4]  # Chat, Email, VoIP (NonVPN)
class_names = {0: "Chat", 1: "Email", 4: "VoIP"}

for label in problematic_classes:
    class_data = test_data[test_labels == label]
    densities = np.mean(class_data > 0, axis=(1, 2))
    
    print(f"\n{class_names[label]} (NonVPN):")
    print(f"  Avg density: {np.mean(densities)*100:.1f}%")
    print(f"  Min density: {np.min(densities)*100:.1f}%")
    print(f"  Samples with < 15% density: {np.sum(densities < 0.15)} / {len(densities)}")
    print(f"  Sparse ratio: {np.sum(densities < 0.15) / len(densities) * 100:.1f}%")