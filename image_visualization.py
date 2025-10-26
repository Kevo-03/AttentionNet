import numpy as np
import matplotlib.pyplot as plt
import random

# Load processed data
images = np.load("processed_test/idx/data.npy")   # shape (N, 28, 28)
labels = np.load("processed_test/idx/labels.npy") # shape (N,)

# Label mapping for display
class_names = {
    0: "Chat (NonVPN)",
    1: "Email (NonVPN)",
    2: "File (NonVPN)",
    3: "P2P (NonVPN)",
    4: "Streaming (NonVPN)",
    5: "VoIP (NonVPN)",
    6: "Chat (VPN)",
    7: "Email (VPN)",
    8: "File (VPN)",
    9: "P2P (VPN)",
    10: "Streaming (VPN)",
    11: "VoIP (VPN)",
}

# How many samples to show per class
SAMPLES = 9

# Prepare figure
plt.figure(figsize=(12, 18))

for label in range(12):
    class_imgs = images[labels == label]

    if len(class_imgs) == 0:
        print(f"No samples for class {label}")
        continue

    sample_idxs = np.random.choice(len(class_imgs), 
                                   size=min(SAMPLES, len(class_imgs)),
                                   replace=False)

    # Create a subplot for this class
    for i, idx in enumerate(sample_idxs):
        plt.subplot(12, SAMPLES, label*SAMPLES + i + 1)
        plt.imshow(class_imgs[idx], cmap="gray")
        plt.axis("off")

        # Only label first image in the row
        if i == 0:
            plt.title(class_names[label], fontsize=8, pad=4)

plt.tight_layout()
plt.show()