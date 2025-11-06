import os
import numpy as np
import matplotlib.pyplot as plt
import random

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
IMAGE_DIR = os.path.join(PROJECT_ROOT,"processed_data/final/train_data.npy")
LABELS_DIR = os.path.join(PROJECT_ROOT,"processed_data/finalso/train_labels.npy")

images = np.load(IMAGE_DIR)   # shape (N, 28, 28)
labels = np.load(LABELS_DIR) # shape (N,)

class_names = {
    0: "Chat (NonVPN)",
    1: "Email (NonVPN)",
    2: "File (NonVPN)",
    3: "Streaming (NonVPN)",
    4: "VoIP (NonVPN)",
    5: "Chat (VPN)",
    6: "Email (VPN)",
    7: "File (VPN)",
    8: "P2P (VPN)",
    9: "Streaming (VPN)",
    10: "VoIP (VPN)",
}

SAMPLES = 9

plt.figure(figsize=(12, 18))

for label in range(12):
    class_imgs = images[labels == label]

    if len(class_imgs) == 0:
        print(f"No samples for class {label}")
        continue

    sample_idxs = np.random.choice(len(class_imgs), 
                                   size=min(SAMPLES, len(class_imgs)),
                                   replace=False)

    for i, idx in enumerate(sample_idxs):
        plt.subplot(12, SAMPLES, label*SAMPLES + i + 1)
        plt.imshow(class_imgs[idx], cmap="gray")
        plt.axis("off")

        if i == 0:
            plt.title(class_names[label], fontsize=8, pad=4)

plt.tight_layout()
plt.show()