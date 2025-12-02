import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))

DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data/final/memory_safe/own_nonVPN_p2p")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model_output/memory_safe/tensor_visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 128

CLASS_NAMES = {
    0: "Chat (NonVPN)", 1: "Email (NonVPN)", 2: "File (NonVPN)",
    3: "P2P (NonVPN)", 4: "Streaming (NonVPN)", 5: "VoIP (NonVPN)",
    6: "Chat (VPN)",   7: "Email (VPN)",   8: "File (VPN)",
    9: "P2P (VPN)",   10: "Streaming (VPN)", 11: "VoIP (VPN)"
}

# Choose whether to visualize augmented or non-augmented tensors
USE_AUGMENTED = True   # True = same as training; False = clean tensors


# ============================================================================
# DATASET (SAME LOGIC AS TRAINING)
# ============================================================================
class TrafficDataset(Dataset):
    def __init__(self, data_path, labels_path, augment=False):
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        aug = img.astype(np.float32)

        # 1) small Gaussian noise (70% of the time)
        if np.random.rand() < 0.7:
            noise = np.random.normal(0.0, 3.0, aug.shape)
            aug = np.clip(aug + noise, 0, 255)

        # 2) tiny random erasing (30% of the time)
        if np.random.rand() < 0.3:
            h, w = aug.shape
            eh = np.random.randint(2, 4)
            ew = np.random.randint(4, 8)
            y = np.random.randint(0, h - eh)
            x = np.random.randint(0, w - ew)
            aug[y:y+eh, x:x+ew] = 0.0

        # 3) small horizontal shift (30% of the time)
        if np.random.rand() < 0.3:
            shift = np.random.randint(-2, 3)  # -2..2
            aug = np.roll(aug, shift, axis=1)
            if shift > 0:
                aug[:, :shift] = 0.0
            elif shift < 0:
                aug[:, shift:] = 0.0

        return aug.astype(np.float32)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.augment:
            img = self._augment_image(img)
        else:
            img = img.astype(np.float32)

        # normalize to [0, 1]
        img = img / 255.0
        # add channel dim -> (1, 28, 28)
        img = np.expand_dims(img, axis=0)

        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_tensor_batch(loader, split: str, n: int = 16):
    images, labels = next(iter(loader))  # images: (B, 1, 28, 28), float32
    print(f"\n[{split.upper()}] Batch from DataLoader")
    print(f"  images.shape = {images.shape}")
    print(f"  images.dtype = {images.dtype}")
    print(f"  images.min() = {images.min().item():.4f}")
    print(f"  images.max() = {images.max().item():.4f}")
    print(f"  labels.shape = {labels.shape}")
    print(f"  labels[:10]  = {labels[:10].tolist()}")

    # choose first n samples
    n = min(n, images.size(0))
    images = images[:n].cpu()      # (n, 1, 28, 28)
    labels = labels[:n].cpu()

    # (n, 1, 28, 28) -> (n, 28, 28) for plotting
    imgs_np = images.squeeze(1).numpy()

    side = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(side, side, figsize=(side * 2.2, side * 2.2))

    axes = np.array(axes).reshape(side, side)

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            img = imgs_np[i]   # already in [0, 1] if all good
            label_idx = int(labels[i].item())
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"{label_idx}: {CLASS_NAMES[label_idx]}", fontsize=7)

    aug_str = "augmented" if USE_AUGMENTED else "clean"
    plt.suptitle(f"{split.upper()} batch tensors ({aug_str})", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fname = f"{split}_tensor_batch_{aug_str}.png"
    save_path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved tensor batch visualization: {save_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("PYTORCH TENSOR VISUALIZATION (WHAT THE MODEL ACTUALLY SEES)")
    print("=" * 80)
    print(f"DATA_DIR   = {DATA_DIR}")
    print(f"OUTPUT_DIR = {OUTPUT_DIR}")
    print(f"USE_AUGMENTED = {USE_AUGMENTED}")
    print("=" * 80)

    # Use TRAIN split by default (what you feed during training)
    train_data_path = os.path.join(DATA_DIR, "train_data_memory_safe_own_nonVPN_p2p.npy")
    train_labels_path = os.path.join(DATA_DIR, "train_labels_memory_safe_own_nonVPN_p2p.npy")

    dataset = TrafficDataset(
        train_data_path,
        train_labels_path,
        augment=USE_AUGMENTED   # True → exactly like training
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    visualize_tensor_batch(loader, split="train", n=16)

    print("\nDone. Open the saved PNG to inspect the tensors.")
    print("=" * 80)


if __name__ == "__main__":
    main()