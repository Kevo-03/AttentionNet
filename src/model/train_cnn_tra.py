import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
from tqdm import tqdm
import json
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================================
# CONFIGURATION
# ============================================================================
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))

DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data/final/memory_safe/own_nonVPN_p2p")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model_output/memory_safe/3_conv/transformer_model_2dpos_encode")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 80  # slightly higher for CNN+Transformer

# Device selection (CPU / CUDA / MPS)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

CLASS_NAMES = {
    0: "Chat (NonVPN)", 1: "Email (NonVPN)", 2: "File (NonVPN)", 
    3: "P2P (NonVPN)",4: "Streaming (NonVPN)", 5: "VoIP (NonVPN)", 
    6: "Chat (VPN)", 7: "Email (VPN)", 8: "File (VPN)", 
    9: "P2P (VPN)",10: "Streaming (VPN)", 11: "VoIP (VPN)"
}

N_CLASSES = len(CLASS_NAMES)  # 12

print("="*80)
print("NETWORK TRAFFIC CLASSIFICATION - CNN+Transformer TRAINING")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {NUM_EPOCHS}")
print("="*80)

# ============================================================================
# DATASET CLASS
# ============================================================================
class TrafficDataset(Dataset):
    def __init__(self, data_path, labels_path, augment=False):
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """
        On-the-fly augmentation tailored for traffic flow images.
        img: (28, 28), uint8
        """
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
            # zero-fill wrapped area
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

import math
import torch
import torch.nn as nn


# ============================================================
# 2D SINUSOIDAL POSITIONAL ENCODING
# ============================================================
class SinusoidalPositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding.

    Channels are split into 4 chunks:
      [sin_y, cos_y, sin_x, cos_x]

    We add this on the CNN feature map (B, C, H, W) before
    flattening to a sequence for the Transformer.
    """
    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        assert channels % 4 == 0, "channels must be divisible by 4 for 2D sinusoidal PE"
        self.channels = channels
        self.height = height
        self.width = width

        pe = self._build_pe(channels, height, width)
        # shape: (1, C, H, W) so it broadcasts across batch
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build_pe(channels: int, height: int, width: int) -> torch.Tensor:
        c = channels
        c_half = c // 2
        c_quarter = c // 4

        # positions
        y_pos = torch.arange(height, dtype=torch.float32).unsqueeze(1)  # (H, 1)
        x_pos = torch.arange(width, dtype=torch.float32).unsqueeze(1)   # (W, 1)

        # frequency terms
        div_term_y = torch.exp(torch.arange(0, c_quarter, dtype=torch.float32)
                               * (-math.log(10000.0) / c_quarter))
        div_term_x = torch.exp(torch.arange(0, c_quarter, dtype=torch.float32)
                               * (-math.log(10000.0) / c_quarter))

        # (H, c_half): [sin_y | cos_y]
        pe_y = torch.zeros(height, c_half, dtype=torch.float32)
        pe_y[:, 0:c_quarter] = torch.sin(y_pos * div_term_y)
        pe_y[:, c_quarter:c_half] = torch.cos(y_pos * div_term_y)

        # (W, c_half): [sin_x | cos_x]
        pe_x = torch.zeros(width, c_half, dtype=torch.float32)
        pe_x[:, 0:c_quarter] = torch.sin(x_pos * div_term_x)
        pe_x[:, c_quarter:c_half] = torch.cos(x_pos * div_term_x)

        # Combine into (H, W, C)
        pe = torch.zeros(height, width, c, dtype=torch.float32)
        # first half = y encoding, broadcast across width
        pe[..., :c_half] = pe_y.unsqueeze(1).expand(height, width, c_half)
        # second half = x encoding, broadcast across height
        pe[..., c_half:] = pe_x.unsqueeze(0).expand(height, width, c_half)

        # (C, H, W) and add batch dim later
        pe = pe.permute(2, 0, 1).unsqueeze(0)
        return pe  # (1, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) with H=height, W=width
        return x + self.pe.to(x.device)


# ============================================================
# CNN + TRANSFORMER WITH 2D SINUSOIDAL PE
# ============================================================
class TrafficCNN_Transformer(nn.Module):
    def __init__(self, num_classes: int = 12):
        super().__init__()

        # -----------------------
        # CNN BACKBONE (your 3-layer version)
        # -----------------------
        # First layer: wide 1D kernel over width
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 25), padding=(0, 12))
        self.bn1 = nn.BatchNorm2d(64)

        # Second layer: another wide 1D kernel
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 15), padding=(0, 7))
        self.bn2 = nn.BatchNorm2d(128)

        # Third layer: 2D kernel
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool_width = nn.MaxPool2d((1, 2))   # halves width
        self.pool_2d = nn.MaxPool2d((2, 2))      # halves H and W

        self.relu = nn.ReLU(inplace=True)
        self.dropout_fc = nn.Dropout(p=0.5)     # heavy dropout for dense layers
        self.dropout_out = nn.Dropout(p=0.3)    # light dropout before final output
        # Input (28,28) → shapes:
        # after conv1 + pool_width:  (64, 28, 14)
        # after conv2 + pool_width:  (128, 28, 7)
        # after conv3 + pool_2d:     (256, 14, 3)

        self.d_model = 256        # channel dim = embedding dim
        self.h = 14
        self.w = 3
        self.seq_len = self.h * self.w  # 42 tokens

        # 2D sinusoidal positional encoding on (C,H,W)
        self.pos2d = SinusoidalPositionalEncoding2D(
            channels=self.d_model,
            height=self.h,
            width=self.w,
        )

        # -----------------------
        # TRANSFORMER
        # -----------------------
        self.num_heads = 4        # 256 / 4 = 64
        self.num_layers = 2

        self.norm_in = nn.LayerNorm(self.d_model)
        self.norm_out = nn.LayerNorm(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            batch_first=True,   # (B, L, d_model)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
        )

        # -----------------------
        # CLASSIFIER HEAD
        # -----------------------
        # after transformer: (B, L, 256) → mean over L → (B, 256)
        self.fc1 = nn.Linear(self.d_model, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)

        # ---- CNN ----
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool_width(x)   # (B, 64, 28, 14)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool_width(x)   # (B, 128, 28, 7)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool_2d(x)      # (B, 256, 14, 3)

        # ---- 2D positional encoding on feature map ----
        #x = self.pos2d(x)        # (B, 256, 14, 3)

        # ---- to sequence for Transformer ----
        x = x.flatten(2)         # (B, 256, 42)
        x = x.permute(0, 2, 1)   # (B, 42, 256)

        # LayerNorm before Transformer
        x = self.norm_in(x)

        # Transformer
        x = self.transformer(x)  # (B, 42, 256)

        # LayerNorm after Transformer
        x = self.norm_out(x)

        # Global average over tokens
        x = x.mean(dim=1)        # (B, 256)

        # Classifier
        x = self.dropout_fc(x)           # strong dropout before FC1
        x = self.relu(self.fc1(x))       # hidden FC layer
        x = self.dropout_out(x)          # lighter dropout before output
        x = self.fc2(x)                  # logits

        return x

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/5] Loading datasets...")

train_dataset = TrafficDataset(
    os.path.join(DATA_DIR, "train_data_memory_safe_own_nonVPN_p2p.npy"),
    os.path.join(DATA_DIR, "train_labels_memory_safe_own_nonVPN_p2p.npy"),
    augment=True,          # <-- turn on for training
)
val_dataset = TrafficDataset(
    os.path.join(DATA_DIR, "val_data_memory_safe_own_nonVPN_p2p.npy"),
    os.path.join(DATA_DIR, "val_labels_memory_safe_own_nonVPN_p2p.npy"),
    augment=False,
)
test_dataset = TrafficDataset(
    os.path.join(DATA_DIR, "test_data_memory_safe_own_nonVPN_p2p.npy"),
    os.path.join(DATA_DIR, "test_labels_memory_safe_own_nonVPN_p2p.npy"),
    augment=False,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train: {len(train_dataset)} samples")
print(f"  Val:   {len(val_dataset)} samples")
print(f"  Test:  {len(test_dataset)} samples")

# ============================================================================
# INITIALIZE MODEL
# ============================================================================
print("\n[2/5] Initializing model...")

model = TrafficCNN_Transformer(num_classes=N_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# ---- Warmup + Cosine ----
WARMUP_EPOCHS = 10  # 3–5 is usually enough for your scale

warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,       # start at 0.1 * LEARNING_RATE
    end_factor=1.0,         # reach full LR at the end of warmup
    total_iters=WARMUP_EPOCHS
)

cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS - WARMUP_EPOCHS,
    eta_min=1e-6
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[WARMUP_EPOCHS]
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/len(loader), 'acc': 100.*correct/total})
    
    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    val_macro_f1 = f1_score(all_targets, all_preds, average='macro')
    return val_loss, val_acc, val_macro_f1

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n[3/5] Training model...")

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'val_macro_f1': []
}

best_acc = 0.0
best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")

early_stop_patience = 10
no_improve_epochs = 0

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 40)

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, val_macro_f1 = validate(model, val_loader, criterion, DEVICE)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_macro_f1'].append(val_macro_f1)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}% | Val Macro F1: {val_macro_f1:.4f}")

    old_lr = optimizer.param_groups[0]['lr']

    # Cosine annealing: step every epoch
    scheduler.step()

    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != old_lr:
        print(f"LR updated: {old_lr:.6f} → {new_lr:.6f}")

    # Early stopping and best model still based on validation accuracy
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"✓ Saved best model (Val Accuracy: {val_acc:.2f})")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        print(f"No improvement for {no_improve_epochs} epoch(s)")

    if no_improve_epochs >= early_stop_patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
        break

# ============================================================================
# EVALUATION ON TEST SET
# ============================================================================
print("\n[4/5] Evaluating on test set...")

# Load best model
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate metrics
test_acc = 100. * np.mean(all_preds == all_labels)
test_macro_f1 = f1_score(all_labels, all_preds, average='macro')

print(f"\n  Test Accuracy: {test_acc:.2f}%")
print(f"  Test Macro F1: {test_macro_f1:.4f}")

# Classification report
print("\n  Classification Report:")
print("-" * 80)
report = classification_report(
    all_labels, all_preds,
    target_names=[CLASS_NAMES[i] for i in range(N_CLASSES)],
    digits=3
)
print(report)

# Save report
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), 'w') as f:
    f.write(f"Test Accuracy: {test_acc:.2f}%\n")
    f.write(f"Test Macro F1: {test_macro_f1:.4f}\n\n")
    f.write(report)

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[5/5] Generating visualizations...")

# 1. Training history
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

ax1, ax2, ax3 = axes

ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3.plot(history['val_macro_f1'], label='Val Macro F1', linewidth=2)
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Macro F1', fontsize=12, fontweight='bold')
ax3.set_title('Validation Macro F1', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: training_history.png")
plt.close()

# 2. Confusion matrix
# 2. Confusion matrix (normalized by true class)
cm = confusion_matrix(all_labels, all_preds)

# Normalize per row (true class)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)  # handle any division-by-zero rows safely

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=[CLASS_NAMES[i] for i in range(N_CLASSES)],
    yticklabels=[CLASS_NAMES[i] for i in range(N_CLASSES)],
    vmin=0.0,
    vmax=1.0
)
plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
plt.ylabel("True Label", fontsize=12, fontweight="bold")
plt.title(
    f"Normalized Confusion Matrix (Row-wise) – Test Acc: {test_acc:.2f}%",
    fontsize=14,
    fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
            dpi=150, bbox_inches="tight")
print("  ✓ Saved: confusion_matrix.png (normalized)")
plt.close()

# 3. Per-class accuracy
from sklearn.metrics import accuracy_score
class_accuracies = []
for i in range(N_CLASSES):
    mask = all_labels == i
    if mask.sum() > 0:
        class_acc = accuracy_score(all_labels[mask], all_preds[mask]) * 100
        class_accuracies.append(class_acc)
    else:
        class_accuracies.append(0)

plt.figure(figsize=(14, 6))
bars = plt.bar(range(N_CLASSES), class_accuracies, color='steelblue', edgecolor='black')
plt.xlabel('Class Label', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Per-Class Accuracy on Test Set', fontsize=14, fontweight='bold')
plt.xticks(range(N_CLASSES), range(N_CLASSES))
plt.ylim(0, 105)
plt.grid(axis='y', alpha=0.3)

for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "per_class_accuracy.png"), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: per_class_accuracy.png")
plt.close()

# Save history
with open(os.path.join(OUTPUT_DIR, "training_history.json"), 'w') as f:
    json.dump(history, f, indent=2)

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"  • best_model.pth (Best validation model)")
print(f"  • training_history.png")
print(f"  • confusion_matrix.png")
print(f"  • per_class_accuracy.png")
print(f"  • classification_report.txt")
print(f"  • training_history.json")
print(f"\nBest Validation Macro F1: {best_acc:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print("="*80)