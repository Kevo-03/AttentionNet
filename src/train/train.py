import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import json
from src.model import TransECANet

# ============================================================================
# CONFIGURATION
# ============================================================================
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))

DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data/final/memory_safe/own_nonVPN_p2p")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model_output/memory_safe/transeca")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = {
    0: "Chat (NonVPN)", 1: "Email (NonVPN)", 2: "File (NonVPN)", 
    3: "P2P (NonVPN)",4: "Streaming (NonVPN)", 5: "VoIP (NonVPN)", 
    6: "Chat (VPN)", 7: "Email (VPN)", 8: "File (VPN)", 
    9: "P2P (VPN)",10: "Streaming (VPN)", 11: "VoIP (VPN)"
}

N_CLASSES = len(CLASS_NAMES)  # 12

print("="*80)
print("NETWORK TRAFFIC CLASSIFICATION - CNN TRAINING")
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
# ============================================================================
# CNN MODEL
# ============================================================================
class TrafficCNN(nn.Module):
    def __init__(self, num_classes=12):
        super(TrafficCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            padding=1  # keeps 28x28
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1  # keeps 14x14
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

        # After conv + pooling:
        # input:  (B, 1, 28, 28)
        # conv1:  (B, 64, 28, 28)
        # pool1:  (B, 64, 14, 14)
        # conv2:  (B, 128, 14, 14)
        # pool2:  (B, 128, 7, 7)
        # Flatten size = 128 * 7 * 7 = 6272
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)   # (B, 64, 14, 14)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)   # (B, 128, 7, 7)

        x = x.view(x.size(0), -1)  # flatten
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
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

model = TransECANet(num_classes=N_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3
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
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n[3/5] Training model...")

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

best_val_acc = 0.0
best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")

early_stop_patience = 8
no_improve_epochs = 0

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 40)

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
    old_lr = optimizer.param_groups[0]['lr']

    scheduler.step(val_acc)

    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != old_lr:
        print(f"✓ Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
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
model.load_state_dict(torch.load(best_model_path))
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

# Calculate metrics
test_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\n  Test Accuracy: {test_acc:.2f}%")

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
    f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
    f.write(report)

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[5/5] Generating visualizations...")

# 1. Training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

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

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: training_history.png")
plt.close()

# 2. Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=range(N_CLASSES),
    yticklabels=range(N_CLASSES)
)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title(f'Confusion Matrix (Test Accuracy: {test_acc:.2f}%)', 
         fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: confusion_matrix.png")
plt.close()

# 3. Per-class accuracy
from sklearn.metrics import accuracy_score
class_accuracies = []
for i in range(N_CLASSES):
    mask = np.array(all_labels) == i
    if mask.sum() > 0:
        class_acc = accuracy_score(np.array(all_labels)[mask], np.array(all_preds)[mask]) * 100
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
print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")
print("="*80)