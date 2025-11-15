"""
TRAINING WITH BALANCED DATASET AND PYTORCH AUGMENTATION
========================================================

Features:
1. Uses balanced dataset from processed_data/balanced/
2. On-the-fly augmentation in PyTorch (different each epoch!)
3. Weighted Random Sampler for oversampling minorities
4. Weighted CrossEntropyLoss for remaining imbalance
5. Semantically valid augmentation for network traffic

Saves to: model_output_balanced/
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))

DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data/balanced_backup")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model_output_balanced")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training parameters
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Augmentation parameters
AUGMENT_STRENGTH = 'moderate'  # 'light', 'moderate', 'heavy'
USE_WEIGHTED_SAMPLER = False
USE_WEIGHTED_LOSS = True

CLASS_NAMES = {
    0: "Chat (NonVPN)", 1: "Email (NonVPN)", 2: "File (NonVPN)", 
    3: "Streaming (NonVPN)", 4: "VoIP (NonVPN)",
    5: "Chat (VPN)", 6: "Email (VPN)", 7: "File (VPN)", 
    8: "P2P (VPN)", 9: "Streaming (VPN)", 10: "VoIP (VPN)"
}

print("="*80)
print("NETWORK TRAFFIC CLASSIFICATION - BALANCED TRAINING")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Augmentation: {AUGMENT_STRENGTH}")
print(f"Weighted sampler: {USE_WEIGHTED_SAMPLER}")
print(f"Weighted loss: {USE_WEIGHTED_LOSS}")
print("="*80)

# ============================================================================
# AUGMENTATION FUNCTIONS
# ============================================================================
def augment_network_traffic(img, strength='moderate'):
    """
    Apply semantically valid augmentation for network traffic.
    
    Techniques:
    1. Small random noise (simulates protocol variations)
    2. Random truncation (simulates incomplete captures)
    3. Small vertical shift (simulates packet reordering)
    
    Args:
        img: 28x28 numpy array
        strength: 'light', 'moderate', or 'heavy'
    
    Returns:
        Augmented 28x28 numpy array
    """
    augmented = img.copy().astype(np.float32)
    
    # 1. RANDOM NOISE (70% chance - let model see some clean data too)
    # Simulates: protocol variations, measurement noise, different implementations
    if np.random.rand() > 0.3:  # 70% apply noise, 30% keep clean
        if strength == 'light':
            noise_level = 5
        elif strength == 'moderate':
            noise_level = 8
        else:  # heavy
            noise_level = 12
        
        noise = np.random.randint(-noise_level, noise_level + 1, augmented.shape)
        augmented = np.clip(augmented + noise, 0, 255)
    
    # 2. RANDOM TRUNCATION (50% chance)
    # Simulates: incomplete traffic capture, shorter flows
    if np.random.rand() > 0.5:
        truncate_len = np.random.randint(int(784 * 0.70), 784)
        flat = augmented.flatten()
        flat[truncate_len:] = 0  # Zero out the rest
        augmented = flat.reshape(28, 28)
    
    # 3. SMALL VERTICAL SHIFT (30% chance)
    # Simulates: minor packet reordering (happens in real networks)
    if np.random.rand() > 0.7:
        shift = np.random.randint(-1, 2)  # Very small shift only
        augmented = np.roll(augmented, shift, axis=0)
    
    return augmented.astype(np.uint8)

# ============================================================================
# DATASET CLASS WITH ON-THE-FLY AUGMENTATION
# ============================================================================
class TrafficDataset(Dataset):
    def __init__(self, data_path, labels_path, augment=False, augment_strength='moderate'):
        """
        Dataset with optional on-the-fly augmentation.
        
        Args:
            data_path: Path to .npy file with images
            labels_path: Path to .npy file with labels
            augment: Whether to apply augmentation
            augment_strength: 'light', 'moderate', or 'heavy'
        """
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)
        self.augment = augment
        self.augment_strength = augment_strength
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        # Apply augmentation if training
        if self.augment:
            image = augment_network_traffic(image, self.augment_strength)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension (1, 28, 28)
        image = np.expand_dims(image, axis=0)
        
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)

# ============================================================================
# CNN MODEL (SIMPLIFIED - 2 conv layers, fewer parameters)
# ============================================================================
class TrafficCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(TrafficCNN, self).__init__()
        
        # Simplified architecture: 2 conv layers with 2D kernels
        # Conv1: 2D kernel for capturing both spatial and sequential patterns
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv2: 2D kernel for higher-level patterns
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d((2, 2))
        
        self.dropout = nn.Dropout(0.5)
        
        # After 2 pooling layers: 28 → 14 → 7
        # Output: 64 channels × 7 × 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Conv block 1: Extract low-level 2D patterns
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # 28×28 → 14×14
        
        # Conv block 2: Extract high-level 2D patterns
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # 14×14 → 7×7
        
        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/len(train_loader), 'acc': 100.*correct/total})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)

# ============================================================================
# MAIN TRAINING
# ============================================================================
def main():
    # Load datasets
    print("\n[Loading] Datasets...")
    train_dataset = TrafficDataset(
        os.path.join(DATA_DIR, "train_data_weighted.npy"),
        os.path.join(DATA_DIR, "train_labels_weighted.npy"),
        augment=True,
        augment_strength=AUGMENT_STRENGTH
    )
    
    val_dataset = TrafficDataset(
        os.path.join(DATA_DIR, "val_data_weighted.npy"),
        os.path.join(DATA_DIR, "val_labels_weighted.npy"),
        augment=False  # No augmentation for validation
    )
    
    test_dataset = TrafficDataset(
        os.path.join(DATA_DIR, "test_data_weighted.npy"),
        os.path.join(DATA_DIR, "test_labels_weighted.npy"),
        augment=False  # No augmentation for test
    )
    
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")
    
    # Create weighted sampler if enabled
    if USE_WEIGHTED_SAMPLER:
        print("\n[Creating] Weighted sampler for oversampling minorities...")
        augmentation_multipliers = np.load(os.path.join(DATA_DIR, "augmentation_multipliers.npy"))
        
        # Create sample weights (weight per sample based on its class)
        sample_weights = np.array([augmentation_multipliers[label] for label in train_dataset.labels])
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    print("\n[Initializing] Model...")
    model = TrafficCNN(num_classes=11).to(DEVICE)
    
    # Create weighted loss if enabled
    if USE_WEIGHTED_LOSS:
        class_weights = np.load(os.path.join(DATA_DIR, "class_weights.npy"))
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(DEVICE))
        print(f"  Using weighted loss with class weights")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"  Using standard CrossEntropyLoss")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    print("\n[Training] Starting...")
    print("="*80)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 10
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 80)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n  Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save training history
    with open(os.path.join(OUTPUT_DIR, "history.json"), 'w') as f:
        json.dump(history, f)
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss')
    ax1.plot(epochs_range, history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_range, history['train_acc'], label='Train Acc')
    ax2.plot(epochs_range, history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=150, bbox_inches='tight')
    
    # Load best model for testing
    print("\n[Testing] Loading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, DEVICE)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Classification report
    report = classification_report(test_labels, test_preds, 
                                   target_names=[CLASS_NAMES[i] for i in range(11)],
                                   digits=3)
    print("\n" + report)
    
    # Save report
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[CLASS_NAMES[i] for i in range(11)],
                yticklabels=[CLASS_NAMES[i] for i in range(11)])
    plt.title(f'Confusion Matrix (Test Accuracy: {test_acc:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
    
    print(f"\n✓ Results saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()

