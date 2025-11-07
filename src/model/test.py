import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# SETUP
# ============================================================================
PROJECT_ROOT = "/Users/kivanc/Desktop/AttentionNet"
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data/final")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model_output/best_model.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model_output")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = {
    0: "Chat (NonVPN)", 1: "Email (NonVPN)", 2: "File (NonVPN)", 
    3: "Streaming (NonVPN)", 4: "VoIP (NonVPN)",
    5: "Chat (VPN)", 6: "Email (VPN)", 7: "File (VPN)", 
    8: "P2P (VPN)", 9: "Streaming (VPN)", 10: "VoIP (VPN)"
}

# ============================================================================
# DATASET
# ============================================================================
class TrafficDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx].astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        label = self.labels[idx]
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)

# ============================================================================
# MODEL (Copy your architecture here - use the same one you trained!)
# ============================================================================
class TrafficCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(TrafficCNN, self).__init__()
        
        # First layer: Wide 1D-style kernel
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 25), padding=(0, 12))
        self.bn1 = nn.BatchNorm2d(64)
        
        # Second layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 15), padding=(0, 7))
        self.bn2 = nn.BatchNorm2d(128)
        
        # Third layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool_width = nn.MaxPool2d((1, 2))
        self.pool_2d = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256 * 14 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool_width(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool_width(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool_2d(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# ============================================================================
# LOAD AND TEST
# ============================================================================
print("="*80)
print("TESTING BEST MODEL")
print("="*80)

# Load test data
print("\n[1/4] Loading test data...")
test_dataset = TrafficDataset(
    os.path.join(DATA_DIR, "test_data.npy"),
    os.path.join(DATA_DIR, "test_labels.npy")
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print(f"  Test samples: {len(test_dataset)}")

# Load model
print("\n[2/4] Loading best model...")
model = TrafficCNN(num_classes=11).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print(f"  Loaded from: {MODEL_PATH}")

# Evaluate
print("\n[3/4] Evaluating...")
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Calculate accuracy
test_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))

print(f"\n{'='*80}")
print(f"TEST ACCURACY: {test_acc:.2f}%")
print(f"{'='*80}")

# Detailed report
print("\nClassification Report:")
print("-"*80)
report = classification_report(
    all_labels, all_preds,
    target_names=[CLASS_NAMES[i] for i in range(11)],
    digits=3
)
print(report)

# Save report
with open(os.path.join(OUTPUT_DIR, "test_results.txt"), 'w') as f:
    f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
    f.write(report)

print(f"\n✓ Results saved to: {OUTPUT_DIR}/test_results.txt")

# Confusion matrix
print("\n[4/4] Generating confusion matrix...")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(11), yticklabels=range(11))
plt.xlabel('Predicted', fontsize=12, fontweight='bold')
plt.ylabel('True', fontsize=12, fontweight='bold')
plt.title(f'Test Set Confusion Matrix (Accuracy: {test_acc:.2f}%)',
         fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "test_confusion_matrix.png"),
           dpi=150, bbox_inches='tight')
print(f"✓ Confusion matrix saved")

print("\n" + "="*80)
print("TESTING COMPLETE!")
print("="*80)