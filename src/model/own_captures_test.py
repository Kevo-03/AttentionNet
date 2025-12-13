import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import seaborn as sns
from tqdm import tqdm
from src.model import TrafficCNN_Backbone, TrafficCNN_Transformer, TrafficCNN_TinyTransformer

# ============================================================================
# CONFIGURATION
# ============================================================================
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))

# Data directory for own captures
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data/own_captures_test")

# Model path - UPDATE THIS to point to your trained model
MODEL_PATH = os.path.join(PROJECT_ROOT, "model_output/memory_safe/hocaya_gosterilcek/p2p_change/2layer_cnn_hybrid_3fc/best_model.pth")

# Output directory for results
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model_output/own_captures_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test parameters
BATCH_SIZE = 128

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

# Class names - NonVPN only (0-5)
CLASS_NAMES = {
    0: "Chat (NonVPN)",
    1: "Email (NonVPN)",
    2: "File (NonVPN)",
    3: "P2P (NonVPN)",
    4: "Streaming (NonVPN)",
    5: "VoIP (NonVPN)",
}

N_CLASSES = 12  # Model was trained on 12 classes

print("="*80)
print("NETWORK TRAFFIC CLASSIFICATION - OWN CAPTURES TEST")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Model path: {MODEL_PATH}")
print(f"Data directory: {DATA_DIR}")
print(f"Batch size: {BATCH_SIZE}")
print("="*80)

# ============================================================================
# DATASET CLASS
# ============================================================================
class TrafficDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.float32)
        label = self.labels[idx]

        # normalize to [0, 1]
        img = img / 255.0
        # add channel dim -> (1, 28, 28)
        img = np.expand_dims(img, axis=0)

        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/4] Loading test dataset...")

test_dataset = TrafficDataset(
    os.path.join(DATA_DIR, "data_memory_safe.npy"),
    os.path.join(DATA_DIR, "labels_memory_safe.npy"),
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"  Test samples: {len(test_dataset)}")

# Check label distribution
labels = test_dataset.labels
unique_labels, counts = np.unique(labels, return_counts=True)
print(f"\n  Label distribution:")
for lbl, count in zip(unique_labels, counts):
    print(f"    {CLASS_NAMES.get(lbl, f'Unknown-{lbl}')}: {count} samples")

# ============================================================================
# LOAD MODEL
# ============================================================================
print("\n[2/4] Loading trained model...")

if not os.path.exists(MODEL_PATH):
    print(f"[!] Model not found at: {MODEL_PATH}")
    print("Please update MODEL_PATH in the script to point to your trained model.")
    exit(1)

# Initialize model architecture (must match training)
model = TrafficCNN_TinyTransformer(num_classes=N_CLASSES).to(DEVICE)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# ============================================================================
# EVALUATION ON TEST SET
# ============================================================================
print("\n[3/4] Evaluating on own captures...")

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Calculate metrics
test_acc = 100. * np.mean(all_preds == all_labels)

# Calculate F1 only for classes present in test set
present_classes = np.unique(all_labels)
test_macro_f1 = f1_score(all_labels, all_preds, average='macro', labels=present_classes)

print(f"\n{'='*80}")
print(f"TEST RESULTS - OWN CAPTURES")
print(f"{'='*80}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print(f"{'='*80}")

# Classification report
print("\n  Classification Report:")
print("-" * 80)
report = classification_report(
    all_labels, all_preds,
    target_names=[CLASS_NAMES[i] for i in present_classes],
    labels=present_classes,
    digits=3
)
print(report)

# Save report
with open(os.path.join(OUTPUT_DIR, "own_captures_test_results.txt"), 'w') as f:
    f.write("="*80 + "\n")
    f.write("TEST RESULTS - OWN CAPTURES\n")
    f.write("="*80 + "\n")
    f.write(f"Test Accuracy: {test_acc:.2f}%\n")
    f.write(f"Test Macro F1: {test_macro_f1:.4f}\n\n")
    f.write("Label Distribution:\n")
    for lbl, count in zip(unique_labels, counts):
        f.write(f"  {CLASS_NAMES.get(lbl, f'Unknown-{lbl}')}: {count} samples\n")
    f.write("\n" + "-"*80 + "\n")
    f.write("Classification Report:\n")
    f.write("-"*80 + "\n")
    f.write(report)

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[4/4] Generating visualizations...")

# 1. Confusion matrix (normalized by true class)
cm = confusion_matrix(all_labels, all_preds, labels=present_classes)

# Normalize per row (true class)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)  # handle any division-by-zero rows safely

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=[CLASS_NAMES[i] for i in present_classes],
    yticklabels=[CLASS_NAMES[i] for i in present_classes],
    vmin=0.0,
    vmax=1.0
)
plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
plt.ylabel("True Label", fontsize=12, fontweight="bold")
plt.title(
    f"Normalized Confusion Matrix - Own Captures Test\nAccuracy: {test_acc:.2f}%",
    fontsize=14,
    fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "own_captures_confusion_matrix.png"),
            dpi=150, bbox_inches="tight")
print("  ✓ Saved: own_captures_confusion_matrix.png (normalized)")
plt.close()

# 2. Per-class accuracy
class_accuracies = []
class_labels_list = []
for i in present_classes:
    mask = all_labels == i
    if mask.sum() > 0:
        class_acc = accuracy_score(all_labels[mask], all_preds[mask]) * 100
        class_accuracies.append(class_acc)
        class_labels_list.append(i)

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(class_labels_list)), class_accuracies, color='steelblue', edgecolor='black')
plt.xlabel('Class', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Per-Class Accuracy - Own Captures Test', fontsize=14, fontweight='bold')
plt.xticks(range(len(class_labels_list)), [CLASS_NAMES[i] for i in class_labels_list], rotation=45, ha='right')
plt.ylim(0, 105)
plt.grid(axis='y', alpha=0.3)

for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "own_captures_per_class_accuracy.png"), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: own_captures_per_class_accuracy.png")
plt.close()

# 3. Prediction confidence distribution
plt.figure(figsize=(12, 6))
max_probs = np.max(all_probs, axis=1)
correct_mask = all_preds == all_labels

plt.hist(max_probs[correct_mask], bins=50, alpha=0.7, label='Correct Predictions', color='green', edgecolor='black')
plt.hist(max_probs[~correct_mask], bins=50, alpha=0.7, label='Incorrect Predictions', color='red', edgecolor='black')
plt.xlabel('Prediction Confidence (Max Probability)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Prediction Confidence Distribution - Own Captures Test', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "own_captures_confidence_distribution.png"), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: own_captures_confidence_distribution.png")
plt.close()

# 4. Detailed per-class metrics visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Precision
precisions = []
recalls = []
f1_scores = []

for i in present_classes:
    tp = np.sum((all_labels == i) & (all_preds == i))
    fp = np.sum((all_labels != i) & (all_preds == i))
    fn = np.sum((all_labels == i) & (all_preds != i))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    precisions.append(precision * 100)
    recalls.append(recall * 100)
    f1_scores.append(f1 * 100)

class_names_list = [CLASS_NAMES[i] for i in present_classes]

# Precision plot
axes[0].bar(range(len(class_labels_list)), precisions, color='skyblue', edgecolor='black')
axes[0].set_xlabel('Class', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Precision (%)', fontsize=11, fontweight='bold')
axes[0].set_title('Precision by Class', fontsize=12, fontweight='bold')
axes[0].set_xticks(range(len(class_labels_list)))
axes[0].set_xticklabels([CLASS_NAMES[i].split()[0] for i in class_labels_list], rotation=45, ha='right')
axes[0].set_ylim(0, 105)
axes[0].grid(axis='y', alpha=0.3)

# Recall plot
axes[1].bar(range(len(class_labels_list)), recalls, color='lightcoral', edgecolor='black')
axes[1].set_xlabel('Class', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Recall (%)', fontsize=11, fontweight='bold')
axes[1].set_title('Recall by Class', fontsize=12, fontweight='bold')
axes[1].set_xticks(range(len(class_labels_list)))
axes[1].set_xticklabels([CLASS_NAMES[i].split()[0] for i in class_labels_list], rotation=45, ha='right')
axes[1].set_ylim(0, 105)
axes[1].grid(axis='y', alpha=0.3)

# F1 plot
axes[2].bar(range(len(class_labels_list)), f1_scores, color='lightgreen', edgecolor='black')
axes[2].set_xlabel('Class', fontsize=11, fontweight='bold')
axes[2].set_ylabel('F1-Score (%)', fontsize=11, fontweight='bold')
axes[2].set_title('F1-Score by Class', fontsize=12, fontweight='bold')
axes[2].set_xticks(range(len(class_labels_list)))
axes[2].set_xticklabels([CLASS_NAMES[i].split()[0] for i in class_labels_list], rotation=45, ha='right')
axes[2].set_ylim(0, 105)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "own_captures_detailed_metrics.png"), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: own_captures_detailed_metrics.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TESTING COMPLETE!")
print("="*80)
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"  • own_captures_test_results.txt")
print(f"  • own_captures_confusion_matrix.png")
print(f"  • own_captures_per_class_accuracy.png")
print(f"  • own_captures_confidence_distribution.png")
print(f"  • own_captures_detailed_metrics.png")
print(f"\nTest Accuracy: {test_acc:.2f}%")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print("\nPer-Class Accuracy:")
for i, (lbl, acc) in enumerate(zip(class_labels_list, class_accuracies)):
    print(f"  {CLASS_NAMES[lbl]}: {acc:.2f}%")
print("="*80)

