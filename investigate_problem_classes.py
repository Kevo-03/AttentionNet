import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load test data
data_dir = Path('/Users/kivanc/Desktop/AttentionNet/processed_data/final')
test_data = np.load(data_dir / 'test_data.npy')
test_labels = np.load(data_dir / 'test_labels.npy')

# Class names
class_names = [
    'Chat (NonVPN)', 'Email (NonVPN)', 'File (NonVPN)', 
    'Streaming (NonVPN)', 'VoIP (NonVPN)',
    'Chat (VPN)', 'Email (VPN)', 'File (VPN)', 
    'P2P (VPN)', 'Streaming (VPN)', 'VoIP (VPN)'
]

# Focus on problematic NonVPN classes: Chat, Email, VoIP
problem_classes = [0, 1, 4]  # Chat, Email, VoIP (NonVPN)
problem_names = ['Chat (NonVPN)', 'Email (NonVPN)', 'VoIP (NonVPN)']

# Compare with their VPN counterparts
vpn_classes = [5, 6, 10]  # Chat, Email, VoIP (VPN)
vpn_names = ['Chat (VPN)', 'Email (VPN)', 'VoIP (VPN)']

# 1. Calculate average density for each class
print("DENSITY ANALYSIS:")
print("=" * 80)
for cls_idx, cls_name in zip(problem_classes + vpn_classes, problem_names + vpn_names):
    mask = test_labels == cls_idx
    samples = test_data[mask]
    densities = (samples > 0).mean(axis=(1, 2)) * 100
    print(f"{cls_name:20} | Mean: {densities.mean():.2f}% | Std: {densities.std():.2f}% | Min: {densities.min():.2f}% | Max: {densities.max():.2f}%")

# 2. Visualize average patterns
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Average Traffic Patterns: NonVPN vs VPN (Problematic Classes)', fontsize=16)

for i, (cls_idx, cls_name) in enumerate(zip(problem_classes, problem_names)):
    mask = test_labels == cls_idx
    samples = test_data[mask]
    avg_pattern = samples.mean(axis=0)
    
    ax = axes[0, i]
    im = ax.imshow(avg_pattern, cmap='hot', aspect='auto')
    ax.set_title(f'{cls_name}\n({mask.sum()} samples)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

for i, (cls_idx, cls_name) in enumerate(zip(vpn_classes, vpn_names)):
    mask = test_labels == cls_idx
    samples = test_data[mask]
    avg_pattern = samples.mean(axis=0)
    
    ax = axes[1, i]
    im = ax.imshow(avg_pattern, cmap='hot', aspect='auto')
    ax.set_title(f'{cls_name}\n({mask.sum()} samples)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig(data_dir / 'problem_classes_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: problem_classes_comparison.png")

# 3. Show random samples from Chat vs Email (NonVPN)
fig, axes = plt.subplots(4, 8, figsize=(20, 10))
fig.suptitle('Chat vs Email (NonVPN) - Random Samples Comparison', fontsize=16)

# Top 2 rows: Chat (NonVPN)
chat_samples = test_data[test_labels == 0]
np.random.seed(42)
chat_indices = np.random.choice(len(chat_samples), 16, replace=False)
for i in range(16):
    ax = axes[i // 8, i % 8]
    ax.imshow(chat_samples[chat_indices[i]], cmap='gray', aspect='auto')
    density = (chat_samples[chat_indices[i]] > 0).mean() * 100
    ax.set_title(f'{density:.1f}%', fontsize=8)
    ax.axis('off')
    if i == 0:
        ax.set_ylabel('Chat (NonVPN)', fontsize=12, rotation=0, labelpad=60)

# Bottom 2 rows: Email (NonVPN)
email_samples = test_data[test_labels == 1]
email_indices = np.random.choice(len(email_samples), 16, replace=False)
for i in range(16):
    ax = axes[2 + i // 8, i % 8]
    ax.imshow(email_samples[email_indices[i]], cmap='gray', aspect='auto')
    density = (email_samples[email_indices[i]] > 0).mean() * 100
    ax.set_title(f'{density:.1f}%', fontsize=8)
    ax.axis('off')
    if i == 0:
        ax.set_ylabel('Email (NonVPN)', fontsize=12, rotation=0, labelpad=60)

plt.tight_layout()
plt.savefig(data_dir / 'chat_vs_email_samples.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: chat_vs_email_samples.png")

# 4. Statistical comparison
print("\n" + "=" * 80)
print("STATISTICAL COMPARISON (Test Set):")
print("=" * 80)

for cls_idx, cls_name in enumerate(class_names):
    mask = test_labels == cls_idx
    samples = test_data[mask]
    
    mean_intensity = samples.mean()
    std_intensity = samples.std()
    sparsity = (samples == 0).mean() * 100
    
    print(f"{cls_name:20} | Samples: {mask.sum():4} | Mean: {mean_intensity:.4f} | Std: {std_intensity:.4f} | Sparsity: {sparsity:.2f}%")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

