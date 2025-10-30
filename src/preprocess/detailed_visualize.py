import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# --- CONFIG ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
IDX_DIR = os.path.join(PROJECT_ROOT, "processed_test/idx")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_test/visualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def load_detailed_data():
    """Load all the detailed data"""
    print("[+] Loading detailed dataset...")
    
    images = np.load(os.path.join(IDX_DIR, "new_data_images.npy"))
    labels = np.load(os.path.join(IDX_DIR, "data_labels_detailed.npy"))
    app_types = np.load(os.path.join(IDX_DIR, "data_app_types.npy"))
    apps = np.load(os.path.join(IDX_DIR, "data_apps.npy"))
    vpn_types = np.load(os.path.join(IDX_DIR, "data_vpn_types.npy"))
    
    print(f"    Loaded {len(images)} samples")
    return images, labels, app_types, apps, vpn_types


def visualize_sample_images_grid(images, labels):
    """
    Visualize sample images in a grid format (like visualize.py)
    Shows multiple samples per detailed label in rows
    Split into multiple figures for better readability
    """
    print("[+] Visualizing sample images grid (multiple samples per label)...")
    
    unique_labels = sorted(np.unique(labels))
    SAMPLES_PER_CLASS = 6  # Number of samples to show per label
    LABELS_PER_FIGURE = 10  # Number of labels per figure to keep images large enough
    
    # Split labels into chunks for multiple figures
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    label_chunks = list(chunks(unique_labels, LABELS_PER_FIGURE))
    print(f"    Creating {len(label_chunks)} separate figures ({LABELS_PER_FIGURE} labels each)")
    
    # Create a figure for each chunk
    for fig_num, label_chunk in enumerate(label_chunks, 1):
        n_classes_in_chunk = len(label_chunk)
        fig_height = max(n_classes_in_chunk * 1.5, 8)
        
        plt.figure(figsize=(18, fig_height))
        
        for row_idx, label in enumerate(label_chunk):
            # Get all images for this label
            class_imgs = images[labels == label]
            
            if len(class_imgs) == 0:
                print(f"    No samples for label: {label}")
                continue
            
            # Sample random images (or all if less than SAMPLES_PER_CLASS)
            n_samples = min(SAMPLES_PER_CLASS, len(class_imgs))
            sample_idxs = np.random.choice(len(class_imgs), size=n_samples, replace=False)
            
            # Create subplots for this row
            for col_idx, idx in enumerate(sample_idxs):
                plt.subplot(n_classes_in_chunk, SAMPLES_PER_CLASS, 
                           row_idx * SAMPLES_PER_CLASS + col_idx + 1)
                plt.imshow(class_imgs[idx], cmap='gray', interpolation='nearest')
                plt.axis('off')
                
                # Only label the first image in each row
                if col_idx == 0:
                    plt.title(label, fontsize=9, pad=3, loc='left', fontweight='bold')
        
        plt.suptitle(f'Detailed Labels - Part {fig_num}/{len(label_chunks)}', 
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save each figure separately
        output_file = os.path.join(OUTPUT_DIR, f"sample_images_grid_part{fig_num}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"    Saved: sample_images_grid_part{fig_num}.png")
        plt.close()


def visualize_distribution_by_app_type(app_types):
    """Visualize distribution of samples by application type"""
    print("[+] Visualizing distribution by app type...")
    
    app_type_counts = Counter(app_types)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    categories = list(app_type_counts.keys())
    counts = list(app_type_counts.values())
    colors = plt.cm.Set3(range(len(categories)))
    
    bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Application Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Samples by Application Type', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distribution_by_app_type.png"), dpi=150, bbox_inches='tight')
    print(f"    Saved: distribution_by_app_type.png")
    plt.close()


def visualize_distribution_by_app(apps):
    """Visualize distribution of samples by specific application"""
    print("[+] Visualizing distribution by specific app...")
    
    app_counts = Counter(apps)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Sort by count descending
    sorted_apps = sorted(app_counts.items(), key=lambda x: x[1], reverse=True)
    apps_list = [item[0] for item in sorted_apps]
    counts = [item[1] for item in sorted_apps]
    
    colors = plt.cm.tab20(range(len(apps_list)))
    
    bars = ax.bar(apps_list, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Application Name', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Samples by Specific Application', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distribution_by_app.png"), dpi=150, bbox_inches='tight')
    print(f"    Saved: distribution_by_app.png")
    plt.close()


def visualize_vpn_comparison(vpn_types, app_types):
    """Compare VPN vs NonVPN across application types"""
    print("[+] Visualizing VPN vs NonVPN comparison...")
    
    # Create contingency table
    vpn_app_pairs = list(zip(vpn_types, app_types))
    
    unique_app_types = sorted(set(app_types))
    unique_vpn_types = sorted(set(vpn_types))
    
    # Count matrix
    matrix = np.zeros((len(unique_vpn_types), len(unique_app_types)))
    for i, vpn in enumerate(unique_vpn_types):
        for j, app in enumerate(unique_app_types):
            count = sum(1 for v, a in vpn_app_pairs if v == vpn and a == app)
            matrix[i, j] = count
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(unique_app_types))
    width = 0.35
    
    for i, vpn in enumerate(unique_vpn_types):
        offset = width * (i - 0.5)
        bars = ax.bar(x + offset, matrix[i], width, label=vpn, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Application Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_title('VPN vs NonVPN Distribution Across Application Types', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_app_types, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "vpn_vs_nonvpn_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"    Saved: vpn_vs_nonvpn_comparison.png")
    plt.close()


def visualize_heatmap_app_vs_apptype(apps, app_types):
    """Create heatmap showing relationship between apps and app types"""
    print("[+] Creating heatmap: App vs App Type...")
    
    unique_apps = sorted(set(apps))
    unique_app_types = sorted(set(app_types))
    
    # Create matrix
    matrix = np.zeros((len(unique_apps), len(unique_app_types)))
    app_apptype_pairs = list(zip(apps, app_types))
    
    for i, app in enumerate(unique_apps):
        for j, app_type in enumerate(unique_app_types):
            count = sum(1 for a, at in app_apptype_pairs if a == app and at == app_type)
            matrix[i, j] = count
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(np.arange(len(unique_app_types)))
    ax.set_yticks(np.arange(len(unique_apps)))
    ax.set_xticklabels(unique_app_types, rotation=45, ha='right')
    ax.set_yticklabels(unique_apps)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Samples', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(unique_apps)):
        for j in range(len(unique_app_types)):
            if matrix[i, j] > 0:
                text = ax.text(j, i, int(matrix[i, j]),
                             ha="center", va="center", color="black", fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Application Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Application Name', fontsize=14, fontweight='bold')
    ax.set_title('Heatmap: Application vs Application Type', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_app_vs_apptype.png"), dpi=150, bbox_inches='tight')
    print(f"    Saved: heatmap_app_vs_apptype.png")
    plt.close()


def visualize_image_patterns_by_app_type(images, app_types):
    """Show average image patterns for each app type"""
    print("[+] Visualizing average patterns by app type...")
    
    unique_app_types = sorted(set(app_types))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, app_type in enumerate(unique_app_types):
        if idx >= len(axes):
            break
            
        # Get all images for this app type
        mask = app_types == app_type
        app_images = images[mask]
        
        # Calculate mean image
        mean_img = np.mean(app_images, axis=0)
        
        im = axes[idx].imshow(mean_img, cmap='viridis', interpolation='nearest')
        axes[idx].set_title(f'{app_type}\n(n={len(app_images)})', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(len(unique_app_types), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Average Image Patterns by Application Type', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "average_patterns_by_app_type.png"), dpi=150, bbox_inches='tight')
    print(f"    Saved: average_patterns_by_app_type.png")
    plt.close()


def visualize_image_patterns_by_app(images, apps):
    """Show average image patterns for each specific app (top 12)"""
    print("[+] Visualizing average patterns by specific app...")
    
    # Get top apps by count
    app_counts = Counter(apps)
    top_apps = [app for app, count in app_counts.most_common(12)]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, app in enumerate(top_apps):
        if idx >= len(axes):
            break
            
        # Get all images for this app
        mask = apps == app
        app_images = images[mask]
        
        # Calculate mean image
        mean_img = np.mean(app_images, axis=0)
        
        im = axes[idx].imshow(mean_img, cmap='viridis', interpolation='nearest')
        axes[idx].set_title(f'{app}\n(n={len(app_images)})', fontsize=10, fontweight='bold')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(len(top_apps), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Average Image Patterns by Specific Application (Top 12)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "average_patterns_by_app.png"), dpi=150, bbox_inches='tight')
    print(f"    Saved: average_patterns_by_app.png")
    plt.close()


def create_summary_report(labels, app_types, apps, vpn_types):
    """Create a text summary report"""
    print("[+] Creating summary report...")
    
    report_path = os.path.join(OUTPUT_DIR, "detailed_summary_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED LABELING SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Samples: {len(labels)}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("VPN TYPE DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        vpn_counts = Counter(vpn_types)
        for vpn, count in sorted(vpn_counts.items()):
            f.write(f"  {vpn}: {count} samples ({count/len(labels)*100:.1f}%)\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("APPLICATION TYPE DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        app_type_counts = Counter(app_types)
        for app_type, count in sorted(app_type_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {app_type}: {count} samples ({count/len(labels)*100:.1f}%)\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("SPECIFIC APPLICATION DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        app_counts = Counter(apps)
        for app, count in sorted(app_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {app}: {count} samples ({count/len(labels)*100:.1f}%)\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("DETAILED LABELS (showing all unique combinations)\n")
        f.write("-" * 80 + "\n")
        label_counts = Counter(labels)
        for label, count in sorted(label_counts.items()):
            f.write(f"  {label}: {count} samples\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"    Saved: detailed_summary_report.txt")


def main():
    """Run all visualizations"""
    print("\n" + "=" * 80)
    print("DETAILED LABEL VISUALIZATION")
    print("=" * 80 + "\n")
    
    # Load data
    images, labels, app_types, apps, vpn_types = load_detailed_data()
    
    # Create all visualizations
    visualize_sample_images_grid(images, labels)  # New grid-style visualization
    visualize_distribution_by_app_type(app_types)
    visualize_distribution_by_app(apps)
    visualize_vpn_comparison(vpn_types, app_types)
    visualize_heatmap_app_vs_apptype(apps, app_types)
    visualize_image_patterns_by_app_type(images, app_types)
    visualize_image_patterns_by_app(images, apps)
    create_summary_report(labels, app_types, apps, vpn_types)
    
    print("\n" + "=" * 80)
    print(f"[✓] All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

