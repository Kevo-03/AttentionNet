"""
Example script demonstrating how to use the detailed labeling data
for different types of analysis and classification tasks.
"""

import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- CONFIG ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
IDX_DIR = os.path.join(PROJECT_ROOT, "processed_test/idx")


def load_detailed_data():
    """Load all the detailed data"""
    print("[+] Loading detailed dataset...")
    
    images = np.load(os.path.join(IDX_DIR, "new_data_images.npy"))
    labels = np.load(os.path.join(IDX_DIR, "data_labels_detailed.npy"))
    app_types = np.load(os.path.join(IDX_DIR, "data_app_types.npy"))
    apps = np.load(os.path.join(IDX_DIR, "data_apps.npy"))
    vpn_types = np.load(os.path.join(IDX_DIR, "data_vpn_types.npy"))
    
    print(f"    Loaded {len(images)} samples")
    print(f"    Image shape: {images.shape}")
    print(f"    Unique detailed labels: {len(np.unique(labels))}")
    print(f"    Unique app types: {len(np.unique(app_types))}")
    print(f"    Unique apps: {len(np.unique(apps))}")
    print(f"    Unique VPN types: {len(np.unique(vpn_types))}")
    
    return images, labels, app_types, apps, vpn_types


def example_1_binary_vpn_classification():
    """
    Example 1: Binary Classification - VPN vs NonVPN
    Simplest task: just detect if traffic is using VPN or not
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Binary VPN Classification")
    print("="*80)
    
    images, _, _, _, vpn_types = load_detailed_data()
    
    # Flatten images for classical ML (optional)
    X = images.reshape(len(images), -1)  # Shape: (N, 784)
    
    # Create binary labels
    le = LabelEncoder()
    y = le.fit_transform(vpn_types)  # 0: NonVPN, 1: VPN (or vice versa)
    
    print(f"\n[+] Label distribution:")
    for label, name in enumerate(le.classes_):
        count = np.sum(y == label)
        print(f"    {name}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n[+] Dataset split:")
    print(f"    Training: {len(X_train)} samples")
    print(f"    Testing: {len(X_test)} samples")
    print(f"\n[+] Ready for binary classification (VPN detection)")
    
    return X_train, X_test, y_train, y_test, le


def example_2_app_type_classification():
    """
    Example 2: Multi-class Classification - Application Types
    Classify traffic into Chat, Email, File, Streaming, VoIP, P2P
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Application Type Classification")
    print("="*80)
    
    images, _, app_types, _, _ = load_detailed_data()
    
    # Flatten images
    X = images.reshape(len(images), -1)
    
    # Encode app types
    le = LabelEncoder()
    y = le.fit_transform(app_types)
    
    print(f"\n[+] Application types ({len(le.classes_)} classes):")
    for label, name in enumerate(le.classes_):
        count = np.sum(y == label)
        print(f"    {label} - {name}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n[+] Dataset split:")
    print(f"    Training: {len(X_train)} samples")
    print(f"    Testing: {len(X_test)} samples")
    print(f"\n[+] Ready for app type classification")
    
    return X_train, X_test, y_train, y_test, le


def example_3_specific_app_classification():
    """
    Example 3: Fine-grained Classification - Specific Applications
    Classify traffic into specific apps (facebook, netflix, skype, etc.)
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Specific Application Classification")
    print("="*80)
    
    images, _, _, apps, _ = load_detailed_data()
    
    # Flatten images
    X = images.reshape(len(images), -1)
    
    # Encode specific apps
    le = LabelEncoder()
    y = le.fit_transform(apps)
    
    print(f"\n[+] Specific applications ({len(le.classes_)} classes):")
    app_counts = Counter(apps)
    for app, count in sorted(app_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {app}: {count} samples ({count/len(apps)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n[+] Dataset split:")
    print(f"    Training: {len(X_train)} samples")
    print(f"    Testing: {len(X_test)} samples")
    print(f"\n[+] Ready for specific app classification")
    
    return X_train, X_test, y_train, y_test, le


def example_4_combined_classification():
    """
    Example 4: Combined Classification - Original system labels
    Combine VPN type and app type into 12 classes (same as step2changed.py)
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Combined VPN + App Type Classification (12 classes)")
    print("="*80)
    
    images, _, app_types, _, vpn_types = load_detailed_data()
    
    # Flatten images
    X = images.reshape(len(images), -1)
    
    # Create combined labels (same as original system)
    label_map = {
        ("NonVPN", "Chat"): 0, ("NonVPN", "Email"): 1,
        ("NonVPN", "File"): 2, ("NonVPN", "P2P"): 3,
        ("NonVPN", "Streaming"): 4, ("NonVPN", "VoIP"): 5,
        ("VPN", "Chat"): 6, ("VPN", "Email"): 7,
        ("VPN", "File"): 8, ("VPN", "P2P"): 9,
        ("VPN", "Streaming"): 10, ("VPN", "VoIP"): 11
    }
    
    y = np.array([
        label_map[(vpn, app_type)] 
        for vpn, app_type in zip(vpn_types, app_types)
    ])
    
    print(f"\n[+] Combined labels (12 classes):")
    reverse_map = {v: f"{k[0]}_{k[1]}" for k, v in label_map.items()}
    for label_id in sorted(reverse_map.keys()):
        count = np.sum(y == label_id)
        print(f"    {label_id} - {reverse_map[label_id]}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n[+] Dataset split:")
    print(f"    Training: {len(X_train)} samples")
    print(f"    Testing: {len(X_test)} samples")
    print(f"\n[+] Ready for combined classification (matches original system)")
    
    return X_train, X_test, y_train, y_test


def example_5_hierarchical_data_prep():
    """
    Example 5: Hierarchical Classification Data Preparation
    Prepare data for multi-level hierarchical classification
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Hierarchical Classification Data Preparation")
    print("="*80)
    
    images, labels, app_types, apps, vpn_types = load_detailed_data()
    
    # Flatten images
    X = images.reshape(len(images), -1)
    
    # Encode each level
    vpn_encoder = LabelEncoder()
    app_type_encoder = LabelEncoder()
    app_encoder = LabelEncoder()
    
    y_vpn = vpn_encoder.fit_transform(vpn_types)
    y_app_type = app_type_encoder.fit_transform(app_types)
    y_app = app_encoder.fit_transform(apps)
    
    print(f"\n[+] Hierarchical structure:")
    print(f"    Level 1 (VPN detection): {len(vpn_encoder.classes_)} classes")
    print(f"        Classes: {list(vpn_encoder.classes_)}")
    print(f"    Level 2 (App type): {len(app_type_encoder.classes_)} classes")
    print(f"        Classes: {list(app_type_encoder.classes_)}")
    print(f"    Level 3 (Specific app): {len(app_encoder.classes_)} classes")
    
    # Split data (same split for all levels)
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_vpn
    )
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_vpn_train, y_vpn_test = y_vpn[train_idx], y_vpn[test_idx]
    y_app_type_train, y_app_type_test = y_app_type[train_idx], y_app_type[test_idx]
    y_app_train, y_app_test = y_app[train_idx], y_app[test_idx]
    
    print(f"\n[+] Dataset split:")
    print(f"    Training: {len(X_train)} samples")
    print(f"    Testing: {len(X_test)} samples")
    print(f"\n[+] Ready for hierarchical classification:")
    print(f"    Step 1: Train VPN detector")
    print(f"    Step 2: Train app type classifier")
    print(f"    Step 3: Train specific app classifier")
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_vpn_train': y_vpn_train, 'y_vpn_test': y_vpn_test,
        'y_app_type_train': y_app_type_train, 'y_app_type_test': y_app_type_test,
        'y_app_train': y_app_train, 'y_app_test': y_app_test,
        'vpn_encoder': vpn_encoder,
        'app_type_encoder': app_type_encoder,
        'app_encoder': app_encoder
    }


def example_6_filter_by_app_type():
    """
    Example 6: Filter data by specific app type
    Focus on just one app type (e.g., Streaming)
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Filter by Specific App Type (Streaming)")
    print("="*80)
    
    images, _, app_types, apps, vpn_types = load_detailed_data()
    
    # Filter for Streaming only
    target_app_type = "Streaming"
    mask = app_types == target_app_type
    
    X = images[mask].reshape(np.sum(mask), -1)
    filtered_apps = apps[mask]
    filtered_vpn = vpn_types[mask]
    
    print(f"\n[+] Filtered to {target_app_type} traffic only:")
    print(f"    Total samples: {len(X)}")
    print(f"    Unique apps: {list(np.unique(filtered_apps))}")
    
    # Now you can classify specific streaming apps
    le = LabelEncoder()
    y = le.fit_transform(filtered_apps)
    
    print(f"\n[+] Apps in {target_app_type} category:")
    for label, name in enumerate(le.classes_):
        count = np.sum(y == label)
        print(f"    {label} - {name}: {count} samples ({count/len(y)*100:.1f}%)")
    
    print(f"\n[+] Use case: Fine-grained streaming service classification")
    print(f"    (netflix vs youtube vs spotify vs vimeo)")


def example_7_vpn_effect_analysis():
    """
    Example 7: Analyze VPN effect on specific apps
    Compare VPN vs NonVPN for same application
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: VPN Effect Analysis")
    print("="*80)
    
    images, _, app_types, apps, vpn_types = load_detailed_data()
    
    # Find apps that exist in both VPN and NonVPN
    vpn_apps = set(apps[vpn_types == "VPN"])
    nonvpn_apps = set(apps[vpn_types == "NonVPN"])
    common_apps = vpn_apps & nonvpn_apps
    
    print(f"\n[+] Apps available in both VPN and NonVPN traffic:")
    print(f"    {sorted(common_apps)}")
    
    # Example: Compare facebook traffic
    if "facebook" in common_apps:
        print(f"\n[+] Example: Facebook traffic analysis")
        
        mask_facebook = apps == "facebook"
        facebook_images = images[mask_facebook]
        facebook_vpn = vpn_types[mask_facebook]
        
        mask_vpn = facebook_vpn == "VPN"
        mask_nonvpn = facebook_vpn == "NonVPN"
        
        print(f"    Facebook NonVPN samples: {np.sum(mask_nonvpn)}")
        print(f"    Facebook VPN samples: {np.sum(mask_vpn)}")
        
        if np.sum(mask_vpn) > 0 and np.sum(mask_nonvpn) > 0:
            avg_nonvpn = np.mean(facebook_images[mask_nonvpn], axis=0)
            avg_vpn = np.mean(facebook_images[mask_vpn], axis=0)
            
            difference = np.abs(avg_vpn - avg_nonvpn)
            avg_diff = np.mean(difference)
            
            print(f"    Average pixel difference: {avg_diff:.4f}")
            print(f"    Use case: Study how VPN encryption affects traffic patterns")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("DETAILED LABELING - USAGE EXAMPLES")
    print("="*80)
    print("\nThis script demonstrates different ways to use the detailed labeling data")
    print("for various classification and analysis tasks.\n")
    
    try:
        # Run all examples
        example_1_binary_vpn_classification()
        example_2_app_type_classification()
        example_3_specific_app_classification()
        example_4_combined_classification()
        example_5_hierarchical_data_prep()
        example_6_filter_by_app_type()
        example_7_vpn_effect_analysis()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED")
        print("="*80)
        print("\nNext steps:")
        print("  1. Choose the classification task that fits your needs")
        print("  2. Implement your model (CNN, Random Forest, etc.)")
        print("  3. Train and evaluate using the prepared data")
        print("  4. Compare results across different granularity levels")
        print("\n")
        
    except FileNotFoundError as e:
        print(f"\n[!] Error: Data files not found.")
        print(f"    Please run 'python src/preprocess/detailed_preprocess.py' first")
        print(f"    Error details: {e}")


if __name__ == "__main__":
    main()

