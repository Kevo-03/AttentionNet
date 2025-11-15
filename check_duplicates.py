import numpy as np
import hashlib
from collections import defaultdict

print("="*80)
print("CHECKING FOR DUPLICATE IMAGES ACROSS CHAT AND EMAIL CLASSES")
print("="*80)

# Load test data (where we saw the confusion)
data = np.load('/Users/kivanc/Desktop/AttentionNet/processed_data/final/test_data_fixed.npy')
labels = np.load('/Users/kivanc/Desktop/AttentionNet/processed_data/final/test_labels_fixed.npy')

# Focus on Chat (0) and Email (1) NonVPN
chat_mask = labels == 0
email_mask = labels == 1

chat_images = data[chat_mask]
email_images = data[email_mask]

print(f"\nTest set samples:")
print(f"  Chat (NonVPN):  {len(chat_images)} images")
print(f"  Email (NonVPN): {len(email_images)} images")

# Calculate SHA1 hashes for all Chat images
print("\n" + "="*80)
print("Computing hashes for Chat images...")
chat_hashes = {}
for idx, img in enumerate(chat_images):
    img_hash = hashlib.sha1(img.tobytes()).hexdigest()
    chat_hashes[img_hash] = idx

print(f"  Unique Chat hashes: {len(chat_hashes)}")
print(f"  Total Chat images:  {len(chat_images)}")
if len(chat_hashes) < len(chat_images):
    print(f"  ⚠️  Duplicates within Chat: {len(chat_images) - len(chat_hashes)}")

# Check if any Email images have same hash as Chat
print("\n" + "="*80)
print("Checking for identical images between Chat and Email...")
identical_count = 0
identical_indices = []

for idx, img in enumerate(email_images):
    img_hash = hashlib.sha1(img.tobytes()).hexdigest()
    if img_hash in chat_hashes:
        identical_count += 1
        identical_indices.append((idx, chat_hashes[img_hash]))

print(f"  Identical images (same hash): {identical_count}")

if identical_count > 0:
    print(f"\n  ❌ PREPROCESSING BUG DETECTED!")
    print(f"  {identical_count} Email images are IDENTICAL to Chat images")
    print(f"  This is a serious labeling or deduplication issue!")
    if identical_count <= 10:
        print(f"\n  Affected pairs (Email idx, Chat idx):")
        for e_idx, c_idx in identical_indices[:10]:
            print(f"    Email[{e_idx}] == Chat[{c_idx}]")
else:
    print(f"  ✅ No identical images found between Chat and Email")

# Now check similarity (not identity)
print("\n" + "="*80)
print("Checking image similarity (visual resemblance)...")

# Sample 100 random pairs and compute similarity
np.random.seed(42)
num_samples = min(100, len(chat_images), len(email_images))
chat_sample_indices = np.random.choice(len(chat_images), num_samples, replace=False)
email_sample_indices = np.random.choice(len(email_images), num_samples, replace=False)

similarities = []
for c_idx, e_idx in zip(chat_sample_indices, email_sample_indices):
    chat_img = chat_images[c_idx].flatten()
    email_img = email_images[e_idx].flatten()
    
    # Cosine similarity
    dot_product = np.dot(chat_img, email_img)
    norm_chat = np.linalg.norm(chat_img)
    norm_email = np.linalg.norm(email_img)
    
    if norm_chat > 0 and norm_email > 0:
        similarity = dot_product / (norm_chat * norm_email)
        similarities.append(similarity)

avg_similarity = np.mean(similarities)
max_similarity = np.max(similarities)
min_similarity = np.min(similarities)

print(f"  Average cosine similarity: {avg_similarity:.4f}")
print(f"  Max similarity: {max_similarity:.4f}")
print(f"  Min similarity: {min_similarity:.4f}")
print(f"\n  Interpretation:")
print(f"    0.0 = completely different")
print(f"    1.0 = identical")

if avg_similarity > 0.7:
    print(f"    ⚠️  High similarity ({avg_similarity:.2f}) - images look very similar")
elif avg_similarity > 0.5:
    print(f"    ⚠️  Moderate similarity ({avg_similarity:.2f}) - some visual overlap")
else:
    print(f"    ✅  Low similarity ({avg_similarity:.2f}) - images are distinguishable")

# Check sparsity patterns
print("\n" + "="*80)
print("Comparing sparsity patterns...")
chat_densities = (chat_images > 0).mean(axis=(1, 2))
email_densities = (email_images > 0).mean(axis=(1, 2))

print(f"  Chat  - Mean density: {chat_densities.mean():.3f}, Std: {chat_densities.std():.3f}")
print(f"  Email - Mean density: {email_densities.mean():.3f}, Std: {email_densities.std():.3f}")

# Count how many images overlap in density ranges
sparse_chat = np.sum(chat_densities < 0.2)
sparse_email = np.sum(email_densities < 0.2)
print(f"\n  Very sparse samples (< 20% density):")
print(f"    Chat:  {sparse_chat} / {len(chat_images)} ({sparse_chat/len(chat_images)*100:.1f}%)")
print(f"    Email: {sparse_email} / {len(email_images)} ({sparse_email/len(email_images)*100:.1f}%)")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)

if identical_count > 0:
    print("❌ PREPROCESSING ERROR: Identical images in different classes!")
    print("   You need to investigate the deduplication or labeling logic.")
elif avg_similarity > 0.6 and abs(chat_densities.mean() - email_densities.mean()) < 0.05:
    print("⚠️  NATURAL SIMILARITY: Chat and Email traffic are inherently similar.")
    print("   Both use small packets, text-based protocols, and similar patterns.")
    print("   This is expected behavior, not a bug.")
else:
    print("✅ DISTINCT CLASSES: Images are sufficiently different.")
    print("   The confusion is due to model limitations, not data issues.")

print("="*80)




