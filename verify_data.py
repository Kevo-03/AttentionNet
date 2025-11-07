import numpy as np
import os
from collections import Counter

# Load the raw preprocessed data
data = np.load('/Users/kivanc/Desktop/AttentionNet/processed_data/idx/data_fixed.npy')
labels = np.load('/Users/kivanc/Desktop/AttentionNet/processed_data/idx/labels_fixed.npy')

# Count samples per class
counts = Counter(labels)

print("="*80)
print("DATA VERIFICATION - Checking if gmailchat fix was applied")
print("="*80)

class_names = {
    0: "Chat (NonVPN)", 1: "Email (NonVPN)", 2: "File (NonVPN)",
    3: "Streaming (NonVPN)", 4: "VoIP (NonVPN)",
    5: "Chat (VPN)", 6: "Email (VPN)", 7: "File (VPN)",
    8: "P2P (VPN)", 9: "Streaming (VPN)", 10: "VoIP (VPN)"
}

print("\nSample counts in data_fixed.npy:")
for label in sorted(counts.keys()):
    print(f"  Label {label:2} - {class_names[label]:25}: {counts[label]:6} samples")

# Check if Email has fewer samples than Chat (expected after fix)
chat_count = counts.get(0, 0)
email_count = counts.get(1, 0)

print("\n" + "="*80)
print("CRITICAL CHECK: Chat (NonVPN) vs Email (NonVPN)")
print("="*80)
print(f"  Chat (NonVPN):  {chat_count:6} samples")
print(f"  Email (NonVPN): {email_count:6} samples")
print(f"  Ratio (Chat/Email): {chat_count/email_count:.2f}x")

print("\n" + "="*80)
if chat_count > email_count * 1.5:  # Chat should have significantly more samples
    print("✅ STATUS: CORRECT")
    print("="*80)
    print("Chat has significantly more samples than Email.")
    print("This indicates gmailchat files were correctly moved to Chat folder")
    print("BEFORE preprocessing was run.")
elif chat_count > email_count:
    print("⚠️ STATUS: LIKELY CORRECT (but close)")
    print("="*80)
    print("Chat has more samples than Email, which is expected.")
    print("The difference is small, but this could be normal.")
else:
    print("❌ STATUS: PROBLEM DETECTED")
    print("="*80)
    print("Email has same or more samples than Chat!")
    print("This suggests gmailchat files were still in Email folder")
    print("when preprocessing was run. You need to rerun preprocessing.")

# Also check file timestamps
print("\n" + "="*80)
print("File modification times:")
print("="*80)
import time
files_to_check = [
    '/Users/kivanc/Desktop/AttentionNet/processed_data/idx/data_fixed.npy',
    '/Users/kivanc/Desktop/AttentionNet/processed_data/final/train_data_fixed.npy',
    '/Users/kivanc/Desktop/AttentionNet/model_output/best_model.pth'
]

for fpath in files_to_check:
    if os.path.exists(fpath):
        mtime = os.path.getmtime(fpath)
        mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        print(f"  {os.path.basename(fpath):30} : {mtime_str}")
    else:
        print(f"  {os.path.basename(fpath):30} : NOT FOUND")

print("="*80)

