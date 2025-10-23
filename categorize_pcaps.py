import os
import shutil

BASE_DIR = "raw_pcaps"   # where VPN/NonVPN folders are
OUTPUT_DIR = "categorized_pcaps"

# Define category patterns based on your dataset
CATEGORY_PATTERNS = {
    "Chat": [
        "aim_chat", "AIMchat", "facebook_chat", "facebookchat",
        "hangout_chat", "hangouts_chat", "icq_chat", "ICQchat", "skype_chat"
    ],
    "Email": ["email", "gmailchat"],
    "File": ["ftps", "sftp", "scp", "skype_file", "skype_files"],
    "P2P": ["bittorrent"],
    "Streaming": ["youtube", "vimeo", "netflix", "spotify"],
    "VoIP": ["voipbuster", "skype_audio", "facebook_audio", "hangouts_audio"],
}

# Create output folders
for vpn_type in ["VPN", "NonVPN"]:
    for cat in CATEGORY_PATTERNS.keys():
        os.makedirs(os.path.join(OUTPUT_DIR, vpn_type, cat), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, vpn_type, "Uncategorized"), exist_ok=True)

# Categorize files
for vpn_type in ["VPN", "NonVPN"]:
    folder_path = os.path.join(BASE_DIR, vpn_type)
    if not os.path.exists(folder_path):
        continue

    for fname in os.listdir(folder_path):
        if not fname.endswith((".pcap", ".pcapng")):
            continue

        src = os.path.join(folder_path, fname)
        moved = False

        # Find matching category
        for category, patterns in CATEGORY_PATTERNS.items():
            if any(p.lower() in fname.lower() for p in patterns):
                dest = os.path.join(OUTPUT_DIR, vpn_type, category, fname)
                shutil.move(src, dest)
                moved = True
                break

        if not moved:
            dest = os.path.join(OUTPUT_DIR, vpn_type, "Uncategorized", fname)
            shutil.move(src, dest)

print("✅ Categorization complete!")