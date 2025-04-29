import academictorrents as at

# Target download directory
target_path = "/gpfs/data/sodicksonlab/gozde/LDM100K"

# Use the hash from the torrent link (the long hex in the URL)
hash_id = "63aeb864bbe2115ded0aa0d7d36334c026f0660b"

# Download the dataset
at_path = at.get(hash_id, target_path=target_path)

print(f"Dataset downloaded to: {at_path}")
