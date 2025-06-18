import pandas as pd
import os

# --- CONFIG ---
input_csvs = {
    "train": "/gpfs/home/unalg01/jepa/src/datasets/Prostate_downstream_train.csv",
    "val":   "/gpfs/home/unalg01/jepa/src/datasets/Prostate_downstream_val.csv",
    "test":  "/gpfs/home/unalg01/jepa/src/datasets/Prostate_downstream_test.csv"
}

output_csvs = {
    "train": "/gpfs/home/unalg01/jepa/src/datasets/Prostate_downstream_train_binary.csv",
    "val":   "/gpfs/home/unalg01/jepa/src/datasets/Prostate_downstream_val_binary.csv",
    "test":  "/gpfs/home/unalg01/jepa/src/datasets/Prostate_downstream_test_binary.csv"
}

log_dir = "/gpfs/home/unalg01/jepa/src/datasets/prostate_unexpected_label_logs"
os.makedirs(log_dir, exist_ok=True)

# --- CLASSIFICATION MAPPING ---
def map_to_binary(label):
    try:
        label = int(label)
    except ValueError:
        return -1  # Non-integer label — mark as unexpected
    if label in [1, 2, 3]:
        return 0  # Low risk
    elif label in [4, 5]:
        return 1  # High risk
    else:
        return -1  # Unexpected label

# --- PROCESS EACH SPLIT ---
for split in ["train", "val", "test"]:
    df = pd.read_csv(input_csvs[split])
    df["binary_label"] = df["label"].apply(map_to_binary)

    # Separate unexpected labels
    df_unexpected = df[df["binary_label"] == -1]
    df_binary = df[df["binary_label"] != -1].copy()

    # Save unexpected labels to log file (if any)
    if not df_unexpected.empty:
        unexpected_log_path = os.path.join(log_dir, f"{split}_unexpected_labels.log")
        df_unexpected.to_csv(unexpected_log_path, index=False)
        print(f"⚠️  {len(df_unexpected)} unexpected labels found in {split} split. Logged to {unexpected_log_path}")

    # Replace original label with binary one
    df_binary["label"] = df_binary["binary_label"]
    df_binary = df_binary.drop(columns=["binary_label"])

    # Log class distribution
    class_counts = df_binary["label"].value_counts().to_dict()
    print(f"\n Split: {split}")
    print(f" Total samples: {len(df_binary)}")
    print(f"   ➤ Class 0 (PIRADS 1–3): {class_counts.get(0, 0)}")
    print(f"   ➤ Class 1 (PIRADS 4–5): {class_counts.get(1, 0)}")

    # Save to CSV
    df_binary.to_csv(output_csvs[split], index=False)
    print(f" ✅ Saved clean split to {output_csvs[split]}")
