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

# --- CLASSIFICATION MAPPING ---
def map_to_binary(label):
    label = int(label)
    if label in [0, 1]:
        return 0
    elif label in [3, 4]:
        return 1
    else:
        return -1  # PIRADS 3 (label 2) → exclude

# --- PROCESS EACH SPLIT ---
for split in ["train", "val", "test"]:
    df = pd.read_csv(input_csvs[split])
    df["binary_label"] = df["label"].apply(map_to_binary)

    # Filter out PIRADS 3 (label == 2 → binary_label == -1)
    df_binary = df[df["binary_label"] != -1].copy()

    # Replace original label with binary one (optional)
    df_binary["label"] = df_binary["binary_label"]
    df_binary = df_binary.drop(columns=["binary_label"])

    # Log class distribution
    class_counts = df_binary["label"].value_counts().to_dict()
    print(f"\n Split: {split}")
    print(f" Total samples: {len(df_binary)}")
    print(f"   ➤ Class 0 (PIRADS 1–2): {class_counts.get(0, 0)}")
    print(f"   ➤ Class 1 (PIRADS 4–5): {class_counts.get(1, 0)}")

    # Save to CSV
    df_binary.to_csv(output_csvs[split], index=False)
    print(f" Saved to {output_csvs[split]}")
