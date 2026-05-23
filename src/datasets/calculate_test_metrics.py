import os
import pandas as pd
import numpy as np

# === CONFIG ===
base_dir = "/gpfs/data/sodicksonlab/gozde/logs"
fold_folders = [
    "mae_test_eval_distributed_2025_10_21__11_19_46",
    "mae_test_eval_distributed_2025_10_21__11_24_41",
    "mae_test_eval_distributed_2025_10_21__11_27_49",
    "mae_test_eval_distributed_2025_10_21__11_29_32",
]
# discarding fold 0    "mae_test_eval_distributed_2025_10_21__11_29_57",

# --- Containers ---
rank0_data, rank1_data = [], []

print("\n===== Searching for test result CSVs =====\n")

for folder in fold_folders:
    csv_dir = os.path.join(base_dir, folder, "csv_logs")
    if not os.path.isdir(csv_dir):
        print(f" Missing folder: {csv_dir}")
        continue

    for rank in ["r0", "r1"]:
        file_path = os.path.join(csv_dir, f"mri-test_{rank}.csv")
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            df["fold"] = folder
            df["rank"] = rank
            if rank == "r0":
                rank0_data.append(df)
            else:
                rank1_data.append(df)
            print(f" Found: {file_path}")
        else:
            print(f" Missing: {file_path}")

# --- Combine data safely ---
if not rank0_data and not rank1_data:
    print("\n No metric files found. Please verify the paths and filenames.")
    exit()

def safe_concat(lst):
    return pd.concat(lst, ignore_index=True) if lst else pd.DataFrame()

rank0_df = safe_concat(rank0_data)
rank1_df = safe_concat(rank1_data)
all_data = pd.concat([rank0_df, rank1_df], ignore_index=True)

# --- Compute summary stats ---
metrics = ["test acc", "test loss", "test recall", "test precision", "test f1", "test AUC"]
summary = {}

for metric in metrics:
    if metric not in all_data.columns:
        continue

    summary[metric] = {
        "mean_all": np.mean(all_data[metric]),
        "std_all": np.std(all_data[metric]),
        "mean_r0": np.mean(rank0_df[metric]) if not rank0_df.empty else np.nan,
        "std_r0": np.std(rank0_df[metric]) if not rank0_df.empty else np.nan,
        "mean_r1": np.mean(rank1_df[metric]) if not rank1_df.empty else np.nan,
        "std_r1": np.std(rank1_df[metric]) if not rank1_df.empty else np.nan,
    }

# --- Output ---
summary_df = pd.DataFrame(summary).T
print("\n\n===== Test Performance Summary Across Folds =====\n")
print(summary_df.to_string(float_format="%.4f"))

summary_path = "mae_test_5fold_summary.csv"
summary_df.to_csv(summary_path)
print(f"\n Saved summary to {summary_path}")
