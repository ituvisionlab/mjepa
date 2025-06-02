import pandas as pd

# --- CONFIG ---
pretrain_csv = "/gpfs/data/prostatelab/NIFTI_csv/Prostate_training_6May2025.csv"
bx_files = {
    "train": "/gpfs/data/prostatelab/NIFTI_csv/Bx_train_set_2025_11_4.csv",
    "val":   "/gpfs/data/prostatelab/NIFTI_csv/Bx_val_set_2025_11_4.csv",
    "test":  "/gpfs/data/prostatelab/NIFTI_csv/Bx_test_set_2025_17_1.csv"
}
log_dir = "/gpfs/home/unalg01/jepa/src/datasets"

# --- LOAD PRETRAIN ACCESSIONS ---
pre_df = pd.read_csv(pretrain_csv, dtype=str)
pre_acc = pre_df["AccNum"].astype(str).str.strip().str.split(".").str[0]
pre_acc_set = set(pre_acc)
print(f"✅ Pretraining set: {len(pre_acc_set)} unique accession numbers")

# --- LOAD Bx SPLITS INTO MEMORY ---
bx_accessions = {}
for split_name, bx_file in bx_files.items():
    df = pd.read_csv(bx_file, dtype=str)
    acc = df["AccessionNumber"].astype(str).str.strip().str.split(".").str[0]
    bx_accessions[split_name] = set(acc)

# --- CHECK OVERLAP WITH PRETRAIN ---
for split_name, acc_set in bx_accessions.items():
    overlap = pre_acc_set & acc_set
    print(f"\n🔍 Overlap with Bx_{split_name}: {len(overlap)} subjects")
    log_file = f"{log_dir}/overlap_pretrain_vs_Bx_{split_name}.log"
    with open(log_file, "w") as f:
        for acc in sorted(overlap):
            f.write(acc + "\n")
    print(f"📝 Saved overlap log to {log_file}")

# --- CHECK OVERLAPS BETWEEN Bx SPLITS ---
split_keys = list(bx_accessions.keys())
for i in range(len(split_keys)):
    for j in range(i + 1, len(split_keys)):
        a, b = split_keys[i], split_keys[j]
        overlap = bx_accessions[a] & bx_accessions[b]
        print(f"\n🚨 Overlap between Bx_{a} and Bx_{b}: {len(overlap)} subjects")
        if overlap:
            log_file = f"{log_dir}/overlap_Bx_{a}_vs_Bx_{b}.log"
            with open(log_file, "w") as f:
                for acc in sorted(overlap):
                    f.write(acc + "\n")
            print(f"📝 Saved overlap log to {log_file}")
