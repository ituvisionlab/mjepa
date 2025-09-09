import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

CSV_PATHS = {
    "pretrain": "/gpfs/home/unalg01/jepa/src/datasets/Prostate_pretraining_single.csv",
    "train": "/gpfs/home/unalg01/jepa/src/datasets/Prostate_downstream_train_single.csv",
    "val": "/gpfs/home/unalg01/jepa/src/datasets/Prostate_downstream_val_single.csv",
    "test": "/gpfs/home/unalg01/jepa/src/datasets/Prostate_downstream_test_single.csv"
}

SAVE_DIR = "/gpfs/home/unalg01/jepa/src/datasets/Prostate_Downstream_QC"
NUM_SAMPLES = 30
RANDOM_SEED = 1974

os.makedirs(SAVE_DIR, exist_ok=True)

for split, csv_path in CSV_PATHS.items():
    df = pd.read_csv(csv_path)
    sampled_df = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=RANDOM_SEED)

    for i, row in sampled_df.iterrows():
        path = row["nii_file_path"]
        sid = row.get("subject_id", "unknown")
        contrast = row.get("contrast", "unknown")
        filename = os.path.basename(path)
        label = row["label"]

        if not os.path.exists(path):
            print(f"[MISSING] {path}")
            continue

        try:
            vol = nib.load(path).get_fdata()
            H, W, T = vol.shape

            # Mid slice along z-axis
            mid_z = T // 2
            slice_img = vol[:, :, mid_z]

            # Normalize for visualization
            slice_img = (slice_img - np.min(slice_img)) / (np.ptp(slice_img) + 1e-6)

            # Plot and save
            fig, ax = plt.subplots()
            ax.imshow(slice_img.T, cmap='gray', origin='lower')  # transpose for anatomical correctness
            ax.set_title(f"{split.upper()} | Sub: {sid} | Label: {label} | Contrast: {contrast}\n{filename} - Z:{mid_z}")
            ax.axis('off')

            out_dir = os.path.join(SAVE_DIR, split)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{split}_{i}_{sid}_{contrast}_{label}.png")
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"[ERROR] Failed on {path}: {e}")
            continue

print(f"[DONE] Visual QC images saved to {SAVE_DIR}")

