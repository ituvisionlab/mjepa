import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import random

CSV_PATH = "/gpfs/home/unalg01/jepa/src/datasets/ucsf_all_nii.csv"
SAVE_DIR = "/gpfs/home/unalg01/jepa/src/datasets/UCSF_QC_midslices_bbox"
NUM_SAMPLES = 50

os.makedirs(SAVE_DIR, exist_ok=True)

seed=1974
df = pd.read_csv(CSV_PATH)
sampled = df.sample(n=NUM_SAMPLES, random_state=seed)

for i, row in sampled.iterrows():
    path = row["nii_file_path"]
    sid = row.get("subject_id", "unknown")
    filename = os.path.basename(path)
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        continue

    try:
        vol = nib.load(path).get_fdata()  # shape: H, W, T
        zmin, zmax = int(row["zmin"]), int(row["zmax"])
        ymin, ymax = int(row["ymin"]), int(row["ymax"])
        xmin, xmax = int(row["xmin"]), int(row["xmax"])

        H, W, T = vol.shape
        if not (0 <= xmin < xmax < H and 0 <= ymin < ymax < W and 0 <= zmin < zmax < T):
            print(f"[SKIP] Invalid bbox for subject {sid}: ({xmin},{xmax},{ymin},{ymax},{zmin},{zmax})")
            continue

        mid_ = (zmin + zmax) // 2
        slice_img = vol[:, :, mid_]

        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-6)

        fig, ax = plt.subplots()
        ax.imshow(slice_img, cmap='gray')

        rect = plt.Rectangle(
            (ymin, xmin),
            ymax - ymin,
            xmax - xmin,
            linewidth=1.5,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.set_title(f"{filename}\nSubject {sid} - MidZ={mid_}")
        ax.axis('off')

        out_path = os.path.join(SAVE_DIR, f"slice_{i}_{filename.replace('.nii.gz','')}.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"[ERROR] Failed on {path}: {e}")
        continue

print(f"[DONE] Saved {len(os.listdir(SAVE_DIR))} midslices with bounding boxes to {SAVE_DIR}")
