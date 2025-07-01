import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# --- Settings ---
FOLDER = "/gpfs/data/sodicksonlab/gozde/SCAN/SCAN_NIFTI/NACC659755"
SAVE_DIR = "/gpfs/home/unalg01/jepa/logs/midslice_visuals"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Traverse all _betmask.nii.gz files ---
for fname in os.listdir(FOLDER):
    if not fname.endswith("_betmask.nii.gz"):
        continue

    fpath = os.path.join(FOLDER, fname)
    mask_path = fpath.replace("_betmask.nii.gz", "_betmask_mask.nii.gz")

    if not os.path.exists(mask_path):
        print(f"[SKIP] No mask for {fname}")
        continue

    try:
        # Load image and mask
        img = nib.load(fpath).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        if not np.any(mask):
            print(f"[EMPTY] Mask is empty: {fname}")
            continue

        # Sanity check image shape
        if img.ndim != 3:
            print(f"[SKIP] Not 3D: {fname}")
            continue

        # Compute bbox from mask
        coords = np.argwhere(mask > 0)
        xmin, ymin, zmin = coords.min(axis=0)
        xmax, ymax, zmax = coords.max(axis=0)

        # Mid-slice along Z (axis 2 in H, W, Z)
        mid_z = (zmin + zmax) // 2
        slice_img = img[:, :, mid_z]

        # Normalize for display
        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-6)

        # Draw
        fig, ax = plt.subplots()
        ax.imshow(slice_img, cmap='gray')

        # bbox: (y, x) → (xmin:row, ymin:col)
        rect = plt.Rectangle(
            (ymin, xmin),
            ymax - ymin,
            xmax - xmin,
            edgecolor='red',
            linewidth=1.5,
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.set_title(f"{fname} | MidZ={mid_z}")
        ax.axis('off')

        out_path = os.path.join(SAVE_DIR, f"midslice_{fname.replace('.nii.gz', '.png')}")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"[ERROR] {fname}: {e}")
        continue

print(f"[DONE] Visuals saved to {SAVE_DIR}")
