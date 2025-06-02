import os
import shutil
import nibabel as nib
import pandas as pd

# === CONFIGURATION ===
root_dir = "/gpfs/data/prostatelab/NIFTI"
bad_dir = "/gpfs/data/prostatelab/NIFTI_BAD"
expected_files = ['adc.nii.gz', 'axt2.nii.gz', 'b1500.nii.gz']
output_csv = "/gpfs/home/unalg01/jepa/validation_logs/valid_subjects_3channel.csv"
log_file = "/gpfs/home/unalg01/jepa/validation_logs/invalid_subjects_3channel.log"

os.makedirs(bad_dir, exist_ok=True)

valid_subjects = []
invalid_logs = []

subject_list = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
total = len(subject_list)
subject_counter = 0

# === PROCESS SUBJECT FOLDERS ===
for subject_id in subject_list:
    subject_counter += 1
    print(f"[{subject_counter}/{total}] Processing subject...")

    subject_path = os.path.join(root_dir, subject_id)
    has_error = False
    subject_entry = {"subject_id": subject_id}

    for scan_type in expected_files:
        nii_path = os.path.join(subject_path, scan_type)

        if not os.path.exists(nii_path):
            invalid_logs.append(f"{subject_id}: missing {scan_type}")
            has_error = True
            break

        try:
            nii = nib.load(nii_path)
            volume = nii.get_fdata()
            if volume.ndim != 3:
                invalid_logs.append(f"{subject_id}: {scan_type} has shape {volume.shape} (not 3D)")
                has_error = True
                break
        except Exception as e:
            invalid_logs.append(f"{subject_id}: error loading {scan_type} → {e}")
            has_error = True
            break

        # Save path for valid entries
        subject_entry[scan_type.split('.')[0]] = nii_path

    if has_error:
        dst_path = os.path.join(bad_dir, subject_id)
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.move(subject_path, dst_path)
    else:
        valid_subjects.append(subject_entry)

# === SAVE OUTPUTS ===
df = pd.DataFrame(valid_subjects)
df.to_csv(output_csv, index=False)
print(f"\n Saved list of {len(df)} valid subjects to: {output_csv}")

with open(log_file, 'w') as f:
    for line in invalid_logs:
        f.write(line + "\n")
print(f" Logged {len(invalid_logs)} invalid subjects to: {log_file}")
