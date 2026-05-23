import os
import pandas as pd
import re

# === Config ===
abide_root = "/gpfs/data/sodicksonlab/gozde/ABIDE"
meta_csv = "/gpfs/data/sodicksonlab/gozde/ABIDE_6_29_2025.csv"
output_csv = "/gpfs/home/unalg01/jepa/src/datasets/ABIDE_master.csv"

# === Load metadata CSV ===
df_meta = pd.read_csv(meta_csv, sep='\t' if '\t' in open(meta_csv).readline() else ',')
df_meta['Image Data ID'] = df_meta['Image Data ID'].astype(str)
df_meta = df_meta.set_index('Image Data ID')

# === Helper to extract ImageID from filename ===
def extract_image_id(filename):
    match = re.search(r'I(\d+)', filename)
    if match:
        return 'I' + match.group(1)
    return None

# === Map label text to numeric ===
def label_from_group(group):
    return 1 if group.strip().lower() == 'autism' else 0

# === Walk and extract data ===
records = []

for root, dirs, files in os.walk(abide_root):
    for file in files:
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            image_id = extract_image_id(file)
            if image_id and image_id in df_meta.index:
                row = df_meta.loc[image_id]
                subject_id = str(row['Subject'])
                group = row['Group']
                sex = row['Sex']
                age = row['Age']
                date_acquired = '2000-01-01'  # Metadata has default dates; adjust if you parse folders
                contrast = row['Modality'] if 'Modality' in row else "MP-RAGE"

                record = {
                    'label': label_from_group(group),
                    'subject_id': subject_id,
                    'contrast': contrast,
                    'date_acquired': date_acquired,
                    'subject_sex': sex,
                    'subject_age': age,
                    'subject_weight': -1,
                    'nii_file_path': os.path.join(root, file),
                    'xmin': -1,
                    'xmax': -1,
                    'ymin': -1,
                    'ymax': -1,
                    'zmin': -1,
                    'zmax': -1
                }
                records.append(record)

# === Save to CSV ===
df_out = pd.DataFrame(records)
df_out.to_csv(output_csv, index=False)
print(f"Saved master CSV with {len(df_out)} entries to {output_csv}")
