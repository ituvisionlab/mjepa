import pandas as pd
import re

# Paths
INPUT_CSV = "SCAN_NIFTI_all_with_betmask_and_bbox.csv"
OUTPUT_CSV = "SCAN_NIFTI_all_with_cleaned_contrast.csv"

# Load data
df = pd.read_csv(INPUT_CSV)

# Contrast extractor
def extract_contrast_from_filename(filename):
    fname = filename.lower()
    if 'flair' in fname:
        return 'FLAIR'
    elif 'mprage' in fname:
        return 'MPRAGE'
    elif 't2' in fname:
        return 'T2'
    elif re.search(r'\bt1\b', fname):
        return 'T1'
    elif 'ir' in fname or 'fspgr' in fname:
        return 'IR'
    elif 'gre' in fname:
        return 'GRE'
    elif 'space' in fname:
        return 'SPACE'
    elif 'star' in fname:
        return 'STAR'
    else:
        return 'OTHER'

# Replace the 'contrast' field with cleaned values
df["contrast"] = df["nii_file_path"].apply(lambda x: extract_contrast_from_filename(str(x)))

# Save output
df.to_csv(OUTPUT_CSV, index=False)
print(f"[DONE] Updated 'contrast' column written to {OUTPUT_CSV}")
