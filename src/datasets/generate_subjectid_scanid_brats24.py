import pandas as pd
import re
from pathlib import Path

in_csv  = "brats24_all_nii.csv"
out_csv = "brats24_pretrain.csv"

df = pd.read_csv(in_csv)

def extract_scan_uid(path):
    path = str(path)

    # 1) ADNI / PPMI style -> real Image ID
    m = re.search(r'(I\d+)', path)
    if m:
        return m.group(1)

    # 2) BraTS style -> case id like BraTS-GLI-02795-100
    m = re.search(r'(BraTS-[A-Za-z]+-\d{5}-\d{3})', path)
    if m:
        return m.group(1)

    # 3) OASIS3 style -> derive from BIDS-ish filename
    fname = Path(path).name
    m_sub = re.search(r'(OAS\d+)', fname)
    m_ses = re.search(r'(?:sess|ses)-([A-Za-z0-9]+)', fname)
    if m_sub and m_ses:
        subject = m_sub.group(1)
        session = m_ses.group(1)
        tail = fname[m_ses.end():]
        tail = tail.split("_betmask")[0].lstrip("_")
        return f"{subject}_{session}_{tail}"

    # 4) fallback
    return Path(path).stem

df["scan_uid"] = df["nii_file_path"].apply(extract_scan_uid)

out_df = (
    df[["subject_id", "scan_uid"]]
    .dropna()
    .drop_duplicates()
    .sort_values(["subject_id", "scan_uid"])
)

out_df.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print("Unique subjects:", out_df["subject_id"].nunique())
print("Unique scan_uids:", len(out_df))
