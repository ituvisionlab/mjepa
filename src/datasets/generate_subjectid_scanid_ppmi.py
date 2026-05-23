import pandas as pd
import re
from pathlib import Path

in_csv  = "ppmi_all_bet_nii_with_bbox.csv"
out_csv = "ppmi_pretrain_ids.csv"

df = pd.read_csv(in_csv)

def extract_scan_uid(path):
    path = str(path)

    # ✅ ADNI / PPMI → real image ID
    m = re.search(r'(I\d+)', path)
    if m:
        return m.group(1)

    # ✅ OASIS3 → build from BIDS filename
    fname = Path(path).name

    m_sub = re.search(r'(OAS\d+)', fname)
    m_ses = re.search(r'(?:sess|ses)-([a-zA-Z0-9]+)', fname)

    if m_sub and m_ses:
        subject = m_sub.group(1)
        session = m_ses.group(1)

        tail = fname[m_ses.end():]
        tail = tail.split("_betmask")[0].lstrip("_")

        return f"{subject}_{session}_{tail}"

    # fallback → filename stem
    return Path(path).stem


df["scan_uid"] = df["nii_file_path"].apply(extract_scan_uid)

out_df = df[["subject_id", "scan_uid"]].drop_duplicates()
out_df = out_df.sort_values(["subject_id", "scan_uid"])

out_df.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print("Unique subjects:", out_df.subject_id.nunique())
print("Unique scans:", len(out_df))
