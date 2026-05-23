import pandas as pd
from pathlib import Path
import re

in_csv  = "oasis3_all_bet.csv"
out_csv = "oasis3_pretrain_ids.csv"

df = pd.read_csv(in_csv)

def make_oasis_scan_id(nii_path: str) -> str:
    """
    Build a reproducible scan_id from an OASIS3 BIDS-like filename.
    Examples:
      sub-OAS30052_sess-d4235_acq-TSE_T2w_betmask.nii.gz -> OAS30052_d4235_acq-TSE_T2w
      sub-OAS30052_sess-d4235_T2star_betmask.nii.gz      -> OAS30052_d4235_T2star
      sub-OAS31029_ses-d2448_T1w_betmask.nii.gz          -> OAS31029_d2448_T1w
    """
    fname = Path(str(nii_path)).name

    # subject (supports sub-OASxxxxx or OASxxxxx somewhere)
    m_sub = re.search(r'(?:sub-)?(OAS\d+)', fname)
    subject = m_sub.group(1) if m_sub else None

    # session (sess-d#### or ses-d####)
    m_ses = re.search(r'(?:sess|ses)-([a-zA-Z0-9]+)', fname)
    ses = m_ses.group(1) if m_ses else None

    # keep the "descriptor tail" between session and _betmask
    # e.g. "_acq-TSE_T2w" or "_T1w" or "_T2star"
    tail = ""
    if m_ses:
        start = m_ses.end()
        m_bet = re.search(r'_betmask', fname)
        end = m_bet.start() if m_bet else len(fname)
        tail = fname[start:end].lstrip("_")

    # fallback: if tail can't be parsed, use filename without extensions
    if not subject or not ses or not tail:
        stem = fname.replace(".nii.gz", "").replace(".nii", "")
        return stem

    return f"{subject}_{ses}_{tail}"

df["scan_uid"] = df["nii_file_path"].apply(make_oasis_scan_id)

out_df = df[["subject_id", "scan_uid"]].drop_duplicates().sort_values(["subject_id", "scan_uid"])
out_df.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print("Unique subjects:", out_df["subject_id"].nunique())
print("Unique scans:", len(out_df))
