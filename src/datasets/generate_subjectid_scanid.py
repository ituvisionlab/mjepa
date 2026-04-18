import pandas as pd
import re
from pathlib import Path

# =========================
# Input / Output
# =========================
in_csv = "abide_downval.csv"

out_dir = Path("downstream_subject_ids")
out_dir.mkdir(parents=True, exist_ok=True)

out_csv = out_dir / "abide_downval_uids.csv"

print("Working directory:", Path.cwd())

# =========================
# Load CSV
# =========================
df = pd.read_csv(in_csv)

# =========================
# Helpers
# =========================
def strip_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem

def extract_scan_uid(path: str) -> str:
    path = str(path)
    fname = Path(path).name
    stem = strip_ext(fname)

    # 1) ADNI / PPMI -> real Image ID
    m = re.search(r'(I\d+)', path)
    if m:
        return m.group(1)

    # 2) BraTS -> case id
    m = re.search(r'(BraTS-[A-Za-z]+-\d{5}-\d{3})', path)
    if m:
        return m.group(1)

    # 3) IXI -> filename base without _betmask
    if fname.startswith("IXI"):
        return re.sub(r'_betmask$', '', stem)

    # 4) MOOD -> numeric file id like 00000
    if "/MOOD/" in path and re.fullmatch(r"\d{5}", stem):
        return stem

    # 5) UCSF-PDGM -> file-level unique (filename stem)
    if "UCSF-PDGM" in stem or "/UCSF/" in path:
        return stem

    # 6) OASIS3 -> derive from BIDS-ish filename
    m_sub = re.search(r'(OAS\d+)', fname)
    m_ses = re.search(r'(?:sess|ses)-([A-Za-z0-9]+)', fname)
    if m_sub and m_ses:
        subject = m_sub.group(1)
        session = m_ses.group(1)
        tail = fname[m_ses.end():]
        tail = tail.split("_betmask")[0].lstrip("_")
        return f"{subject}_{session}_{tail}"

    # 7) fallback
    return re.sub(r'_betmask$', '', stem)

# =========================
# Build output
# =========================
df["scan_uid"] = df["nii_file_path"].apply(extract_scan_uid)

out_df = (
    df[["subject_id", "scan_uid"]]
    .dropna()
    .drop_duplicates()
    .sort_values(["subject_id", "scan_uid"])
)

# =========================
# Save
# =========================
out_df.to_csv(out_csv, index=False)

print("Saved to:", out_csv.resolve())
print("Unique subjects:", out_df["subject_id"].nunique())
print("Unique scan_uids:", len(out_df))
