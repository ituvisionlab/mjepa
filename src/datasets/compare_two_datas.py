import pandas as pd
from pathlib import Path

def load_scan_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalize column names if needed
    df.columns = [c.strip() for c in df.columns]

    # common expected columns (adjust if yours differ)
    # subject_id, contrast, date_acquired, nii_file_path
    for col in ["subject_id", "contrast", "date_acquired", "nii_file_path"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col} in {path}. Found: {df.columns.tolist()}")

    # clean types / whitespace
    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["contrast"] = df["contrast"].astype(str).str.strip()
    df["date_acquired"] = df["date_acquired"].astype(str).str.strip()
    df["nii_file_path"] = df["nii_file_path"].astype(str).str.strip()

    # extract basename (robust to moved folders)
    df["basename"] = df["nii_file_path"].apply(lambda p: Path(p).name)

    # parse date (optional but helpful)
    df["date_acquired_dt"] = pd.to_datetime(df["date_acquired"], errors="coerce")

    # a few useful keys
    df["key_strict"] = df["basename"]
    df["key_fallback"] = (
        df["subject_id"].astype(str) + "||" +
        df["contrast"].astype(str) + "||" +
        df["date_acquired"].astype(str)
    )

    return df
def remove_overlap_from_second(csv_a: str, csv_b: str,
                              name_a="A", name_b="B",
                              save_name_b="cleaned_B.csv"):

    A = load_scan_csv(csv_a)
    B = load_scan_csv(csv_b)

    # find overlapping subjects
    overlap_subjects = set(A["subject_id"]) & set(B["subject_id"])

    print("\n=== REMOVING OVERLAP FROM SECOND DATASET ===")
    print(f"Overlapping subjects: {len(overlap_subjects)}")

    # only clean B (second dataset)
    B_clean = B[~B["subject_id"].isin(overlap_subjects)].copy()

    print(f"{name_b}: {len(B)} -> {len(B_clean)} rows after cleaning")

    # save
    B_clean.to_csv(save_name_b, index=False)

    print(f"\nSaved: {save_name_b}")

    return save_name_b
def summarize(df: pd.DataFrame, name: str):
    print(f"\n=== {name} ===")
    print(f"Lines/volumes: {len(df)}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")
    print("Contrasts:", df["contrast"].value_counts(dropna=False).to_dict())
    # duplicates (useful sanity)
    print(f"Duplicate basenames: {(df['basename'].duplicated().sum())}")
    print(f"Duplicate fallback keys: {(df['key_fallback'].duplicated().sum())}")

def compare(csv_a: str, csv_b: str, name_a="A", name_b="B"):
    A = load_scan_csv(csv_a)
    B = load_scan_csv(csv_b)

    summarize(A, name_a)
    summarize(B, name_b)

    # ---- overlap by strict key (basename)
    setA_strict = set(A["key_strict"])
    setB_strict = set(B["key_strict"])
    strict_overlap = setA_strict & setB_strict

    # ---- overlap by fallback key (subject+contrast+date)
    setA_fallback = set(A["key_fallback"])
    setB_fallback = set(B["key_fallback"])
    fallback_overlap = setA_fallback & setB_fallback

    print("\n=== OVERLAP ===")
    print(f"Strict overlap (same basename): {len(strict_overlap)}")
    print(f"Fallback overlap (subject+contrast+date): {len(fallback_overlap)}")

    # subject-level overlap
    subj_overlap = set(A["subject_id"]) & set(B["subject_id"])
    print(f"Subject overlap: {len(subj_overlap)}")

    # show examples of overlaps / non-overlaps
    print("\nExamples (strict overlap basenames):")
    print(list(sorted(strict_overlap))[:10])

    # compute lines that overlap (strict)
    A_strict_overlap = A[A["key_strict"].isin(strict_overlap)].copy()
    B_strict_overlap = B[B["key_strict"].isin(strict_overlap)].copy()

    # compute lines that are unique to each (strict)
    A_only_strict = A[~A["key_strict"].isin(setB_strict)].copy()
    B_only_strict = B[~B["key_strict"].isin(setA_strict)].copy()

    print("\n=== UNIQUE LINES (strict) ===")
    print(f"{name_a} only (basename not in {name_b}): {len(A_only_strict)}")
    print(f"{name_b} only (basename not in {name_a}): {len(B_only_strict)}")

    # optional: save reports
    A_only_strict.to_csv(f"{name_a}_only_strict.csv", index=False)
    B_only_strict.to_csv(f"{name_b}_only_strict.csv", index=False)
    A_strict_overlap.to_csv(f"{name_a}_overlap_strict.csv", index=False)
    B_strict_overlap.to_csv(f"{name_b}_overlap_strict.csv", index=False)

    print("\nSaved:")
    print(f" - {name_a}_only_strict.csv, {name_b}_only_strict.csv")
    print(f" - {name_a}_overlap_strict.csv, {name_b}_overlap_strict.csv")

# ---- usage ----
compare(
    "nc_ad_test.csv",
    "adni_cv_5folds_stratified/adni_pretrain.csv",
    name_a="old_cluster",
    name_b="new_cluster"
)

"""
clean_new = remove_overlap_from_second(
    "nc_ad_train_k16.csv",
    "nc_ad_test.csv",
    name_a="old_cluster",
    name_b="new_cluster",
    save_name_b="nc_ad_test_cleaned.csv"
)
 
compare(
    "nc_ad_train_k16.csv",
    "nc_ad_test_cleaned.csv",
    name_a="old_cluster",
    name_b="new_clean"
)
"""