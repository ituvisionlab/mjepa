#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
import pandas as pd


REQUIRED_COLS = ["subject_id", "contrast", "date_acquired", "nii_file_path"]


def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing columns: {missing}. Found columns: {df.columns.tolist()}"
        )

    # normalize
    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["contrast"] = df["contrast"].astype(str).str.strip()
    df["date_acquired"] = df["date_acquired"].astype(str).str.strip()
    df["nii_file_path"] = df["nii_file_path"].astype(str).str.strip()

    # robust to moved folders
    df["basename"] = df["nii_file_path"].apply(lambda p: Path(p).name)

    # keys
    df["key_strict"] = df["basename"]
    df["key_fallback"] = (
        df["subject_id"] + "||" + df["contrast"] + "||" + df["date_acquired"]
    )

    return df


def find_downstream_csvs(downstream_dir: str):
    """
    Looks for files like:
      scan_fold0_downtest_nc_ad.csv
      scan_fold0_downtrain_nc_mci.csv
      scan_fold0_downval_nc_ad.csv
    in downstream_dir (recursive).
    """
    p = Path(downstream_dir)
    files = sorted(p.rglob("*_fold*_down*_nc_*.csv"))
    files += sorted(p.rglob("*_fold*_downtest_nc_*.csv"))
    files = sorted(set(files))
    return [str(f) for f in files]


def parse_fold_and_split(path: str):
    name = Path(path).name
    # fold number
    m = re.search(r"_fold(\d+)_", name)
    fold = int(m.group(1)) if m else None

    split = None
    for s in ["downtrain", "downval", "downtest"]:
        if f"_{s}_" in name:
            split = s
            break

    task = None
    for t in ["nc_ad", "nc_mci"]:
        if name.endswith(f"_{t}.csv"):
            task = t
            break

    return fold, split, task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", required=True, help="Path to scan_pretrain.csv")
    ap.add_argument(
        "--downstream_dir",
        required=True,
        help="Directory containing scan_fold*_*.csv downstream files",
    )
    ap.add_argument(
        "--report_csv",
        default="pretrain_leakage_report.csv",
        help="Output CSV report path",
    )
    ap.add_argument(
        "--save_leaked_rows",
        action="store_true",
        help="If set, writes leaked row extracts per downstream file",
    )
    args = ap.parse_args()

    pretrain_path = args.pretrain
    downstream_dir = args.downstream_dir

    pre = load_df(pretrain_path)
    pre_strict = set(pre["key_strict"])
    pre_fallback = set(pre["key_fallback"])
    pre_subjects = set(pre["subject_id"])

    ds_files = find_downstream_csvs(downstream_dir)
    if not ds_files:
        raise SystemExit(
            f"No downstream CSVs found under {downstream_dir}. "
            "Expected names like scan_fold0_downtrain_nc_ad.csv"
        )

    rows = []
    total_union_strict = set()
    total_union_fallback = set()
    total_union_subjects = set()

    # For overall leaked scan names
    overall_leaked_strict = set()
    overall_leaked_fallback = set()
    overall_leaked_subjects = set()

    for f in ds_files:
        ds = load_df(f)

        ds_strict = set(ds["key_strict"])
        ds_fallback = set(ds["key_fallback"])
        ds_subjects = set(ds["subject_id"])

        leak_strict = pre_strict & ds_strict
        leak_fallback = pre_fallback & ds_fallback
        leak_subjects = pre_subjects & ds_subjects

        total_union_strict |= ds_strict
        total_union_fallback |= ds_fallback
        total_union_subjects |= ds_subjects

        overall_leaked_strict |= leak_strict
        overall_leaked_fallback |= leak_fallback
        overall_leaked_subjects |= leak_subjects

        fold, split, task = parse_fold_and_split(f)

        rows.append(
            {
                "file": f,
                "fold": fold,
                "split": split,
                "task": task,
                "downstream_rows": len(ds),
                "downstream_subjects": ds["subject_id"].nunique(),
                # leakage counts
                "leak_scans_strict_basename": len(leak_strict),
                "leak_rows_fallback_key": len(leak_fallback),
                "leak_subjects": len(leak_subjects),
                # leakage rates
                "leak_rate_scans_strict_vs_downstream": (len(leak_strict) / max(1, len(ds_strict))),
                "leak_rate_subjects_vs_downstream": (len(leak_subjects) / max(1, len(ds_subjects))),
            }
        )

        if args.save_leaked_rows and leak_strict:
            leaked = ds[ds["key_strict"].isin(leak_strict)].copy()
            out = Path(args.report_csv).with_suffix("").name + "_leaked_rows"
            Path(out).mkdir(parents=True, exist_ok=True)
            leaked_out = Path(out) / (Path(f).stem + "__LEAKED_STRICT.csv")
            leaked.to_csv(leaked_out, index=False)

    report = pd.DataFrame(rows).sort_values(["fold", "task", "split", "file"], na_position="last")
    report.to_csv(args.report_csv, index=False)

    # Overall totals
    print("\n=== PRETRAIN SUMMARY ===")
    print(f"Pretrain rows: {len(pre)}")
    print(f"Pretrain subjects: {pre['subject_id'].nunique()}")
    print(f"Pretrain unique basenames: {len(pre_strict)}")

    print("\n=== DOWNSTREAM SUMMARY (ALL FILES UNION) ===")
    print(f"Downstream files found: {len(ds_files)}")
    print(f"Union unique basenames: {len(total_union_strict)}")
    print(f"Union subjects: {len(total_union_subjects)}")

    print("\n=== TOTAL LEAKAGE (PRETRAIN ∩ ANY DOWNSTREAM) ===")
    print(f"Leaked scans (STRICT by basename): {len(overall_leaked_strict)}")
    print(f"Leaked fallback-keys (subject+contrast+date): {len(overall_leaked_fallback)}")
    print(f"Leaked subjects: {len(overall_leaked_subjects)}")

    # Also show rates
    print("\n=== LEAKAGE RATES ===")
    print(f"Strict scan leakage rate vs pretrain basenames: {len(overall_leaked_strict) / max(1, len(pre_strict)):.4f}")
    print(f"Subject leakage rate vs pretrain subjects: {len(overall_leaked_subjects) / max(1, len(pre_subjects)):.4f}")

    # Small peek at worst offenders
    print("\n=== TOP 10 DOWNSTREAM FILES BY STRICT SCAN LEAKAGE ===")
    print(
        report.sort_values("leak_scans_strict_basename", ascending=False)
        .head(10)[
            ["fold", "task", "split", "downstream_rows", "downstream_subjects",
             "leak_scans_strict_basename", "leak_subjects"]
        ]
        .to_string(index=False)
    )

    print(f"\n[OK] Wrote report: {args.report_csv}")
    if args.save_leaked_rows:
        print("[OK] Wrote leaked row extracts to: <reportname>_leaked_rows/")


if __name__ == "__main__":
    main()
