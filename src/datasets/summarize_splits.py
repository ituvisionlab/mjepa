import pandas as pd
import os
from glob import glob

split_dir = "adni_cv_folds_stratified"
csvs = sorted(glob(os.path.join(split_dir, "*.csv")))
summary = []

for path in csvs:
    if "subject_split_map" in path:
        continue

    try:
        df = pd.read_csv(path)
        if "label" not in df.columns or "subject_id" not in df.columns:
            raise ValueError("Missing required columns")

        label_counts = df["label"].value_counts().to_dict()
        subject_count = df["subject_id"].nunique()
        total_volumes = len(df)

        summary.append({
            "file": os.path.basename(path),
            "subjects": subject_count,
            "volumes": total_volumes,
            "label_0_volumes": label_counts.get(0, 0),
            "label_1_volumes": label_counts.get(1, 0),
        })

    except Exception as e:
        print(f"⚠️ Skipped {os.path.basename(path)} due to error: {e}")

summary_df = pd.DataFrame(summary).sort_values("file")
print(summary_df.to_string(index=False))

summary_df.to_csv(os.path.join(split_dir, "adni_split_summary.csv"), index=False)
print(f"\n✅ Summary saved to: {os.path.join(split_dir, 'adni_split_summary.csv')}")
