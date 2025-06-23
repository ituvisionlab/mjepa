# === Modified Summary Logger for ADNI Split ===

import pandas as pd

# Load pretrain and downstream CSVs
pretrain_df = pd.read_csv("adni_cv_folds_stratified/adni_pretrain.csv")
downstream_df = pd.read_csv("adni_cv_folds_stratified/adni_downstream.csv")

# Combine all for full label distribution
full_df = pd.concat([pretrain_df, downstream_df], ignore_index=True)

# Per-scan label distributions
scan_label_counts = full_df['label'].value_counts().sort_index()
scan_label_percent = scan_label_counts / scan_label_counts.sum() * 100

# Per-subject label distributions
subject_labels = full_df[['subject_id', 'label']].drop_duplicates()
subj_label_counts = subject_labels['label'].value_counts().sort_index()
subj_label_percent = subj_label_counts / subj_label_counts.sum() * 100

# Build summary table
df_summary = pd.DataFrame({
    "Label": scan_label_counts.index,
    "Scans": scan_label_counts.values,
    "Scan_%": scan_label_percent.round(2).values,
    "Subjects": subj_label_counts.values,
    "Subject_%": subj_label_percent.round(2).values
})

# Save summary to CSV
df_summary.to_csv("adni_cv_folds_stratified/adni_split_summary.csv", index=False)
print("✅ Saved split summary with scan- and subject-level label distributions.")
