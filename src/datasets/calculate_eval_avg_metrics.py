import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# Input CSV path
csv_file0 = "mae_eval_distributed_2025_04_08__10_29_01" #"mjepa_eval_distributed_2025_04_21__10_53_00" #NC/MCI
csv_file = f"/gpfs/data/sodicksonlab/gozde/logs/{csv_file0}/csv_logs/mri-probe_r0.csv"

# Read the CSV
df = pd.read_csv(csv_file)

# Separate epoch column
epoch = df['epoch']
metrics_df = df.drop(columns=["epoch"])

# Scale accuracy columns from % to [0,1]
accuracy_columns = ["train acc", "val acc"]
for col in accuracy_columns:
    if col in metrics_df.columns:
        metrics_df[col] = metrics_df[col] / 100.0

# Ignore the first 25% of epochs for statistics
num_epochs = len(df)
start_index = math.floor(0.25 * num_epochs)

df_filtered = df.iloc[start_index:].reset_index(drop=True)
metrics_df_filtered = metrics_df.iloc[start_index:].reset_index(drop=True)

# Calculate averages after ignoring first 25%
average_metrics = metrics_df_filtered.mean()
output_dir = os.path.dirname(csv_file)

# Save averages to CSV
average_csv_path = os.path.join(output_dir, "averages_filtered.csv")
average_metrics.to_csv(average_csv_path, header=["mean"], index_label="metric")
print(f"Filtered averages saved to: {average_csv_path}")

# Track max values and epochs
summary_lines = []

# Max val accuracy
if "val acc" in metrics_df_filtered.columns:
    max_val_acc = metrics_df_filtered["val acc"].max()
    max_val_acc_idx = metrics_df_filtered["val acc"].idxmax()
    max_val_acc_epoch = df_filtered.loc[max_val_acc_idx, "epoch"]
    acc_summary = f"Max val acc after 25% warmup: {max_val_acc:.4f} at epoch {max_val_acc_epoch}"
    summary_lines.append(acc_summary)
    print(acc_summary)

    # Get all val metrics at that epoch
    val_metrics_at_max_acc = df_filtered.loc[max_val_acc_idx, df_filtered.columns.str.startswith("val ")]
    summary_lines.append("Validation metrics at max val acc epoch:")
    summary_lines.extend([f"{col}: {val_metrics_at_max_acc[col]}" for col in val_metrics_at_max_acc.index])
else:
    summary_lines.append("Validation accuracy column not found.")

# Max val F1
f1_col_name = "val f1" if "val f1" in metrics_df_filtered.columns else None
if f1_col_name:
    max_val_f1 = metrics_df_filtered[f1_col_name].max()
    max_val_f1_idx = metrics_df_filtered[f1_col_name].idxmax()
    max_val_f1_epoch = df_filtered.loc[max_val_f1_idx, "epoch"]
    f1_summary = f"Max val F1 after 25% warmup: {max_val_f1:.4f} at epoch {max_val_f1_epoch}"
    summary_lines.append(f1_summary)
    print(f1_summary)

    # Get all val metrics at that epoch
    val_metrics_at_max_f1 = df_filtered.loc[max_val_f1_idx, df_filtered.columns.str.startswith("val ")]
    summary_lines.append("Validation metrics at max val F1 epoch:")
    summary_lines.extend([f"{col}: {val_metrics_at_max_f1[col]}" for col in val_metrics_at_max_f1.index])
else:
    summary_lines.append("Validation F1 column not found.")

# Save to text file
summary_path = os.path.join(output_dir, "max_metrics.txt")
with open(summary_path, "w") as f:
    for line in summary_lines:
        f.write(str(line) + "\n")
print(f"Max metrics saved to: {summary_path}")


# Split metrics into train and val groups
train_metrics = {col: metrics_df[col] for col in metrics_df.columns if "train" in col}
val_metrics = {col: metrics_df[col] for col in metrics_df.columns if "val" in col}

# --- Plot Train Metrics ---
plt.figure(figsize=(12, 6))
for label, values in train_metrics.items():
    plt.plot(epoch, values, label=label)
plt.xlabel("Epoch")
plt.ylabel("Metric Value (0–1)")
plt.title("Training Metrics Over Epochs")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
train_plot_path = os.path.join(output_dir, "train_metrics_plot.png")
plt.savefig(train_plot_path)
print(f"Training plot saved to: {train_plot_path}")
# plt.close()

# --- Plot Val Metrics ---
plt.figure(figsize=(12, 6))
for label, values in val_metrics.items():
    plt.plot(epoch, values, label=label)
plt.xlabel("Epoch")
plt.ylabel("Metric Value (0–1)")
plt.title("Validation Metrics Over Epochs")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
val_plot_path = os.path.join(output_dir, "val_metrics_plot.png")
plt.savefig(val_plot_path)
print(f"Validation plot saved to: {val_plot_path}")
plt.show()
