import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np

# Input CSV path
csv_file0 = "mjepa_eval_distributed_2025_05_09__10_01_21"
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

# --------------------- PLOTTING ---------------------

# Split metrics into train and val groups (EXCLUDE losses)
train_metrics = {col: metrics_df[col] for col in metrics_df.columns if "train" in col and "loss" not in col}
val_metrics = {col: metrics_df[col] for col in metrics_df.columns if "val" in col and "loss" not in col}

# Function to annotate max point
def annotate_max_point(ax, x, y, label):
    max_idx = np.argmax(y)
    max_x = x.iloc[max_idx]
    max_y = y.iloc[max_idx]
    ax.plot(max_x, max_y, 'ro')  # red dot
    ax.annotate(f"Max {label}\n{max_y:.4f} @ {max_x}", xy=(max_x, max_y), xytext=(max_x, max_y + 0.02),
                arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=8)

# --- Plot Train Metrics (without loss) ---
fig, ax = plt.subplots(figsize=(12, 6))
for label, values in train_metrics.items():
    ax.plot(epoch, values, label=label)
    annotate_max_point(ax, epoch, values, label)

ax.set_xlabel("Epoch")
ax.set_ylabel("Metric Value (0–1)")
ax.set_title("Training Metrics (Acc/F1 etc.) Over Epochs")
ax.legend(loc="best")
ax.grid(True)
plt.tight_layout()
train_plot_path = os.path.join(output_dir, "train_metrics_plot.png")
plt.savefig(train_plot_path)
#plt.show()
#plt.close()
print(f"Training metrics plot saved to: {train_plot_path}")

# --- Plot Val Metrics (without loss) ---
fig, ax = plt.subplots(figsize=(12, 6))
for label, values in val_metrics.items():
    ax.plot(epoch, values, label=label)
    annotate_max_point(ax, epoch, values, label)

ax.set_xlabel("Epoch")
ax.set_ylabel("Metric Value (0–1)")
ax.set_title("Validation Metrics (Acc/F1 etc.) Over Epochs")
ax.legend(loc="best")
ax.grid(True)
plt.tight_layout()
val_plot_path = os.path.join(output_dir, "val_metrics_plot.png")
plt.savefig(val_plot_path)
# plt.close()
print(f"Validation metrics plot saved to: {val_plot_path}")

# --- Plot Combined Smoothed Train & Val Loss ---
def smooth_curve(values, window_size=5):
    """Simple moving average smoothing"""
    if len(values) < window_size:
        return values  # Don't smooth if not enough points
    return np.convolve(values, np.ones(window_size) / window_size, mode='valid')

fig, ax = plt.subplots(figsize=(12, 6))
window_size = 5

# Plot smoothed train loss
if "train loss" in metrics_df.columns:
    smoothed_train_loss = smooth_curve(metrics_df["train loss"].values, window_size)
    smoothed_epoch = epoch[window_size - 1:]  # Align epochs after smoothing
    ax.plot(smoothed_epoch, smoothed_train_loss, label="Train Loss (smoothed)", color="blue")
    min_idx = np.argmin(smoothed_train_loss)
    min_x = smoothed_epoch.iloc[min_idx]
    min_y = smoothed_train_loss[min_idx]
    ax.plot(min_x, min_y, 'go')  # green dot
    ax.annotate(f"Min Train Loss\n{min_y:.4f} @ {min_x}", xy=(min_x, min_y), xytext=(min_x, min_y + 0.02),
                arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=8)

# Plot smoothed val loss
if "val loss" in metrics_df.columns:
    smoothed_val_loss = smooth_curve(metrics_df["val loss"].values, window_size)
    ax.plot(smoothed_epoch, smoothed_val_loss, label="Val Loss (smoothed)", color="orange", linestyle="--")
    min_idx = np.argmin(smoothed_val_loss)
    min_x = smoothed_epoch.iloc[min_idx]
    min_y = smoothed_val_loss[min_idx]
    ax.plot(min_x, min_y, 'go')  # green dot
    ax.annotate(f"Min Val Loss\n{min_y:.4f} @ {min_x}", xy=(min_x, min_y), xytext=(min_x, min_y + 0.02),
                arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=8)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss Value")
ax.set_title("Smoothed Training and Validation Loss Over Epochs")
ax.legend(loc="best")
ax.grid(True)
plt.tight_layout()
loss_plot_path = os.path.join(output_dir, "train_val_loss_plot.png")
plt.savefig(loss_plot_path)
#plt.close()
plt.show()
print(f"Training + Validation Loss plot (smoothed) saved to: {loss_plot_path}")
