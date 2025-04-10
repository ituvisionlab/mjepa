import pandas as pd
import matplotlib.pyplot as plt
import os

# Input CSV path
csv_file = "/gpfs/data/sodicksonlab/gozde/logs/mjepa_eval_distributed_2025_04_03__21_22_28/csv_logs/mri-probe_r0.csv"

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

# Calculate averages
average_metrics = metrics_df.mean()
output_dir = os.path.dirname(csv_file)

# Save averages to CSV
average_csv_path = os.path.join(output_dir, "averages.csv")
average_metrics.to_csv(average_csv_path, header=["mean"], index_label="metric")
print(f"Averages saved to: {average_csv_path}")

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
plt.show()  # This keeps all plots open until you manually close them
#plt.show(block=False)  # Non-blocking
# plt.close()
