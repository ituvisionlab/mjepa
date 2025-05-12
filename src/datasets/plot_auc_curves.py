import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

def plot_saved_auc_curves(
    save_dir, step="latest", num_classes=2, figsize=(8, 6), show=True, save_img=False
):
    """
    Loads .npy outputs/labels saved during eval and plots ROC curves.

    Args:
        save_dir (str): Path where 'outputs_stepX.npy' and 'labels_stepX.npy' are saved.
        step (str or int): Step identifier used in filenames (e.g. 'step0' or 'latest').
        num_classes (int): Number of classes in the classification task.
        figsize (tuple): Size of the matplotlib figure.
        show (bool): Whether to display the plot.
        save_img (bool): Whether to save the plot as a PNG in the same directory.

    Returns:
        None
    """
    outputs_path = os.path.join(save_dir, f"outputs_step{step}.npy")
    labels_path = os.path.join(save_dir, f"labels_step{step}.npy")

    try:
        outputs = np.load(outputs_path)
        labels = np.load(labels_path)
    except Exception as e:
        print(f"[ERROR] Could not load files: {e}")
        return

    if num_classes == 2:
        labels_bin = label_binarize(labels, classes=[0, 1])
    else:
        labels_bin = label_binarize(labels, classes=np.arange(num_classes))

    # Plot ROC curves
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=figsize)
    for i in range(num_classes):
        plt.plot(
            fpr[i], tpr[i],
            label=f"Class {i} (AUC = {roc_auc[i]:.2f})"
        )
    plt.plot(
        fpr["macro"], tpr["macro"],
        label=f"Macro Average (AUC = {roc_auc['macro']:.2f})",
        linestyle='--', color='black'
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve at Step {step}")
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_img:
        out_path = os.path.join(save_dir, f"roc_curve_step{step}.png")
        plt.savefig(out_path)
        print(f"[INFO] ROC curve saved to {out_path}")

    if show:
        plt.show()
    else:
        plt.close()

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
import pandas as pd

def generate_per_class_roc_auc_table(save_dir, step="latest", num_classes=2):
    """
    Computes per-class ROC curves and AUC scores from saved outputs and labels.

    Args:
        save_dir (str): Path where 'outputs_stepX.npy' and 'labels_stepX.npy' are saved.
        step (str or int): Step identifier (e.g., 0, 1, 'latest').
        num_classes (int): Total number of classes.

    Returns:
        pd.DataFrame: AUC scores and fpr/tpr arrays per class.
    """
    outputs_path = os.path.join(save_dir, f"outputs_step{step}.npy")
    labels_path = os.path.join(save_dir, f"labels_step{step}.npy")

    try:
        outputs = np.load(outputs_path)
        labels = np.load(labels_path)
    except Exception as e:
        print(f"[ERROR] Failed to load AUC data: {e}")
        return None

    labels_bin = label_binarize(labels, classes=np.arange(num_classes))
    roc_data = []

    for i in range(num_classes):
        try:
            fpr, tpr, _ = roc_curve(labels_bin[:, i], outputs[:, i])
            auc_score = auc(fpr, tpr)
        except ValueError as e:
            fpr, tpr, auc_score = None, None, float('nan')
            print(f"[WARNING] Class {i} AUC could not be computed: {e}")

        roc_data.append({
            'Class': i,
            'AUC': auc_score,
            'FPR': fpr,
            'TPR': tpr
        })

    df = pd.DataFrame([{'Class': d['Class'], 'AUC': d['AUC']} for d in roc_data])
    print(df.to_markdown(index=False))
    return df, roc_data

if __name__ == '__main__':
   plot_saved_auc_curves(
    save_dir="/gpfs/home/unalg01/jepa/logs/auc_debug",  
    step=0,
    num_classes=2,  
    save_img=True,
    show=True
)

df, roc_data = generate_per_class_roc_auc_table(
    save_dir="/gpfs/home/unalg01/jepa/logs/auc_debug",
    step=0,
    num_classes=2
)

# You’ll get a nicely formatted table like:
# |   Class |   AUC |
# |--------:|------:|
# |       0 | 0.91  |
# |       1 | 0.89  |
# |       2 | 0.95  |
# Each entry in roc_data contains:
# {
#     'Class': 1,
#     'AUC': 0.92,
#     'FPR': array([...]),
#     'TPR': array([...])
# }
