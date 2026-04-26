"""
Plot the test confusion matrix for the BEST model.

Best model is identified by highest overall accuracy on the test split.
The test split confusion matrix for that run is then plotted and saved.

Input  : outputs/master_training_outputs/all_confusion_matrices_long.csv
Output : outputs/test_confusion_matrix_bestmodel.png

Example Run:
python scripts/visualization/plot_confusion_matrix_best_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_CSV  = "outputs/master_training_with_outputs/all_confusion_matrices_long.csv"
OUTPUT_PNG = "outputs/test_confusion_matrix_bestmodel.png"

os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)

# ── Identify best model (highest accuracy on test split) ─────────────────────
test_all = df[df["split"] == "test"]

diag_counts = (
    test_all[test_all["true_class_id"] == test_all["pred_class_id"]]
    .groupby("run_name")["count"]
    .sum()
)
total_counts = test_all.groupby("run_name")["count"].sum()
test_accuracy = diag_counts / total_counts

best_run = test_accuracy.idxmax()
best_acc  = test_accuracy[best_run]

# Grab metadata for the best run
meta = df[df["run_name"] == best_run].iloc[0]
best_model        = meta["model"]
best_model_family = meta["model_family"]
best_feature_set  = meta["feature_set"]

print(f"Best run       : {best_run}")
print(f"Model          : {best_model}  |  Family: {best_model_family}")
print(f"Feature set    : {best_feature_set}")
print(f"Test accuracy  : {best_acc:.4f} ({best_acc*100:.2f}%)")

# ── Extract test confusion matrix for that run ───────────────────────────────
test_data = df[(df["run_name"] == best_run) & (df["split"] == "test")]

if test_data.empty:
    raise ValueError(f"No test data found for run '{best_run}'.")

classes = sorted(df["true_class_id"].unique())
n = len(classes)

cm_df = test_data.pivot_table(
    index="true_class_id",
    columns="pred_class_id",
    values="count",
    aggfunc="sum",
    fill_value=0,
).reindex(index=classes, columns=classes, fill_value=0)

cm = cm_df.values.astype(float)

row_sums = cm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
cm_norm = cm / row_sums

test_acc = cm.diagonal().sum() / cm.sum()
print(f"Test accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")

# ── Plot ─────────────────────────────────────────────────────────────────────
cmap = LinearSegmentedColormap.from_list("bw_blue", ["#ffffff", "#1565C0"])

fig, ax = plt.subplots(figsize=(max(8, n * 0.85), max(7, n * 0.75)))
im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Row-normalised recall", fontsize=11)
cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

tick_labels = [str(c) for c in classes]
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(tick_labels, fontsize=9)

for i in range(n):
    for j in range(n):
        pct    = cm_norm[i, j]
        raw    = int(cm[i, j])
        color  = "white" if pct > 0.55 else "black"
        weight = "bold" if i == j else "normal"
        ax.text(
            j, i,
            f"{pct:.1%}\n({raw:,})",
            ha="center", va="center",
            fontsize=7.5, color=color, fontweight=weight,
        )

ax.set_xlabel("Predicted class", fontsize=12, labelpad=8)
ax.set_ylabel("True class",      fontsize=12, labelpad=8)
ax.set_title(
    f"Test Confusion Matrix — Best Model (by Test Accuracy)\n"
    f"{best_model} · {best_feature_set} · {best_run}\n"
    f"Test acc: {best_acc:.2%}",
    fontsize=12, pad=14,
)

plt.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUTPUT_PNG}")
plt.close(fig)