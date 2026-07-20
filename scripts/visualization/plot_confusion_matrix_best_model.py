"""
Plot the test confusion matrix for the BEST model.

Best model is identified by highest overall accuracy on the test split.
The test split confusion matrix for that run is then plotted and saved.

Input  : outputs/master_training_with_outputs/all_confusion_matrices_long.csv
Output : outputs/figures/test_confusion_matrix_bestmodel.png

Example Run:
    python scripts/visualization/plot_confusion_matrix_best_model.py

Complete Example Run:
    python scripts/visualization/plot_confusion_matrix_best_model.py \
        --add-title \
        --output-plot outputs/figures/test_confusion_matrix_bestmodel.png

Reproduction and AOI adaptation
-------------------------------
Workflow role: Turn prepared rasters, vectors, and tables into thesis-ready figures.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--add-title``, ``--output-plot``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace raster/vector/palette paths with target-AOI products and verify matching CRS, extent, class IDs, units, and map annotations before publication.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_CSV = "outputs/master_training_with_outputs/all_confusion_matrices_long.csv"
OUTPUT_PNG = "outputs/figures/test_confusion_matrix_bestmodel.png"


# ── Command-line arguments ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Plot the test confusion matrix for the best model."
)
parser.add_argument(
    "--add-title",
    action="store_true",
    help="Show title and subtitle on top of the plot.",
)
parser.add_argument(
    "--output-plot",
    default=OUTPUT_PNG,
    help=f"Output plot PNG path. Default: {OUTPUT_PNG}",
)
args = parser.parse_args()

OUTPUT_PNG = args.output_plot

output_dir = os.path.dirname(OUTPUT_PNG)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# ── LULC class names ─────────────────────────────────────────────────────────
LULC_NAMES = {
    1: "Urban / Institutional Built-up",
    2: "Rural Settlement (Homestead Vegetation)",
    3: "Transport & Coastal Embankments",
    4: "Cropland (All Crop Intensities)",
    5: "Tree-based Agroforestry & Orchard",
    6: "Aquaculture & Inland Ponds",
    7: "Canals & Drainage Network",
    8: "Rivers & Estuarine Channels",
    9: "Mangrove Forest",
    10: "Bare / Exposed Coastal Land",
}

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)

required_cols = {
    "run_name",
    "split",
    "true_class_id",
    "pred_class_id",
    "count",
    "model",
    "model_family",
    "feature_set",
}

missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns in CSV: {sorted(missing_cols)}")

# ── Identify best model: highest accuracy on test split ──────────────────────
test_all = df[df["split"] == "test"].copy()

if test_all.empty:
    raise ValueError("No rows found for split == 'test'.")

diag_counts = (
    test_all[test_all["true_class_id"] == test_all["pred_class_id"]]
    .groupby("run_name")["count"]
    .sum()
)

total_counts = test_all.groupby("run_name")["count"].sum()

# Fill missing diagonal counts with 0 for models with no correct predictions
test_accuracy = diag_counts.reindex(total_counts.index, fill_value=0) / total_counts

best_run = test_accuracy.idxmax()
best_acc = test_accuracy.loc[best_run]

# Grab metadata for the best run
meta = df[df["run_name"] == best_run].iloc[0]
best_model = meta["model"]
best_model_family = meta["model_family"]
best_feature_set = meta["feature_set"]

print(f"Best run       : {best_run}")
print(f"Model          : {best_model}  |  Family: {best_model_family}")
print(f"Feature set    : {best_feature_set}")
print(f"Test accuracy  : {best_acc:.4f} ({best_acc * 100:.2f}%)")

# ── Extract test confusion matrix for that run ───────────────────────────────
test_data = df[(df["run_name"] == best_run) & (df["split"] == "test")].copy()

if test_data.empty:
    raise ValueError(f"No test data found for run '{best_run}'.")

# Use all classes present in either true or predicted class columns
classes = sorted(
    set(df["true_class_id"].dropna().astype(int).unique())
    | set(df["pred_class_id"].dropna().astype(int).unique())
)

n = len(classes)

cm_df = (
    test_data.pivot_table(
        index="true_class_id",
        columns="pred_class_id",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    .reindex(index=classes, columns=classes, fill_value=0)
)

cm = cm_df.values.astype(float)

row_sums = cm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
cm_norm = cm / row_sums

test_acc = cm.diagonal().sum() / cm.sum()
print(f"Recomputed test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

# ── Plot ─────────────────────────────────────────────────────────────────────
cmap = LinearSegmentedColormap.from_list("bw_bottle_green", ["#ffffff", "#006A4E"])

fig_width = max(12, n * 1.15)
fig_height = max(7, n * 0.75)

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Row-normalised recall", fontsize=11)
cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

# Axis tick labels: class IDs only
tick_labels = [str(c) for c in classes]

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(tick_labels, fontsize=9)

# Cell annotations
for i in range(n):
    for j in range(n):
        pct = cm_norm[i, j]
        raw = int(cm[i, j])

        color = "white" if pct > 0.55 else "black"
        weight = "bold" if i == j else "normal"

        ax.text(
            j,
            i,
            f"{pct:.1%}\n({raw:,})",
            ha="center",
            va="center",
            fontsize=7.5,
            color=color,
            fontweight=weight,
        )

ax.set_xlabel("Predicted class", fontsize=12, labelpad=8)
ax.set_ylabel("True class", fontsize=12, labelpad=8)

if args.add_title:
    ax.set_title(
        "Test Confusion Matrix — Best Model by Test Accuracy\n"
        f"Best Model: {best_model}, Best Feature Set: {best_feature_set}\n"
        f"Overall Test Accuracy: {best_acc:.2%}",
        fontsize=12,
        pad=14,
    )

plt.tight_layout()

fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUTPUT_PNG}")

plt.close(fig)
