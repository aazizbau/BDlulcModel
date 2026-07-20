"""
Make accuracy table and visualization for the BEST model on the test split.

Best model is identified by highest overall accuracy on the test split.

Metrics produced:
- Producer's Accuracy = Recall = TP / actual class total
- User's Accuracy     = Precision = TP / predicted class total
- F1-score            = harmonic mean of precision and recall
- Support             = number of true samples/pixels per class
- Overall Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted Precision
- Weighted Recall
- Weighted F1

Input:
    outputs/master_training_with_outputs/all_confusion_matrices_long.csv

Outputs:
    outputs/test_accuracy_bestmodel_plot.png
    outputs/test_accuracy_bestmodel_table.png
    outputs/test_accuracy_bestmodel_table.csv

Example Run:
    python scripts/analysis/make_table_and_visualize_accuary.py

Complete Example Run:
    python scripts/analysis/make_table_and_visualize_accuary.py \
        --add-title \
        --output-plot outputs/figures/test_accuracy_bestmodel_plot.png \
        --output-table outputs/figures/test_accuracy_bestmodel_table.png \
        --output-csv outputs/figures/test_accuracy_bestmodel_table.csv

Reproduction and AOI adaptation
-------------------------------
Workflow role: Derive quantitative summaries, accuracy assessments, or change statistics from prepared model outputs.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--add-title``, ``--output-plot``, ``--output-table``, ``--output-csv``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace input result paths with outputs generated for the new AOI, and retain the same class-ID definitions when comparing results.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
"""

import argparse
import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_CSV = "outputs/master_training_with_outputs/all_confusion_matrices_long.csv"

OUTPUT_PLOT = "outputs/figures/test_accuracy_bestmodel_plot.png"
OUTPUT_TABLE_PNG = "outputs/figures/test_accuracy_bestmodel_table.png"
OUTPUT_TABLE_CSV = "outputs/figures/test_accuracy_bestmodel_table.csv"


# ── Command-line arguments ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Make accuracy table and visualization for the best test model."
)
parser.add_argument(
    "--add-title",
    action="store_true",
    help="Show title and subtitle on top of the plot and table.",
)
parser.add_argument(
    "--output-plot",
    default=OUTPUT_PLOT,
    help=f"Output plot PNG path. Default: {OUTPUT_PLOT}",
)
parser.add_argument(
    "--output-table",
    default=OUTPUT_TABLE_PNG,
    help=f"Output table PNG path. Default: {OUTPUT_TABLE_PNG}",
)
parser.add_argument(
    "--output-csv",
    default=OUTPUT_TABLE_CSV,
    help=f"Output table CSV path. Default: {OUTPUT_TABLE_CSV}",
)
args = parser.parse_args()

OUTPUT_PLOT = args.output_plot
OUTPUT_TABLE_PNG = args.output_table
OUTPUT_TABLE_CSV = args.output_csv

for output_path in [OUTPUT_PLOT, OUTPUT_TABLE_PNG, OUTPUT_TABLE_CSV]:
    output_dir = os.path.dirname(output_path)
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


# ── Helper functions ─────────────────────────────────────────────────────────
def safe_divide(numerator, denominator):
    """Safely divide arrays/scalars and return 0 where denominator is 0."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def wrap_label(text, width=28):
    """Wrap long class names for cleaner plotting."""
    return "\n".join(textwrap.wrap(str(text), width=width))


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


# ── Identify best model by highest test overall accuracy ─────────────────────
test_all = df[df["split"] == "test"].copy()

if test_all.empty:
    raise ValueError("No rows found for split == 'test'.")

diag_counts = (
    test_all[test_all["true_class_id"] == test_all["pred_class_id"]]
    .groupby("run_name")["count"]
    .sum()
)

total_counts = test_all.groupby("run_name")["count"].sum()

test_accuracy = diag_counts.reindex(total_counts.index, fill_value=0) / total_counts

best_run = test_accuracy.idxmax()
best_acc = test_accuracy.loc[best_run]

meta = df[df["run_name"] == best_run].iloc[0]
best_model = meta["model"]
best_model_family = meta["model_family"]
best_feature_set = meta["feature_set"]

print("Best model selected by highest overall test accuracy")
print(f"Best run        : {best_run}")
print(f"Model           : {best_model}")
print(f"Model family    : {best_model_family}")
print(f"Feature set     : {best_feature_set}")
print(f"Overall accuracy: {best_acc:.4f} ({best_acc * 100:.2f}%)")


# ── Extract test confusion matrix for best run ───────────────────────────────
test_data = df[(df["run_name"] == best_run) & (df["split"] == "test")].copy()

if test_data.empty:
    raise ValueError(f"No test data found for best run: {best_run}")

classes = sorted(
    set(test_data["true_class_id"].dropna().astype(int).unique())
    | set(test_data["pred_class_id"].dropna().astype(int).unique())
)

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

tp = np.diag(cm)
actual_total = cm.sum(axis=1)
predicted_total = cm.sum(axis=0)
total_samples = cm.sum()

producer_accuracy = safe_divide(tp, actual_total)      # Recall
user_accuracy = safe_divide(tp, predicted_total)       # Precision
f1_score = safe_divide(
    2 * user_accuracy * producer_accuracy,
    user_accuracy + producer_accuracy,
)

overall_accuracy = safe_divide(tp.sum(), total_samples).item()

macro_user_accuracy = np.mean(user_accuracy)
macro_producer_accuracy = np.mean(producer_accuracy)
macro_f1 = np.mean(f1_score)

weighted_user_accuracy = safe_divide(
    np.sum(user_accuracy * actual_total),
    np.sum(actual_total),
).item()

weighted_producer_accuracy = safe_divide(
    np.sum(producer_accuracy * actual_total),
    np.sum(actual_total),
).item()

weighted_f1 = safe_divide(
    np.sum(f1_score * actual_total),
    np.sum(actual_total),
).item()


# ── Build output table ───────────────────────────────────────────────────────
table_df = pd.DataFrame(
    {
        "Class ID": classes,
        "Class Name": [LULC_NAMES.get(c, f"Class {c}") for c in classes],
        "Producer's Accuracy / Recall (%)": producer_accuracy * 100,
        "User's Accuracy / Precision (%)": user_accuracy * 100,
        "F1-score (%)": f1_score * 100,
        "Support / True Count": actual_total.astype(int),
        "Predicted Count": predicted_total.astype(int),
        "Correct Count": tp.astype(int),
    }
)

summary_rows = pd.DataFrame(
    [
        {
            "Class ID": "Overall",
            "Class Name": "Overall Accuracy",
            "Producer's Accuracy / Recall (%)": overall_accuracy * 100,
            "User's Accuracy / Precision (%)": overall_accuracy * 100,
            "F1-score (%)": overall_accuracy * 100,
            "Support / True Count": int(total_samples),
            "Predicted Count": int(total_samples),
            "Correct Count": int(tp.sum()),
        },
        {
            "Class ID": "Macro",
            "Class Name": "Macro Average",
            "Producer's Accuracy / Recall (%)": macro_producer_accuracy * 100,
            "User's Accuracy / Precision (%)": macro_user_accuracy * 100,
            "F1-score (%)": macro_f1 * 100,
            "Support / True Count": int(total_samples),
            "Predicted Count": int(total_samples),
            "Correct Count": int(tp.sum()),
        },
        {
            "Class ID": "Weighted",
            "Class Name": "Weighted Average",
            "Producer's Accuracy / Recall (%)": weighted_producer_accuracy * 100,
            "User's Accuracy / Precision (%)": weighted_user_accuracy * 100,
            "F1-score (%)": weighted_f1 * 100,
            "Support / True Count": int(total_samples),
            "Predicted Count": int(total_samples),
            "Correct Count": int(tp.sum()),
        },
    ]
)

final_table_df = pd.concat([table_df, summary_rows], ignore_index=True)

# Save CSV
final_table_df.to_csv(OUTPUT_TABLE_CSV, index=False)

print(f"\nSaved table CSV → {OUTPUT_TABLE_CSV}")


# ── Plot Producer's Accuracy, User's Accuracy, and F1-score ──────────────────
x = np.arange(len(classes))
bar_width = 0.25

fig, ax = plt.subplots(figsize=(15, 7))

producer_bars = ax.bar(
    x - bar_width,
    producer_accuracy * 100,
    width=bar_width,
    label="Producer's Accuracy / Recall",
)

user_bars = ax.bar(
    x,
    user_accuracy * 100,
    width=bar_width,
    label="User's Accuracy / Precision",
)

f1_bars = ax.bar(
    x + bar_width,
    f1_score * 100,
    width=bar_width,
    label="F1-score",
)

for bars in [producer_bars, user_bars, f1_bars]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.0,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            path_effects=[pe.Stroke(linewidth=2.2, foreground="0.90"), pe.Stroke(linewidth=0.8, foreground="0.15"), pe.Normal()],
        )

ax.axhline(
    overall_accuracy * 100,
    linestyle="--",
    linewidth=1.5,
    label=f"Overall Accuracy: {overall_accuracy * 100:.2f}%",
)

x_labels = [
    f"{c}\n{wrap_label(LULC_NAMES.get(c, f'Class {c}'), width=18)}"
    for c in classes
]

ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=8)

ax.set_ylabel("Accuracy / Score (%)", fontsize=12)
ax.set_xlabel("LULC Class", fontsize=12)

ax.set_ylim(0, 105)
ax.grid(axis="y", linestyle="--", alpha=0.35)

if args.add_title:
    ax.set_title(
        "Producer's Accuracy, User's Accuracy, and F1-score — Best Test Model\n"
        f"Best Model: {best_model}, Feature Set: {best_feature_set}, Overall Accuracy: {overall_accuracy * 100:.2f}%",
        fontsize=13,
        pad=14,
    )

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    frameon=True,
    fontsize=10,
)

plt.tight_layout()
fig.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved plot PNG  → {OUTPUT_PLOT}")


# ── Save table as PNG ────────────────────────────────────────────────────────
display_df = final_table_df.copy()

for col in [
    "Producer's Accuracy / Recall (%)",
    "User's Accuracy / Precision (%)",
    "F1-score (%)",
]:
    display_df[col] = display_df[col].map(lambda v: f"{v:.2f}")

for col in ["Support / True Count", "Predicted Count", "Correct Count"]:
    display_df[col] = display_df[col].map(lambda v: f"{int(v):,}")

# Shorter column names for the PNG table
display_df = display_df.rename(
    columns={
        "Class ID": "Class",
        "Class Name": "Name",
        "Producer's Accuracy / Recall (%)": "Producer's\nAccuracy (%)",
        "User's Accuracy / Precision (%)": "User's\nAccuracy (%)",
        "F1-score (%)": "F1\n(%)",
        "Support / True Count": "Support",
        "Predicted Count": "Predicted",
        "Correct Count": "Correct",
    }
)

fig_height = max(6, 0.45 * len(display_df) + 2)
fig, ax = plt.subplots(figsize=(16, fig_height))
ax.axis("off")

if args.add_title:
    title = (
        "Accuracy Assessment Table — Best Test Model\n"
        f"Best Model: {best_model} | Feature Set: {best_feature_set} | "
        f"Overall Accuracy: {overall_accuracy * 100:.2f}%"
    )
    ax.set_title(title, fontsize=14, pad=18)

table = ax.table(
    cellText=display_df.values,
    colLabels=display_df.columns,
    cellLoc="center",
    colLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.45)

# Adjust column widths
col_widths = {
    0: 0.07,  # Class
    1: 0.32,  # Name
    2: 0.13,  # Producer's Accuracy
    3: 0.13,  # User's Accuracy
    4: 0.09,  # F1
    5: 0.10,  # Support
    6: 0.10,  # Predicted
    7: 0.10,  # Correct
}

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("0.75")

    if col in col_widths:
        cell.set_width(col_widths[col])

    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#E6E6E6")
    elif row > len(classes):
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#F2F2F2")

    if col == 1 and row > 0:
        cell.set_text_props(ha="left")

plt.tight_layout()
fig.savefig(OUTPUT_TABLE_PNG, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved table PNG → {OUTPUT_TABLE_PNG}")

print("\nDone.")
