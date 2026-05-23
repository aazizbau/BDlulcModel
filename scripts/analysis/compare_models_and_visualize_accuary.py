"""
Compare best test accuracy results among model families.

For each model family, this script identifies the best run based on highest
overall accuracy on the test split. It then calculates accuracy assessment
metrics from the test confusion matrix and visualizes model-family comparison.

Compared model families:
    CNN1D, FTTransformer, LightGBM, MLP, ResMLP, XGBoost

Metrics produced:
    - Overall Accuracy
    - Macro Producer's Accuracy / Macro Recall
    - Macro User's Accuracy / Macro Precision
    - Macro F1-score
    - Weighted Producer's Accuracy / Weighted Recall
    - Weighted User's Accuracy / Weighted Precision
    - Weighted F1-score
    - Total support
    - Correct count

Input:
    outputs/master_training_with_outputs/all_confusion_matrices_long.csv

Outputs:
    outputs/figures/test_accuracy_comapre_models_plot.png
    outputs/figures/test_accuracy_compare_models_table.png
    outputs/figures/test_accuracy_compare_models_table.csv

Example Run:
    python scripts/analysis/compare_models_and_visualize_accuary.py

Complete Example Run:
    python scripts/analysis/compare_models_and_visualize_accuary.py \
        --add-title \
        --output-plot outputs/figures/test_accuracy_comapre_models_plot.png \
        --output-table outputs/figures/test_accuracy_compare_models_table.png \
        --output-csv outputs/figures/test_accuracy_compare_models_table.csv
"""

import argparse
import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_CSV = "outputs/master_training_with_outputs/all_confusion_matrices_long.csv"

OUTPUT_DIR = "outputs/figures"

OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "test_accuracy_comapre_models_plot.png")
OUTPUT_TABLE_PNG = os.path.join(OUTPUT_DIR, "test_accuracy_compare_models_table.png")
OUTPUT_TABLE_CSV = os.path.join(OUTPUT_DIR, "test_accuracy_compare_models_table.csv")


# ── Command-line arguments ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Compare best test accuracy results among model families."
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


# ── Expected model-family order ──────────────────────────────────────────────
MODEL_FAMILY_ORDER = [
    "CNN1D",
    "FTTransformer",
    "LightGBM",
    "MLP",
    "ResMLP",
    "XGBoost",
]


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


def clean_family_name(name):
    """
    Standardize possible model-family spelling variations.
    This helps if the CSV contains names such as cnn1d, CNN1D, lightgbm, etc.
    """
    if pd.isna(name):
        return name

    s = str(name).strip()

    mapping = {
        "cnn1d": "CNN1D",
        "cnn": "CNN1D",
        "fttransformer": "FTTransformer",
        "ft_transformer": "FTTransformer",
        "lightgbm": "LightGBM",
        "lgbm": "LightGBM",
        "mlp": "MLP",
        "resmlp": "ResMLP",
        "res_mlp": "ResMLP",
        "xgboost": "XGBoost",
        "xgb": "XGBoost",
    }

    key = s.lower().replace("-", "_").replace(" ", "_")

    return mapping.get(key, s)


def calculate_metrics_for_run(test_data):
    """
    Calculate accuracy metrics from one run's test confusion-matrix rows.
    """
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

    producer_accuracy = safe_divide(tp, actual_total)     # Recall
    user_accuracy = safe_divide(tp, predicted_total)      # Precision

    f1_score = safe_divide(
        2 * user_accuracy * producer_accuracy,
        user_accuracy + producer_accuracy,
    )

    overall_accuracy = safe_divide(tp.sum(), total_samples).item()

    macro_producer_accuracy = np.mean(producer_accuracy)
    macro_user_accuracy = np.mean(user_accuracy)
    macro_f1 = np.mean(f1_score)

    weighted_producer_accuracy = safe_divide(
        np.sum(producer_accuracy * actual_total),
        np.sum(actual_total),
    ).item()

    weighted_user_accuracy = safe_divide(
        np.sum(user_accuracy * actual_total),
        np.sum(actual_total),
    ).item()

    weighted_f1 = safe_divide(
        np.sum(f1_score * actual_total),
        np.sum(actual_total),
    ).item()

    return {
        "Overall Accuracy (%)": overall_accuracy * 100,
        "Macro Producer's Accuracy / Recall (%)": macro_producer_accuracy * 100,
        "Macro User's Accuracy / Precision (%)": macro_user_accuracy * 100,
        "Macro F1-score (%)": macro_f1 * 100,
        "Weighted Producer's Accuracy / Recall (%)": weighted_producer_accuracy * 100,
        "Weighted User's Accuracy / Precision (%)": weighted_user_accuracy * 100,
        "Weighted F1-score (%)": weighted_f1 * 100,
        "Total Support": int(total_samples),
        "Correct Count": int(tp.sum()),
        "Number of Classes": len(classes),
    }


def wrap_label(text, width=16):
    """Wrap long labels for x-axis."""
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


# ── Prepare test data ────────────────────────────────────────────────────────
test_all = df[df["split"] == "test"].copy()

if test_all.empty:
    raise ValueError("No rows found for split == 'test'.")

test_all["model_family_clean"] = test_all["model_family"].apply(clean_family_name)

available_families = sorted(test_all["model_family_clean"].dropna().unique())

print("Available model families in test split:")
for fam in available_families:
    print(f"  - {fam}")


# ── Identify best run within each model family ───────────────────────────────
results = []

for family in MODEL_FAMILY_ORDER:
    fam_data = test_all[test_all["model_family_clean"] == family].copy()

    if fam_data.empty:
        print(f"Warning: No test data found for model family: {family}")
        continue

    diag_counts = (
        fam_data[fam_data["true_class_id"] == fam_data["pred_class_id"]]
        .groupby("run_name")["count"]
        .sum()
    )

    total_counts = fam_data.groupby("run_name")["count"].sum()

    run_accuracy = diag_counts.reindex(total_counts.index, fill_value=0) / total_counts

    best_run = run_accuracy.idxmax()
    best_acc = run_accuracy.loc[best_run]

    best_run_data = fam_data[fam_data["run_name"] == best_run].copy()

    if best_run_data.empty:
        print(f"Warning: Could not extract data for best run: {best_run}")
        continue

    meta = best_run_data.iloc[0]

    metrics = calculate_metrics_for_run(best_run_data)

    results.append(
        {
            "Model Family": family,
            "Best Run": best_run,
            "Model": meta["model"],
            "Feature Set": meta["feature_set"],
            **metrics,
        }
    )

    print("\nBest run for model family")
    print(f"Model family    : {family}")
    print(f"Best run        : {best_run}")
    print(f"Model           : {meta['model']}")
    print(f"Feature set     : {meta['feature_set']}")
    print(f"Overall accuracy: {best_acc:.4f} ({best_acc * 100:.2f}%)")


if not results:
    raise ValueError("No model-family comparison results could be created.")


# ── Build comparison table ───────────────────────────────────────────────────
table_df = pd.DataFrame(results)

# Preserve requested family order
table_df["Model Family"] = pd.Categorical(
    table_df["Model Family"],
    categories=MODEL_FAMILY_ORDER,
    ordered=True,
)

table_df = table_df.sort_values("Model Family").reset_index(drop=True)

# Ranking by overall accuracy
table_df["Rank by Overall Accuracy"] = (
    table_df["Overall Accuracy (%)"]
    .rank(method="dense", ascending=False)
    .astype(int)
)

# Move rank near the front
front_cols = [
    "Rank by Overall Accuracy",
    "Model Family",
    "Model",
    "Feature Set",
    "Best Run",
]

metric_cols = [c for c in table_df.columns if c not in front_cols]
table_df = table_df[front_cols + metric_cols]

table_df.to_csv(OUTPUT_TABLE_CSV, index=False)
print(f"\nSaved table CSV → {OUTPUT_TABLE_CSV}")


# ── Plot model-family comparison ─────────────────────────────────────────────
x = np.arange(len(table_df))
bar_width = 0.22

overall = table_df["Overall Accuracy (%)"].values
macro_f1 = table_df["Macro F1-score (%)"].values
weighted_f1 = table_df["Weighted F1-score (%)"].values

fig, ax = plt.subplots(figsize=(13, 7))

ax.bar(
    x - bar_width,
    overall,
    width=bar_width,
    label="Overall Accuracy",
)

ax.bar(
    x,
    macro_f1,
    width=bar_width,
    label="Macro F1-score",
)

ax.bar(
    x + bar_width,
    weighted_f1,
    width=bar_width,
    label="Weighted F1-score",
)

# Add value labels above bars
for xpos, values in [
    (x - bar_width, overall),
    (x, macro_f1),
    (x + bar_width, weighted_f1),
]:
    for xi, yi in zip(xpos, values):
        ax.text(
            xi,
            yi + 1.0,
            f"{yi:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

x_labels = [wrap_label(fam, width=14) for fam in table_df["Model Family"]]

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=10)

ax.set_ylabel("Accuracy / Score (%)", fontsize=12)
ax.set_xlabel("Model Family", fontsize=12)

ax.set_ylim(0, 110)
ax.grid(axis="y", linestyle="--", alpha=0.35)

best_idx = table_df["Overall Accuracy (%)"].idxmax()
best_family = table_df.loc[best_idx, "Model Family"]
best_overall = table_df.loc[best_idx, "Overall Accuracy (%)"]
best_feature_set = table_df.loc[best_idx, "Feature Set"]

if args.add_title:
    ax.set_title(
        "Comparison of Best Test Models by Model Family\n"
        f"Best Overall: {best_family} | Feature Set: {best_feature_set} | "
        f"Overall Accuracy: {best_overall:.2f}%",
        fontsize=13,
        pad=14,
    )

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=3,
    frameon=True,
    fontsize=10,
)

plt.tight_layout()
fig.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved plot PNG  → {OUTPUT_PLOT}")


# ── Save table as PNG ────────────────────────────────────────────────────────
display_df = table_df.copy()

percent_cols = [
    "Overall Accuracy (%)",
    "Macro Producer's Accuracy / Recall (%)",
    "Macro User's Accuracy / Precision (%)",
    "Macro F1-score (%)",
    "Weighted Producer's Accuracy / Recall (%)",
    "Weighted User's Accuracy / Precision (%)",
    "Weighted F1-score (%)",
]

for col in percent_cols:
    display_df[col] = display_df[col].map(lambda v: f"{v:.2f}")

for col in ["Total Support", "Correct Count"]:
    display_df[col] = display_df[col].map(lambda v: f"{int(v):,}")

# Shorter column names for PNG table
display_df = display_df.rename(
    columns={
        "Rank by Overall Accuracy": "Rank",
        "Model Family": "Family",
        "Feature Set": "Feature\nSet",
        "Best Run": "Best\nRun",
        "Overall Accuracy (%)": "Overall\nAcc. (%)",
        "Macro Producer's Accuracy / Recall (%)": "Macro\nProducer (%)",
        "Macro User's Accuracy / Precision (%)": "Macro\nUser (%)",
        "Macro F1-score (%)": "Macro\nF1 (%)",
        "Weighted Producer's Accuracy / Recall (%)": "Weighted\nProducer (%)",
        "Weighted User's Accuracy / Precision (%)": "Weighted\nUser (%)",
        "Weighted F1-score (%)": "Weighted\nF1 (%)",
        "Total Support": "Support",
        "Correct Count": "Correct",
        "Number of Classes": "Classes",
    }
)

# Keep table readable by wrapping long run names
if "Best\nRun" in display_df.columns:
    display_df["Best\nRun"] = display_df["Best\nRun"].map(
        lambda s: "\n".join(textwrap.wrap(str(s), width=28))
    )

if "Feature\nSet" in display_df.columns:
    display_df["Feature\nSet"] = display_df["Feature\nSet"].map(
        lambda s: "\n".join(textwrap.wrap(str(s), width=18))
    )

fig_height = max(4.5, 0.65 * len(display_df) + 2.5)
fig, ax = plt.subplots(figsize=(20, fig_height))
ax.axis("off")

if args.add_title:
    title = (
        "Accuracy Assessment Comparison Table — Best Test Run per Model Family\n"
        f"Best Overall: {best_family} | Overall Accuracy: {best_overall:.2f}%"
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
table.set_fontsize(7.8)
table.scale(1, 1.8)

# Column width tuning
col_widths = {
    0: 0.045,  # Rank
    1: 0.075,  # Family
    2: 0.075,  # Model
    3: 0.085,  # Feature Set
    4: 0.170,  # Best Run
    5: 0.075,  # Overall Accuracy
    6: 0.075,  # Macro Producer
    7: 0.075,  # Macro User
    8: 0.075,  # Macro F1
    9: 0.080,  # Weighted Producer
    10: 0.080,  # Weighted User
    11: 0.080,  # Weighted F1
    12: 0.070,  # Support
    13: 0.070,  # Correct
    14: 0.055,  # Classes
}

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("0.75")

    if col in col_widths:
        cell.set_width(col_widths[col])

    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#E6E6E6")
    else:
        rank_value = display_df.iloc[row - 1]["Rank"]
        if str(rank_value) == "1":
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(weight="bold")

plt.tight_layout()
fig.savefig(OUTPUT_TABLE_PNG, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved table PNG → {OUTPUT_TABLE_PNG}")

print("\nDone.")
