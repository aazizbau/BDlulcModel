"""
Compare feature sets (AE64 vs AE64_plus10indices) across model families
based on test overall accuracy.

For each (model_family, feature_set) combination this script identifies the
best run based on highest overall accuracy on the test split, calculates
accuracy-assessment metrics, and visualises the feature-set comparison.

Compared model families:
    CNN1D, FTTransformer, LightGBM, MLP, ResMLP, XGBoost

Compared feature sets:
    AE64, AE64_plus10indices

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
    outputs/figures/test_accuracy_compare_featureset_plot.png
    outputs/figures/test_accuracy_compare_featureset_table.png
    outputs/figures/test_accuracy_compare_featureset_table.csv

Example Run:
    python scripts/analysis/compare_models_featureset_and_visualize_accuary.py
"""

import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_CSV = "outputs/master_training_with_outputs/all_confusion_matrices_long.csv"
OUTPUT_DIR = "outputs/figures"
OUTPUT_PLOT     = os.path.join(OUTPUT_DIR, "test_accuracy_compare_featureset_plot.png")
OUTPUT_TABLE_PNG = os.path.join(OUTPUT_DIR, "test_accuracy_compare_featureset_table.png")
OUTPUT_TABLE_CSV = os.path.join(OUTPUT_DIR, "test_accuracy_compare_featureset_table.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Expected orders ────────────────────────────────────────────────────────────
MODEL_FAMILY_ORDER = [
    "CNN1D",
    "FTTransformer",
    "LightGBM",
    "MLP",
    "ResMLP",
    "XGBoost",
]

FEATURE_SETS = ["AE64", "AE64_plus10indices"]

FEATURE_SET_COLORS = {
    "AE64":               "#4C72B0",
    "AE64_plus10indices": "#DD8452",
}

FEATURE_SET_DISPLAY = {
    "AE64":               "AE64",
    "AE64_plus10indices": "AE64 + 10 Indices",
}


# ── Helper functions ──────────────────────────────────────────────────────────
def safe_divide(numerator, denominator):
    numerator   = np.asarray(numerator,   dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator, denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def clean_family_name(name):
    if pd.isna(name):
        return name
    s = str(name).strip()
    mapping = {
        "cnn1d":        "CNN1D",
        "cnn":          "CNN1D",
        "fttransformer":"FTTransformer",
        "ft_transformer":"FTTransformer",
        "lightgbm":     "LightGBM",
        "lgbm":         "LightGBM",
        "mlp":          "MLP",
        "resmlp":       "ResMLP",
        "res_mlp":      "ResMLP",
        "xgboost":      "XGBoost",
        "xgb":          "XGBoost",
    }
    key = s.lower().replace("-", "_").replace(" ", "_")
    return mapping.get(key, s)


def calculate_metrics_for_run(test_data):
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

    tp              = np.diag(cm)
    actual_total    = cm.sum(axis=1)
    predicted_total = cm.sum(axis=0)
    total_samples   = cm.sum()

    producer_accuracy = safe_divide(tp, actual_total)
    user_accuracy     = safe_divide(tp, predicted_total)
    f1_score          = safe_divide(
        2 * user_accuracy * producer_accuracy,
        user_accuracy + producer_accuracy,
    )
    overall_accuracy          = safe_divide(tp.sum(), total_samples).item()
    macro_producer_accuracy   = np.mean(producer_accuracy)
    macro_user_accuracy       = np.mean(user_accuracy)
    macro_f1                  = np.mean(f1_score)
    weighted_producer_accuracy = safe_divide(
        np.sum(producer_accuracy * actual_total), np.sum(actual_total)
    ).item()
    weighted_user_accuracy    = safe_divide(
        np.sum(user_accuracy * actual_total), np.sum(actual_total)
    ).item()
    weighted_f1               = safe_divide(
        np.sum(f1_score * actual_total), np.sum(actual_total)
    ).item()

    return {
        "Overall Accuracy (%)":                         overall_accuracy          * 100,
        "Macro Producer's Accuracy / Recall (%)":       macro_producer_accuracy   * 100,
        "Macro User's Accuracy / Precision (%)":        macro_user_accuracy       * 100,
        "Macro F1-score (%)":                           macro_f1                  * 100,
        "Weighted Producer's Accuracy / Recall (%)":    weighted_producer_accuracy * 100,
        "Weighted User's Accuracy / Precision (%)":     weighted_user_accuracy    * 100,
        "Weighted F1-score (%)":                        weighted_f1               * 100,
        "Total Support":                                int(total_samples),
        "Correct Count":                                int(tp.sum()),
        "Number of Classes":                            len(classes),
    }


def wrap_label(text, width=16):
    return "\n".join(textwrap.wrap(str(text), width=width))


# ── Load & validate data ──────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)

required_cols = {
    "run_name", "split", "true_class_id", "pred_class_id",
    "count", "model", "model_family", "feature_set",
}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns in CSV: {sorted(missing_cols)}")


# ── Prepare test data ─────────────────────────────────────────────────────────
test_all = df[df["split"] == "test"].copy()
if test_all.empty:
    raise ValueError("No rows found for split == 'test'.")

test_all["model_family_clean"] = test_all["model_family"].apply(clean_family_name)

print("Available model families in test split:")
for fam in sorted(test_all["model_family_clean"].dropna().unique()):
    print(f"  - {fam}")

print("\nAvailable feature sets in test split:")
for fs in sorted(test_all["feature_set"].dropna().unique()):
    print(f"  - {fs}")


# ── Identify best run per (model family, feature set) ─────────────────────────
results = []

for family in MODEL_FAMILY_ORDER:
    fam_data = test_all[test_all["model_family_clean"] == family].copy()
    if fam_data.empty:
        print(f"Warning: No test data found for model family: {family}")
        continue

    for feature_set in FEATURE_SETS:
        fs_data = fam_data[fam_data["feature_set"] == feature_set].copy()
        if fs_data.empty:
            print(f"Warning: No test data found for {family} / {feature_set}")
            continue

        diag_counts  = (
            fs_data[fs_data["true_class_id"] == fs_data["pred_class_id"]]
            .groupby("run_name")["count"]
            .sum()
        )
        total_counts = fs_data.groupby("run_name")["count"].sum()
        run_accuracy = (
            diag_counts.reindex(total_counts.index, fill_value=0) / total_counts
        )

        best_run      = run_accuracy.idxmax()
        best_acc      = run_accuracy.loc[best_run]
        best_run_data = fs_data[fs_data["run_name"] == best_run].copy()

        if best_run_data.empty:
            print(f"Warning: Could not extract data for best run: {best_run}")
            continue

        meta    = best_run_data.iloc[0]
        metrics = calculate_metrics_for_run(best_run_data)

        results.append({
            "Model Family": family,
            "Feature Set":  feature_set,
            "Best Run":     best_run,
            "Model":        meta["model"],
            **metrics,
        })

        print(f"\nBest run — {family} / {feature_set}")
        print(f"  Best run        : {best_run}")
        print(f"  Model           : {meta['model']}")
        print(f"  Overall accuracy: {best_acc:.4f} ({best_acc * 100:.2f}%)")

if not results:
    raise ValueError("No (model-family, feature-set) comparison results could be created.")


# ── Build comparison table ────────────────────────────────────────────────────
table_df = pd.DataFrame(results)

table_df["Model Family"] = pd.Categorical(
    table_df["Model Family"], categories=MODEL_FAMILY_ORDER, ordered=True
)
table_df["Feature Set"] = pd.Categorical(
    table_df["Feature Set"], categories=FEATURE_SETS, ordered=True
)
table_df = table_df.sort_values(["Model Family", "Feature Set"]).reset_index(drop=True)

# Delta Overall Accuracy: AE64_plus10indices − AE64 per family
oa_pivot = table_df.pivot_table(
    index="Model Family", columns="Feature Set", values="Overall Accuracy (%)"
)
delta_map = {}
if "AE64" in oa_pivot.columns and "AE64_plus10indices" in oa_pivot.columns:
    delta_map = (oa_pivot["AE64_plus10indices"] - oa_pivot["AE64"]).to_dict()

table_df["Δ Overall Accuracy (%)"] = table_df.apply(
    lambda r: delta_map.get(r["Model Family"], float("nan"))
    if r["Feature Set"] == "AE64_plus10indices"
    else float("nan"),
    axis=1,
)

# Rank within each feature set by overall accuracy
table_df["Rank (within Feature Set)"] = (
    table_df.groupby("Feature Set")["Overall Accuracy (%)"]
    .rank(method="dense", ascending=False)
    .astype(int)
)

front_cols = [
    "Rank (within Feature Set)",
    "Model Family",
    "Feature Set",
    "Model",
    "Best Run",
]
metric_cols = [c for c in table_df.columns if c not in front_cols]
table_df = table_df[front_cols + metric_cols]

table_df.to_csv(OUTPUT_TABLE_CSV, index=False)
print(f"\nSaved table CSV → {OUTPUT_TABLE_CSV}")


# ── Grouped bar plot (Overall Accuracy per feature set per model family) ───────
families_present = [f for f in MODEL_FAMILY_ORDER
                    if f in table_df["Model Family"].values]

x         = np.arange(len(families_present))
bar_width = 0.35
n_fs      = len(FEATURE_SETS)
offsets   = np.linspace(-(n_fs - 1) * bar_width / 2,
                         (n_fs - 1) * bar_width / 2, n_fs)

fig, ax = plt.subplots(figsize=(13, 7))

for offset, feature_set in zip(offsets, FEATURE_SETS):
    fs_rows = table_df[table_df["Feature Set"] == feature_set].set_index("Model Family")
    oa_vals = [
        fs_rows.loc[fam, "Overall Accuracy (%)"] if fam in fs_rows.index else 0.0
        for fam in families_present
    ]
    ax.bar(
        x + offset,
        oa_vals,
        width=bar_width,
        label=FEATURE_SET_DISPLAY[feature_set],
        color=FEATURE_SET_COLORS[feature_set],
    )
    for xi, yi in zip(x + offset, oa_vals):
        ax.text(xi, yi + 0.6, f"{yi:.1f}", ha="center", va="bottom",
                fontsize=8, rotation=0)

# Annotate delta above each family group
for xi, fam in zip(x, families_present):
    delta = delta_map.get(fam, None)
    if delta is None:
        continue
    sign  = "+" if delta >= 0 else ""
    color = "#2ca02c" if delta >= 0 else "#d62728"
    ax.text(
        xi,
        max(
            table_df.loc[table_df["Model Family"] == fam, "Overall Accuracy (%)"].max(),
            0.0,
        ) + 3.5,
        f"Δ {sign}{delta:.1f}",
        ha="center",
        va="bottom",
        fontsize=8,
        color=color,
        fontweight="bold",
    )

x_labels = [wrap_label(fam, width=14) for fam in families_present]
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_ylabel("Overall Accuracy (%)", fontsize=12)
ax.set_xlabel("Model Family", fontsize=12)
ax.set_ylim(0, 115)
ax.grid(axis="y", linestyle="--", alpha=0.35)

best_row    = table_df.loc[table_df["Overall Accuracy (%)"].idxmax()]
best_family = best_row["Model Family"]
best_fs     = best_row["Feature Set"]
best_oa     = best_row["Overall Accuracy (%)"]

ax.set_title(
    "Feature Set Comparison: AE64 vs AE64 + 10 Indices — Best Test Run per Model Family\n"
    f"Best Overall: {best_family} | Feature Set: {FEATURE_SET_DISPLAY[best_fs]} | "
    f"Overall Accuracy: {best_oa:.2f}%  (Δ = AE64 + 10 Indices − AE64)",
    fontsize=12,
    pad=14,
)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
          frameon=True, fontsize=10)

plt.tight_layout()
fig.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved plot PNG  → {OUTPUT_PLOT}")


# ── Save table as PNG ─────────────────────────────────────────────────────────
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

display_df["Δ Overall Accuracy (%)"] = display_df["Δ Overall Accuracy (%)"].map(
    lambda v: f"{v:+.2f}" if pd.notna(v) else "—"
)
for col in ["Total Support", "Correct Count"]:
    display_df[col] = display_df[col].map(lambda v: f"{int(v):,}")

display_df = display_df.rename(
    columns={
        "Rank (within Feature Set)":                    "Rank",
        "Model Family":                                 "Family",
        "Feature Set":                                  "Feature\nSet",
        "Best Run":                                     "Best\nRun",
        "Overall Accuracy (%)":                         "Overall\nAcc. (%)",
        "Δ Overall Accuracy (%)":                       "Δ OA\n(%)",
        "Macro Producer's Accuracy / Recall (%)":       "Macro\nProducer (%)",
        "Macro User's Accuracy / Precision (%)":        "Macro\nUser (%)",
        "Macro F1-score (%)":                           "Macro\nF1 (%)",
        "Weighted Producer's Accuracy / Recall (%)":    "Weighted\nProducer (%)",
        "Weighted User's Accuracy / Precision (%)":     "Weighted\nUser (%)",
        "Weighted F1-score (%)":                        "Weighted\nF1 (%)",
        "Total Support":                                "Support",
        "Correct Count":                                "Correct",
        "Number of Classes":                            "Classes",
    }
)

if "Best\nRun" in display_df.columns:
    display_df["Best\nRun"] = display_df["Best\nRun"].map(
        lambda s: "\n".join(textwrap.wrap(str(s), width=26))
    )
if "Feature\nSet" in display_df.columns:
    display_df["Feature\nSet"] = display_df["Feature\nSet"].map(
        lambda s: "\n".join(textwrap.wrap(str(s), width=16))
    )

fig_height = max(5.0, 0.72 * len(display_df) + 2.5)
fig, ax = plt.subplots(figsize=(22, fig_height))
ax.axis("off")

title = (
    "Accuracy Assessment Comparison — Feature Set AE64 vs AE64 + 10 Indices "
    "(Best Test Run per Model Family × Feature Set)\n"
    f"Best Overall: {best_family} | Feature Set: {FEATURE_SET_DISPLAY[best_fs]} | "
    f"Overall Accuracy: {best_oa:.2f}%"
)
ax.set_title(title, fontsize=13, pad=18)

table = ax.table(
    cellText=display_df.values,
    colLabels=display_df.columns,
    cellLoc="center",
    colLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(7.5)
table.scale(1, 1.85)

col_widths = {
    0:  0.040,   # Rank
    1:  0.072,   # Family
    2:  0.080,   # Feature Set
    3:  0.070,   # Model
    4:  0.160,   # Best Run
    5:  0.072,   # Overall Acc
    6:  0.055,   # Δ OA
    7:  0.072,   # Macro Producer
    8:  0.072,   # Macro User
    9:  0.072,   # Macro F1
    10: 0.078,   # Weighted Producer
    11: 0.078,   # Weighted User
    12: 0.078,   # Weighted F1
    13: 0.065,   # Support
    14: 0.065,   # Correct
    15: 0.050,   # Classes
}

# Row-background: alternate shading per model-family block, highlight Δ > 0
family_col_idx = list(display_df.columns).index("Family")
delta_col_idx  = list(display_df.columns).index("Δ OA\n(%)")

family_colors  = ["#FFFFFF", "#EEF4FB"]  # alternate per family
prev_family    = None
family_toggle  = 0

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("0.75")
    if col in col_widths:
        cell.set_width(col_widths[col])

    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#D0D9E8")
        continue

    family_val = display_df.iloc[row - 1, family_col_idx]
    if family_val != prev_family:
        family_toggle = 1 - family_toggle
        prev_family   = family_val
    base_color = family_colors[family_toggle]
    cell.set_facecolor(base_color)

    # Highlight Δ OA column: green if positive, red if negative
    if col == delta_col_idx:
        delta_text = display_df.iloc[row - 1, delta_col_idx]
        if delta_text != "—":
            try:
                val = float(delta_text.replace("+", ""))
                if val > 0:
                    cell.set_facecolor("#D4EDDA")
                    cell.set_text_props(color="#155724", weight="bold")
                elif val < 0:
                    cell.set_facecolor("#F8D7DA")
                    cell.set_text_props(color="#721C24", weight="bold")
            except ValueError:
                pass

    # Bold the rank-1 row within each feature set
    rank_col_idx = list(display_df.columns).index("Rank")
    if str(display_df.iloc[row - 1, rank_col_idx]) == "1":
        cell.set_text_props(weight="bold")

plt.tight_layout()
fig.savefig(OUTPUT_TABLE_PNG, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved table PNG → {OUTPUT_TABLE_PNG}")

print("\nDone.")
