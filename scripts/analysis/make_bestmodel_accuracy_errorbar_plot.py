#!/usr/bin/env python3
"""
Make an error-bar accuracy plot for the best test model.

The best model is identified by highest overall accuracy on the test split.
For that model, the script plots class-wise Producer's Accuracy, User's
Accuracy, and F1-score with 95% bootstrap confidence intervals estimated from
the test confusion matrix.

Input:
    outputs/master_training_with_outputs/all_confusion_matrices_long.csv

Outputs:
    outputs/figures/test_accuracy_bestmodel_errorbar_plot.png
    outputs/figures/test_accuracy_bestmodel_errorbar_table.csv

Complete Example Run:
    python scripts/analysis/make_bestmodel_accuracy_errorbar_plot.py \
        --add-title \
        --output-plot outputs/figures/test_accuracy_bestmodel_errorbar_plot.png \
        --output-csv outputs/figures/test_accuracy_bestmodel_errorbar_table.csv \
        --bootstrap 1000 \
        --seed 42

Reproduction and AOI adaptation
-------------------------------
Workflow role: Derive quantitative summaries, accuracy assessments, or change statistics from prepared model outputs.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--add-title``, ``--output-plot``, ``--output-csv``, ``--bootstrap``, ``--seed``. Run the ``--help`` command below for required values, defaults, and accepted choices.
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

from __future__ import annotations

import argparse
import os
import textwrap

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = "outputs/master_training_with_outputs/all_confusion_matrices_long.csv"
OUTPUT_PLOT = "outputs/figures/test_accuracy_bestmodel_errorbar_plot.png"
OUTPUT_CSV = "outputs/figures/test_accuracy_bestmodel_errorbar_table.csv"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make best-model class-wise accuracy plot with bootstrap error bars."
    )
    parser.add_argument(
        "--add-title",
        action="store_true",
        help="Show title and subtitle on top of the plot.",
    )
    parser.add_argument(
        "--output-plot",
        default=OUTPUT_PLOT,
        help=f"Output plot PNG path. Default: {OUTPUT_PLOT}",
    )
    parser.add_argument(
        "--output-csv",
        default=OUTPUT_CSV,
        help=f"Output CSV path. Default: {OUTPUT_CSV}",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap replicates for confidence intervals (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic bootstrap sampling (default: 42).",
    )
    return parser.parse_args()


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


def calculate_metrics(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm)
    actual_total = cm.sum(axis=1)
    predicted_total = cm.sum(axis=0)

    producer_accuracy = safe_divide(tp, actual_total)
    user_accuracy = safe_divide(tp, predicted_total)
    f1_score = safe_divide(
        2 * user_accuracy * producer_accuracy,
        user_accuracy + producer_accuracy,
    )

    return producer_accuracy, user_accuracy, f1_score


def bootstrap_confidence_intervals(
    cm: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if n_bootstrap <= 0:
        raise ValueError("--bootstrap must be greater than 0.")

    rng = np.random.default_rng(seed)
    n_classes = cm.shape[0]

    producer_samples = np.zeros((n_bootstrap, n_classes), dtype=float)
    user_samples = np.zeros((n_bootstrap, n_classes), dtype=float)
    f1_samples = np.zeros((n_bootstrap, n_classes), dtype=float)

    row_totals = cm.sum(axis=1).astype(int)
    row_probs = np.zeros_like(cm, dtype=float)
    for row_idx, row_total in enumerate(row_totals):
        if row_total > 0:
            row_probs[row_idx] = cm[row_idx] / row_total

    for boot_idx in range(n_bootstrap):
        boot_cm = np.zeros_like(cm, dtype=float)
        for row_idx, row_total in enumerate(row_totals):
            if row_total > 0:
                boot_cm[row_idx] = rng.multinomial(row_total, row_probs[row_idx])

        producer, user, f1 = calculate_metrics(boot_cm)
        producer_samples[boot_idx] = producer * 100
        user_samples[boot_idx] = user * 100
        f1_samples[boot_idx] = f1 * 100

    return {
        "producer": (
            np.percentile(producer_samples, 2.5, axis=0),
            np.percentile(producer_samples, 97.5, axis=0),
        ),
        "user": (
            np.percentile(user_samples, 2.5, axis=0),
            np.percentile(user_samples, 97.5, axis=0),
        ),
        "f1": (
            np.percentile(f1_samples, 2.5, axis=0),
            np.percentile(f1_samples, 97.5, axis=0),
        ),
    }


def add_bar_and_ci_labels(
    ax,
    bars,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    text_effects = [
        pe.Stroke(linewidth=2.4, foreground="white"),
        pe.Stroke(linewidth=0.7, foreground="0.15"),
        pe.Normal(),
    ]

    for idx, bar in enumerate(bars):
        x = bar.get_x() + bar.get_width() / 2
        value = float(values[idx])
        hi = float(upper[idx])

        ax.text(
            x,
            min(hi + 1.4, ax.get_ylim()[1] - 0.2),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.5,
            path_effects=text_effects,
            zorder=6,
        )


def main() -> None:
    args = parse_args()

    for output_path in [args.output_plot, args.output_csv]:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

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

    producer_accuracy, user_accuracy, f1_score = calculate_metrics(cm)
    producer_pct = producer_accuracy * 100
    user_pct = user_accuracy * 100
    f1_pct = f1_score * 100

    tp = np.diag(cm)
    actual_total = cm.sum(axis=1)
    predicted_total = cm.sum(axis=0)
    total_samples = cm.sum()
    overall_accuracy = safe_divide(tp.sum(), total_samples).item()

    print(f"Bootstrap replicates: {args.bootstrap}")
    ci = bootstrap_confidence_intervals(cm, args.bootstrap, args.seed)

    producer_lower, producer_upper = ci["producer"]
    user_lower, user_upper = ci["user"]
    f1_lower, f1_upper = ci["f1"]

    table_df = pd.DataFrame(
        {
            "Class ID": classes,
            "Class Name": [LULC_NAMES.get(c, f"Class {c}") for c in classes],
            "Producer's Accuracy / Recall (%)": producer_pct,
            "Producer's Accuracy / Recall Lower 95% (%)": producer_lower,
            "Producer's Accuracy / Recall Upper 95% (%)": producer_upper,
            "User's Accuracy / Precision (%)": user_pct,
            "User's Accuracy / Precision Lower 95% (%)": user_lower,
            "User's Accuracy / Precision Upper 95% (%)": user_upper,
            "F1-score (%)": f1_pct,
            "F1-score Lower 95% (%)": f1_lower,
            "F1-score Upper 95% (%)": f1_upper,
            "Support / True Count": actual_total.astype(int),
            "Predicted Count": predicted_total.astype(int),
            "Correct Count": tp.astype(int),
        }
    )
    table_df.to_csv(args.output_csv, index=False)
    print(f"Saved table CSV → {args.output_csv}")

    x = np.arange(len(classes))
    bar_width = 0.25

    max_upper = max(float(producer_upper.max()), float(user_upper.max()), float(f1_upper.max()))
    y_max = max(105.0, min(115.0, max_upper + 8.0))

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_ylim(0, y_max)

    producer_yerr = np.vstack([producer_pct - producer_lower, producer_upper - producer_pct])
    user_yerr = np.vstack([user_pct - user_lower, user_upper - user_pct])
    f1_yerr = np.vstack([f1_pct - f1_lower, f1_upper - f1_pct])

    producer_bars = ax.bar(
        x - bar_width,
        producer_pct,
        width=bar_width,
        yerr=producer_yerr,
        capsize=3,
        ecolor="black",
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        label="Producer's Accuracy / Recall",
    )
    user_bars = ax.bar(
        x,
        user_pct,
        width=bar_width,
        yerr=user_yerr,
        capsize=3,
        ecolor="black",
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        label="User's Accuracy / Precision",
    )
    f1_bars = ax.bar(
        x + bar_width,
        f1_pct,
        width=bar_width,
        yerr=f1_yerr,
        capsize=3,
        ecolor="black",
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        label="F1-score",
    )

    add_bar_and_ci_labels(ax, producer_bars, producer_pct, producer_lower, producer_upper)
    add_bar_and_ci_labels(ax, user_bars, user_pct, user_lower, user_upper)
    add_bar_and_ci_labels(ax, f1_bars, f1_pct, f1_lower, f1_upper)

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
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    if args.add_title:
        ax.set_title(
            "Producer's Accuracy, User's Accuracy, and F1-score with 95% CI — Best Test Model\n"
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
    fig.savefig(args.output_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot PNG  → {args.output_plot}")
    print("Done.")


if __name__ == "__main__":
    main()
