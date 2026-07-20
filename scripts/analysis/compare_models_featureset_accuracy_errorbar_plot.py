#!/usr/bin/env python3
"""
Compare feature sets across model families with error bars.

For each (model_family, feature_set) combination this script identifies the
best run based on highest overall accuracy on the test split, calculates
accuracy-assessment metrics, and visualizes the feature-set comparison with
95% bootstrap confidence intervals.

Compared model families:
    CNN1D, FTTransformer, LightGBM, MLP, ResMLP, XGBoost

Compared feature sets:
    AE64, AE64_plus10indices

Input:
    outputs/master_training_with_outputs/all_confusion_matrices_long.csv

Outputs:
    outputs/figures/test_accuracy_compare_featureset_errorbar_plot.png
    outputs/figures/test_accuracy_compare_featureset_errorbar_table.png
    outputs/figures/test_accuracy_compare_featureset_errorbar_table.csv

Complete Example Run:
    python scripts/analysis/compare_models_featureset_accuracy_errorbar_plot.py \
        --add-title \
        --output-plot outputs/figures/test_accuracy_compare_featureset_errorbar_plot.png \
        --output-table outputs/figures/test_accuracy_compare_featureset_errorbar_table.png \
        --output-csv outputs/figures/test_accuracy_compare_featureset_errorbar_table.csv \
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
The command-line interface exposes ``--add-title``, ``--output-plot``, ``--output-table``, ``--output-csv``, ``--bootstrap``, ``--seed``. Run the ``--help`` command below for required values, defaults, and accepted choices.
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
OUTPUT_DIR = "outputs/figures"
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "test_accuracy_compare_featureset_errorbar_plot.png")
OUTPUT_TABLE_PNG = os.path.join(OUTPUT_DIR, "test_accuracy_compare_featureset_errorbar_table.png")
OUTPUT_TABLE_CSV = os.path.join(OUTPUT_DIR, "test_accuracy_compare_featureset_errorbar_table.csv")

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
    "AE64": "#4C72B0",
    "AE64_plus10indices": "#DD8452",
}

FEATURE_SET_DISPLAY = {
    "AE64": "AE64",
    "AE64_plus10indices": "AE64 + 10 Indices",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare feature sets across model families with bootstrap error bars."
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
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def clean_family_name(name):
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


def clean_feature_set_name(name):
    if pd.isna(name):
        return name
    s = str(name).strip()
    mapping = {
        "ae64": "AE64",
        "ae64_plus10indices": "AE64_plus10indices",
    }
    return mapping.get(s.lower(), s)


def wrap_label(text, width=16):
    return "\n".join(textwrap.wrap(str(text), width=width))


def confusion_matrix_for_run(test_data: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
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
    return cm_df.values.astype(float), classes


def calculate_metrics_from_cm(cm: np.ndarray) -> dict[str, float]:
    tp = np.diag(cm)
    actual_total = cm.sum(axis=1)
    predicted_total = cm.sum(axis=0)
    total_samples = cm.sum()

    producer_accuracy = safe_divide(tp, actual_total)
    user_accuracy = safe_divide(tp, predicted_total)
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
        "Number of Classes": int(cm.shape[0]),
    }


def bootstrap_confidence_intervals(
    cm: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    if n_bootstrap <= 0:
        raise ValueError("--bootstrap must be greater than 0.")

    rng = np.random.default_rng(seed)
    row_totals = cm.sum(axis=1).astype(int)
    row_probs = np.zeros_like(cm, dtype=float)

    for row_idx, row_total in enumerate(row_totals):
        if row_total > 0:
            row_probs[row_idx] = cm[row_idx] / row_total

    sampled = {
        "Overall Accuracy (%)": np.zeros(n_bootstrap, dtype=float),
        "Macro F1-score (%)": np.zeros(n_bootstrap, dtype=float),
        "Weighted F1-score (%)": np.zeros(n_bootstrap, dtype=float),
    }

    for boot_idx in range(n_bootstrap):
        boot_cm = np.zeros_like(cm, dtype=float)
        for row_idx, row_total in enumerate(row_totals):
            if row_total > 0:
                boot_cm[row_idx] = rng.multinomial(row_total, row_probs[row_idx])
        metrics = calculate_metrics_from_cm(boot_cm)
        for metric_name in sampled:
            sampled[metric_name][boot_idx] = metrics[metric_name]

    return {
        metric_name: (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        )
        for metric_name, values in sampled.items()
    }


def add_value_labels(ax, bars, values: list[float], uppers: list[float]) -> None:
    text_effects = [
        pe.Stroke(linewidth=2.2, foreground="white"),
        pe.Stroke(linewidth=0.7, foreground="0.15"),
        pe.Normal(),
    ]
    y_top = ax.get_ylim()[1]
    for bar, value, upper in zip(bars, values, uppers):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(float(upper) + 1.1, y_top - 0.2),
            f"{float(value):.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
            path_effects=text_effects,
            zorder=6,
        )


def save_table_png(
    table_df: pd.DataFrame,
    output_table: str,
    add_title: bool,
    best_family,
    best_fs,
    best_oa: float,
) -> None:
    display_df = table_df.copy()

    percent_cols = [col for col in display_df.columns if col.endswith("(%)")]
    for col in percent_cols:
        display_df[col] = display_df[col].map(lambda v: f"{float(v):.2f}")

    display_df["Δ Overall Accuracy (%)"] = display_df["Δ Overall Accuracy (%)"].map(
        lambda v: f"{float(v):+.2f}" if pd.notna(v) else "—"
    )
    for col in ["Total Support", "Correct Count"]:
        display_df[col] = display_df[col].map(lambda v: f"{int(v):,}")

    display_df = display_df.rename(
        columns={
            "Rank (within Feature Set)": "Rank",
            "Model Family": "Family",
            "Feature Set": "Feature\nSet",
            "Best Run": "Best\nRun",
            "Overall Accuracy (%)": "Overall\nAcc. (%)",
            "Overall Accuracy Lower 95% (%)": "Overall\nLow (%)",
            "Overall Accuracy Upper 95% (%)": "Overall\nHigh (%)",
            "Δ Overall Accuracy (%)": "Δ OA\n(%)",
            "Macro Producer's Accuracy / Recall (%)": "Macro\nProducer (%)",
            "Macro User's Accuracy / Precision (%)": "Macro\nUser (%)",
            "Macro F1-score (%)": "Macro\nF1 (%)",
            "Macro F1-score Lower 95% (%)": "Macro F1\nLow (%)",
            "Macro F1-score Upper 95% (%)": "Macro F1\nHigh (%)",
            "Weighted Producer's Accuracy / Recall (%)": "Weighted\nProducer (%)",
            "Weighted User's Accuracy / Precision (%)": "Weighted\nUser (%)",
            "Weighted F1-score (%)": "Weighted\nF1 (%)",
            "Weighted F1-score Lower 95% (%)": "Weighted F1\nLow (%)",
            "Weighted F1-score Upper 95% (%)": "Weighted F1\nHigh (%)",
            "Total Support": "Support",
            "Correct Count": "Correct",
            "Number of Classes": "Classes",
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
    fig, ax = plt.subplots(figsize=(24, fig_height))
    ax.axis("off")

    if add_title:
        title = (
            "Accuracy Assessment Comparison with 95% CI — Feature Set AE64 vs AE64 + 10 Indices\n"
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
    table.set_fontsize(6.8)
    table.scale(1, 1.85)

    family_col_idx = list(display_df.columns).index("Family")
    delta_col_idx = list(display_df.columns).index("Δ OA\n(%)")
    rank_col_idx = list(display_df.columns).index("Rank")
    family_colors = ["#FFFFFF", "#EEF4FB"]
    prev_family = None
    family_toggle = 0

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.75")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#D0D9E8")
            continue

        family_val = display_df.iloc[row - 1, family_col_idx]
        if family_val != prev_family:
            family_toggle = 1 - family_toggle
            prev_family = family_val
        cell.set_facecolor(family_colors[family_toggle])

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

        if str(display_df.iloc[row - 1, rank_col_idx]) == "1":
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    fig.savefig(output_table, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    for output_path in [args.output_plot, args.output_table, args.output_csv]:
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

    test_all["model_family_clean"] = test_all["model_family"].apply(clean_family_name)
    test_all["feature_set_clean"] = test_all["feature_set"].apply(clean_feature_set_name)

    print("Available model families in test split:")
    for fam in sorted(test_all["model_family_clean"].dropna().unique()):
        print(f"  - {fam}")

    print("\nAvailable feature sets in test split:")
    for fs in sorted(test_all["feature_set_clean"].dropna().unique()):
        print(f"  - {fs}")

    results = []
    combo_idx = 0

    for family in MODEL_FAMILY_ORDER:
        fam_data = test_all[test_all["model_family_clean"] == family].copy()
        if fam_data.empty:
            print(f"Warning: No test data found for model family: {family}")
            continue

        for feature_set in FEATURE_SETS:
            fs_data = fam_data[fam_data["feature_set_clean"] == feature_set].copy()
            if fs_data.empty:
                print(f"Warning: No test data found for {family} / {feature_set}")
                continue

            diag_counts = (
                fs_data[fs_data["true_class_id"] == fs_data["pred_class_id"]]
                .groupby("run_name")["count"]
                .sum()
            )
            total_counts = fs_data.groupby("run_name")["count"].sum()
            run_accuracy = diag_counts.reindex(total_counts.index, fill_value=0) / total_counts

            best_run = run_accuracy.idxmax()
            best_acc = run_accuracy.loc[best_run]
            best_run_data = fs_data[fs_data["run_name"] == best_run].copy()

            if best_run_data.empty:
                print(f"Warning: Could not extract data for best run: {best_run}")
                continue

            meta = best_run_data.iloc[0]
            cm, classes = confusion_matrix_for_run(best_run_data)
            metrics = calculate_metrics_from_cm(cm)
            ci = bootstrap_confidence_intervals(
                cm,
                n_bootstrap=args.bootstrap,
                seed=args.seed + combo_idx,
            )
            combo_idx += 1

            results.append(
                {
                    "Model Family": family,
                    "Feature Set": feature_set,
                    "Best Run": best_run,
                    "Model": meta["model"],
                    "Overall Accuracy (%)": metrics["Overall Accuracy (%)"],
                    "Overall Accuracy Lower 95% (%)": ci["Overall Accuracy (%)"][0],
                    "Overall Accuracy Upper 95% (%)": ci["Overall Accuracy (%)"][1],
                    "Macro Producer's Accuracy / Recall (%)": metrics["Macro Producer's Accuracy / Recall (%)"],
                    "Macro User's Accuracy / Precision (%)": metrics["Macro User's Accuracy / Precision (%)"],
                    "Macro F1-score (%)": metrics["Macro F1-score (%)"],
                    "Macro F1-score Lower 95% (%)": ci["Macro F1-score (%)"][0],
                    "Macro F1-score Upper 95% (%)": ci["Macro F1-score (%)"][1],
                    "Weighted Producer's Accuracy / Recall (%)": metrics["Weighted Producer's Accuracy / Recall (%)"],
                    "Weighted User's Accuracy / Precision (%)": metrics["Weighted User's Accuracy / Precision (%)"],
                    "Weighted F1-score (%)": metrics["Weighted F1-score (%)"],
                    "Weighted F1-score Lower 95% (%)": ci["Weighted F1-score (%)"][0],
                    "Weighted F1-score Upper 95% (%)": ci["Weighted F1-score (%)"][1],
                    "Total Support": metrics["Total Support"],
                    "Correct Count": metrics["Correct Count"],
                    "Number of Classes": len(classes),
                }
            )

            print(f"\nBest run — {family} / {feature_set}")
            print(f"  Best run        : {best_run}")
            print(f"  Model           : {meta['model']}")
            print(f"  Overall accuracy: {best_acc:.4f} ({best_acc * 100:.2f}%)")

    if not results:
        raise ValueError("No (model-family, feature-set) comparison results could be created.")

    table_df = pd.DataFrame(results)
    table_df["Model Family"] = pd.Categorical(
        table_df["Model Family"],
        categories=MODEL_FAMILY_ORDER,
        ordered=True,
    )
    table_df["Feature Set"] = pd.Categorical(
        table_df["Feature Set"],
        categories=FEATURE_SETS,
        ordered=True,
    )
    table_df = table_df.sort_values(["Model Family", "Feature Set"]).reset_index(drop=True)

    oa_pivot = table_df.pivot_table(
        index="Model Family",
        columns="Feature Set",
        values="Overall Accuracy (%)",
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

    table_df["Rank (within Feature Set)"] = (
        table_df.groupby("Feature Set", observed=False)["Overall Accuracy (%)"]
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
    table_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved table CSV → {args.output_csv}")

    families_present = [
        f for f in MODEL_FAMILY_ORDER
        if f in table_df["Model Family"].values
    ]
    x = np.arange(len(families_present))
    bar_width = 0.35
    offsets = np.linspace(
        -(len(FEATURE_SETS) - 1) * bar_width / 2,
        (len(FEATURE_SETS) - 1) * bar_width / 2,
        len(FEATURE_SETS),
    )

    max_upper = float(table_df["Overall Accuracy Upper 95% (%)"].max())
    y_max = max(105.0, min(115.0, max_upper + 7.0))

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_ylim(0, y_max)

    for offset, feature_set in zip(offsets, FEATURE_SETS):
        fs_rows = table_df[table_df["Feature Set"] == feature_set].set_index("Model Family")
        oa_vals = [
            fs_rows.loc[fam, "Overall Accuracy (%)"] if fam in fs_rows.index else 0.0
            for fam in families_present
        ]
        oa_low = [
            fs_rows.loc[fam, "Overall Accuracy Lower 95% (%)"] if fam in fs_rows.index else 0.0
            for fam in families_present
        ]
        oa_high = [
            fs_rows.loc[fam, "Overall Accuracy Upper 95% (%)"] if fam in fs_rows.index else 0.0
            for fam in families_present
        ]
        bars = ax.bar(
            x + offset,
            oa_vals,
            width=bar_width,
            yerr=np.vstack([np.asarray(oa_vals) - np.asarray(oa_low), np.asarray(oa_high) - np.asarray(oa_vals)]),
            capsize=3,
            ecolor="black",
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
            label=FEATURE_SET_DISPLAY[feature_set],
            color=FEATURE_SET_COLORS[feature_set],
        )
        add_value_labels(ax, bars, oa_vals, oa_high)

    for xi, fam in zip(x, families_present):
        delta = delta_map.get(fam, None)
        if delta is None:
            continue
        sign = "+" if delta >= 0 else ""
        color = "#2ca02c" if delta >= 0 else "#d62728"
        family_max = table_df.loc[
            table_df["Model Family"] == fam,
            "Overall Accuracy Upper 95% (%)",
        ].max()
        ax.text(
            xi,
            min(float(family_max) + 3.0, y_max - 0.4),
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
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    best_row = table_df.loc[table_df["Overall Accuracy (%)"].idxmax()]
    best_family = best_row["Model Family"]
    best_fs = best_row["Feature Set"]
    best_oa = best_row["Overall Accuracy (%)"]

    if args.add_title:
        ax.set_title(
            "Feature Set Comparison with 95% CI: AE64 vs AE64 + 10 Indices\n"
            f"Best Overall: {best_family} | Feature Set: {FEATURE_SET_DISPLAY[best_fs]} | "
            f"Overall Accuracy: {best_oa:.2f}%  (Δ = AE64 + 10 Indices − AE64)",
            fontsize=12,
            pad=14,
        )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(args.output_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot PNG  → {args.output_plot}")

    save_table_png(
        table_df=table_df,
        output_table=args.output_table,
        add_title=args.add_title,
        best_family=best_family,
        best_fs=best_fs,
        best_oa=float(best_oa),
    )
    print(f"Saved table PNG → {args.output_table}")
    print("\nDone.")


if __name__ == "__main__":
    main()
