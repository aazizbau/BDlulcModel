#!/usr/bin/env python3
"""
Create one thesis-ready CSV from test-selected spatial bootstrap results.

Example:
    python scripts/testdataset_spatial_block_bootstrap/reporting/24_create_thesis_results_csv.py \
        --output-root outputs/testdataset_spatial_block_bootstrap \
        --output-csv outputs/testdataset_spatial_block_bootstrap/summaries/thesis_results_spatial_ci.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = Path("outputs/testdataset_spatial_block_bootstrap")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Defaults to "
            "<output-root>/summaries/thesis_results_spatial_ci.csv."
        ),
    )
    return parser.parse_args()


def result_row(
    *,
    section: str,
    metric: str,
    observed: float,
    bootstrap_mean: float,
    lower: float,
    upper: float,
    model_family: str = "",
    comparison: str = "",
    feature_set: str = "",
    class_id: int | str = "",
    class_name: str = "",
    run_name: str = "",
    n_blocks: int = 0,
    n_bootstrap: int = 0,
    unit: str = "percent",
) -> dict[str, object]:
    lower_margin = observed - lower
    upper_margin = upper - observed
    half_width = (upper - lower) / 2.0
    unit_suffix = "%" if unit == "percent" else " pp"
    return {
        "section": section,
        "model_family": model_family,
        "comparison": comparison,
        "feature_set": feature_set,
        "class_id": class_id,
        "class_name": class_name,
        "metric": metric,
        "unit": unit,
        "observed": observed,
        "bootstrap_mean": bootstrap_mean,
        "lower_95": lower,
        "upper_95": upper,
        "lower_margin": lower_margin,
        "upper_margin": upper_margin,
        "ci_half_width": half_width,
        "ci_total_width": upper - lower,
        "formatted_estimate_pm": (
            f"{observed:.2f}{unit_suffix} +/- {half_width:.2f} percentage points"
        ),
        "formatted_estimate_ci": (
            f"{observed:.2f}{unit_suffix} (95% CI: {lower:.2f} to {upper:.2f}{unit_suffix})"
        ),
        "n_blocks": n_blocks,
        "n_bootstrap": n_bootstrap,
        "run_name": run_name,
    }


def add_best_model_summary(
    rows: list[dict[str, object]],
    model_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    class_distribution: pd.DataFrame,
) -> None:
    mlp = model_summary[model_summary["model_family"] == "MLP"]
    for record in mlp.itertuples(index=False):
        rows.append(
            result_row(
                section="selected_best_model_summary",
                model_family=record.model_family,
                feature_set=record.feature_set,
                metric=record.metric,
                observed=record.observed,
                bootstrap_mean=record.bootstrap_mean,
                lower=record.lower_95,
                upper=record.upper_95,
                n_blocks=record.n_blocks,
                n_bootstrap=record.n_bootstrap,
                run_name=record.run_name,
            )
        )

    for metric in ["Producer's Accuracy / Recall", "User's Accuracy / Precision"]:
        observed = class_summary.loc[class_summary["metric"] == metric, "observed"].mean()
        replicate_values = (
            class_distribution[class_distribution["metric"] == metric]
            .groupby("replicate", sort=True)["value"]
            .mean()
        )
        first = class_summary.iloc[0]
        rows.append(
            result_row(
                section="selected_best_model_summary",
                model_family=first["model_family"],
                feature_set=first["feature_set"],
                metric=f"Macro {metric}",
                observed=observed,
                bootstrap_mean=replicate_values.mean(),
                lower=replicate_values.quantile(0.025),
                upper=replicate_values.quantile(0.975),
                n_blocks=int(first["n_blocks"]),
                n_bootstrap=len(replicate_values),
                run_name=first["run_name"],
            )
        )


def add_model_comparison(
    rows: list[dict[str, object]],
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
) -> None:
    for record in summary.itertuples(index=False):
        rows.append(
            result_row(
                section="model_family_comparison",
                model_family=record.model_family,
                feature_set=record.feature_set,
                metric=record.metric,
                observed=record.observed,
                bootstrap_mean=record.bootstrap_mean,
                lower=record.lower_95,
                upper=record.upper_95,
                n_blocks=record.n_blocks,
                n_bootstrap=record.n_bootstrap,
                run_name=record.run_name,
            )
        )

    accuracy_summary = summary[summary["metric"] == "Overall Accuracy"].set_index(
        "model_family"
    )
    accuracy_distribution = distribution[
        distribution["metric"] == "Overall Accuracy"
    ].pivot(index="replicate", columns="model_family", values="value")
    for competitor in accuracy_summary.index:
        if competitor == "MLP":
            continue
        differences = accuracy_distribution["MLP"] - accuracy_distribution[competitor]
        observed = (
            accuracy_summary.loc["MLP", "observed"]
            - accuracy_summary.loc[competitor, "observed"]
        )
        rows.append(
            result_row(
                section="mlp_accuracy_gap",
                model_family="MLP",
                comparison=f"MLP minus {competitor}",
                metric="Overall Accuracy Difference",
                observed=observed,
                bootstrap_mean=differences.mean(),
                lower=differences.quantile(0.025),
                upper=differences.quantile(0.975),
                n_blocks=int(accuracy_summary.loc["MLP", "n_blocks"]),
                n_bootstrap=len(differences),
                unit="percentage_points",
            )
        )


def add_feature_comparison(
    rows: list[dict[str, object]], summary: pd.DataFrame
) -> None:
    configurations = [
        ("AE64", "observed_ae64", "ae64_bootstrap_mean", "ae64_lower_95", "ae64_upper_95"),
        (
            "AE64_plus10indices",
            "observed_plusindices",
            "plusindices_bootstrap_mean",
            "plusindices_lower_95",
            "plusindices_upper_95",
        ),
    ]
    for record in summary.to_dict("records"):
        for feature_set, observed, mean, lower, upper in configurations:
            rows.append(
                result_row(
                    section="feature_set_comparison",
                    model_family=record["model_family"],
                    feature_set=feature_set,
                    metric="Overall Accuracy",
                    observed=record[observed],
                    bootstrap_mean=record[mean],
                    lower=record[lower],
                    upper=record[upper],
                    n_blocks=record["n_blocks"],
                    n_bootstrap=record["n_bootstrap"],
                )
            )
        rows.append(
            result_row(
                section="feature_set_effect",
                model_family=record["model_family"],
                comparison="AE64_plus10indices minus AE64",
                metric="Overall Accuracy Difference",
                observed=record["observed_delta"],
                bootstrap_mean=record["delta_bootstrap_mean"],
                lower=record["delta_lower_95"],
                upper=record["delta_upper_95"],
                n_blocks=record["n_blocks"],
                n_bootstrap=record["n_bootstrap"],
                unit="percentage_points",
            )
        )


def add_classwise(
    rows: list[dict[str, object]], summary: pd.DataFrame
) -> None:
    for record in summary.itertuples(index=False):
        rows.append(
            result_row(
                section="best_model_classwise",
                model_family=record.model_family,
                feature_set=record.feature_set,
                class_id=record.class_id,
                class_name=record.class_name,
                metric=record.metric,
                observed=record.observed,
                bootstrap_mean=record.bootstrap_mean,
                lower=record.lower_95,
                upper=record.upper_95,
                n_blocks=record.n_blocks,
                n_bootstrap=record.total_replicates,
                run_name=record.run_name,
            )
        )


def main() -> None:
    args = parse_args()
    root = resolve_path(args.output_root)
    output = (
        resolve_path(args.output_csv)
        if args.output_csv
        else root / "summaries" / "thesis_results_spatial_ci.csv"
    )
    summary_dir = root / "summaries"
    distribution_dir = root / "bootstrap_distributions"

    model_summary = pd.read_csv(summary_dir / "model_comparison_spatial_ci.csv")
    feature_summary = pd.read_csv(summary_dir / "featureset_comparison_spatial_ci.csv")
    class_summary = pd.read_csv(summary_dir / "bestmodel_classwise_spatial_ci.csv")
    model_distribution = pd.read_csv(
        distribution_dir / "model_comparison_bootstrap_metrics.csv.gz"
    )
    class_distribution = pd.read_csv(
        distribution_dir / "bestmodel_classwise_bootstrap_metrics.csv.gz"
    )

    rows: list[dict[str, object]] = []
    add_best_model_summary(rows, model_summary, class_summary, class_distribution)
    add_model_comparison(rows, model_summary, model_distribution)
    add_feature_comparison(rows, feature_summary)
    add_classwise(rows, class_summary)

    result = pd.DataFrame(rows)
    numeric_columns = [
        "observed",
        "bootstrap_mean",
        "lower_95",
        "upper_95",
        "lower_margin",
        "upper_margin",
        "ci_half_width",
        "ci_total_width",
    ]
    result[numeric_columns] = result[numeric_columns].round(6)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    print(f"Saved: {output}")
    print(f"Rows: {len(result):,}")
    print("Sections:")
    for section, count in result["section"].value_counts(sort=False).items():
        print(f"  {section}: {count}")


if __name__ == "__main__":
    main()
