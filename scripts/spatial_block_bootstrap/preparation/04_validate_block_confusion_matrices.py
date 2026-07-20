#!/usr/bin/env python3
"""Strictly validate reconstructed block confusion matrices before bootstrap.

Reproduction and AOI adaptation
-------------------------------
Workflow role: Estimate confidence intervals by resampling the original spatial test blocks.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--output-root``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Regenerate block IDs, predictions, and selected-run metadata from the new AOI spatial split before resampling; never reuse this project's block inventory.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.

Reproducible invocation
~~~~~~~~~~~~~~~~~~~~~~~
Inspect the complete interface before supplying AOI-specific paths::

    python scripts/spatial_block_bootstrap/preparation/04_validate_block_confusion_matrices.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from common.block_confusion_utils import block_tensor  # noqa: E402
from common.constants import DEFAULT_OUTPUT_ROOT, resolve_path  # noqa: E402
from common.metric_utils import metrics_from_cm  # noqa: E402
from common.output_utils import read_table, write_json  # noqa: E402
from common.validation_utils import read_saved_confusion_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def bool_text(value: bool) -> str:
    return "PASS" if value else "FAIL"


def load_prediction(path: str) -> pd.DataFrame:
    actual = Path(path)
    if actual.suffixes[-2:] == [".csv", ".gz"]:
        return pd.read_csv(actual)
    return pd.read_parquet(actual)


def main() -> None:
    args = parse_args()
    output_root = resolve_path(args.output_root)
    selected = pd.read_csv(output_root / "metadata" / "selected_runs.csv")
    inventory = pd.read_csv(output_root / "metadata" / "test_block_inventory.csv")
    block_long = read_table(
        output_root
        / "block_confusion_matrices"
        / "all_selected_runs_block_confusion_long.parquet"
    )

    inventory_by_run = inventory.set_index("run_name")
    report_lines = [
        "SPATIAL BLOCK BOOTSTRAP VALIDATION REPORT",
        "=" * 80,
        "",
    ]
    statuses: dict[str, bool] = {}
    block_sets: dict[str, set[str]] = {}
    true_signatures: dict[str, pd.Series] = {}

    for row in selected.itertuples(index=False):
        run_blocks = sorted(
            block_long.loc[block_long["run_name"] == row.run_name, "block_id"].unique()
        )
        block_sets[row.run_name] = set(run_blocks)
        tensor = block_tensor(block_long, row.run_name, run_blocks)
        reconstructed = tensor.sum(axis=0)
        original = read_saved_confusion_matrix(resolve_path(row.confusion_matrix_test_csv))
        original_metrics = metrics_from_cm(original)
        reconstructed_metrics = metrics_from_cm(reconstructed)

        support_match = int(original.sum()) == int(reconstructed.sum())
        correct_match = int(np.diag(original).sum()) == int(
            np.diag(reconstructed).sum()
        )
        exact_match = np.array_equal(original, reconstructed)
        max_difference = int(np.abs(original - reconstructed).max())
        oa_match = np.isclose(
            original_metrics["overall_accuracy"],
            reconstructed_metrics["overall_accuracy"],
            atol=0.0,
            rtol=0.0,
        )
        macro_match = np.isclose(
            original_metrics["macro_f1"],
            reconstructed_metrics["macro_f1"],
            atol=0.0,
            rtol=0.0,
        )
        weighted_match = np.isclose(
            original_metrics["weighted_f1"],
            reconstructed_metrics["weighted_f1"],
            atol=0.0,
            rtol=0.0,
        )
        status = all(
            [support_match, correct_match, exact_match, oa_match, macro_match, weighted_match]
        )
        statuses[row.run_name] = status

        prediction_path = inventory_by_run.loc[row.run_name, "prediction_path"]
        predictions = load_prediction(prediction_path)
        true_signatures[row.run_name] = (
            predictions.groupby(["block_id", "true_class_id"]).size().sort_index()
        )

        report_lines.extend(
            [
                f"Run: {row.run_name}",
                f"Model family: {row.model_family}",
                f"Feature set: {row.feature_set}",
                f"Number of test blocks: {len(run_blocks):,}",
                f"Number of test observations: {len(predictions):,}",
                f"Original support: {int(original.sum()):,}",
                f"Reconstructed support: {int(reconstructed.sum()):,}",
                f"Original correct count: {int(np.diag(original).sum()):,}",
                f"Reconstructed correct count: {int(np.diag(reconstructed).sum()):,}",
                f"Maximum confusion-matrix cell difference: {max_difference}",
                f"Original Overall Accuracy: {float(original_metrics['overall_accuracy']):.12f}",
                f"Reconstructed Overall Accuracy: {float(reconstructed_metrics['overall_accuracy']):.12f}",
                f"Original Macro F1: {float(original_metrics['macro_f1']):.12f}",
                f"Reconstructed Macro F1: {float(reconstructed_metrics['macro_f1']):.12f}",
                f"Original Weighted F1: {float(original_metrics['weighted_f1']):.12f}",
                f"Reconstructed Weighted F1: {float(reconstructed_metrics['weighted_f1']):.12f}",
                f"Support match: {bool_text(support_match)}",
                f"Correct count match: {bool_text(correct_match)}",
                f"Confusion matrix exact match: {bool_text(exact_match)}",
                f"Overall Accuracy match: {bool_text(oa_match)}",
                f"Macro F1 match: {bool_text(macro_match)}",
                f"Weighted F1 match: {bool_text(weighted_match)}",
                f"FINAL STATUS: {bool_text(status)}",
                "-" * 80,
                "",
            ]
        )

    reference_run = selected.iloc[0]["run_name"]
    block_ids_consistent = all(
        blocks == block_sets[reference_run] for blocks in block_sets.values()
    )
    report_lines.extend(
        [
            "CROSS-RUN CHECKS",
            "=" * 80,
            f"Same test block IDs across all runs: {bool_text(block_ids_consistent)}",
        ]
    )
    for feature_set, group in selected.groupby("feature_set"):
        runs = group["run_name"].tolist()
        reference = true_signatures[runs[0]]
        consistent = all(reference.equals(true_signatures[name]) for name in runs[1:])
        report_lines.append(
            f"Same block-wise true labels within {feature_set}: {bool_text(consistent)}"
        )
        if not consistent:
            for name in runs:
                statuses[name] = False

    report_lines.append(
        "Cross-feature-set true-label masks are reported separately because "
        "index nodata filtering can remove observations while retaining the same blocks."
    )
    overall_pass = all(statuses.values()) and block_ids_consistent
    report_lines.extend(["", f"OVERALL VALIDATION STATUS: {bool_text(overall_pass)}"])

    report_path = output_root / "metadata" / "validation_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_json(
        output_root / "metadata" / "validation_status.json",
        {
            "passed": overall_pass,
            "block_ids_consistent": block_ids_consistent,
            "run_status": statuses,
        },
    )
    print(f"Saved: {report_path}")
    print(f"Validation status: {bool_text(overall_pass)}")
    if not overall_pass:
        raise SystemExit("Validation failed; spatial bootstrap is blocked.")


if __name__ == "__main__":
    main()
