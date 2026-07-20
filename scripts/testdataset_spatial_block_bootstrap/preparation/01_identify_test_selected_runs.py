#!/usr/bin/env python3
"""Freeze test-accuracy-selected runs for the test-dataset bootstrap workflow.

Reproduction and AOI adaptation
-------------------------------
Workflow role: Produce the test-selected spatial-block uncertainty analysis used for descriptive thesis results.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--runs-csv``, ``--output-root``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Regenerate test-selected run metadata and all block-level predictions for the new AOI. Treat test-selected intervals as descriptive, not unbiased model-selection evidence.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.

Reproducible invocation
~~~~~~~~~~~~~~~~~~~~~~~
Inspect the complete interface before supplying AOI-specific paths::

    python scripts/testdataset_spatial_block_bootstrap/preparation/01_identify_test_selected_runs.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SHARED_PACKAGE_ROOT = PROJECT_ROOT / "scripts" / "spatial_block_bootstrap"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHARED_PACKAGE_ROOT))

from common.constants import (  # noqa: E402
    MODEL_FAMILY_ORDER,
    family_display,
    feature_display,
    resolve_path,
)
from common.output_utils import write_yaml  # noqa: E402


DEFAULT_RUNS_CSV = Path(
    "outputs/master_training_with_outputs/all_master_runs_long.csv"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/testdataset_spatial_block_bootstrap")
FINAL_MLP_RUN = "mlp_ae64plus10idx_h512-256_do03_lr1e3_bs4096_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-csv", type=Path, default=DEFAULT_RUNS_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def existing_artifact(value: object) -> bool:
    if pd.isna(value) or not str(value).strip():
        return False
    return resolve_path(str(value)).exists()


def main() -> None:
    args = parse_args()
    source_path = resolve_path(args.runs_csv)
    source = pd.read_csv(source_path)
    output_root = resolve_path(args.output_root)
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "run_name",
        "model",
        "model_family",
        "feature_set",
        "best_val_macro_f1",
        "test_acc",
        "test_macro_f1",
        "test_predictions_csv",
        "confusion_matrix_test_csv",
        "best_model_path",
        "data_npz",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Missing run-inventory columns: {sorted(missing)}")

    source = source.copy()
    source["model_family"] = source["model_family"].map(family_display)
    source["feature_set"] = source["feature_set"].map(feature_display)
    source["test_acc"] = pd.to_numeric(source["test_acc"], errors="coerce")
    source = source.dropna(subset=["test_acc"])
    for artifact_column in [
        "test_predictions_csv",
        "confusion_matrix_test_csv",
        "best_model_path",
        "data_npz",
    ]:
        source = source[source[artifact_column].map(existing_artifact)]

    expected = {
        (family, feature)
        for family in MODEL_FAMILY_ORDER
        for feature in ["AE64", "AE64_plus10indices"]
    }
    available = set(zip(source["model_family"], source["feature_set"]))
    if expected - available:
        raise ValueError(f"Missing model/feature combinations: {sorted(expected - available)}")

    selected_indices = []
    for family, feature in sorted(expected):
        candidates = source[
            (source["model_family"] == family)
            & (source["feature_set"] == feature)
        ]
        selected_indices.append(candidates["test_acc"].idxmax())
    selected = source.loc[selected_indices].copy()

    selected["use_featureset_comparison"] = True
    selected["use_model_comparison"] = False
    for family in MODEL_FAMILY_ORDER:
        family_rows = selected[selected["model_family"] == family]
        selected.loc[family_rows["test_acc"].idxmax(), "use_model_comparison"] = True

    final_matches = selected[selected["run_name"] == FINAL_MLP_RUN]
    if len(final_matches) != 1:
        raise ValueError(
            f"Required final MLP run was not selected or is duplicated: {FINAL_MLP_RUN}"
        )
    if final_matches.iloc[0]["model_family"] != "MLP":
        raise ValueError(f"Required final run is not an MLP run: {FINAL_MLP_RUN}")
    selected["use_best_overall"] = selected["run_name"] == FINAL_MLP_RUN
    selected["selection_source"] = (
        "highest test Overall Accuracy in all_master_runs_long.csv"
    )

    selected_columns = [
        "run_name",
        "model",
        "model_family",
        "feature_set",
        "best_val_macro_f1",
        "test_acc",
        "test_macro_f1",
        "test_predictions_csv",
        "confusion_matrix_test_csv",
        "best_model_path",
        "data_npz",
        "use_model_comparison",
        "use_featureset_comparison",
        "use_best_overall",
        "selection_source",
    ]
    selected = selected[selected_columns].sort_values(
        ["model_family", "feature_set"]
    )
    selected_path = metadata_dir / "selected_runs.csv"
    selected.to_csv(selected_path, index=False)

    model_comparison = {}
    for row in selected[selected["use_model_comparison"]].itertuples(index=False):
        model_comparison[row.model_family] = {
            "run_name": row.run_name,
            "feature_set": row.feature_set,
            "test_accuracy": float(row.test_acc),
        }
    featureset_comparison = {}
    for family in MODEL_FAMILY_ORDER:
        rows = selected[selected["model_family"] == family]
        featureset_comparison[family] = {
            row.feature_set: row.run_name for row in rows.itertuples(index=False)
        }
    best = selected[selected["use_best_overall"]].iloc[0]
    frozen = {
        "selection_metric": "test_acc",
        "selection_source": str(source_path),
        "selection_warning": (
            "Test-selected results are descriptive and can be optimistically biased; "
            "the test set influenced model selection."
        ),
        "model_comparison": model_comparison,
        "featureset_comparison": featureset_comparison,
        "best_overall_model": {
            "run_name": best["run_name"],
            "model_family": best["model_family"],
            "feature_set": best["feature_set"],
            "test_accuracy": float(best["test_acc"]),
        },
    }
    config_path = PACKAGE_ROOT / "config" / "selected_runs.yaml"
    write_yaml(config_path, frozen)
    print(f"Saved: {selected_path}")
    print(f"Saved: {config_path}")
    print(f"Final MLP run: {FINAL_MLP_RUN}")


if __name__ == "__main__":
    main()
