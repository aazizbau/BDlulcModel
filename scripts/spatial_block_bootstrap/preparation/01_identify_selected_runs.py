#!/usr/bin/env python3
"""Freeze validation-selected model runs for all spatial bootstrap analyses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from common.constants import (  # noqa: E402
    DEFAULT_BEST_RUNS_CSV,
    DEFAULT_OUTPUT_ROOT,
    MODEL_FAMILY_ORDER,
    family_display,
    feature_display,
    resolve_path,
)
from common.output_utils import write_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-runs-csv", type=Path, default=DEFAULT_BEST_RUNS_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(resolve_path(args.best_runs_csv))
    output_root = resolve_path(args.output_root)
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "run_name",
        "model_family",
        "feature_set",
        "best_val_macro_f1",
        "test_predictions_csv",
        "confusion_matrix_test_csv",
        "best_model_path",
        "data_npz",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Missing selected-run columns: {sorted(missing)}")

    source = source.copy()
    source["model_family"] = source["model_family"].map(family_display)
    source["feature_set"] = source["feature_set"].map(feature_display)

    duplicate = source.duplicated(["model_family", "feature_set"], keep=False)
    if duplicate.any():
        raise ValueError(
            "best_runs_by_group.csv must contain one validation-selected run per "
            "model-family/feature-set combination."
        )

    combinations = set(zip(source["model_family"], source["feature_set"]))
    expected = {
        (family, feature)
        for family in MODEL_FAMILY_ORDER
        for feature in ["AE64", "AE64_plus10indices"]
    }
    if combinations != expected:
        raise ValueError(f"Missing combinations: {sorted(expected - combinations)}")

    source["use_featureset_comparison"] = True
    source["use_model_comparison"] = False
    for family in MODEL_FAMILY_ORDER:
        family_rows = source[source["model_family"] == family]
        selected_index = family_rows["best_val_macro_f1"].astype(float).idxmax()
        source.loc[selected_index, "use_model_comparison"] = True

    best_index = source["best_val_macro_f1"].astype(float).idxmax()
    source["use_best_overall"] = False
    source.loc[best_index, "use_best_overall"] = True
    source["selection_source"] = "best validation macro F1 in best_runs_by_group.csv"

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
    selected = source[selected_columns].sort_values(
        ["model_family", "feature_set"]
    )
    selected_path = metadata_dir / "selected_runs.csv"
    selected.to_csv(selected_path, index=False)

    model_comparison = {}
    for row in selected[selected["use_model_comparison"]].itertuples(index=False):
        model_comparison[row.model_family] = {
            "run_name": row.run_name,
            "feature_set": row.feature_set,
        }
    featureset_comparison = {}
    for family in MODEL_FAMILY_ORDER:
        rows = selected[selected["model_family"] == family]
        featureset_comparison[family] = {
            row.feature_set: row.run_name for row in rows.itertuples(index=False)
        }
    best = selected[selected["use_best_overall"]].iloc[0]
    frozen = {
        "selection_metric": "best_val_macro_f1",
        "selection_source": str(resolve_path(args.best_runs_csv)),
        "model_comparison": model_comparison,
        "featureset_comparison": featureset_comparison,
        "best_overall_model": {
            "run_name": best["run_name"],
            "model_family": best["model_family"],
            "feature_set": best["feature_set"],
        },
    }
    write_yaml(PACKAGE_ROOT / "config" / "selected_runs.yaml", frozen)
    print(f"Saved: {selected_path}")
    print(f"Saved: {PACKAGE_ROOT / 'config' / 'selected_runs.yaml'}")


if __name__ == "__main__":
    main()
