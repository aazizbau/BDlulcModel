#!/usr/bin/env python3
"""Reproduction and AOI adaptation
-------------------------------
Workflow role: Extract spatially split samples, train a classifier, or orchestrate hyperparameter experiments.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--family``, ``--master-csv``, ``--runs``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace NPZ/raster/vector inputs with samples extracted from the new AOI, preserve spatially disjoint splits, and review class IDs, feature order, block size, budgets, and random seeds.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.

Reproducible invocation
~~~~~~~~~~~~~~~~~~~~~~~
Inspect the complete interface before supplying AOI-specific paths::

    python scripts/training/build_master_csv.py --help
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

JST = timezone(timedelta(hours=9))

COMMON_PREFIX_FIELDS = [
    "run_dir",
    "timestamp_jst",
    "data_npz",
    "summary_json",
    "model",
    "model_type",
    "seed",
    "input_dim",
    "expected_input_dim",
    "num_classes",
]

COMMON_SAMPLE_FIELDS = [
    "train_samples",
    "val_samples",
    "test_samples",
]

COMMON_METRIC_FIELDS = [
    "best_val_loss",
    "best_val_acc",
    "best_val_macro_f1",
    "best_val_balanced_acc",
    "test_evaluated",
    "test_loss",
    "test_acc",
    "test_macro_f1",
    "test_balanced_acc",
    "total_train_seconds",
]

COMMON_ARTIFACT_FIELDS = [
    "history_csv",
    "best_model_path",
    "feature_importance_gain_csv",
    "confusion_matrix_val_csv",
    "val_predictions_csv",
    "per_class_metrics_val_csv",
    "confusion_matrix_test_csv",
    "test_predictions_csv",
    "per_class_metrics_test_csv",
]

COMMON_LABEL_FIELDS = [
    "train_present_labels",
    "train_missing_labels",
    "val_present_labels",
    "val_missing_labels",
    "test_present_labels",
    "test_missing_labels",
]

FAMILY_CONFIG: dict[str, dict[str, Any]] = {
    "mlp": {
        "default_model": "MLP",
        "family_fields": [
            "hidden_dims",
            "dropout",
            "batch_size",
            "epochs_requested",
            "epochs_completed",
            "lr",
            "weight_decay",
            "label_smoothing",
            "scheduler",
            "scheduler_factor",
            "scheduler_patience",
            "best_epoch",
        ],
    },
    "resmlp": {
        "default_model": "ResMLP",
        "family_fields": [
            "hidden_dims",
            "dropout",
            "batch_size",
            "epochs_requested",
            "epochs_completed",
            "lr",
            "weight_decay",
            "label_smoothing",
            "scheduler",
            "scheduler_factor",
            "scheduler_patience",
            "best_epoch",
        ],
    },
    "cnn1d": {
        "default_model": "CNN1D",
        "family_fields": [
            "channels",
            "kernels",
            "head_dim",
            "dropout",
            "batch_size",
            "epochs_requested",
            "epochs_completed",
            "lr",
            "weight_decay",
            "label_smoothing",
            "scheduler",
            "scheduler_factor",
            "scheduler_patience",
            "best_epoch",
        ],
    },
    "fttransformer": {
        "default_model": "FT-Transformer",
        "family_fields": [
            "d_token",
            "n_blocks",
            "n_heads",
            "attention_dropout",
            "ff_dropout",
            "residual_dropout",
            "ff_multiplier",
            "batch_size",
            "grad_accum_steps",
            "effective_batch_size",
            "amp_enabled",
            "epochs_requested",
            "epochs_completed",
            "lr",
            "weight_decay",
            "label_smoothing",
            "scheduler",
            "scheduler_factor",
            "scheduler_patience",
            "best_epoch",
        ],
    },
    "lgbm": {
        "default_model": "LightGBM",
        "family_fields": [
            "learning_rate",
            "n_estimators_requested",
            "best_iteration",
            "num_leaves",
            "max_depth",
            "min_child_samples",
            "subsample",
            "subsample_freq",
            "colsample_bytree",
            "min_split_gain",
            "reg_alpha",
            "reg_lambda",
            "max_bin",
            "early_stopping_rounds",
            "eval_every",
            "n_jobs",
            "force_col_wise",
            "force_row_wise",
            "deterministic",
            "train_loss",
            "train_acc",
            "train_macro_f1",
            "train_balanced_acc",
        ],
    },
    "xgboost": {
        "default_model": "XGBoost",
        "family_fields": [
            "learning_rate",
            "n_estimators_requested",
            "best_iteration",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "max_bin",
            "early_stopping_rounds",
            "eval_every",
            "n_jobs",
            "tree_method",
            "grow_policy",
            "train_loss",
            "train_acc",
            "train_macro_f1",
            "train_balanced_acc",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a standardized master CSV from run summary files.")
    parser.add_argument("--family", required=True, choices=sorted(FAMILY_CONFIG))
    parser.add_argument("--master-csv", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    return parser.parse_args()


def list_to_pipe(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(str(x) for x in value)
    return "" if value is None else str(value)


def hidden_dims_to_dash(value: Any) -> str:
    if isinstance(value, list):
        return "-".join(str(x) for x in value)
    return "" if value is None else str(value)


def flatten_presence(value: Any, key: str) -> str:
    if not isinstance(value, dict):
        return ""
    items = value.get(key, [])
    return "|".join(str(x) for x in items)


def extract_value(field: str, summary: dict[str, Any], run_dir: Path, default_model: str) -> Any:
    if field == "run_dir":
        return str(run_dir)
    if field == "timestamp_jst":
        return summary.get("created_at_jst", "")
    if field == "data_npz":
        return summary.get("data", "")
    if field == "summary_json":
        return str(run_dir / "summary.json")
    if field == "model":
        return summary.get("model", default_model)
    if field == "model_type":
        return summary.get("model_type", "")
    if field == "hidden_dims":
        return hidden_dims_to_dash(summary.get("hidden_dims", ""))
    if field in {"channels", "kernels"}:
        return list_to_pipe(summary.get(field, ""))
    if field == "feature_importance_gain_csv":
        return summary.get("feature_importance_gain_csv", "")
    if field.startswith("train_") and field.endswith("_labels"):
        source = summary.get("train_label_presence", {})
        subkey = "present" if "present" in field else "missing"
        return flatten_presence(source, subkey)
    if field.startswith("val_") and field.endswith("_labels"):
        source = summary.get("val_label_presence", {})
        subkey = "present" if "present" in field else "missing"
        return flatten_presence(source, subkey)
    if field.startswith("test_") and field.endswith("_labels"):
        source = summary.get("test_label_presence", {})
        subkey = "present" if "present" in field else "missing"
        return flatten_presence(source, subkey)
    if field == "notes":
        return ""
    return summary.get(field, "")


def num_or_inf(value: Any, negative: bool = False) -> float:
    if value in ("", None):
        return float("-inf") if negative else float("inf")
    try:
        return float(value)
    except Exception:
        return float("-inf") if negative else float("inf")


def main() -> None:
    args = parse_args()
    cfg = FAMILY_CONFIG[args.family]
    fieldnames = (
        COMMON_PREFIX_FIELDS
        + cfg["family_fields"]
        + COMMON_SAMPLE_FIELDS
        + COMMON_METRIC_FIELDS
        + COMMON_ARTIFACT_FIELDS
        + COMMON_LABEL_FIELDS
        + ["notes"]
    )

    rows: list[dict[str, Any]] = []
    for run_name in args.runs:
        run_dir = Path("runs") / run_name
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue

        with summary_path.open() as f:
            summary = json.load(f)

        row = {
            field: extract_value(field, summary, run_dir, cfg["default_model"])
            for field in fieldnames
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            -num_or_inf(row["best_val_macro_f1"], negative=True),
            -num_or_inf(row["test_macro_f1"], negative=True),
            num_or_inf(row["best_val_loss"], negative=False),
        )
    )

    args.master_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.master_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    ts = datetime.now(JST).isoformat(timespec="seconds")
    print(f"[{ts}] Wrote master CSV: {args.master_csv}")
    print(f"[{ts}] Rows: {len(rows)}")


if __name__ == "__main__":
    main()
