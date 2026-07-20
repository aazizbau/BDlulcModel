#!/usr/bin/env python3
"""
Freeze the selected best configuration for each model family.

This script reads the existing master best-run table, chooses the best run from
each model family using validation macro F1, and writes one YAML file per family
under scripts/final_10seed_experiment/configs/.

Complete Example Run
--------------------
python scripts/final_10seed_experiment/prepare_selected_configs.py \
    --best-runs-csv outputs/master_training_with_outputs/best_runs_by_group.csv \
    --output-dir scripts/final_10seed_experiment/configs

Reproduction and AOI adaptation
-------------------------------
Workflow role: Run or summarize repeated-seed experiments for empirical model-performance uncertainty.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--best-runs-csv``, ``--output-dir``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Regenerate selected configurations and seed outputs from the new AOI training data; do not mix summaries from different spatial splits.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from common.experiment_constants import (
    DEFAULT_BEST_RUNS_CSV,
    MODEL_FAMILY_ORDER,
    TRAINING_SCRIPT_STEMS,
    feature_set_suffix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create frozen selected-configuration YAML files for the final 10-seed experiment."
    )
    parser.add_argument(
        "--best-runs-csv",
        type=Path,
        default=Path(DEFAULT_BEST_RUNS_CSV),
        help=f"Input best-runs CSV. Default: {DEFAULT_BEST_RUNS_CSV}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/final_10seed_experiment/configs"),
        help="Output directory for YAML configs.",
    )
    return parser.parse_args()


def clean_for_yaml(value: Any) -> Any:
    if isinstance(value, list):
        return [clean_for_yaml(item) for item in value]
    if isinstance(value, tuple):
        return [clean_for_yaml(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_for_yaml(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def training_script_for(model_family: str, feature_set: str) -> str:
    stem = TRAINING_SCRIPT_STEMS[model_family]
    suffix = feature_set_suffix(feature_set)
    return f"scripts/training/{stem}_{suffix}_from_npz.py"


def main() -> None:
    args = parse_args()
    best_runs_csv = resolve_path(args.best_runs_csv)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(best_runs_csv)
    required = {
        "model_family",
        "feature_set",
        "model",
        "run_name",
        "run_dir",
        "summary_json",
        "best_val_macro_f1",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {best_runs_csv}: {sorted(missing)}")

    selected = (
        df.sort_values(["model_family", "best_val_macro_f1"], ascending=[True, False])
        .groupby("model_family", as_index=False)
        .head(1)
    )

    written = []
    for _, row in selected.iterrows():
        model_family = str(row["model_family"]).lower()
        if model_family not in MODEL_FAMILY_ORDER:
            continue

        summary_path = resolve_path(Path(str(row["summary_json"])))
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        config = {
            "model_family": model_family,
            "model": str(row["model"]),
            "feature_set": str(row["feature_set"]),
            "training_script": training_script_for(model_family, str(row["feature_set"])),
            "data": str(summary["data"]),
            "source_best_run_name": str(row["run_name"]),
            "source_best_run_dir": str(row["run_dir"]),
            "selection_metric": "best_val_macro_f1",
            "selection_metric_value": float(row["best_val_macro_f1"]),
            "args": {key: clean_for_yaml(value) for key, value in summary["args"].items()},
        }

        output_path = output_dir / f"{model_family}_best.yaml"
        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        written.append(output_path)

    manifest = pd.DataFrame(
        {
            "config_path": [str(path.relative_to(PROJECT_ROOT)) for path in written],
            "model_family": [path.stem.replace("_best", "") for path in written],
        }
    )
    manifest.to_csv(output_dir / "selected_config_manifest.csv", index=False)

    print("Wrote frozen configs:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
