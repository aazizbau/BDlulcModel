#!/usr/bin/env python3
"""
Run the complete spatial block bootstrap workflow.

Complete example:
    python scripts/spatial_block_bootstrap/run_all_spatial_bootstrap.py \
        --output-root outputs/spatial_block_bootstrap \
        --bootstrap 5000 \
        --seed 42 \
        --add-title

Reproduction and AOI adaptation
-------------------------------
Workflow role: Estimate confidence intervals by resampling the original spatial test blocks.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--output-root``, ``--bootstrap``, ``--seed``, ``--chunk``, ``--add-title``, ``--dry-run``. Run the ``--help`` command below for required values, defaults, and accepted choices.
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
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common.constants import DEFAULT_OUTPUT_ROOT, PROJECT_ROOT, resolve_path


STAGES = [
    "preparation/01_identify_selected_runs.py",
    "preparation/02_export_test_predictions_by_block.py",
    "preparation/03_create_block_confusion_matrices.py",
    "preparation/04_validate_block_confusion_matrices.py",
    "preparation/05_generate_shared_bootstrap_indices.py",
    "bootstrap/10_run_model_comparison_bootstrap.py",
    "bootstrap/11_run_featureset_comparison_bootstrap.py",
    "bootstrap/12_run_bestmodel_classwise_bootstrap.py",
    "visualization/20_plot_model_comparison_spatial_ci.py",
    "visualization/21_plot_featureset_comparison_spatial_ci.py",
    "visualization/22_plot_bestmodel_classwise_spatial_ci.py",
    "visualization/23_create_spatial_bootstrap_tables.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all spatial block bootstrap stages.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--add-title", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve_path(args.output_root)
    package_root = Path(__file__).resolve().parent

    for relative_script in STAGES:
        command = [
            sys.executable,
            str(package_root / relative_script),
            "--output-root",
            str(output_root),
        ]
        if relative_script.endswith("02_export_test_predictions_by_block.py"):
            command.extend(["--chunk", str(args.chunk)])
        if relative_script.endswith("05_generate_shared_bootstrap_indices.py"):
            command.extend(
                ["--bootstrap", str(args.bootstrap), "--seed", str(args.seed)]
            )
        if args.add_title and relative_script.startswith("visualization/"):
            command.append("--add-title")

        print("=" * 88, flush=True)
        print("CMD:", " ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    print("Spatial block bootstrap workflow complete.", flush=True)


if __name__ == "__main__":
    main()
