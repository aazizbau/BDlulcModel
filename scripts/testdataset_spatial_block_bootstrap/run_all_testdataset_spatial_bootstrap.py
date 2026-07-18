#!/usr/bin/env python3
"""
Run the test-performance-selected spatial block bootstrap workflow.

Complete example:
    python scripts/testdataset_spatial_block_bootstrap/run_all_testdataset_spatial_bootstrap.py \
        --output-root outputs/testdataset_spatial_block_bootstrap \
        --bootstrap 5000 \
        --seed 42 \
        --add-title
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
SHARED_ROOT = PROJECT_ROOT / "scripts" / "spatial_block_bootstrap"
DEFAULT_OUTPUT_ROOT = Path("outputs/testdataset_spatial_block_bootstrap")

STAGES = [
    PACKAGE_ROOT / "preparation" / "01_identify_test_selected_runs.py",
    SHARED_ROOT / "preparation" / "02_export_test_predictions_by_block.py",
    SHARED_ROOT / "preparation" / "03_create_block_confusion_matrices.py",
    SHARED_ROOT / "preparation" / "04_validate_block_confusion_matrices.py",
    SHARED_ROOT / "preparation" / "05_generate_shared_bootstrap_indices.py",
    SHARED_ROOT / "bootstrap" / "10_run_model_comparison_bootstrap.py",
    SHARED_ROOT / "bootstrap" / "11_run_featureset_comparison_bootstrap.py",
    SHARED_ROOT / "bootstrap" / "12_run_bestmodel_classwise_bootstrap.py",
    PACKAGE_ROOT / "visualization" / "20_plot_model_comparison_spatial_ci.py",
    PACKAGE_ROOT / "visualization" / "21_plot_featureset_comparison_spatial_ci.py",
    PACKAGE_ROOT / "visualization" / "22_plot_bestmodel_classwise_spatial_ci.py",
    SHARED_ROOT / "visualization" / "23_create_spatial_bootstrap_tables.py",
    PACKAGE_ROOT / "reporting" / "24_create_thesis_results_csv.py",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run test-selected spatial block bootstrap analyses."
    )
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
    bootstrap_config = PACKAGE_ROOT / "config" / "bootstrap_config.yaml"

    for script in STAGES:
        command = [
            sys.executable,
            str(script),
            "--output-root",
            str(output_root),
        ]
        if script.name == "02_export_test_predictions_by_block.py":
            command.extend(["--chunk", str(args.chunk)])
        if script.name == "05_generate_shared_bootstrap_indices.py":
            command.extend(
                [
                    "--config",
                    str(bootstrap_config),
                    "--bootstrap",
                    str(args.bootstrap),
                    "--seed",
                    str(args.seed),
                ]
            )
        if args.add_title and script.parent.name == "visualization":
            command.append("--add-title")

        print("=" * 88, flush=True)
        print("CMD:", " ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    print("Test-dataset spatial block bootstrap workflow complete.", flush=True)


if __name__ == "__main__":
    main()
