#!/usr/bin/env python3
"""Create complete 10x10 confusion matrices for every selected run and test block."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from common.block_confusion_utils import predictions_to_block_long  # noqa: E402
from common.constants import DEFAULT_OUTPUT_ROOT, resolve_path  # noqa: E402
from common.output_utils import read_table, write_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve_path(args.output_root)
    inventory = pd.read_csv(output_root / "metadata" / "test_block_inventory.csv")
    selected = pd.read_csv(output_root / "metadata" / "selected_runs.csv")

    parts = []
    for row in inventory.itertuples(index=False):
        prediction_path = Path(row.prediction_path)
        if prediction_path.suffixes[-2:] == [".csv", ".gz"]:
            predictions = pd.read_csv(prediction_path)
        else:
            predictions = read_table(
                output_root
                / "test_predictions_by_block"
                / row.model_family
                / f"{row.feature_set}.parquet"
            )
        print(f"Building block confusion matrices: {row.run_name}")
        parts.append(predictions_to_block_long(predictions))

    all_blocks = pd.concat(parts, ignore_index=True)
    destination = output_root / "block_confusion_matrices"
    all_path = write_table(
        all_blocks, destination / "all_selected_runs_block_confusion_long.parquet"
    )
    all_blocks.to_csv(
        destination / "all_selected_runs_block_confusion_long.csv", index=False
    )

    subsets = {
        "model_comparison_block_confusions.parquet": selected.loc[
            selected["use_model_comparison"].astype(bool), "run_name"
        ],
        "featureset_comparison_block_confusions.parquet": selected.loc[
            selected["use_featureset_comparison"].astype(bool), "run_name"
        ],
        "bestmodel_block_confusions.parquet": selected.loc[
            selected["use_best_overall"].astype(bool), "run_name"
        ],
    }
    for filename, run_names in subsets.items():
        subset = all_blocks[all_blocks["run_name"].isin(set(run_names))]
        print(f"Saved: {write_table(subset, destination / filename)}")
    print(f"Saved: {all_path}")


if __name__ == "__main__":
    main()
