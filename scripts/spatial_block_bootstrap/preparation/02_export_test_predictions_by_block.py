#!/usr/bin/env python3
"""Attach reconstructed original 1 km block IDs to saved test predictions.

Reproduction and AOI adaptation
-------------------------------
Workflow role: Estimate confidence intervals by resampling the original spatial test blocks.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--output-root``, ``--chunk``. Run the ``--help`` command below for required values, defaults, and accepted choices.
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

    python scripts/spatial_block_bootstrap/preparation/02_export_test_predictions_by_block.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from common.constants import DEFAULT_OUTPUT_ROOT, resolve_path  # noqa: E402
from common.data_utils import reconstruct_test_sample_metadata  # noqa: E402
from common.naming_utils import safe_stem  # noqa: E402
from common.output_utils import write_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chunk", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve_path(args.output_root)
    selected = pd.read_csv(output_root / "metadata" / "selected_runs.csv")
    destination = output_root / "test_predictions_by_block"
    destination.mkdir(parents=True, exist_ok=True)

    metadata_by_npz: dict[str, pd.DataFrame] = {}
    for npz_path in selected["data_npz"].drop_duplicates():
        print(f"Reconstructing original test blocks for: {npz_path}")
        metadata_by_npz[npz_path] = reconstruct_test_sample_metadata(
            npz_path, chunk_size=args.chunk
        )

    inventory_rows = []
    for row in selected.itertuples(index=False):
        sample_metadata = metadata_by_npz[row.data_npz]
        predictions = pd.read_csv(resolve_path(row.test_predictions_csv))
        if len(predictions) != len(sample_metadata):
            raise RuntimeError(
                f"Prediction count mismatch for {row.run_name}: "
                f"{len(predictions)} vs {len(sample_metadata)}."
            )
        predicted_true = predictions["y_true"].to_numpy(dtype=np.uint8)
        recovered_true = sample_metadata["true_class_id"].to_numpy(dtype=np.uint8)
        if not np.array_equal(predicted_true, recovered_true):
            raise RuntimeError(f"Saved prediction order mismatch for {row.run_name}.")

        exported = sample_metadata.copy()
        exported["pred_class_id"] = predictions["y_pred"].to_numpy(dtype=np.uint8)
        exported.insert(0, "feature_set", row.feature_set)
        exported.insert(0, "model", row.model)
        exported.insert(0, "model_family", row.model_family)
        exported.insert(0, "run_name", row.run_name)

        path = destination / row.model_family / f"{safe_stem(row.feature_set)}.parquet"
        written = write_table(exported, path)
        inventory_rows.append(
            {
                "run_name": row.run_name,
                "model_family": row.model_family,
                "feature_set": row.feature_set,
                "prediction_path": str(written),
                "test_observations": len(exported),
                "test_blocks": exported["block_id"].nunique(),
            }
        )
        print(f"Saved: {written}")

    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(output_root / "metadata" / "test_block_inventory.csv", index=False)
    print(f"Saved: {output_root / 'metadata' / 'test_block_inventory.csv'}")


if __name__ == "__main__":
    main()
