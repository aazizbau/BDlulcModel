#!/usr/bin/env python3
"""Run spatial block bootstrap for class-wise metrics of the selected best model.

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

    python scripts/spatial_block_bootstrap/bootstrap/12_run_bestmodel_classwise_bootstrap.py --help
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
from common.bootstrap_utils import aggregate_bootstrap_cms, percentile_summary  # noqa: E402
from common.constants import CLASS_IDS, CLASS_NAMES, DEFAULT_OUTPUT_ROOT  # noqa: E402
from common.metric_utils import metrics_from_cm  # noqa: E402
from common.output_utils import write_table  # noqa: E402
from common.workflow_utils import load_bootstrap_context  # noqa: E402


METRICS = {
    "Producer's Accuracy / Recall": "producer_accuracy",
    "User's Accuracy / Precision": "user_accuracy",
    "F1-score": "f1_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, selected, block_long, block_ids, indices, settings = load_bootstrap_context(
        args.output_root
    )
    best = selected[selected["use_best_overall"].astype(bool)].iloc[0]
    lower = float(settings["bootstrap"]["lower_percentile"])
    upper = float(settings["bootstrap"]["upper_percentile"])
    minimum_valid = int(settings["undefined_metric"]["minimum_valid_replicates"])

    tensor = block_tensor(block_long, best["run_name"], block_ids)
    observed_metrics = metrics_from_cm(tensor.sum(axis=0))
    bootstrap_cms = aggregate_bootstrap_cms(tensor, indices)

    distributions = []
    values_by_metric = {
        name: np.full((len(indices), len(CLASS_IDS)), np.nan, dtype=float)
        for name in METRICS
    }
    for replicate, cm in enumerate(bootstrap_cms):
        current = metrics_from_cm(cm)
        for display, key in METRICS.items():
            values_by_metric[display][replicate] = np.asarray(current[key]) * 100.0

    summary_rows = []
    for metric_name, metric_key in METRICS.items():
        values = values_by_metric[metric_name]
        observed = np.asarray(observed_metrics[metric_key]) * 100.0
        for class_index, class_id in enumerate(CLASS_IDS):
            current = values[:, class_index]
            summary = percentile_summary(current, lower, upper)
            summary_rows.append(
                {
                    "run_name": best["run_name"],
                    "model_family": best["model_family"],
                    "feature_set": best["feature_set"],
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "metric": metric_name,
                    "observed": float(observed[class_index]),
                    **summary,
                    "total_replicates": len(indices),
                    "interval_unstable": summary["valid_replicates"] < minimum_valid,
                    "n_blocks": len(block_ids),
                }
            )
            distributions.append(
                pd.DataFrame(
                    {
                        "replicate": np.arange(len(indices)),
                        "run_name": best["run_name"],
                        "model_family": best["model_family"],
                        "feature_set": best["feature_set"],
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "metric": metric_name,
                        "value": current,
                    }
                )
            )

    distribution_df = pd.concat(distributions, ignore_index=True)
    distribution_path = write_table(
        distribution_df,
        root
        / "bootstrap_distributions"
        / "bestmodel_classwise_bootstrap_metrics.parquet",
    )
    summary_path = root / "summaries" / "bestmodel_classwise_spatial_ci.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved: {distribution_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
