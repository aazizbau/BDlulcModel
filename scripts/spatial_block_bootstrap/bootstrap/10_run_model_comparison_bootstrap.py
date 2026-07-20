#!/usr/bin/env python3
"""Run paired spatial block bootstrap for selected model-family configurations.

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

    python scripts/spatial_block_bootstrap/bootstrap/10_run_model_comparison_bootstrap.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from common.block_confusion_utils import block_tensor  # noqa: E402
from common.bootstrap_utils import percentile_summary, scalar_distribution  # noqa: E402
from common.constants import DEFAULT_OUTPUT_ROOT  # noqa: E402
from common.metric_utils import metrics_from_cm, scalar_metric_rows  # noqa: E402
from common.output_utils import write_table  # noqa: E402
from common.workflow_utils import load_bootstrap_context  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, selected, block_long, block_ids, indices, settings = load_bootstrap_context(
        args.output_root
    )
    selected = selected[selected["use_model_comparison"].astype(bool)]
    lower = float(settings["bootstrap"]["lower_percentile"])
    upper = float(settings["bootstrap"]["upper_percentile"])

    distributions = []
    summaries = []
    for row in selected.itertuples(index=False):
        tensor = block_tensor(block_long, row.run_name, block_ids)
        observed = scalar_metric_rows(metrics_from_cm(tensor.sum(axis=0)))
        distribution = scalar_distribution(tensor, indices)
        distribution.insert(1, "run_name", row.run_name)
        distribution.insert(2, "model_family", row.model_family)
        distribution.insert(3, "feature_set", row.feature_set)
        distributions.append(distribution)

        for metric, observed_value in observed.items():
            values = distribution.loc[distribution["metric"] == metric, "value"]
            summary = percentile_summary(values.to_numpy(), lower, upper)
            summaries.append(
                {
                    "model_family": row.model_family,
                    "run_name": row.run_name,
                    "feature_set": row.feature_set,
                    "metric": metric,
                    "observed": observed_value * 100.0,
                    **summary,
                    "n_blocks": len(block_ids),
                    "n_bootstrap": len(indices),
                }
            )
        print(f"Completed: {row.model_family} / {row.feature_set}")

    distribution_df = pd.concat(distributions, ignore_index=True)
    distribution_path = write_table(
        distribution_df,
        root
        / "bootstrap_distributions"
        / "model_comparison_bootstrap_metrics.parquet",
    )
    summary_path = root / "summaries" / "model_comparison_spatial_ci.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Saved: {distribution_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
