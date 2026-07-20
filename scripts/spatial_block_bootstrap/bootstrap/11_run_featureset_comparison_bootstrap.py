#!/usr/bin/env python3
"""Run paired spatial block bootstrap for AE64 versus AE64 plus indices.

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

    python scripts/spatial_block_bootstrap/bootstrap/11_run_featureset_comparison_bootstrap.py --help
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
from common.constants import DEFAULT_OUTPUT_ROOT, MODEL_FAMILY_ORDER  # noqa: E402
from common.metric_utils import metrics_from_cm  # noqa: E402
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
    selected = selected[selected["use_featureset_comparison"].astype(bool)]
    lower = float(settings["bootstrap"]["lower_percentile"])
    upper = float(settings["bootstrap"]["upper_percentile"])

    distributions = []
    observed_lookup: dict[tuple[str, str], float] = {}
    for row in selected.itertuples(index=False):
        tensor = block_tensor(block_long, row.run_name, block_ids)
        observed_lookup[(row.model_family, row.feature_set)] = (
            float(metrics_from_cm(tensor.sum(axis=0))["overall_accuracy"]) * 100.0
        )
        distribution = scalar_distribution(tensor, indices)
        distribution = distribution[distribution["metric"] == "Overall Accuracy"].copy()
        distribution.insert(1, "run_name", row.run_name)
        distribution.insert(2, "model_family", row.model_family)
        distribution.insert(3, "feature_set", row.feature_set)
        distributions.append(distribution)
        print(f"Completed: {row.model_family} / {row.feature_set}")

    distribution_df = pd.concat(distributions, ignore_index=True)
    summary_rows = []
    difference_parts = []
    for family in MODEL_FAMILY_ORDER:
        family_values = distribution_df[distribution_df["model_family"] == family]
        ae64 = family_values[family_values["feature_set"] == "AE64"].set_index(
            "replicate"
        )["value"]
        plus = family_values[
            family_values["feature_set"] == "AE64_plus10indices"
        ].set_index("replicate")["value"]
        paired = pd.concat([ae64.rename("ae64"), plus.rename("plusindices")], axis=1)
        paired["difference"] = paired["plusindices"] - paired["ae64"]
        paired = paired.reset_index()
        paired.insert(1, "model_family", family)
        difference_parts.append(paired)

        ae_summary = percentile_summary(paired["ae64"].to_numpy(), lower, upper)
        plus_summary = percentile_summary(
            paired["plusindices"].to_numpy(), lower, upper
        )
        delta_summary = percentile_summary(
            paired["difference"].to_numpy(), lower, upper
        )
        observed_ae = observed_lookup[(family, "AE64")]
        observed_plus = observed_lookup[(family, "AE64_plus10indices")]
        summary_rows.append(
            {
                "model_family": family,
                "observed_ae64": observed_ae,
                "ae64_bootstrap_mean": ae_summary["bootstrap_mean"],
                "ae64_lower_95": ae_summary["lower_95"],
                "ae64_upper_95": ae_summary["upper_95"],
                "observed_plusindices": observed_plus,
                "plusindices_bootstrap_mean": plus_summary["bootstrap_mean"],
                "plusindices_lower_95": plus_summary["lower_95"],
                "plusindices_upper_95": plus_summary["upper_95"],
                "observed_delta": observed_plus - observed_ae,
                "delta_bootstrap_mean": delta_summary["bootstrap_mean"],
                "delta_lower_95": delta_summary["lower_95"],
                "delta_upper_95": delta_summary["upper_95"],
                "n_blocks": len(block_ids),
                "n_bootstrap": len(indices),
            }
        )

    distribution_path = write_table(
        distribution_df,
        root / "bootstrap_distributions" / "featureset_bootstrap_metrics.parquet",
    )
    difference_df = pd.concat(difference_parts, ignore_index=True)
    difference_path = write_table(
        difference_df,
        root
        / "bootstrap_distributions"
        / "featureset_difference_bootstrap_metrics.parquet",
    )
    summary = pd.DataFrame(summary_rows)
    summary_dir = root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_dir / "featureset_comparison_spatial_ci.csv", index=False)
    summary[
        [
            "model_family",
            "observed_delta",
            "delta_bootstrap_mean",
            "delta_lower_95",
            "delta_upper_95",
            "n_blocks",
            "n_bootstrap",
        ]
    ].to_csv(summary_dir / "featureset_difference_spatial_ci.csv", index=False)
    print(f"Saved: {distribution_path}")
    print(f"Saved: {difference_path}")


if __name__ == "__main__":
    main()
