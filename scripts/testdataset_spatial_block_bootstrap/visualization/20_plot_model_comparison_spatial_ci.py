#!/usr/bin/env python3
"""Plot test-selected model-family metrics with spatial bootstrap intervals.

Reproduction and AOI adaptation
-------------------------------
Workflow role: Produce the test-selected spatial-block uncertainty analysis used for descriptive thesis results.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--output-root``, ``--output-plot``, ``--add-title``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Regenerate test-selected run metadata and all block-level predictions for the new AOI. Treat test-selected intervals as descriptive, not unbiased model-selection evidence.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.

Reproducible invocation
~~~~~~~~~~~~~~~~~~~~~~~
Inspect the complete interface before supplying AOI-specific paths::

    python scripts/testdataset_spatial_block_bootstrap/visualization/20_plot_model_comparison_spatial_ci.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SHARED_ROOT = Path(__file__).resolve().parents[2] / "spatial_block_bootstrap"
sys.path.insert(0, str(SHARED_ROOT))

from common.constants import DEFAULT_OUTPUT_ROOT, MODEL_FAMILY_ORDER, resolve_path  # noqa: E402
from common.plot_utils import asymmetric_yerr  # noqa: E402
from test_plot_utils import (  # noqa: E402
    add_ci_labels,
    double_figure_text,
    double_height_figsize,
)


METRIC_ORDER = ["Overall Accuracy", "Macro F1-score", "Weighted F1-score"]
COLORS = ["#2878B5", "#F28E2B", "#2CA02C"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-plot", type=Path, default=None)
    parser.add_argument("--add-title", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_path(args.output_root)
    output = (
        resolve_path(args.output_plot)
        if args.output_plot
        else root / "figures" / "model_comparison_spatial_block_ci.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(root / "summaries" / "model_comparison_spatial_ci.csv")

    x = np.arange(len(MODEL_FAMILY_ORDER))
    width = 0.25
    fig, ax = plt.subplots(figsize=double_height_figsize((13.5, 7.2)))
    maximum = 0.0
    for metric_index, (metric, color) in enumerate(zip(METRIC_ORDER, COLORS)):
        rows = (
            summary[summary["metric"] == metric]
            .set_index("model_family")
            .loc[MODEL_FAMILY_ORDER]
        )
        observed = rows["observed"].to_numpy(float)
        lower = rows["lower_95"].to_numpy(float)
        upper = rows["upper_95"].to_numpy(float)
        bars = ax.bar(
            x + (metric_index - 1) * width,
            observed,
            width,
            color=color,
            label=metric,
            yerr=asymmetric_yerr(observed, lower, upper),
            capsize=3,
            ecolor="black",
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        add_ci_labels(ax, bars, observed, lower, upper)
        maximum = max(maximum, float(np.nanmax(upper)))

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_FAMILY_ORDER)
    ax.set_ylabel("Accuracy / Score (%)")
    ax.set_xlabel("Model Family")
    ax.set_ylim(0, max(105.0, maximum + 8.0))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=10)
    if args.add_title:
        n_bootstrap = int(summary["n_bootstrap"].iloc[0])
        ax.set_title(
            "Test-Selected Model-Family Test Performance\n"
            f"95% confidence intervals from {n_bootstrap:,} paired\n"
            "spatial block bootstrap replicates",
            pad=14,
        )
    double_figure_text(fig)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
