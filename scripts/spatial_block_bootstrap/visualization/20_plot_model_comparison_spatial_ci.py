#!/usr/bin/env python3
"""Plot model-family metrics with paired spatial block bootstrap intervals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from common.constants import DEFAULT_OUTPUT_ROOT, MODEL_FAMILY_ORDER, resolve_path  # noqa: E402
from common.plot_utils import add_ci_labels, asymmetric_yerr  # noqa: E402


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
    output = resolve_path(args.output_plot) if args.output_plot else root / "figures" / "model_comparison_spatial_block_ci.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(root / "summaries" / "model_comparison_spatial_ci.csv")

    x = np.arange(len(MODEL_FAMILY_ORDER))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    maximum = 0.0
    for metric_index, (metric, color) in enumerate(zip(METRIC_ORDER, COLORS)):
        rows = summary[summary["metric"] == metric].set_index("model_family").loc[MODEL_FAMILY_ORDER]
        observed = rows["observed"].to_numpy(float)
        lower = rows["lower_95"].to_numpy(float)
        upper = rows["upper_95"].to_numpy(float)
        offset = (metric_index - 1) * width
        bars = ax.bar(
            x + offset,
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
            "Selected Model-Family Test Performance\n"
            f"95% confidence intervals from {n_bootstrap:,} paired spatial block bootstrap replicates",
            pad=14,
        )
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
