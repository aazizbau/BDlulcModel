#!/usr/bin/env python3
"""Plot test-selected feature-set comparison with spatial bootstrap intervals."""

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
from common.plot_utils import TEXT_EFFECTS, asymmetric_yerr  # noqa: E402
from test_plot_utils import add_ci_labels  # noqa: E402


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
        else root / "figures" / "featureset_comparison_spatial_block_ci.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = (
        pd.read_csv(root / "summaries" / "featureset_comparison_spatial_ci.csv")
        .set_index("model_family")
        .loc[MODEL_FAMILY_ORDER]
    )

    x = np.arange(len(MODEL_FAMILY_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    series = [
        ("AE64", "observed_ae64", "ae64_lower_95", "ae64_upper_95", "#4C72B0", -width / 2),
        ("AE64 + 10 Indices", "observed_plusindices", "plusindices_lower_95", "plusindices_upper_95", "#DD8452", width / 2),
    ]
    maximum = 0.0
    for label, observed_col, lower_col, upper_col, color, offset in series:
        observed = summary[observed_col].to_numpy(float)
        lower = summary[lower_col].to_numpy(float)
        upper = summary[upper_col].to_numpy(float)
        bars = ax.bar(
            x + offset,
            observed,
            width,
            color=color,
            label=label,
            yerr=asymmetric_yerr(observed, lower, upper),
            capsize=4,
            ecolor="black",
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        add_ci_labels(ax, bars, observed, lower, upper, fontsize=7)
        maximum = max(maximum, float(np.nanmax(upper)))

    for index, row in enumerate(summary.itertuples()):
        annotation_y = max(row.ae64_upper_95, row.plusindices_upper_95) + 5.0
        ax.text(
            index,
            annotation_y,
            f"Delta {row.observed_delta:+.2f} pp\n"
            f"95% CI: {row.delta_lower_95:+.2f} to {row.delta_upper_95:+.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            path_effects=TEXT_EFFECTS,
        )
        maximum = max(maximum, annotation_y + 5.0)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_FAMILY_ORDER)
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_xlabel("Model Family")
    ax.set_ylim(0, max(105.0, maximum + 2.0))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10)
    if args.add_title:
        n_bootstrap = int(summary["n_bootstrap"].iloc[0])
        ax.set_title(
            "AE64 versus AE64 + 10 Spectral Indices (Test-Selected Runs)\n"
            f"95% confidence intervals from {n_bootstrap:,} paired spatial block bootstrap replicates",
            pad=14,
        )
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
