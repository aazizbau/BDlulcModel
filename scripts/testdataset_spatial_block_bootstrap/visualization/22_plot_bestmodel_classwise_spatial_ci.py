#!/usr/bin/env python3
"""Plot test-selected best-model class metrics with spatial bootstrap intervals."""

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

from common.constants import CLASS_IDS, DEFAULT_OUTPUT_ROOT, resolve_path  # noqa: E402
from common.plot_utils import asymmetric_yerr, wrap_label  # noqa: E402
from test_plot_utils import add_ci_labels  # noqa: E402


METRICS = [
    "Producer's Accuracy / Recall",
    "User's Accuracy / Precision",
    "F1-score",
]
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
        else root / "figures" / "bestmodel_classwise_spatial_block_ci.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(root / "summaries" / "bestmodel_classwise_spatial_ci.csv")

    x = np.arange(len(CLASS_IDS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(15.5, 7.5))
    maximum = 0.0
    for metric_index, (metric, color) in enumerate(zip(METRICS, COLORS)):
        rows = summary[summary["metric"] == metric].set_index("class_id").loc[CLASS_IDS]
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
        add_ci_labels(ax, bars, observed, lower, upper, fontsize=5.8)
        maximum = max(maximum, float(np.nanmax(upper)))

    class_rows = summary.drop_duplicates("class_id").set_index("class_id").loc[CLASS_IDS]
    labels = [
        f"{class_id}\n{wrap_label(class_rows.loc[class_id, 'class_name'], 18)}"
        for class_id in CLASS_IDS
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy / Score (%)")
    ax.set_xlabel("LULC Class")
    ax.set_ylim(0, max(105.0, maximum + 8.0))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=10)
    if args.add_title:
        first = summary.iloc[0]
        ax.set_title(
            "Class-wise Accuracy Metrics for the Test-Selected Best Model\n"
            f"{first['model_family']} | {first['feature_set']} | "
            "spatial block bootstrap 95% confidence intervals",
            pad=14,
        )
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
