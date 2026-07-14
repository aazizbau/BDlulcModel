#!/usr/bin/env python3
"""
Plot empirical error bars from final 10-seed experiment metrics.

Error bars are seed-to-seed standard deviations from the completed repeated
runs, not bootstrap intervals.

Complete Example Run
--------------------
python scripts/final_10seed_experiment/plot_10seed_errorbars.py \
    --metrics-csv outputs/final_10seed_experiment/combined/all_run_metrics.csv \
    --output-dir outputs/final_10seed_experiment/figures \
    --add-title
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.experiment_constants import DEFAULT_OUTPUT_ROOT, MODEL_FAMILY_DISPLAY, MODEL_FAMILY_ORDER

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS = {
    "overall_accuracy": "Overall Accuracy",
    "macro_f1": "Macro F1-score",
    "weighted_f1": "Weighted F1-score",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 10-seed empirical model-family error bars.")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path(DEFAULT_OUTPUT_ROOT) / "combined" / "all_run_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_ROOT) / "figures",
    )
    parser.add_argument("--add-title", action="store_true")
    return parser.parse_args()


def ordered_families(df: pd.DataFrame) -> list[str]:
    present = set(df["model_family"].astype(str))
    ordered = [family for family in MODEL_FAMILY_ORDER if family in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def summarize(df: pd.DataFrame, metric: str, families: list[str]) -> pd.DataFrame:
    rows = []
    for family in families:
        group = df[df["model_family"] == family]
        values = group[metric].astype(float) * 100.0
        rows.append(
            {
                "model_family": family,
                "label": MODEL_FAMILY_DISPLAY.get(family, family),
                "mean": values.mean(),
                "sd": values.std(ddof=1) if len(values) > 1 else 0.0,
                "n": int(values.count()),
            }
        )
    return pd.DataFrame(rows)


def add_labels(ax, bars, means, sds) -> None:
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    for bar, mean, sd in zip(bars, means, sds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + sd + 0.6,
            f"{mean:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            path_effects=halo,
        )


def plot_single_metric(df: pd.DataFrame, metric: str, output_dir: Path, add_title: bool) -> None:
    families = ordered_families(df)
    summary = summarize(df, metric, families)
    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(
        x,
        summary["mean"],
        yerr=summary["sd"],
        capsize=5,
        ecolor="black",
        color="#4C72B0",
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    add_labels(ax, bars, summary["mean"], summary["sd"])
    if add_title:
        ax.set_title(f"{METRICS[metric]} by Model Family across 10 Fixed Seeds", fontsize=14, pad=12)
    ax.set_ylabel(f"{METRICS[metric]} (%)")
    ax.set_xlabel("Model family")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.35)
    ax.set_ylim(0, max(105, float((summary["mean"] + summary["sd"]).max()) + 5))
    fig.tight_layout()
    path = output_dir / f"{metric}_mean_sd.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_grouped_metrics(df: pd.DataFrame, output_dir: Path, add_title: bool) -> None:
    families = ordered_families(df)
    x = np.arange(len(families))
    width = 0.25
    offsets = [-width, 0, width]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for offset, (metric, label), color in zip(offsets, METRICS.items(), colors):
        summary = summarize(df, metric, families)
        bars = ax.bar(
            x + offset,
            summary["mean"],
            width=width,
            yerr=summary["sd"],
            capsize=4,
            ecolor="black",
            color=color,
            label=label,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        add_labels(ax, bars, summary["mean"], summary["sd"])

    if add_title:
        ax.set_title("Final 10-Seed Test Performance with Empirical Error Bars", fontsize=14, pad=12)
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Model family")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_FAMILY_DISPLAY.get(f, f) for f in families], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    path = output_dir / "test_metrics_grouped_errorbars.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    args = parse_args()
    metrics_csv = resolve_path(args.metrics_csv)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_csv)
    if df.empty:
        raise ValueError(f"No rows found in {metrics_csv}")

    for metric in METRICS:
        plot_single_metric(df, metric, output_dir, args.add_title)
    plot_grouped_metrics(df, output_dir, args.add_title)


if __name__ == "__main__":
    main()
