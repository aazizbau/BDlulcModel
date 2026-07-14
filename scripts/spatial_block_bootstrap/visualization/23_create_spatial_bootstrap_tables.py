#!/usr/bin/env python3
"""Create PNG and Excel tables from spatial bootstrap confidence summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from common.constants import DEFAULT_OUTPUT_ROOT, resolve_path  # noqa: E402
from common.excel_utils import write_xlsx  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--add-title", action="store_true")
    return parser.parse_args()


def save_table_png(frame: pd.DataFrame, path: Path, title: str | None) -> None:
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        if "bootstrap" in column or column in {"n_blocks", "valid_replicates", "invalid_replicates", "total_replicates"}:
            continue
        display[column] = display[column].map(lambda value: f"{value:.2f}" if pd.notna(value) else "NA")
    width = max(12.0, 1.25 * len(display.columns))
    height = max(4.0, 0.42 * len(display) + 1.8)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, pad=14)
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.0, 1.35)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("0.75")
        if row == 0:
            cell.set_facecolor("#E6E6E6")
            cell.set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = resolve_path(args.output_root)
    summary_dir = root / "summaries"
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    model = pd.read_csv(summary_dir / "model_comparison_spatial_ci.csv")
    feature = pd.read_csv(summary_dir / "featureset_comparison_spatial_ci.csv")
    difference = pd.read_csv(summary_dir / "featureset_difference_spatial_ci.csv")
    classwise = pd.read_csv(summary_dir / "bestmodel_classwise_spatial_ci.csv")

    save_table_png(
        model,
        figure_dir / "model_comparison_spatial_block_ci_table.png",
        "Model-Family Spatial Bootstrap Confidence Intervals" if args.add_title else None,
    )
    save_table_png(
        feature,
        figure_dir / "featureset_comparison_spatial_block_ci_table.png",
        "Feature-Set Spatial Bootstrap Confidence Intervals" if args.add_title else None,
    )
    save_table_png(
        classwise,
        figure_dir / "bestmodel_classwise_spatial_block_ci_table.png",
        "Best-Model Class-wise Spatial Bootstrap Confidence Intervals" if args.add_title else None,
    )

    workbook = summary_dir / "spatial_bootstrap_summary.xlsx"
    write_xlsx(
        workbook,
        {
            "Model comparison": model,
            "Feature comparison": feature,
            "Feature differences": difference,
            "Best model classwise": classwise,
        },
    )
    print(f"Saved: {workbook}")


if __name__ == "__main__":
    main()
