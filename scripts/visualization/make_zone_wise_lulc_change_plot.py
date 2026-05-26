#!/usr/bin/env python3
"""
Create a zone-wise grouped bar plot of major LULC class areas for 2017 and 2024.

Inputs
------
- outputs/figures/bd_coastal_infer_lulc_2017.csv
- outputs/figures/bd_coastal_infer_lulc_2024.csv

Output
------
- outputs/figures/<zone>_zone_lulc_area_2017_vs_2024.png

Example
-------
python scripts/visualization/make_zone_wise_lulc_change_plot.py \
    --zone western
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_2017 = Path("outputs/figures/bd_coastal_infer_lulc_2017.csv")
DEFAULT_CSV_2024 = Path("outputs/figures/bd_coastal_infer_lulc_2024.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/figures")
ZONE_CHOICES = ("western", "central", "eastern")

MAJOR_CLASS_ORDER = [9, 6, 4, 5, 2, 3, 1, 10]

CLASS_LABELS = {
    1: "Urban /\nInstitutional Built-up",
    2: "Rural Settlement",
    3: "Transport &\nCoastal Embankments",
    4: "Cropland",
    5: "Tree-based\nAgroforestry & Orchard",
    6: "Aquaculture &\nInland Ponds",
    9: "Mangrove Forest",
    10: "Bare / Exposed\nCoastal Land",
}

ZONE_TITLES = {
    "western": "Western Coastal Zone",
    "central": "Central Coastal Zone",
    "eastern": "Eastern Coastal Zone",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_output_path(zone: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{zone}_zone_lulc_area_2017_vs_2024.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create zone-wise LULC class area comparison plot for 2017 and 2024."
    )
    parser.add_argument("--zone", required=True, choices=ZONE_CHOICES, help="Coastal zone to plot.")
    parser.add_argument("--csv-2017", type=Path, default=DEFAULT_CSV_2017, help="2017 LULC area CSV.")
    parser.add_argument("--csv-2024", type=Path, default=DEFAULT_CSV_2024, help="2024 LULC area CSV.")
    parser.add_argument("--output-plot", type=Path, default=None, help="Optional output PNG path.")
    return parser.parse_args()


def load_zone_year(csv_path: Path, zone: str, year: int) -> pd.Series:
    df = pd.read_csv(csv_path)
    required = {"year", "zone", "class_id", "area_km2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    zone_df = df[(df["year"] == year) & (df["zone"].astype(str).str.lower() == zone)].copy()
    if zone_df.empty:
        raise ValueError(f"No rows found for zone={zone!r}, year={year} in {csv_path}")

    zone_df["class_id"] = zone_df["class_id"].astype(int)
    return zone_df.set_index("class_id")["area_km2"].reindex(MAJOR_CLASS_ORDER, fill_value=0.0)


def add_bar_labels(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(20, height * 0.015),
            f"{height:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    args = parse_args()
    csv_2017 = resolve_path(args.csv_2017)
    csv_2024 = resolve_path(args.csv_2024)
    output_plot = resolve_path(args.output_plot or default_output_path(args.zone))

    area_2017 = load_zone_year(csv_2017, args.zone, 2017)
    area_2024 = load_zone_year(csv_2024, args.zone, 2024)

    x = np.arange(len(MAJOR_CLASS_ORDER))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 7))
    bars_2017 = ax.bar(x - width / 2, area_2017.values, width=width, label="2017")
    bars_2024 = ax.bar(x + width / 2, area_2024.values, width=width, label="2024")

    add_bar_labels(ax, bars_2017)
    add_bar_labels(ax, bars_2024)

    ax.set_title(f"Major LULC Class Areas in the {ZONE_TITLES[args.zone]}, 2017 and 2024", fontsize=14, pad=14)
    ax.set_ylabel("Area (km²)", fontsize=12)
    ax.set_xlabel("LULC class", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in MAJOR_CLASS_ORDER], rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(title="Year")

    max_height = max(float(area_2017.max()), float(area_2024.max()), 1.0)
    ax.set_ylim(0, max_height * 1.12)

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_plot}")


if __name__ == "__main__":
    main()
