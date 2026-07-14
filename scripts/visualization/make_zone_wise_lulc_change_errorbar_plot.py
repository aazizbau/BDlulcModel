#!/usr/bin/env python3
"""
Create a zone-wise grouped bar plot of major LULC class areas for 2017 and
2024 with bootstrap error bars.

The uncertainty interval is estimated from the zone-wise class pixel counts.
For each year, class counts are resampled using a multinomial distribution,
converted back to area, and summarized with 2.5th and 97.5th percentiles.

Inputs
------
- outputs/figures/bd_coastal_infer_lulc_2017.csv
- outputs/figures/bd_coastal_infer_lulc_2024.csv

Outputs
-------
- outputs/figures/<zone>_zone_lulc_area_2017_vs_2024_errorbar.png
- outputs/figures/<zone>_zone_lulc_area_2017_vs_2024_errorbar.csv

Example
-------
python scripts/visualization/make_zone_wise_lulc_change_errorbar_plot.py \
    --zone western \
    --add-title

Complete Example Run
--------------------
python scripts/visualization/make_zone_wise_lulc_change_errorbar_plot.py \
    --zone western \
    --add-title \
    --output-plot outputs/figures/western_zone_lulc_area_2017_vs_2024_errorbar.png \
    --output-csv outputs/figures/western_zone_lulc_area_2017_vs_2024_errorbar.csv \
    --bootstrap 1000 \
    --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patheffects as pe
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


def default_output_plot_path(zone: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{zone}_zone_lulc_area_2017_vs_2024_errorbar.png"


def default_output_csv_path(zone: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{zone}_zone_lulc_area_2017_vs_2024_errorbar.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create zone-wise LULC class area comparison plot for 2017 and "
            "2024 with bootstrap error bars."
        )
    )
    parser.add_argument("--zone", required=True, choices=ZONE_CHOICES, help="Coastal zone to plot.")
    parser.add_argument("--csv-2017", type=Path, default=DEFAULT_CSV_2017, help="2017 LULC area CSV.")
    parser.add_argument("--csv-2024", type=Path, default=DEFAULT_CSV_2024, help="2024 LULC area CSV.")
    parser.add_argument(
        "--output-plot",
        "--outptut-plot",
        dest="output_plot",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Default: "
            "outputs/figures/<zone>_zone_lulc_area_2017_vs_2024_errorbar.png"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Default: "
            "outputs/figures/<zone>_zone_lulc_area_2017_vs_2024_errorbar.csv"
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap replicates for confidence intervals (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic bootstrap sampling (default: 42).",
    )
    parser.add_argument("--add-title", action="store_true", help="Show title on top of the plot.")
    return parser.parse_args()


def load_zone_year(csv_path: Path, zone: str, year: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "year",
        "zone",
        "class_id",
        "class_name",
        "pixel_count",
        "area_km2",
        "zone_lulc_pixel_count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    zone_df = df[(df["year"] == year) & (df["zone"].astype(str).str.lower() == zone)].copy()
    if zone_df.empty:
        raise ValueError(f"No rows found for zone={zone!r}, year={year} in {csv_path}")

    zone_df["class_id"] = zone_df["class_id"].astype(int)
    return zone_df.sort_values("class_id")


def pixel_area_km2(zone_df: pd.DataFrame) -> float:
    valid = zone_df[zone_df["pixel_count"] > 0].copy()
    if valid.empty:
        return 0.0001
    return float((valid["area_km2"] / valid["pixel_count"]).median())


def summarize_zone_year(
    zone_df: pd.DataFrame,
    zone: str,
    year: int,
    bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    all_classes = sorted(zone_df["class_id"].astype(int).unique())
    class_to_index = {class_id: index for index, class_id in enumerate(all_classes)}

    counts = zone_df.set_index("class_id")["pixel_count"].reindex(all_classes, fill_value=0).to_numpy(dtype=int)
    total_count = int(counts.sum())
    if total_count <= 0:
        raise ValueError(f"No pixels found for zone={zone!r}, year={year}.")

    probabilities = counts / total_count
    area_per_pixel = pixel_area_km2(zone_df)
    samples = rng.multinomial(total_count, probabilities, size=bootstrap)
    sample_areas = samples.astype(float) * area_per_pixel
    observed_area = counts.astype(float) * area_per_pixel

    class_names = zone_df.set_index("class_id")["class_name"].to_dict()
    zone_total_from_csv = int(zone_df["zone_lulc_pixel_count"].iloc[0])

    rows = []
    for class_id in MAJOR_CLASS_ORDER:
        index = class_to_index.get(class_id)
        if index is None:
            lower_ci = upper_ci = area_km2 = 0.0
            pixel_count = 0
        else:
            lower_ci, upper_ci = np.percentile(sample_areas[:, index], [2.5, 97.5])
            area_km2 = observed_area[index]
            pixel_count = int(counts[index])

        rows.append(
            {
                "zone": zone,
                "year": year,
                "class_id": class_id,
                "class_name": class_names.get(class_id, CLASS_LABELS[class_id].replace("\n", " ")),
                "area_km2": area_km2,
                "area_lower_95_km2": float(lower_ci),
                "area_upper_95_km2": float(upper_ci),
                "pixel_count": pixel_count,
                "zone_lulc_pixel_count": zone_total_from_csv,
                "pixel_area_km2": area_per_pixel,
            }
        )

    return pd.DataFrame(rows)


def add_bar_labels(ax, bars, upper_values) -> None:
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    for bar, upper in zip(bars, upper_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(height, upper) + max(20, height * 0.015),
            f"{height:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            path_effects=halo,
        )


def main() -> None:
    args = parse_args()
    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be greater than 0.")

    csv_2017 = resolve_path(args.csv_2017)
    csv_2024 = resolve_path(args.csv_2024)
    output_plot = resolve_path(args.output_plot or default_output_plot_path(args.zone))
    output_csv = resolve_path(args.output_csv or default_output_csv_path(args.zone))

    rng = np.random.default_rng(args.seed)

    summary_2017 = summarize_zone_year(
        load_zone_year(csv_2017, args.zone, 2017),
        args.zone,
        2017,
        args.bootstrap,
        rng,
    )
    summary_2024 = summarize_zone_year(
        load_zone_year(csv_2024, args.zone, 2024),
        args.zone,
        2024,
        args.bootstrap,
        rng,
    )
    summary_df = pd.concat([summary_2017, summary_2024], ignore_index=True)

    area_2017 = summary_2017["area_km2"].to_numpy()
    area_2024 = summary_2024["area_km2"].to_numpy()
    lower_2017 = summary_2017["area_lower_95_km2"].to_numpy()
    upper_2017 = summary_2017["area_upper_95_km2"].to_numpy()
    lower_2024 = summary_2024["area_lower_95_km2"].to_numpy()
    upper_2024 = summary_2024["area_upper_95_km2"].to_numpy()

    x = np.arange(len(MAJOR_CLASS_ORDER))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 7))
    bars_2017 = ax.bar(
        x - width / 2,
        area_2017,
        width=width,
        yerr=np.vstack([area_2017 - lower_2017, upper_2017 - area_2017]),
        capsize=4,
        ecolor="black",
        error_kw={"elinewidth": 1.1, "capthick": 1.1},
        label="2017",
    )
    bars_2024 = ax.bar(
        x + width / 2,
        area_2024,
        width=width,
        yerr=np.vstack([area_2024 - lower_2024, upper_2024 - area_2024]),
        capsize=4,
        ecolor="black",
        error_kw={"elinewidth": 1.1, "capthick": 1.1},
        label="2024",
    )

    add_bar_labels(ax, bars_2017, upper_2017)
    add_bar_labels(ax, bars_2024, upper_2024)

    if args.add_title:
        ax.set_title(
            f"Major LULC Class Areas in the {ZONE_TITLES[args.zone]}, 2017 and 2024",
            fontsize=14,
            pad=14,
        )
    ax.set_ylabel("Area (km²)", fontsize=12)
    ax.set_xlabel("LULC class", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in MAJOR_CLASS_ORDER], rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(title="Year", fontsize=12, title_fontsize=12)

    max_height = max(float(upper_2017.max()), float(upper_2024.max()), 1.0)
    ax.set_ylim(0, max_height * 1.12)

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)

    fig.tight_layout()
    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {output_plot}")
    print(f"Saved CSV : {output_csv}")


if __name__ == "__main__":
    main()
