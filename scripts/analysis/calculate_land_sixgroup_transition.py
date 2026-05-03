#!/usr/bin/env python3
"""
Summarise the six LULC transition types (2017 → 2024) from the per-map
total CSVs produced by visualize_sixlulc_transition_2017vs2024.py.

Reads
-----
outputs/figures/lulc_transition_2017_vs_2024_map1_*_total.csv
outputs/figures/lulc_transition_2017_vs_2024_map2_*_total.csv
...
outputs/figures/lulc_transition_2017_vs_2024_map6_*_total.csv

Each CSV has three rows (class_id 0 = Unchanged, 1 = focus transition,
2 = Other Change).  This script extracts the class_id == 1 row from each
file and assembles a single summary table.

Output
------
outputs/analysis/lulc_sixgroup_transition_summary.csv

Example
-------
python scripts/analysis/calculate_land_sixgroup_transition.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = Path("outputs/figures")
DEFAULT_OUTPUT = Path("outputs/analysis/lulc_sixgroup_transition_summary.csv")

MAP_KEYS = [
    "map1_urban_infrastructure_expansion",
    "map2_rural_settlement_expansion",
    "map3_productive_land_conversion",
    "map4_water_expansion_erosion",
    "map5_ecological_recovery",
    "map6_ecological_degradation",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarise six LULC transition types from per-map total CSVs."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the per-map _total.csv files.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output summary CSV path.",
    )
    return p.parse_args()


def find_csv(input_dir: Path, map_key: str) -> Path:
    pattern = f"lulc_transition_2017_vs_2024_{map_key}_total.csv"
    candidate = input_dir / pattern
    if candidate.exists():
        return candidate
    # Fallback: glob in case filenames embed extra words
    matches = sorted(input_dir.glob(f"lulc_transition_2017_vs_2024_{map_key}*_total.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No CSV found for '{map_key}' in {input_dir}.\n"
            f"Expected: {candidate}"
        )
    return matches[0]


def extract_transition_row(csv_path: Path, map_key: str) -> dict:
    df = pd.read_csv(csv_path)
    row = df[df["is_changed_class"] == 1].iloc[0]
    # is_changed_class==1 rows: class_id==1 is focus transition, class_id==2 is Other Change.
    # We want only the focus transition (class_id==1 / class_label != "Other Change").
    focus = df[(df["is_changed_class"] == 1) & (df["class_id"] == 1)]
    if focus.empty:
        raise ValueError(f"No focus transition row (class_id=1) found in {csv_path}")
    row = focus.iloc[0]
    return {
        "map_key": map_key,
        "transition_label": row["focus_label"],
        "area_km2": round(row["area_km2"], 4),
        "percent_of_valid_area": round(row["percent_of_valid_area"], 4),
        "percent_of_changed_area": round(row["percent_of_changed_area"], 4),
        "total_valid_area_km2": round(row["total_valid_area_km2"], 4),
        "total_changed_area_km2": round(row["total_changed_area_km2"], 4),
    }


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_path = resolve_path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    rows = []
    for map_key in MAP_KEYS:
        csv_path = find_csv(input_dir, map_key)
        print(f"Reading: {csv_path.name}")
        rows.append(extract_transition_row(csv_path, map_key))

    summary = pd.DataFrame(rows)

    # Add a totals footer row
    total_row = {
        "map_key": "TOTAL",
        "transition_label": "All Six Transition Types",
        "area_km2": round(summary["area_km2"].sum(), 4),
        "percent_of_valid_area": round(summary["percent_of_valid_area"].sum(), 4),
        "percent_of_changed_area": round(summary["percent_of_changed_area"].sum(), 4),
        "total_valid_area_km2": summary["total_valid_area_km2"].iloc[0],
        "total_changed_area_km2": summary["total_changed_area_km2"].iloc[0],
    }
    summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"\nSummary ({len(summary) - 1} transitions + totals row):")
    print(summary.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
