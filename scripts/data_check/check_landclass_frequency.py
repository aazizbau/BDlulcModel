"""
Check hierarchical land-class frequency for an upazila GPKG.

Example use: 
python scripts/data_check/check_landclass_frequency.py --upazila manpura

"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd


UPAZILA_GPKG = {
    "manpura": Path("assets/maps/manpura_landuse.gpkg"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize maj_class/sub_class frequencies for a landuse GPKG."
    )
    parser.add_argument(
        "--upazila",
        choices=sorted(UPAZILA_GPKG.keys()),
        default="manpura",
        help="Upazila name to select the GPKG (default: manpura).",
    )
    parser.add_argument(
        "--gpkg",
        type=Path,
        default=None,
        help="Optional explicit GPKG path (overrides --upazila).",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Layer name for multi-layer geopackages (default: first layer).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("maj_sub_class_frequency.csv"),
        help="CSV output path (default: maj_sub_class_frequency.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpkg_path = args.gpkg or UPAZILA_GPKG[args.upazila]
    if not gpkg_path.exists():
        raise SystemExit(f"GPKG not found: {gpkg_path}")

    gdf = gpd.read_file(gpkg_path, layer=args.layer)
    if "maj_class" not in gdf.columns or "sub_class" not in gdf.columns:
        raise SystemExit("Expected columns 'maj_class' and 'sub_class' in the GPKG.")

    freq = (
        gdf[["maj_class", "sub_class"]]
        .dropna()
        .groupby(["maj_class", "sub_class"])
        .size()
        .reset_index(name="count")
        .sort_values(["maj_class", "count"], ascending=[True, False])
    )
    freq["percent_within_maj"] = (
        freq["count"]
        / freq.groupby("maj_class")["count"].transform("sum")
        * 100
    ).round(2)

    print("\n=== MAJ_CLASS → SUB_CLASS (WITH FREQUENCY) ===")
    print(freq.to_string(index=False))
    freq.to_csv(args.output, index=False)
    print(f"\nSaved CSV to {args.output}")


if __name__ == "__main__":
    main()
