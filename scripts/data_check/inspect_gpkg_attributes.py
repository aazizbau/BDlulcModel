#!/usr/bin/env python3
"""
Inspect attribute (non-geometry) fields of a GeoPackage (GPKG).

Usage examples:
  python scripts/data_check/inspect_gpkg_attributes.py \
      --input assets/maps/manpura_landuse.gpkg

  python scripts/data_check/inspect_gpkg_attributes.py \
      --input assets/maps/manpura_landuse.gpkg \
      --layer landuse \
      --head 10 \
      --export-csv outputs/manpura_attributes.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect attribute fields of a GPKG file")
    parser.add_argument("--input", required=True, help="Path to input GPKG file")
    parser.add_argument("--layer", default=None, help="Layer name (optional)")
    parser.add_argument("--head", type=int, default=5, help="Number of rows to preview")
    parser.add_argument("--export-csv", default=None, help="Export attribute table to CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"GPKG not found: {input_path}")

    if args.layer:
        gdf = gpd.read_file(input_path, layer=args.layer)
    else:
        gdf = gpd.read_file(input_path)

    print("\n==============================")
    print(f"File   : {input_path}")
    print(f"Rows   : {len(gdf)}")
    print(f"CRS    : {gdf.crs}")
    print("==============================\n")

    attr_cols = [c for c in gdf.columns if c != gdf.geometry.name]

    print("ATTRIBUTE FIELDS:")
    for col in attr_cols:
        print(f" - {col:<25} dtype={gdf[col].dtype}")

    print("\n------------------------------")
    print(f"PREVIEW (first {args.head} rows)")
    print("------------------------------")
    print(gdf[attr_cols].head(args.head))

    for key in ["maj_class", "sub_class", "class10_name", "class10_id"]:
        if key in gdf.columns:
            print("\n------------------------------")
            print(f"UNIQUE VALUES: {key}")
            print("------------------------------")
            print(gdf[key].value_counts())

    if args.export_csv:
        out_csv = Path(args.export_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        gdf[attr_cols].to_csv(out_csv, index=False)
        print(f"\n[OK] Attribute table exported to: {out_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
