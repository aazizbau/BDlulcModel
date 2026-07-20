#!/usr/bin/env python3
"""
Calculate total area from a polygon vector in square kilometers.

Input:
- vector file in EPSG:4326

Method:
- reproject to EPSG:6933 (WGS 84 / NSIDC EASE-Grid 2.0 Global)
- this CRS is equal-area, so it is preferable for area calculation
- calculate polygon area in square meters
- convert to square kilometers

Example:
python scripts/analysis/calculate_area_from_vector.py \
    --input assets/maps/bd_coastal_map_solid_gp.gpkg

Reproduction and AOI adaptation
-------------------------------
Workflow role: Derive quantitative summaries, accuracy assessments, or change statistics from prepared model outputs.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--input``, ``--layer``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace input result paths with outputs generated for the new AOI, and retain the same class-ID definitions when comparing results.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

AREA_EPSG = 6933  # WGS 84 / NSIDC EASE-Grid 2.0 Global (equal-area)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate total polygon area in square kilometers."
    )
    parser.add_argument(
        "--input",
        default="assets/maps/bd_coastal_map_solid_gp.gpkg",
        help="Input vector file path (e.g. GPKG, Shapefile, GeoJSON).",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Optional layer name for multi-layer input such as GPKG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    gdf = gpd.read_file(input_path, layer=args.layer)

    if gdf.empty:
        raise ValueError("Input vector contains no features.")

    if gdf.crs is None:
        raise ValueError("Input vector has no CRS defined.")

    if gdf.crs.to_epsg() != 4326:
        raise ValueError(
            f"Expected input CRS EPSG:4326, but got: {gdf.crs}"
        )

    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError("No valid geometry found in input.")

    if len(gdf) != 1:
        raise ValueError(
            f"Expected one polygon feature, but found {len(gdf)} features."
        )

    gdf_proj = gdf.to_crs(epsg=AREA_EPSG)

    area_m2 = float(gdf_proj.geometry.area.iloc[0])
    area_km2 = area_m2 / 1_000_000.0

    print(f"Input file        : {input_path}")
    print(f"Input CRS         : {gdf.crs}")
    print(f"Area CRS          : EPSG:{AREA_EPSG}")
    print(f"Area (m^2)        : {area_m2:,.2f}")
    print(f"Area (km^2)       : {area_km2:,.4f}")


if __name__ == "__main__":
    main()
