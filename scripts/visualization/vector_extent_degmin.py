#!/usr/bin/env python3
"""
Print the geographic extent of a vector layer in degree-minute format.

Input must be in EPSG:4326 (longitude/latitude in decimal degrees).

Example:
    python scripts/visualization/vector_extent_degmin.py \
        --input assets/maps/bd_coastal_map_solid_gp.gpkg

Output example:
    [2026-03-30T23:18:00+09:00] Lower latitude : 20°15.42' N
    [2026-03-30T23:18:00+09:00] Higher latitude: 23°48.17' N
    [2026-03-30T23:18:00+09:00] Lower longitude: 88°01.55' E
    [2026-03-30T23:18:00+09:00] Higher longitude: 92°41.20' E
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import geopandas as gpd


TZ = ZoneInfo("Asia/Tokyo")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ts() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{ts()}] {message}")


def deg_to_degmin(value: float, kind: str) -> str:
    """
    Convert decimal degrees to degree-minute string.

    kind:
        'lat' -> N/S
        'lon' -> E/W
    """
    if kind not in {"lat", "lon"}:
        raise ValueError("kind must be 'lat' or 'lon'")

    abs_val = abs(value)
    degrees = int(abs_val)
    minutes = (abs_val - degrees) * 60.0

    if kind == "lat":
        hemi = "N" if value >= 0 else "S"
    else:
        hemi = "E" if value >= 0 else "W"

    return f"{degrees}°{minutes:05.2f}' {hemi}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find vector extent and print lower/higher latitude/longitude in degree-minute format."
    )
    parser.add_argument(
        "--input",
        required=True,
        help='Input vector file, e.g. "assets/maps/bd_coastal_map_solid_gp.gpkg"',
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Optional layer name for multi-layer GPKG.",
    )
    args = parser.parse_args()

    input_path = resolve_path(args.input)

    log(f"Reading vector: {input_path}")
    gdf = gpd.read_file(input_path, layer=args.layer)

    if gdf.empty:
        log("ERROR: input layer is empty.")
        return 1

    if gdf.crs is None:
        log("ERROR: input has no CRS defined.")
        return 1

    epsg = gdf.crs.to_epsg()
    if epsg != 4326:
        log(f"ERROR: input CRS is {gdf.crs}, expected EPSG:4326.")
        return 1

    minx, miny, maxx, maxy = gdf.total_bounds

    log(f"Lower latitude : {deg_to_degmin(miny, 'lat')}")
    log(f"Higher latitude: {deg_to_degmin(maxy, 'lat')}")
    log(f"Lower longitude: {deg_to_degmin(minx, 'lon')}")
    log(f"Higher longitude: {deg_to_degmin(maxx, 'lon')}")

    log(
        f"Decimal degree bounds: min_lon={minx:.8f}, min_lat={miny:.8f}, "
        f"max_lon={maxx:.8f}, max_lat={maxy:.8f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
