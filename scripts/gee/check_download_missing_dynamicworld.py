#!/usr/bin/env python3
"""
Check which Dynamic World yearly-mode tiles are missing for the BD coastal AOI
and download only the missing ones.

This script follows the same AOI, tiling, naming, and export logic as:
    scripts/gee/download_dynamicworld_lulc_yearly_mode.py

Usage
-----
python scripts/gee/check_download_missing_dynamicworld.py \
    --year 2017 \
    --project ee-project-id \
    --aoi configs/bd_coastal_aoi.yaml \
    --output data/raw/dynamicworld/bd_coastal_dynamicworld_2017_mode.tif

Dry-run example
---------------
python scripts/gee/check_download_missing_dynamicworld.py \
    --year 2017 \
    --project ee-project-id \
    --aoi configs/bd_coastal_aoi.yaml \
    --output data/raw/dynamicworld/bd_coastal_dynamicworld_2017_mode.tif \
    --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator, Sequence, Tuple

import ee
import geemap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gee.aoi import AOI, load_aoi


DATASET_ID = "GOOGLE/DYNAMICWORLD/V1"
BAND_NAME = "label"
GEE_PROJECT_ENV = "GEE_PROJECT_ID"
DYNAMICWORLD_NODATA = -1


def ts() -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Dynamic World yearly-mode tile coverage and download only missing tiles."
    )
    parser.add_argument("--year", type=int, required=True, help="Target year.")
    parser.add_argument(
        "--aoi",
        type=Path,
        default=Path("configs/bd_coastal_aoi.yaml"),
        help="AOI YAML config path. Default: configs/bd_coastal_aoi.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Base output GeoTIFF path. Each tile appends _rXXX_cXXX before the suffix, "
            'e.g. "bd_coastal_dynamicworld_2017_mode_r000_c000.tif".'
        ),
    )
    parser.add_argument("--crs", default="EPSG:4326", help="Output CRS passed to Earth Engine download.")
    parser.add_argument("--scale", type=int, default=10, help="Output pixel size.")
    parser.add_argument(
        "--project",
        default=os.environ.get(GEE_PROJECT_ENV),
        help=f'Optional Earth Engine project ID (default: env "{GEE_PROJECT_ENV}").',
    )
    parser.add_argument("--tile-deg", type=float, default=0.25, help="Tile size in degrees. Default: 0.25.")
    parser.add_argument("--dry-run", action="store_true", help="Only report missing tiles without downloading.")
    return parser.parse_args(argv)


def initialize_earth_engine(project: str | None = None) -> None:
    project = project or os.environ.get(GEE_PROJECT_ENV)
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except ee.EEException:
        log("Earth Engine not initialized; attempting authentication.")
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    else:
        log("Google Earth Engine initialized.")


def build_aoi(aoi: AOI) -> ee.Geometry:
    return ee.Geometry.Polygon([aoi.bbox_polygon()], proj="EPSG:4326", geodesic=False)


def build_yearly_mode_image(year: int, geometry: ee.Geometry) -> ee.Image:
    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"
    collection = (
        ee.ImageCollection(DATASET_ID)
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
    )
    size = collection.size().getInfo()
    if size == 0:
        raise RuntimeError(f"No Dynamic World images found for year {year}.")
    log(f"Found {size} Dynamic World images for {year}. Building yearly mode composite ...")
    return (
        collection.select(BAND_NAME)
        .mode()
        .rename(BAND_NAME)
        .clip(geometry)
        .unmask(DYNAMICWORLD_NODATA)
        .toInt16()
    )


def get_bounds_from_polygon(polygon: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    lons = [pt[0] for pt in polygon]
    lats = [pt[1] for pt in polygon]
    return min(lons), min(lats), max(lons), max(lats)


def iterate_tiles(
    polygon: Sequence[Sequence[float]],
    geometry: ee.Geometry,
    tile_deg: float,
) -> Iterator[Tuple[int, int, Tuple[float, float, float, float]]]:
    min_lon, min_lat, max_lon, max_lat = get_bounds_from_polygon(polygon)
    if tile_deg <= 0:
        raise ValueError("Tile size must be greater than zero.")

    row = 0
    lat_start = min_lat
    while lat_start < max_lat:
        lat_end = min(lat_start + tile_deg, max_lat)
        col = 0
        lon_start = min_lon
        while lon_start < max_lon:
            lon_end = min(lon_start + tile_deg, max_lon)
            rect = ee.Geometry.Rectangle(
                [lon_start, lat_start, lon_end, lat_end],
                proj="EPSG:4326",
                geodesic=False,
            )
            tile_region = rect.intersection(geometry, ee.ErrorMargin(1))
            area = tile_region.area(maxError=1).getInfo()
            if area and area > 0:
                yield row, col, (lon_start, lat_start, lon_end, lat_end)
            lon_start += tile_deg
            col += 1
        lat_start += tile_deg
        row += 1


def tile_output_path(output_base: Path, row: int, col: int) -> Path:
    suffix = output_base.suffix or ".tif"
    stem = output_base.stem if output_base.suffix else output_base.name
    parent = output_base.parent if output_base.suffix else output_base
    return parent / f"{stem}_r{row:03d}_c{col:03d}{suffix}"


def make_tile_geometry(bounds: Tuple[float, float, float, float]) -> ee.Geometry:
    lon_start, lat_start, lon_end, lat_end = bounds
    return ee.Geometry.Rectangle(
        [lon_start, lat_start, lon_end, lat_end],
        proj="EPSG:4326",
        geodesic=False,
    )


def is_missing_tile(path: Path) -> bool:
    return (not path.exists()) or path.stat().st_size == 0


def export_tile(
    image: ee.Image,
    tile_region: ee.Geometry,
    tile_path: Path,
    crs: str,
    scale: int,
) -> None:
    tile_image = image.clip(tile_region).unmask(DYNAMICWORLD_NODATA).toInt16()
    geemap.ee_export_image(
        tile_image,
        filename=str(tile_path),
        region=tile_region,
        crs=crs,
        scale=scale,
        file_per_band=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    aoi_path = resolve_path(args.aoi)
    output_base = resolve_path(args.output)

    if args.tile_deg <= 0:
        log("ERROR: --tile-deg must be > 0.")
        return 1

    output_base.parent.mkdir(parents=True, exist_ok=True)

    aoi = load_aoi(aoi_path)
    initialize_earth_engine(project=args.project)
    geometry = build_aoi(aoi)
    tiles = list(iterate_tiles(aoi.bbox_polygon(), geometry, args.tile_deg))

    missing_tiles = []
    for row, col, tile_bounds in tiles:
        tile_path = tile_output_path(output_base, row, col)
        if is_missing_tile(tile_path):
            missing_tiles.append((row, col, tile_bounds, tile_path))

    log(f"AOI config: {aoi_path}")
    log(f"AOI name: {aoi.name}")
    log(f"Dataset: {DATASET_ID}")
    log(f"Band: {BAND_NAME}")
    log(f"Year: {args.year}")
    log(f"Output base: {output_base}")
    log(f"Output CRS: {args.crs}")
    log(f"Scale: {args.scale}")
    log(f"Tile size (deg): {args.tile_deg}")
    log(f"Expected number of tiles: {len(tiles)}")
    log(f"Missing tiles: {len(missing_tiles)}")

    if not missing_tiles:
        log("All expected tiles are already present.")
        return 0

    for row, col, _, tile_path in missing_tiles:
        log(f"Missing tile: r{row:03d} c{col:03d} -> {tile_path}")

    if args.dry_run:
        log("Dry run enabled. No downloads started.")
        return 0

    image = build_yearly_mode_image(args.year, geometry)

    try:
        for idx, (row, col, tile_bounds, tile_path) in enumerate(missing_tiles, start=1):
            tile_region = make_tile_geometry(tile_bounds)
            log(f"[{idx}/{len(missing_tiles)}] Downloading missing tile r{row:03d} c{col:03d}")
            export_tile(
                image=image,
                tile_region=tile_region,
                tile_path=tile_path,
                crs=args.crs,
                scale=args.scale,
            )
    except Exception as exc:
        log(f"Failed: {exc}")
        return 3

    log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
