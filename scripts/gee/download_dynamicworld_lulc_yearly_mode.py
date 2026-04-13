#!/usr/bin/env python3
"""
Download Dynamic World yearly mode LULC for the BD coastal AOI, exported tile by tile.

Dynamic World classes in the `label` band:
    0 = water
    1 = trees
    2 = grass
    3 = flooded_vegetation
    4 = crops
    5 = shrub_and_scrub
    6 = built
    7 = bare
    8 = snow_and_ice

Usage
-----
python scripts/gee/download_dynamicworld_lulc_yearly_mode.py \
    --year 2024 \
    --project ee-project-id \
    --aoi configs/bd_coastal_aoi.yaml \
    --output data/raw/dynamicworld/bd_coastal_dynamicworld_2024_mode.tif

Optional example
----------------
python scripts/gee/download_dynamicworld_lulc_yearly_mode.py \
    --year 2017 \
    --project ee-project-id \
    --aoi configs/bd_coastal_aoi.yaml \
    --output data/raw/dynamicworld/bd_coastal_dynamicworld_2017_mode.tif \
    --crs EPSG:4326 \
    --scale 10 \
    --tile-deg 0.25
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple

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


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def initialize_earth_engine(project: str | None = None) -> None:
    project = project or os.environ.get(GEE_PROJECT_ENV)
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except ee.EEException:
        print("Authenticating to Google Earth Engine ...")
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    else:
        print("Google Earth Engine initialized.")


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

    print(f"Found {size} Dynamic World images for {year}. Building yearly mode composite ...")
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
) -> Iterable[Tuple[int, int, ee.Geometry]]:
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
                yield row, col, tile_region
            lon_start += tile_deg
            col += 1
        lat_start += tile_deg
        row += 1


def resolve_output_template(base: Path) -> Tuple[Path, str, str]:
    if base.suffix:
        return base.parent, base.stem, base.suffix
    return base, "tile", ".tif"


def export_tiled_image(
    image: ee.Image,
    polygon: Sequence[Sequence[float]],
    geometry: ee.Geometry,
    output: Path,
    crs: str,
    scale: int,
    tile_deg: float,
    overwrite: bool,
) -> None:
    tiles = list(iterate_tiles(polygon, geometry, tile_deg))
    if not tiles:
        raise RuntimeError("No tiles generated for the provided geometry.")

    parent, stem, suffix = resolve_output_template(output)
    parent.mkdir(parents=True, exist_ok=True)

    for idx, (row, col, tile_region) in enumerate(tiles, start=1):
        tile_path = parent / f"{stem}_r{row:03d}_c{col:03d}{suffix}"
        if tile_path.exists() and not overwrite:
            print(f"[{idx}/{len(tiles)}] Skipping existing tile -> {tile_path}")
            continue
        print(f"[{idx}/{len(tiles)}] Exporting tile row={row} col={col} -> {tile_path} ...")
        tile_image = image.clip(tile_region).unmask(DYNAMICWORLD_NODATA).toInt16()
        geemap.ee_export_image(
            tile_image,
            filename=str(tile_path),
            region=tile_region,
            crs=crs,
            scale=scale,
            file_per_band=False,
        )
    print("All Dynamic World tiles processed.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download yearly mode Dynamic World LULC for the BD coastal AOI."
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Target year for the annual mode composite.",
    )
    parser.add_argument(
        "--aoi",
        type=Path,
        default=Path("configs/bd_coastal_aoi.yaml"),
        help="AOI YAML config path. Default: configs/bd_coastal_aoi.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Base output path. Each tile appends _rXXX_cXXX before the suffix. "
            "Default: data/raw/dynamicworld/bd_coastal_dynamicworld_<year>_mode.tif"
        ),
    )
    parser.add_argument(
        "--crs",
        type=str,
        default="EPSG:4326",
        help="CRS for the export (default: EPSG:4326).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=10,
        help="Pixel resolution in meters for export (default: 10m).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=os.environ.get(GEE_PROJECT_ENV),
        help=f'Optional Earth Engine project ID for initialization (default: env "{GEE_PROJECT_ENV}").',
    )
    parser.add_argument(
        "--tile-deg",
        type=float,
        default=0.25,
        help="Tile size in degrees (default: 0.25).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tile outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    aoi_path = resolve_path(args.aoi)
    aoi = load_aoi(aoi_path)

    output = args.output or Path(f"data/raw/dynamicworld/bd_coastal_dynamicworld_{args.year}_mode.tif")
    output = resolve_path(output)

    initialize_earth_engine(project=args.project)
    geometry = build_aoi(aoi)
    mode_image = build_yearly_mode_image(args.year, geometry)

    print(f"AOI config   : {aoi_path}")
    print(f"AOI name     : {aoi.name}")
    print(f"Dataset      : {DATASET_ID}")
    print(f"Band         : {BAND_NAME}")
    print(f"Year         : {args.year}")
    print(f"Output base  : {output}")
    print(f"Output CRS   : {args.crs}")
    print(f"Scale        : {args.scale}")
    print(f"Tile size    : {args.tile_deg} degree")

    export_tiled_image(
        mode_image,
        aoi.bbox_polygon(),
        geometry,
        output=output,
        crs=args.crs,
        scale=args.scale,
        tile_deg=args.tile_deg,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
