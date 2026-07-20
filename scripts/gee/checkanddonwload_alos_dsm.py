#!/usr/bin/env python3
"""
Check which ALOS World 3D 30m (AW3D30) DSM tiles are missing for the
Bangladesh coastal AOI and download only the missing ones.

This script follows the same AOI, tiling, naming, and Earth Engine export
logic as:
    scripts/gee/downoload_alos_dsm.py

Usage
-----
Example:
python scripts/gee/checkanddonwload_alos_dsm.py \
    --project ee-project-id \
    --aoi configs/bd_coastal_aoi.yaml \
    --output data/raw/dem/bd_coastal_aw3d30_v41_dsm.tif
    
Dry-run example
python scripts/gee/checkanddonwload_alos_dsm.py \
    --project ee-project-id \
    --aoi configs/bd_coastal_aoi.yaml \
    --output data/raw/dem/bd_coastal_aw3d30_v41_dsm.tif
    --dry-run

Reproduction and AOI adaptation
-------------------------------
Workflow role: Acquire or prepare Earth Engine products such as AlphaEarth, Dynamic World, or ALOS DSM data.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--aoi``, ``--output``, ``--scale``, ``--crs``, ``--tile-deg``, ``--timeout``, ``--project``, ``--dry-run``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace the Earth Engine project, AOI vector or bounds, year, scale, CRS, and export directory. Authenticate the account that owns the target project first.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterator, Tuple

import ee
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gee.aoi import AOI, load_aoi


DATASET_ID = "JAXA/ALOS/AW3D30/V4_1"
BAND_NAME = "DSM"
GEE_PROJECT_ENV = "GEE_PROJECT_ID"


def ts() -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check ALOS AW3D30 DSM tile coverage for the BD coastal AOI and "
            "download only missing tiles."
        )
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
        required=True,
        help=(
            "Base output GeoTIFF path. Each tile appends _rXXX_cXXX before the suffix, "
            'e.g. "bd_coastal_aw3d30_v41_dsm_r000_c000.tif".'
        ),
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=30.0,
        help="Requested output pixel size in meters/degrees according to CRS. Default: 30.",
    )
    parser.add_argument(
        "--crs",
        default="EPSG:4326",
        help="Output CRS passed to Earth Engine download. Default: EPSG:4326.",
    )
    parser.add_argument(
        "--tile-deg",
        type=float,
        default=0.25,
        help="Tile size in degrees for bbox tiling. Default: 0.25.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="HTTP download timeout in seconds. Default: 600.",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get(GEE_PROJECT_ENV),
        help=f'Optional Earth Engine project ID (default: env "{GEE_PROJECT_ENV}").',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report missing tiles without downloading them.",
    )
    return parser.parse_args()


def init_ee(project: str | None) -> None:
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize(project=None)
        log("Initialized Earth Engine.")
        return
    except Exception:
        log("Earth Engine not initialized; attempting authentication.")

    try:
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize(project=None)
        log("Authenticated and initialized Earth Engine.")
    except Exception as exc:
        raise RuntimeError(
            "Failed to authenticate/initialize Earth Engine. "
            "Run 'earthengine authenticate' first and ensure your EE access is active."
        ) from exc


def build_aoi(aoi: AOI) -> ee.Geometry:
    return ee.Geometry.Polygon([aoi.bbox_polygon()], proj="EPSG:4326", geodesic=False)


def build_image(region: ee.Geometry) -> ee.Image:
    collection = ee.ImageCollection(DATASET_ID).select(BAND_NAME)
    image = collection.mosaic().clip(region).rename(BAND_NAME)
    return image


def aoi_bounds(aoi: AOI) -> Tuple[float, float, float, float]:
    polygon = aoi.bbox_polygon()
    lons = [pt[0] for pt in polygon[:-1]]
    lats = [pt[1] for pt in polygon[:-1]]
    return min(lons), min(lats), max(lons), max(lats)


def iterate_tiles(aoi: AOI, tile_deg: float) -> Iterator[Tuple[int, int, Tuple[float, float, float, float]]]:
    min_lon, min_lat, max_lon, max_lat = aoi_bounds(aoi)

    row = 0
    lat_start = min_lat
    while lat_start < max_lat:
        lat_end = min(lat_start + tile_deg, max_lat)
        col = 0
        lon_start = min_lon
        while lon_start < max_lon:
            lon_end = min(lon_start + tile_deg, max_lon)
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


def download_image(
    image: ee.Image,
    region: ee.Geometry,
    output_path: Path,
    scale: float,
    crs: str,
    timeout: int,
) -> None:
    params = {
        "name": output_path.stem,
        "bands": [BAND_NAME],
        "region": json.dumps(region.getInfo()),
        "scale": scale,
        "crs": crs,
        "format": "GEO_TIFF",
    }

    log("Requesting Earth Engine download URL...")
    url = image.getDownloadURL(params)
    log("Download URL received. Starting download...")

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    log(f"Saved: {output_path}")


def main() -> int:
    args = parse_args()
    aoi_path = resolve_path(args.aoi)
    output_base = resolve_path(args.output)

    if args.tile_deg <= 0:
        log("ERROR: --tile-deg must be > 0.")
        return 1

    output_base.parent.mkdir(parents=True, exist_ok=True)

    aoi = load_aoi(aoi_path)
    tiles = list(iterate_tiles(aoi, args.tile_deg))

    missing_tiles = []
    for row, col, tile_bounds in tiles:
        tile_path = tile_output_path(output_base, row, col)
        if is_missing_tile(tile_path):
            missing_tiles.append((row, col, tile_bounds, tile_path))

    log(f"AOI config: {aoi_path}")
    log(f"AOI name: {aoi.name}")
    log(f"Dataset: {DATASET_ID}")
    log(f"Band: {BAND_NAME}")
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

    init_ee(args.project)
    region = build_aoi(aoi)
    image = build_image(region)

    try:
        for idx, (row, col, tile_bounds, tile_path) in enumerate(missing_tiles, start=1):
            tile_region = make_tile_geometry(tile_bounds)
            log(f"[{idx}/{len(missing_tiles)}] Downloading missing tile r{row:03d} c{col:03d}")
            download_image(
                image=image.clip(tile_region),
                region=tile_region,
                output_path=tile_path,
                scale=args.scale,
                crs=args.crs,
                timeout=args.timeout,
            )
    except requests.HTTPError as exc:
        log(f"HTTP error while downloading: {exc}")
        return 2
    except Exception as exc:
        log(f"Failed: {exc}")
        return 3

    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
