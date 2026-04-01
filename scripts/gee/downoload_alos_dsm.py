#!/usr/bin/env python3
"""
Download ALOS World 3D 30m (AW3D30) DSM data from Google Earth Engine
for a fixed Bangladesh coastal AOI, exported tile by tile.

Note
----
The latest ALOS elevation/topography product currently available in the
Google Earth Engine public catalog is:
    JAXA/ALOS/AW3D30/V4_1
This is a Digital Surface Model (DSM), not a bare-earth DEM.

Usage
-----
python scripts/gee/downoload_alos_dsm.py \
    --project ee-project-id \
    --aoi configs/bd_coastal_aoi.yaml \
    --output data/raw/dem/bd_coastal_aw3d30_v41_dsm.tif

Optional examples:
python scripts/gee/downoload_alos_dsm.py \
    --project ee-project-id \
    --aoi configs/bd_coastal_aoi.yaml \
    --output data/raw/dem/bd_coastal_aw3d30_v41_dsm.tif \
    --scale 30 \
    --crs EPSG:4326 \
    --tile-deg 0.25

Authentication
--------------
If Earth Engine is not already authenticated on your machine, run:
    earthengine authenticate
or let the script attempt interactive authentication.
"""

from __future__ import annotations

import argparse
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
            "Download the latest ALOS AW3D30 DSM layer available in Earth Engine "
            "for the BD coastal AOI, exported tile by tile."
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing tile outputs if they already exist.",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get(GEE_PROJECT_ENV),
        help=f'Optional Earth Engine project ID (default: env "{GEE_PROJECT_ENV}").',
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


def iterate_tiles(aoi: AOI, tile_deg: float) -> Iterator[Tuple[int, int, ee.Geometry]]:
    min_lon, min_lat, max_lon, max_lat = aoi_bounds(aoi)

    row = 0
    lat_start = min_lat
    while lat_start < max_lat:
        lat_end = min(lat_start + tile_deg, max_lat)
        col = 0
        lon_start = min_lon
        while lon_start < max_lon:
            lon_end = min(lon_start + tile_deg, max_lon)
            yield row, col, ee.Geometry.Rectangle(
                [lon_start, lat_start, lon_end, lat_end],
                proj="EPSG:4326",
                geodesic=False,
            )
            lon_start += tile_deg
            col += 1
        lat_start += tile_deg
        row += 1


def tile_output_path(output_base: Path, row: int, col: int) -> Path:
    suffix = output_base.suffix or ".tif"
    stem = output_base.stem if output_base.suffix else output_base.name
    parent = output_base.parent if output_base.suffix else output_base
    return parent / f"{stem}_r{row:03d}_c{col:03d}{suffix}"


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
        "region": region.toGeoJSONString(),
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

    init_ee(args.project)

    region = build_aoi(aoi)
    image = build_image(region)
    tiles = list(iterate_tiles(aoi, args.tile_deg))

    log(f"AOI config: {aoi_path}")
    log(f"AOI name: {aoi.name}")
    log(f"Dataset: {DATASET_ID}")
    log(f"Band: {BAND_NAME}")
    log(f"Output base: {output_base}")
    log(f"Output CRS: {args.crs}")
    log(f"Scale: {args.scale}")
    log(f"Tile size (deg): {args.tile_deg}")
    log(f"Number of tiles: {len(tiles)}")

    try:
        for idx, (row, col, tile_region) in enumerate(tiles, start=1):
            tile_path = tile_output_path(output_base, row, col)
            if tile_path.exists() and not args.overwrite:
                log(f"[{idx}/{len(tiles)}] Skipping existing tile: {tile_path}")
                continue

            log(f"[{idx}/{len(tiles)}] Downloading tile r{row:03d} c{col:03d}")
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
