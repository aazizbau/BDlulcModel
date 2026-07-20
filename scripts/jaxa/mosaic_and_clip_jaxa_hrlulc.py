#!/usr/bin/env python3
"""
Mosaic JAXA HRLULC tiles that intersect the BD coastal AOI and clip to the coastal boundary.

Inputs
------
- data/raw/JAXA_HRLULC/2023SEA_v25.09
- configs/bd_coastal_aoi.yaml
- assets/maps/bd_coastal_map_solid_gp.gpkg

Expected tile name pattern
--------------------------
- LC_N17E097.tif
- LC_N20E088.tif

Output
------
- data/processed/jaxa_hrlulc/bd_coastal_jaxa_hrlulc_2023_clipped.tif

Example
-------
python scripts/jaxa/mosaic_and_clip_jaxa_hrlulc.py \
    --input data/raw/JAXA_HRLULC/2023SEA_v25.09 \
    --aoi configs/bd_coastal_aoi.yaml \
    --clip-vector assets/maps/bd_coastal_map_solid_gp.gpkg \
    --output data/processed/jaxa_hrlulc/bd_coastal_jaxa_hrlulc_2023_clipped.tif

Reproduction and AOI adaptation
-------------------------------
Workflow role: Prepare JAXA HRLULC reference data for harmonized comparison.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--input``, ``--aoi``, ``--clip-vector``, ``--output``, ``--overwrite``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace source tiles and clipping boundary; use nearest-neighbour resampling for categorical classes and revise harmonization if needed.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gee.aoi import AOI, load_aoi


TZ = ZoneInfo("Asia/Tokyo")
TARGET_CRS = "EPSG:4326"
DEFAULT_INPUT = Path("data/raw/JAXA_HRLULC/2023SEA_v25.09")
DEFAULT_AOI = Path("configs/bd_coastal_aoi.yaml")
DEFAULT_CLIP_VECTOR = Path("assets/maps/bd_coastal_map_solid_gp.gpkg")
DEFAULT_OUTPUT = Path("data/processed/jaxa_hrlulc/bd_coastal_jaxa_hrlulc_2023_clipped.tif")
TILE_REGEX = re.compile(r"^LC_([NS])(\d{2})([EW])(\d{3})\.tif$", re.IGNORECASE)


def ts() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mosaic JAXA HRLULC tiles intersecting an AOI and clip to a vector boundary."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Directory containing JAXA HRLULC tiles.",
    )
    parser.add_argument(
        "--aoi",
        type=Path,
        default=DEFAULT_AOI,
        help="AOI YAML config path.",
    )
    parser.add_argument(
        "--clip-vector",
        type=Path,
        default=DEFAULT_CLIP_VECTOR,
        help="Vector file used to clip the mosaic.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output clipped GeoTIFF path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def aoi_bounds(aoi: AOI) -> tuple[float, float, float, float]:
    polygon = aoi.bbox_polygon()
    lons = [pt[0] for pt in polygon[:-1]]
    lats = [pt[1] for pt in polygon[:-1]]
    return min(lons), min(lats), max(lons), max(lats)


def tile_bounds_from_name(path: Path) -> tuple[float, float, float, float] | None:
    match = TILE_REGEX.match(path.name)
    if match is None:
        return None

    lat_hemi, lat_deg_txt, lon_hemi, lon_deg_txt = match.groups()
    lat0 = int(lat_deg_txt)
    lon0 = int(lon_deg_txt)
    if lat_hemi.upper() == "S":
        lat0 = -lat0
    if lon_hemi.upper() == "W":
        lon0 = -lon0

    lat1 = lat0 + 1
    lon1 = lon0 + 1
    ymin = min(lat0, lat1)
    ymax = max(lat0, lat1)
    xmin = min(lon0, lon1)
    xmax = max(lon0, lon1)
    return xmin, ymin, xmax, ymax


def intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    axmin, aymin, axmax, aymax = a
    bxmin, bymin, bxmax, bymax = b
    return not (axmax <= bxmin or axmin >= bxmax or aymax <= bymin or aymin >= bymax)


def select_tiles(input_dir: Path, aoi: AOI) -> list[Path]:
    bounds = aoi_bounds(aoi)
    candidates = sorted(input_dir.glob("LC_*.tif"))
    if not candidates:
        raise FileNotFoundError(f'No JAXA HRLULC tiles found in "{input_dir}" matching "LC_*.tif".')

    selected: list[Path] = []
    skipped_nonmatching: list[str] = []
    for path in candidates:
        tile_bounds = tile_bounds_from_name(path)
        if tile_bounds is None:
            skipped_nonmatching.append(path.name)
            continue
        if intersects(tile_bounds, bounds):
            selected.append(path)

    if not selected:
        raise FileNotFoundError(
            f"No JAXA HRLULC tiles intersect the AOI bounds {bounds} in {input_dir}."
        )

    if skipped_nonmatching:
        log(f"Skipped {len(skipped_nonmatching)} files with unexpected tile names.")
    return selected


def reproject_array_to_4326(array: np.ndarray, meta: dict) -> tuple[np.ndarray, dict]:
    src_crs = meta["crs"]
    src_crs_text = src_crs.to_string() if hasattr(src_crs, "to_string") else str(src_crs)
    if src_crs_text == TARGET_CRS:
        return array, meta

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        TARGET_CRS,
        meta["width"],
        meta["height"],
        *rasterio.transform.array_bounds(meta["height"], meta["width"], meta["transform"]),
    )

    dst = np.full(
        (meta["count"], dst_height, dst_width),
        meta.get("nodata", 0),
        dtype=array.dtype,
    )

    for band_idx in range(meta["count"]):
        reproject(
            source=array[band_idx],
            destination=dst[band_idx],
            src_transform=meta["transform"],
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            src_nodata=meta.get("nodata"),
            dst_nodata=meta.get("nodata"),
            resampling=Resampling.nearest,
        )

    out_meta = meta.copy()
    out_meta.update(
        {
            "crs": TARGET_CRS,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
        }
    )
    return dst, out_meta


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input)
    aoi_path = resolve_path(args.aoi)
    clip_vector = resolve_path(args.clip_vector)
    output_path = resolve_path(args.output)

    if output_path.exists() and not args.overwrite:
        log(f"Output already exists: {output_path}")
        log("Use --overwrite to replace it.")
        return 1

    if not input_dir.exists():
        log(f"ERROR: input directory does not exist: {input_dir}")
        return 1
    if not aoi_path.exists():
        log(f"ERROR: AOI config does not exist: {aoi_path}")
        return 1
    if not clip_vector.exists():
        log(f"ERROR: clip vector does not exist: {clip_vector}")
        return 1

    aoi = load_aoi(aoi_path)
    tile_paths = select_tiles(input_dir, aoi)
    log(f"AOI config: {aoi_path}")
    log(f"AOI name: {aoi.name}")
    log(f"Selected {len(tile_paths)} JAXA HRLULC tiles.")
    for path in tile_paths:
        log(f"  tile: {path.name}")

    srcs = [rasterio.open(path) for path in tile_paths]
    try:
        log("Mosaicking tiles ...")
        mosaic_arr, mosaic_transform = merge(srcs)
        mosaic_meta = srcs[0].meta.copy()
        mosaic_meta.update(
            {
                "height": mosaic_arr.shape[1],
                "width": mosaic_arr.shape[2],
                "transform": mosaic_transform,
                "count": mosaic_arr.shape[0],
            }
        )
        if mosaic_meta.get("nodata") is None:
            mosaic_meta["nodata"] = 0
    finally:
        for src in srcs:
            src.close()

    log(f"Reading clip vector: {clip_vector}")
    gdf = gpd.read_file(clip_vector)
    if gdf.empty:
        log("ERROR: clip vector is empty.")
        return 1
    if gdf.crs is None:
        log("ERROR: clip vector has no CRS.")
        return 1

    target_clip = gdf.to_crs(mosaic_meta["crs"])
    shapes = [geom.__geo_interface__ for geom in target_clip.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        log("ERROR: clip vector has no valid geometries.")
        return 1

    log("Clipping mosaic ...")
    with MemoryFile() as memfile:
        with memfile.open(**mosaic_meta) as ds:
            ds.write(mosaic_arr)
            clipped_arr, clipped_transform = mask(ds, shapes=shapes, crop=True, nodata=mosaic_meta["nodata"])

    clipped_meta = mosaic_meta.copy()
    clipped_meta.update(
        {
            "height": clipped_arr.shape[1],
            "width": clipped_arr.shape[2],
            "transform": clipped_transform,
            "count": clipped_arr.shape[0],
        }
    )

    log(f"Ensuring output CRS is {TARGET_CRS} ...")
    final_arr, final_meta = reproject_array_to_4326(clipped_arr, clipped_meta)
    final_meta.update(
        {
            "driver": "GTiff",
            "compress": "DEFLATE",
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **final_meta) as dst:
        dst.write(final_arr)

    log(f"Saved clipped JAXA HRLULC mosaic: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
