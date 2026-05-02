#!/usr/bin/env python3
"""
Mosaic Manik LULC tiles and clip them to the Bangladesh coastal AOI.

Inputs
------
- assets/maps/N*E*_<year>_v*_10m.tif   (e.g. N20E092_2023_v25.09_10m.tif)
- assets/maps/bd_coastal_map_solid_gp.gpkg

Output
------
- data/processed/manik_lulc/bd_coastal_manik_lulc_<year>_clipped.tif

Example
-------
python scripts/manik_lulc/mosaic_and_clip_manik_lulc.py \
    --year 2023 \
    --input assets/maps \
    --clip-vector assets/maps/bd_coastal_map_solid_gp.gpkg \
    --output data/processed/manik_lulc/bd_coastal_manik_lulc_2023_clipped.tif
"""

from __future__ import annotations

import argparse
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


TZ = ZoneInfo("Asia/Tokyo")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_CRS = "EPSG:4326"
DEFAULT_INPUT = Path("assets/maps")
DEFAULT_CLIP_VECTOR = Path("assets/maps/bd_coastal_map_solid_gp.gpkg")
DEFAULT_OUTPUT_DIR = Path("data/processed/manik_lulc")
MANIK_NODATA = 0


def ts() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_output_path(year: int) -> Path:
    return DEFAULT_OUTPUT_DIR / f"bd_coastal_manik_lulc_{year}_clipped.tif"


def default_pattern(year: int) -> str:
    return f"N*E*_{year}_v*_10m.tif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mosaic Manik LULC tiles and clip them to the coastal AOI."
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Target year, e.g. 2023.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input directory containing Manik LULC tiles.",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Optional tile filename glob pattern inside input directory.",
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
        default=None,
        help="Output clipped GeoTIFF path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def list_tiles(input_dir: Path, pattern: str) -> list[Path]:
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f'No Manik LULC tiles found in "{input_dir}" matching pattern "{pattern}".'
        )
    return paths


def reproject_to_4326(array: np.ndarray, meta: dict) -> tuple[np.ndarray, dict]:
    src_crs = meta["crs"]
    if str(src_crs) == TARGET_CRS or getattr(src_crs, "to_string", lambda: str(src_crs))() == TARGET_CRS:
        return array, meta

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        TARGET_CRS,
        meta["width"],
        meta["height"],
        *rasterio.transform.array_bounds(meta["height"], meta["width"], meta["transform"]),
    )

    nodata = meta.get("nodata", MANIK_NODATA)
    dst = np.full(
        (meta["count"], dst_height, dst_width),
        nodata,
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
            src_nodata=nodata,
            dst_nodata=nodata,
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
    clip_vector = resolve_path(args.clip_vector)
    output_path = resolve_path(args.output or default_output_path(args.year))
    pattern = args.pattern or default_pattern(args.year)

    if output_path.exists() and not args.overwrite:
        log(f"Output already exists: {output_path}")
        log("Use --overwrite to replace it.")
        return 1

    if not input_dir.exists():
        log(f"ERROR: input directory does not exist: {input_dir}")
        return 1

    if not clip_vector.exists():
        log(f"ERROR: clip vector does not exist: {clip_vector}")
        return 1

    tile_paths = list_tiles(input_dir, pattern)
    log(f"Found {len(tile_paths)} Manik LULC tiles for year {args.year}.")

    srcs = [rasterio.open(path) for path in tile_paths]
    try:
        log("Mosaicking tiles ...")
        mosaic_arr, mosaic_transform = merge(srcs)
        mosaic_meta = srcs[0].meta.copy()
        src_dtype = srcs[0].dtypes[0]
        src_nodata = srcs[0].nodata
    finally:
        for src in srcs:
            src.close()

    nodata = src_nodata if src_nodata is not None else MANIK_NODATA
    mosaic_meta.update(
        {
            "height": mosaic_arr.shape[1],
            "width": mosaic_arr.shape[2],
            "transform": mosaic_transform,
            "count": mosaic_arr.shape[0],
            "dtype": src_dtype,
            "nodata": nodata,
        }
    )
    mosaic_arr = mosaic_arr.astype(src_dtype, copy=False)

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
            clipped_arr, clipped_transform = mask(ds, shapes=shapes, crop=True, nodata=nodata)

    clipped_meta = mosaic_meta.copy()
    clipped_meta.update(
        {
            "height": clipped_arr.shape[1],
            "width": clipped_arr.shape[2],
            "transform": clipped_transform,
            "count": clipped_arr.shape[0],
        }
    )
    clipped_arr = clipped_arr.astype(src_dtype, copy=False)

    log(f"Ensuring output CRS is {TARGET_CRS} ...")
    final_arr, final_meta = reproject_to_4326(clipped_arr, clipped_meta)
    final_meta.update(
        {
            "driver": "GTiff",
            "dtype": src_dtype,
            "nodata": nodata,
            "compress": "DEFLATE",
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **final_meta) as dst:
        dst.write(final_arr)

    log(f"Saved clipped Manik LULC mosaic: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
