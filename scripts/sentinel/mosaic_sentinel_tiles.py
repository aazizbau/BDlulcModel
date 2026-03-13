"""
Mosaic Sentinel-2 L2A band tiles into a single GeoTIFF for a given year/band/resolution.
"""

from __future__ import annotations

import argparse
import math
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.windows import from_bounds, transform as window_transform
from rasterio.windows import Window
from rasterio.warp import Resampling, reproject, transform_bounds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def log(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_base = Path(os.environ.get("BD_COASTAL_TILES_DIR", "data/raw/bd_coastal_tiles"))
    parser = argparse.ArgumentParser(
        description="Mosaic Sentinel-2 JP2 tiles for a specific band/year/resolution."
    )
    parser.add_argument("--base-dir", type=Path, default=default_base, help="Root directory containing yearly Sentinel SAFE folders.")
    parser.add_argument("--year", type=int, required=True, help="Year subdirectory to search under the base directory.")
    parser.add_argument("--band", default="B02", help="Band identifier (e.g., B02, B08, B11).")
    parser.add_argument("--resolution", type=float, default=10, choices={10, 20, 60}, help="Resolution in meters (10, 20, or 60).")
    parser.add_argument("--output", type=Path, default=None, help="Output GeoTIFF path. Default: <base>/<year>/coastal_<year>_<band>.tif")
    parser.add_argument("--gdal-cache-mb", type=int, default=1024, help="GDAL block cache size in MB.")
    parser.add_argument("--progress-interval", type=int, default=50, help="Log progress every N tiles.")
    parser.add_argument(
        "--target-crs",
        default="EPSG:32646",
        help="CRS for the output mosaic (e.g., EPSG:32646). Default: EPSG:32646",
    )
    parser.add_argument(
        "--resampling",
        default="nearest",
        choices=list(Resampling.__members__.keys()),
        help="Resampling method used when reprojecting tiles (default: nearest).",
    )
    return parser.parse_args(argv)


def build_default_output(base_dir: Path, year: int, band: str, resolution: float) -> Path:
    res_str = str(int(resolution)).zfill(2)
    return base_dir / str(year) / f"coastal_{year}_{res_str}_{band}.tif"


def list_band_paths(base_dir: Path, year: int, band: str, resolution: int) -> list[Path]:
    year_dir = base_dir / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(f"Year directory not found: {year_dir}")

    res_str = f"{resolution}m"
    pattern = f"*_{band}_{res_str}.jp2"
    matches = sorted(year_dir.glob(f"**/{pattern}"))
    if not matches:
        raise FileNotFoundError(f"No JP2 files found for pattern {pattern} under {year_dir}")
    return matches


def compute_mosaic_metadata(paths: list[Path], *, target_crs: CRS | None, resolution: float) -> dict:
    minx = math.inf
    miny = math.inf
    maxx = -math.inf
    maxy = -math.inf
    ref_meta = None
    output_crs = target_crs

    for path in paths:
        with rasterio.open(path) as ds:
            if ref_meta is None:
                ref_meta = ds.meta.copy()
                if output_crs is None:
                    output_crs = ds.crs
            elif output_crs is None and ds.crs != ref_meta["crs"]:
                raise ValueError(
                    f"CRS mismatch for {path}. Specify --target-crs to reproject all tiles into a single CRS."
                )

            bounds = transform_bounds(ds.crs, output_crs, *ds.bounds, densify_pts=21)
            minx = min(minx, bounds[0])
            miny = min(miny, bounds[1])
            maxx = max(maxx, bounds[2])
            maxy = max(maxy, bounds[3])

    if ref_meta is None:
        raise RuntimeError("No tiles found.")
    if output_crs is None:
        raise RuntimeError("Unable to resolve output CRS.")

    pixel_size = float(resolution)
    width = int(math.ceil((maxx - minx) / pixel_size))
    height = int(math.ceil((maxy - miny) / pixel_size))
    transform = from_origin(minx, maxy, pixel_size, pixel_size)

    meta = ref_meta.copy()
    meta.update({"width": width, "height": height, "transform": transform, "crs": output_crs})
    if meta.get("nodata") is None:
        meta["nodata"] = 0
    return meta


def reproject_tile(
    dst,
    tile_path,
    *,
    dst_transform,
    dst_crs,
    nodata,
    resampling,
):


    with rasterio.open(tile_path) as src:
        src_nodata = src.nodata if src.nodata is not None else nodata

        bounds = transform_bounds(
            src.crs, dst_crs, *src.bounds, densify_pts=21
        )

        # Initial snapped window
        dst_window = from_bounds(
            bounds[0], bounds[1], bounds[2], bounds[3],
            dst_transform
        ).round_offsets().round_lengths()

        # Expand by 1 pixel
        dst_window = Window(
            col_off=dst_window.col_off - 1,
            row_off=dst_window.row_off - 1,
            width=dst_window.width + 2,
            height=dst_window.height + 2,
        )

        # Clip to mosaic bounds (CRITICAL)
        mosaic_window = Window(0, 0, dst.width, dst.height)
        dst_window = dst_window.intersection(mosaic_window)


        if dst_window.width <= 0 or dst_window.height <= 0:
            return

        window_affine = window_transform(dst_window, dst_transform)

        h = int(dst_window.height)
        w = int(dst_window.width)

        tmp = np.full((h, w), nodata, dtype=src.dtypes[0])

        reproject(
            source=rasterio.band(src, 1),
            destination=tmp,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=window_affine,
            dst_crs=dst_crs,
            dst_nodata=nodata,
            resampling=resampling,
            init_dest_nodata=True,
        )


        # Read existing mosaic data
        existing = dst.read(1, window=dst_window)

        # Only write valid pixels
        valid = tmp != nodata
        if existing.shape != tmp.shape:
            raise RuntimeError(
                f"Shape mismatch: tmp={tmp.shape}, existing={existing.shape}, window={dst_window}"
            )

        out = np.where(valid, tmp, existing)

        dst.write(out, 1, window=dst_window)



def mosaic_tiles(
    tile_paths: list[Path],
    output: Path,
    *,
    gdal_cache_mb: int,
    progress_interval: int,
    target_crs: CRS | None,
    resolution: float,
    resampling: Resampling,
) -> None:
    meta = compute_mosaic_metadata(tile_paths, target_crs=target_crs, resolution=resolution)
    output.parent.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "GDAL_CACHEMAX": gdal_cache_mb,
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "RASTERIO_NUM_THREADS": "ALL_CPUS",
    }

    total = len(tile_paths)
    with rasterio.Env(**env_kwargs):
        with rasterio.open(output, "w+", **meta) as dst:
            for idx, tile in enumerate(tile_paths, start=1):
                reproject_tile(
                    dst,
                    tile,
                    dst_transform=meta["transform"],
                    dst_crs=meta["crs"],
                    nodata=meta["nodata"],
                    resampling=resampling,
                )
                if idx % progress_interval == 0 or idx == total:
                    log(f"Wrote {idx}/{total} tiles ...")
    log(f"Saved mosaic to {output}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output or build_default_output(args.base_dir, args.year, args.band, args.resolution)
    progress_interval = max(1, args.progress_interval)
    target_crs = CRS.from_string(args.target_crs) if args.target_crs else None
    resampling = Resampling[args.resampling]

    band_paths = list_band_paths(args.base_dir, args.year, args.band, int(args.resolution))
    log(f"Found {len(band_paths)} tiles for band {args.band} ({args.resolution}m). Building mosaic ...")
    mosaic_tiles(
        band_paths,
        output,
        gdal_cache_mb=args.gdal_cache_mb,
        progress_interval=progress_interval,
        target_crs=target_crs,
        resolution=args.resolution,
        resampling=resampling,
    )


if __name__ == "__main__":
    main()
