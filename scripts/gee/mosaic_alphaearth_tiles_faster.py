"""
Mosaic AlphaEarth tiles for the Bangladesh coastal AOI into a single GeoTIFF.

Optimizations (implemented):
- Avoid per-tile from_bounds() math; compute dst window offsets with integer math.
- Early whole-tile read/write before any block-window iteration overhead.
- Output GeoTIFF tuned for multi-band performance: tiled + band interleave + 512x512 blocks.
- Stronger GDAL env knobs for many-file workloads (disable readdir scans, swath size).
- Time-based progress logging to reduce I/O overhead.

Usage:
    python scripts/gee/mosaic_alphaearth_tiles_faster.py \
        --input-base data/raw/embeddings/bd_coastal_alphaearth_2024.tif \
        --output data/interim/bd_coastal_alphaearth_2024_mosaic.tif
        --gdal-cache-mb 4096

Reproduction and AOI adaptation
-------------------------------
Workflow role: Acquire or prepare Earth Engine products such as AlphaEarth, Dynamic World, or ALOS DSM data.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--input-base``, ``--output``, ``--gdal-cache-mb``, ``--progress-interval``, ``--start-tile``, ``--max-tile-mb``, ``--log-every-seconds``. Run the ``--help`` command below for required values, defaults, and accepted choices.
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
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gee import download_alphaearth_embeddings as downloader


def log(message: str) -> None:
    """Print a message with an ISO timestamp."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}")


def list_tile_paths(base: Path) -> list[Path]:
    """Return sorted tile paths matching the downloader naming scheme."""
    parent, stem, suffix = downloader.resolve_output_template(base)
    pattern = f"{stem}_r*_c*{suffix}"
    paths = sorted(parent.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No tiles found matching pattern {pattern} in directory {parent}"
        )
    return paths


def compute_mosaic_metadata(tile_paths: list[Path]) -> dict:
    """Compute mosaic metadata (bounds, transform, dimensions) from tiles."""
    minx = math.inf
    miny = math.inf
    maxx = -math.inf
    maxy = -math.inf

    ref_meta = None
    for path in tile_paths:
        with rasterio.open(path) as ds:
            bounds = ds.bounds
            minx = min(minx, bounds.left)
            miny = min(miny, bounds.bottom)
            maxx = max(maxx, bounds.right)
            maxy = max(maxy, bounds.top)

            if ref_meta is None:
                ref_meta = ds.meta.copy()
            else:
                if ds.crs != ref_meta["crs"]:
                    raise ValueError(f"CRS mismatch for {path}")
                # Keep the original checks: consistent resolution / transform scale
                if (ds.transform.a != ref_meta["transform"].a) or (
                    ds.transform.e != ref_meta["transform"].e
                ):
                    raise ValueError(f"Resolution/transform mismatch for {path}")

    if ref_meta is None:
        raise RuntimeError("No tiles found.")

    pixel_width = ref_meta["transform"].a
    pixel_height = -ref_meta["transform"].e

    width = int(math.ceil((maxx - minx) / pixel_width))
    height = int(math.ceil((maxy - miny) / pixel_height))

    transform = from_origin(minx, maxy, pixel_width, pixel_height)

    meta = ref_meta.copy()
    meta.update(
        {
            "width": width,
            "height": height,
            "transform": transform,
            # Write-side performance tuning for big multi-band mosaics
            "tiled": True,
            "interleave": "band",
            "blockxsize": 512,
            "blockysize": 512,
            # Big mosaics are likely; let GDAL choose BigTIFF as needed
            "BIGTIFF": "IF_SAFER",
        }
    )

    if meta.get("nodata") is None:
        meta["nodata"] = 0

    return meta


def _dst_window_for_tile(src_bounds, dst_transform, src_width: int, src_height: int) -> Window:
    """
    Compute the destination Window using integer math (fast) assuming
    shared CRS + pixel size + grid alignment.

    dst_transform is affine with:
      a = pixel width
      e = negative pixel height
      c = origin x (top-left)
      f = origin y (top-left)
    """
    # Column offset from x
    col_off = int(round((src_bounds.left - dst_transform.c) / dst_transform.a))
    # Row offset from y (note: transform.e is negative)
    row_off = int(round((dst_transform.f - src_bounds.top) / abs(dst_transform.e)))

    return Window(col_off, row_off, src_width, src_height)


def _stream_tile(
    dst: rasterio.io.DatasetWriter,
    tile_path: Path,
    dst_transform,
    *,
    max_tile_mb: int,
) -> None:
    """Write a source tile into the destination (whole-tile if small; else block streaming)."""
    with rasterio.open(tile_path) as src:
        dst_window = _dst_window_for_tile(src.bounds, dst_transform, src.width, src.height)

        # Fast size estimate before any block iteration
        dtype_size = np.dtype(src.dtypes[0]).itemsize
        tile_mb = (src.width * src.height * max(1, src.count) * dtype_size) / (1024 * 1024)

        # Whole-tile read/write is often fastest for moderate tiles (reduces Python overhead)
        if max_tile_mb > 0 and tile_mb <= max_tile_mb:
            dst.write(src.read(), window=dst_window)
            return

        # Otherwise, stream by blocks to cap RAM
        # Use band 1 to iterate block windows (layout is the same for all bands)
        band_index = 1
        for _, block_window in src.block_windows(band_index):
            data = src.read(window=block_window)
            dst_block = Window(
                col_off=block_window.col_off + dst_window.col_off,
                row_off=block_window.row_off + dst_window.row_off,
                width=block_window.width,
                height=block_window.height,
            )
            dst.write(data, window=dst_block)


def mosaic_tiles(
    tile_paths: list[Path],
    output: Path,
    *,
    gdal_cache_mb: int,
    progress_interval: int,
    start_tile: int,
    max_tile_mb: int,
    log_every_seconds: float = 5.0,
) -> None:
    """Stream tiles into a single mosaic GeoTIFF."""
    meta = compute_mosaic_metadata(tile_paths)
    output.parent.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        # Cache + threads
        "GDAL_CACHEMAX": int(gdal_cache_mb),
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "RASTERIO_NUM_THREADS": "ALL_CPUS",
        # Many-tile workloads: avoid directory scanning overhead
        "GDAL_DISABLE_READDIR_ON_OPEN": "TRUE",
        # Helpful for large sequential IO
        "GDAL_SWATH_SIZE": str(64 * 1024 * 1024),  # 64MB
    }

    total_tiles = len(tile_paths)
    if start_tile < 1 or start_tile > total_tiles:
        raise ValueError(
            f"start_tile must be between 1 and {total_tiles}, got {start_tile}"
        )

    resume = start_tile > 1
    open_mode = "r+" if resume else "w"

    if resume:
        if not output.exists():
            raise FileNotFoundError(
                f"Cannot resume because {output} does not exist. "
                "Re-run without --start-tile or delete the corrupt output."
            )

        with rasterio.open(output) as existing:
            existing_transform = existing.transform
            if existing_transform != meta["transform"]:
                raise ValueError(
                    "Existing mosaic transform does not match computed metadata. "
                    "Delete the mosaic or ensure you are resuming with the same "
                    "input tiles."
                )
            if existing.width != meta["width"] or existing.height != meta["height"]:
                raise ValueError(
                    "Existing mosaic dimensions differ from expected tiles. "
                    "Delete the mosaic and restart."
                )

        log(
            f"Resuming mosaic at tile {start_tile}/{total_tiles} "
            f"using existing file {output}"
        )

    last_log = time.time()

    with rasterio.Env(**env_kwargs):
        with rasterio.open(output, open_mode, **(meta if open_mode == "w" else {})) as dst:
            processed = start_tile - 1
            remaining_paths = tile_paths[start_tile - 1 :]

            for tile_path in remaining_paths:
                processed += 1
                _stream_tile(dst, tile_path, meta["transform"], max_tile_mb=max_tile_mb)

                now = time.time()
                # Log either every N tiles, or at least every few seconds (whichever comes first)
                if (
                    processed == total_tiles
                    or (progress_interval > 0 and processed % progress_interval == 0)
                    or (now - last_log) >= log_every_seconds
                ):
                    log(f"Wrote {processed}/{total_tiles} tiles ...")
                    last_log = now

    log(f"Saved mosaic to {output}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mosaic AlphaEarth tiles faster.")
    parser.add_argument(
        "--input-base",
        type=Path,
        default=Path("data/raw/embeddings/bd_coastal_alphaearth_2024.tif"),
        help="Base path used for per-tile downloads.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/bd_coastal_alphaearth_2024_mosaic.tif"),
        help="Output GeoTIFF path.",
    )
    parser.add_argument(
        "--gdal-cache-mb",
        type=int,
        default=512,
        help="GDAL block cache size in MB (default: 512).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=250,
        help="Log progress every N tiles (default: 250).",
    )
    parser.add_argument(
        "--start-tile",
        type=int,
        default=1,
        help=(
            "Resume mosaicking from this 1-based tile index. "
            "Use the log output to determine the next tile number to process."
        ),
    )
    parser.add_argument(
        "--max-tile-mb",
        type=int,
        default=256,
        help=(
            "Read whole tiles into memory when under this size in MB "
            "(default: 256). Set to 0 to always stream blocks."
        ),
    )
    parser.add_argument(
        "--log-every-seconds",
        type=float,
        default=5.0,
        help="Time-based progress logging interval in seconds (default: 5).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    tile_paths = list_tile_paths(args.input_base)
    log(f"Found {len(tile_paths)} tiles. Building mosaic ...")

    mosaic_tiles(
        tile_paths,
        args.output,
        gdal_cache_mb=max(1, args.gdal_cache_mb),
        progress_interval=max(1, args.progress_interval),
        start_tile=max(1, args.start_tile),
        max_tile_mb=max(0, args.max_tile_mb),
        log_every_seconds=max(0.5, float(args.log_every_seconds)),
    )


if __name__ == "__main__":
    main()