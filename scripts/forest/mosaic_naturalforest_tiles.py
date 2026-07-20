"""
Mosaic natural forest tiles for the Bangladesh coastal AOI into a single GeoTIFF.

Usage:
    python scripts/forest/mosaic_naturalforest_tiles.py \
        --input-base data/raw/forest/bd_coastal_naturalforest_2020.tif \
        --output data/interim/bd_coastal_naturalforest_2020_mosaic.tif

Reproduction and AOI adaptation
-------------------------------
Workflow role: Download, validate, and mosaic natural-forest reference data.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--input-base``, ``--output``, ``--gdal-cache-mb``, ``--progress-interval``, ``--start-tile``, ``--max-tile-mb``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace Earth Engine project/AOI settings and local tile directories, then verify complete tile coverage before mosaicking.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.forest import download_naturalforest as downloader


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
    meta.update({"width": width, "height": height, "transform": transform})

    if meta.get("nodata") is None:
        meta["nodata"] = 0

    return meta


def _stream_tile(
    dst: rasterio.io.DatasetWriter,
    tile_path: Path,
    dst_transform,
    *,
    max_tile_mb: int,
) -> None:
    """Write a source tile into the destination using block-wise streaming."""
    with rasterio.open(tile_path) as src:
        dst_window = from_bounds(
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top,
            dst_transform,
            precision=6,
        ).round_offsets().round_lengths()

        if dst_window.width != src.width or dst_window.height != src.height:
            raise ValueError(f"Window mismatch for {tile_path}")

        dtype_size = np.dtype(src.dtypes[0]).itemsize
        tile_mb = (src.width * src.height * src.count * dtype_size) / (1024 * 1024)
        if max_tile_mb > 0 and tile_mb <= max_tile_mb:
            data = src.read()
            dst.write(data, window=dst_window)
            return

        band_index = 1 if src.count else 1
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
) -> None:
    """Stream tiles into a single mosaic GeoTIFF."""
    meta = compute_mosaic_metadata(tile_paths)
    output.parent.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "GDAL_CACHEMAX": gdal_cache_mb,
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "RASTERIO_NUM_THREADS": "ALL_CPUS",
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

    with rasterio.Env(**env_kwargs):
        with rasterio.open(
            output, open_mode, **(meta if open_mode == "w" else {})
        ) as dst:
            processed = start_tile - 1
            remaining_paths = tile_paths[start_tile - 1 :]
            for tile_path in remaining_paths:
                processed += 1
                _stream_tile(
                    dst,
                    tile_path,
                    meta["transform"],
                    max_tile_mb=max_tile_mb,
                )

                if processed % progress_interval == 0 or processed == total_tiles:
                    log(f"Wrote {processed}/{total_tiles} tiles ...")

    log(f"Saved mosaic to {output}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mosaic natural forest tiles.")
    parser.add_argument(
        "--input-base",
        type=Path,
        default=Path("data/raw/forest/bd_coastal_naturalforest_2020.tif"),
        help="Base path used for per-tile downloads.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/bd_coastal_naturalforest_2020_mosaic.tif"),
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    tile_paths = list_tile_paths(args.input_base)
    log(f"Found {len(tile_paths)} tiles. Building mosaic ...")
    mosaic_tiles(
        tile_paths,
        args.output,
        gdal_cache_mb=args.gdal_cache_mb,
        progress_interval=max(1, args.progress_interval),
        start_tile=max(1, args.start_tile),
        max_tile_mb=max(0, args.max_tile_mb),
    )


if __name__ == "__main__":
    main()
