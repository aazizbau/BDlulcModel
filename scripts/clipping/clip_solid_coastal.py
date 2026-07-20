"""
Clip the coastal Sentinel mosaic to the solid coastal AOI using GDAL streaming tools.

Example:
    python scripts/clipping/clip_solid_coastal.py --year 2017 --band B11

Reproduction and AOI adaptation
-------------------------------
Workflow role: Clip or reproject raster products to the coastal study boundary and analysis grid.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--year``, ``--band``, ``--base-dir``, ``--resolution``, ``--input``, ``--vector``, ``--output-dir``, ``--output-name``, ``--layer``, ``--cache-mb``, ``--compress``, ``--threads``, ``--block-size``, ``--resampling``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace the boundary vector and source raster paths. Select a projected CRS appropriate for the new AOI before area, distance, or 10 m grid operations.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path
from shutil import which


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASE_DIR = Path("data/raw/coastal_tiles")
DEFAULT_RASTER = DEFAULT_BASE_DIR / "2017" / "coastal_2017_10_B02.tif"
DEFAULT_VECTOR = Path("assets/maps/bd_coastal_map_solid_gp.gpkg")
DEFAULT_OUTPUT_DIR = Path("data/processed/clipping/2017")
DEFAULT_OUTPUT_NAME = "coastal_2017_10_B02_solid.tif"


def log(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}")


def ensure_tool(name: str) -> None:
    if which(name) is None:
        raise SystemExit(f"Required tool '{name}' was not found in PATH.")


def run_command(cmd: list[str], env: dict | None = None) -> None:
    log("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip the coastal Sentinel mosaic using the solid coastal AOI."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year folder under the coastal tiles root (optional).",
    )
    parser.add_argument(
        "--band",
        default=None,
        help="Band identifier like B02/B11 (optional).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=f"Base directory for coastal mosaics (default: {DEFAULT_BASE_DIR}).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=10,
        help="Pixel resolution suffix used in filenames (default: 10).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input raster path (default: derived from --year/--band or fallback).",
    )
    parser.add_argument(
        "--vector",
        type=Path,
        default=DEFAULT_VECTOR,
        help=f"AOI vector path (default: {DEFAULT_VECTOR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the clipped output (default: derived from --year or fallback).",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename (default: derived from --year/--band or fallback).",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Layer name for multi-layer vector files (default: first layer).",
    )
    parser.add_argument(
        "--cache-mb",
        type=int,
        default=1024,
        help="GDAL cache size in MB (default: 1024).",
    )
    parser.add_argument(
        "--compress",
        default="ZSTD",
        help="Compression codec for outputs (default: ZSTD).",
    )
    parser.add_argument(
        "--threads",
        default="ALL_CPUS",
        help="Thread hint passed to GDAL tools (default: ALL_CPUS).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Internal tile size for outputs (default: 512).",
    )
    parser.add_argument(
        "--resampling",
        default="near",
        help="gdalwarp resampling method (default: near).",
    )
    return parser.parse_args()


def clip_with_gdalwarp(
    src: Path,
    vector: Path,
    output: Path,
    *,
    layer: str | None,
    cache_mb: int,
    compress: str,
    threads: str,
    block_size: int,
    resampling: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    warp_cmd: list[str] = [
        "gdalwarp",
        "-overwrite",
        "-of",
        "GTiff",
        "-cutline",
        str(vector),
        "-crop_to_cutline",
        "-multi",
        "-wo",
        f"NUM_THREADS={threads}",
        "-co",
        "TILED=YES",
        "-co",
        f"BLOCKXSIZE={block_size}",
        "-co",
        f"BLOCKYSIZE={block_size}",
        "-co",
        f"COMPRESS={compress}",
        "-co",
        "BIGTIFF=YES",
        "-co",
        f"NUM_THREADS={threads}",
        "-r",
        resampling,
    ]

    if layer:
        warp_cmd.extend(["-cl", layer])

    warp_cmd.extend([str(src), str(output)])

    env = os.environ.copy()
    env["GDAL_CACHEMAX"] = str(cache_mb)

    log(f"Clipping {src} with {vector} -> {output}")
    run_command(warp_cmd, env=env)


def main() -> None:
    ensure_tool("gdalwarp")

    args = parse_args()
    args.base_dir = resolve_path(args.base_dir)
    args.vector = resolve_path(args.vector)

    if args.input is None:
        if args.year is not None and args.band is not None:
            args.input = (
                args.base_dir
                / str(args.year)
                / f"coastal_{args.year}_{args.resolution:02d}_{args.band}.tif"
            )
        else:
            args.input = resolve_path(DEFAULT_RASTER)
    else:
        args.input = resolve_path(args.input)

    if args.output_dir is None:
        args.output_dir = (
            args.base_dir / str(args.year)
            if args.year is not None
            else resolve_path(DEFAULT_OUTPUT_DIR)
        )
    else:
        args.output_dir = resolve_path(args.output_dir)

    if args.output_name is None:
        if args.year is not None and args.band is not None:
            args.output_name = (
                f"coastal_{args.year}_{args.resolution:02d}_{args.band}_solid.tif"
            )
        else:
            args.output_name = DEFAULT_OUTPUT_NAME

    output = args.output_dir / args.output_name

    if not args.input.exists():
        raise SystemExit(f"Input raster not found: {args.input}")
    if not args.vector.exists():
        raise SystemExit(f"Vector AOI not found: {args.vector}")

    clip_with_gdalwarp(
        args.input,
        args.vector,
        output,
        layer=args.layer,
        cache_mb=args.cache_mb,
        compress=args.compress,
        threads=args.threads,
        block_size=args.block_size,
        resampling=args.resampling,
    )
    log(f"Saved clipped raster to {output}")


if __name__ == "__main__":
    main()
