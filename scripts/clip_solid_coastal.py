"""
Clip the coastal Sentinel mosaic to the solid coastal AOI using GDAL streaming tools.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path
from shutil import which


DEFAULT_RASTER = Path("/media/abdul-aziz/345E19F75E19B29A/bd_coastal_tiles/2017/coastal_2017_10_B02.tif")
DEFAULT_VECTOR = Path("/media/abdul-aziz/sdb7/masters_research/bd_coastal_map/bd_coastal_map_solid_gp.gpkg")
DEFAULT_OUTPUT_DIR = Path("/media/abdul-aziz/345E19F75E19B29A/bd_coastal_tiles/2017")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip the coastal Sentinel mosaic using the solid coastal AOI."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RASTER,
        help=f"Input raster path (default: {DEFAULT_RASTER}).",
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
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the clipped output (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output filename (default: {DEFAULT_OUTPUT_NAME}).",
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
