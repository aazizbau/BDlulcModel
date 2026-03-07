"""
Clip the Bangladesh coastal AlphaEarth mosaic to the dissolved coastal districts AOI.

This script optionally builds a temporary Cloud-Optimized GeoTIFF (COG) and then
uses GDAL's streaming tools to clip all 64 AlphaEarth bands without exhausting
RAM.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path
from shutil import which

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RASTER = Path("data/interim/bd_coastal_alphaearth_2024_mosaic.tif")
DEFAULT_VECTOR = Path("assets/maps/bd_coastal_districts_disso_gp.gpkg")
DEFAULT_OUTPUT = Path("processed/features/bd_coastal_alphaearth_2024.tif")
DEFAULT_COG = Path("data/interim/bd_coastal_alphaearth_2024_mosaic_cog.tif")


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


def parse_overview_levels(levels: str | None) -> list[int]:
    if not levels:
        return []
    return [int(value.strip()) for value in levels.split(",") if value.strip()]


def build_cog(
    src: Path,
    cog_path: Path,
    *,
    compress: str,
    block_size: int,
    threads: str,
    add_overviews: bool,
    overview_levels: list[int],
    overview_resampling: str,
) -> Path:
    if cog_path.exists():
        log(f"Using existing COG: {cog_path}")
        return cog_path

    log(f"Creating Cloud-Optimized GeoTIFF at {cog_path}")
    translate_cmd = [
        "gdal_translate",
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
        str(src),
        str(cog_path),
    ]
    run_command(translate_cmd)

    if add_overviews and overview_levels:
        log(
            "Adding overviews "
            + ",".join(map(str, overview_levels))
            + f" using {overview_resampling}"
        )
        over_cmd = [
            "gdaladdo",
            "-r",
            overview_resampling,
            str(cog_path),
            *[str(level) for level in overview_levels],
        ]
        run_command(over_cmd)

    return cog_path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip the AlphaEarth coastal mosaic using GDAL streaming tools.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_RASTER, help="Input mosaic path.")
    parser.add_argument("--vector", type=Path, default=DEFAULT_VECTOR, help="Clipping AOI vector path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output raster path.")
    parser.add_argument("--layer", default=None, help="Layer name inside the vector file (optional).")
    parser.add_argument("--cache-mb", type=int, default=1024, help="GDAL cache size in MB.")
    parser.add_argument("--compress", default="ZSTD", help="Compression codec (default: ZSTD).")
    parser.add_argument("--threads", default="ALL_CPUS", help="Thread hint passed to GDAL tools.")
    parser.add_argument("--block-size", type=int, default=512, help="Internal tile size (default: 512).")
    parser.add_argument("--resampling", default="near", help="gdalwarp resampling method (default: near).")
    parser.add_argument("--cog-path", type=Path, default=DEFAULT_COG, help="Intermediate COG path.")
    parser.add_argument("--skip-cog", action="store_true", help="Skip building the COG and clip the original mosaic directly.")
    parser.add_argument(
        "--overview-levels",
        default="2,4,8,16",
        help="Comma-separated overview factors for the COG (default: 2,4,8,16).",
    )
    parser.add_argument(
        "--overview-resampling",
        default="lanczos",
        help="Resampling algorithm for overviews (default: lanczos).",
    )
    parser.add_argument(
        "--no-overviews",
        action="store_true",
        help="Do not add overviews even when creating the COG.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_tool("gdalwarp")
    ensure_tool("gdal_translate")
    ensure_tool("gdaladdo")

    args = parse_args()
    args.input = resolve_path(args.input)
    args.vector = resolve_path(args.vector)
    args.output = resolve_path(args.output)
    args.cog_path = resolve_path(args.cog_path)

    if not args.input.exists():
        raise SystemExit(f"Input raster not found: {args.input}")
    if not args.vector.exists():
        raise SystemExit(f"Vector AOI not found: {args.vector}")

    raster = args.input
    if not args.skip_cog:
        overview_levels = parse_overview_levels(None if args.no_overviews else args.overview_levels)
        raster = build_cog(
            args.input,
            args.cog_path,
            compress=args.compress,
            block_size=args.block_size,
            threads=args.threads,
            add_overviews=bool(overview_levels),
            overview_levels=overview_levels,
            overview_resampling=args.overview_resampling,
        )

    clip_with_gdalwarp(
        raster,
        args.vector,
        args.output,
        layer=args.layer,
        cache_mb=args.cache_mb,
        compress=args.compress,
        threads=args.threads,
        block_size=args.block_size,
        resampling=args.resampling,
    )
    log(f"Saved clipped raster to {args.output}")


if __name__ == "__main__":
    main()
