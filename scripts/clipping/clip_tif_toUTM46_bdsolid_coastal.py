#!/usr/bin/env python3
"""
Clip a BD coastal Sentinel raster to the solid coastal AOI and reproject to UTM 46N.

Examples
--------
Auto-derived input/output:
    python scripts/clipping/clip_tif_toUTM46_bdsolid_coastal.py --year 2017 --band B2

Explicit input/output:
    python scripts/clipping/clip_tif_toUTM46_bdsolid_coastal.py \
        --year 2017 \
        --band B11 \
        --input data/raw/sentinel_gemini/BD_COASTAL_BBOX/S2_2017_octdec_B11_resampled_10m_lzw.tif \
        --output data/interim/S2_2017_B11_10m_utm46_bdcoastal_solid.tif

Custom CRS:
    python scripts/clipping/clip_tif_toUTM46_bdsolid_coastal.py --year 2017 --band B2 --crs EPSG:32646

Reproduction and AOI adaptation
-------------------------------
Workflow role: Clip or reproject raster products to the coastal study boundary and analysis grid.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--year``, ``--band``, ``--crs``, ``--input``, ``--output``, ``--input-dir``, ``--output-dir``, ``--vector``, ``--layer``, ``--dst-nodata``, ``--resampling``, ``--cache-mb``, ``--compress``, ``--threads``, ``--block-size``. Run the ``--help`` command below for required values, defaults, and accepted choices.
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

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sentinel_gemini" / "BD_COASTAL_BBOX"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"
DEFAULT_VECTOR = PROJECT_ROOT / "assets" / "maps" / "bd_coastal_map_solid_gp.gpkg"
DEFAULT_CRS = "EPSG:32646"


def log(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}")


def ensure_tool(name: str) -> None:
    if which(name) is None:
        raise SystemExit(f"Required tool '{name}' was not found in PATH.")


def run_command(cmd: list[str], env: dict | None = None) -> None:
    log("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_band(band: str) -> str:
    """
    Normalize user input like B2, b2, B02 -> B2
    """
    band = band.strip().upper()
    if not band.startswith("B"):
        raise SystemExit(f"Invalid band '{band}'. Expected values like B2, B3, B4, B8, B11, B12.")

    suffix = band[1:]
    if not suffix.isdigit():
        raise SystemExit(f"Invalid band '{band}'. Expected numeric suffix after 'B'.")

    return f"B{int(suffix)}"


def default_input_path(year: int, band: str, input_dir: Path) -> Path:
    """
    Derive the default input path from year and band.

    For B11/B12:
        S2_<year>_octdec_<band>_resampled_10m_lzw.tif
    For others:
        S2_<year>_octdec_<band>_mosaic_10m_lzw.tif
    """
    if band in {"B11", "B12"}:
        name = f"S2_{year}_octdec_{band}_resampled_10m_lzw.tif"
    else:
        name = f"S2_{year}_octdec_{band}_mosaic_10m_lzw.tif"
    return input_dir / name


def default_output_path(year: int, band: str, output_dir: Path) -> Path:
    """
    Derive standardized output path.
    """
    name = f"S2_{year}_{band}_10m_utm46_bdcoastal_solid.tif"
    return output_dir / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip BD coastal Sentinel raster to solid coastal AOI and reproject to UTM 46N."
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year of the raster, e.g. 2017.",
    )
    parser.add_argument(
        "--band",
        required=True,
        help="Band name, e.g. B2, B3, B4, B8, B11, B12.",
    )

    parser.add_argument(
        "--crs",
        default=DEFAULT_CRS,
        help=f"Target CRS for output (default: {DEFAULT_CRS}).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional explicit input raster path. If omitted, derived from --year and --band.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output raster path. If omitted, derived from --year and --band.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Base input directory for auto-derived input path (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Base output directory for auto-derived output path (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--vector",
        type=Path,
        default=DEFAULT_VECTOR,
        help=f"Solid coastal AOI vector path (default: {DEFAULT_VECTOR}).",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Optional layer name for multi-layer vector files.",
    )
    parser.add_argument(
        "--dst-nodata",
        type=float,
        default=None,
        help="Optional destination nodata value. If omitted, GDAL default behavior is used.",
    )
    parser.add_argument(
        "--resampling",
        default="near",
        help="Resampling method for gdalwarp (default: near).",
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
        help="Compression codec for output GeoTIFF (default: ZSTD).",
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
        help="Internal tile block size (default: 512).",
    )

    return parser.parse_args()


def clip_and_reproject(
    src: Path,
    vector: Path,
    output: Path,
    *,
    target_crs: str,
    layer: str | None,
    dst_nodata: float | None,
    resampling: str,
    cache_mb: int,
    compress: str,
    threads: str,
    block_size: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "gdalwarp",
        "-overwrite",
        "-of", "GTiff",
        "-cutline", str(vector),
        "-crop_to_cutline",
        "-t_srs", target_crs,
        "-multi",
        "-wo", f"NUM_THREADS={threads}",
        "-r", resampling,
        "-co", "TILED=YES",
        "-co", f"BLOCKXSIZE={block_size}",
        "-co", f"BLOCKYSIZE={block_size}",
        "-co", f"COMPRESS={compress}",
        "-co", "BIGTIFF=YES",
        "-co", f"NUM_THREADS={threads}",
    ]

    if layer:
        cmd.extend(["-cl", layer])

    if dst_nodata is not None:
        cmd.extend(["-dstnodata", str(dst_nodata)])

    cmd.extend([str(src), str(output)])

    env = os.environ.copy()
    env["GDAL_CACHEMAX"] = str(cache_mb)

    log(f"Clipping and reprojecting: {src}")
    log(f"Using vector AOI: {vector}")
    log(f"Target CRS: {target_crs}")
    log(f"Output: {output}")

    run_command(cmd, env=env)


def validate_output_with_gdalinfo(output: Path) -> None:
    """
    Print final CRS, raster size, and pixel size using gdalinfo.
    """
    log(f"Validating output with gdalinfo: {output}")

    cmd = ["gdalinfo", str(output)]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    info = result.stdout.splitlines()

    crs_lines: list[str] = []
    raster_size_line: str | None = None
    pixel_size_line: str | None = None

    capture_crs = False
    for line in info:
        stripped = line.strip()

        if stripped.startswith("Size is "):
            raster_size_line = stripped

        if stripped.startswith("Pixel Size ="):
            pixel_size_line = stripped

        if stripped.startswith("Coordinate System is:"):
            capture_crs = True
            crs_lines.append(stripped)
            continue

        if capture_crs:
            if not stripped:
                capture_crs = False
            else:
                crs_lines.append(stripped)

    log("GDALINFO VALIDATION SUMMARY")
    if crs_lines:
        for line in crs_lines:
            print(line)
    else:
        print("Coordinate System is: [not found]")

    print(raster_size_line if raster_size_line else "Size is [not found]")
    print(pixel_size_line if pixel_size_line else "Pixel Size = [not found]")


def main() -> None:
    ensure_tool("gdalwarp")
    ensure_tool("gdalinfo")

    args = parse_args()
    band = normalize_band(args.band)

    args.input_dir = resolve_path(args.input_dir)
    args.output_dir = resolve_path(args.output_dir)
    args.vector = resolve_path(args.vector)
    if args.input is not None:
        args.input = resolve_path(args.input)
    if args.output is not None:
        args.output = resolve_path(args.output)

    input_path = args.input if args.input is not None else default_input_path(
        year=args.year,
        band=band,
        input_dir=args.input_dir,
    )

    output_path = args.output if args.output is not None else default_output_path(
        year=args.year,
        band=band,
        output_dir=args.output_dir,
    )

    if not input_path.exists():
        raise SystemExit(f"Input raster not found: {input_path}")

    if not args.vector.exists():
        raise SystemExit(f"Vector AOI not found: {args.vector}")

    clip_and_reproject(
        src=input_path,
        vector=args.vector,
        output=output_path,
        target_crs=args.crs,
        layer=args.layer,
        dst_nodata=args.dst_nodata,
        resampling=args.resampling,
        cache_mb=args.cache_mb,
        compress=args.compress,
        threads=args.threads,
        block_size=args.block_size,
    )

    log(f"Saved output to: {output_path}")
    validate_output_with_gdalinfo(output_path)


if __name__ == "__main__":
    main()
