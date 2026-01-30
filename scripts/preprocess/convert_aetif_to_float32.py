#!/usr/bin/env python3
"""
Convert a GeoTIFF to Float32 with optimized tiling and compression.

Equivalent to:

gdal_translate \
  -ot Float32 -a_nodata 0 \
  -co TILED=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 \
  -co COMPRESS=ZSTD -co ZSTD_LEVEL=15 -co PREDICTOR=3 \
  input.tif output.tif

Example:
    python scripts/preprocess/convert_aetif_to_float32.py \
        --input  data/interim/bd_coastal_fourupazila_alphaearth_2023_mosaic.tif \
        --output data/interim/bd_coastal_fourupazila_alphaearth_2023_mosaic_f32.tif
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from osgeo import gdal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GeoTIFF to Float32 with ZSTD compression and tiling"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input GeoTIFF (Float64 or other)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output GeoTIFF (Float32)",
    )
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}")


def main() -> None:
    args = parse_args()

    in_path = args.input
    out_path = args.output

    if not in_path.exists():
        sys.exit(f"[ERROR] Input file not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    log("Converting to Float32")
    log(f"Input : {in_path}")
    log(f"Output: {out_path}")

    gdal.UseExceptions()

    translate_options = gdal.TranslateOptions(
        outputType=gdal.GDT_Float32,
        noData=0,
        creationOptions=[
            "TILED=YES",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
            "COMPRESS=ZSTD",
            "ZSTD_LEVEL=15",
            "PREDICTOR=3",
            "BIGTIFF=YES",
        ],
    )

    gdal.Translate(
        destName=str(out_path),
        srcDS=str(in_path),
        options=translate_options,
    )

    log("Conversion completed successfully")


if __name__ == "__main__":
    main()
