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

Reproduction and AOI adaptation
-------------------------------
Workflow role: Normalize raster datatype or metadata before alignment and feature extraction.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--input``, ``--output``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace input/output raster paths and verify that conversion preserves CRS, transform, nodata, band order, and pixel dimensions.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
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
