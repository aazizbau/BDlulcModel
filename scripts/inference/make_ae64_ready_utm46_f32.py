#!/usr/bin/env python3
"""
Prepare a full-coast AlphaEarth raster for inference by converting it to:
- Float32
- EPSG:32646
- 10 m

Default behavior for a given year:
- input : data/processed/features/bd_coastal_alphaearth_<year>_clipped.tif
- output: data/interim/bd_coastal_alphaearth_<year>_utm46_f32.tif

Example:
python scripts/inference/make_ae64_ready_utm46_f32.py \
    --year 2017

Explicit paths:
python scripts/inference/make_ae64_ready_utm46_f32.py \
    --year 2024 \
    --input data/processed/features/bd_coastal_alphaearth_2024_clipped.tif \
    --output data/interim/bd_coastal_alphaearth_2024_utm46_f32.tif
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from osgeo import gdal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DST_CRS = "EPSG:32646"
DEFAULT_RES = 10.0


def log(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_input_for_year(year: int) -> Path:
    return Path(f"data/processed/features/bd_coastal_alphaearth_{year}_clipped.tif")


def default_output_for_year(year: int) -> Path:
    return Path(f"data/interim/bd_coastal_alphaearth_{year}_utm46_f32.tif")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare AlphaEarth raster for inference in EPSG:32646 @ 10m Float32."
    )
    p.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to prepare, e.g. 2017 or 2024.",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional explicit input AlphaEarth raster.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output raster.",
    )
    p.add_argument(
        "--dst-crs",
        type=str,
        default=DEFAULT_DST_CRS,
        help=f"Destination CRS (default: {DEFAULT_DST_CRS}).",
    )
    p.add_argument(
        "--res",
        type=float,
        default=DEFAULT_RES,
        help=f"Output resolution in meters (default: {DEFAULT_RES}).",
    )
    p.add_argument(
        "--resampling",
        type=str,
        default="bilinear",
        choices=["near", "bilinear", "cubic", "cubicspline", "lanczos", "average"],
        help="Resampling method (default: bilinear).",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads GDAL may use (default: 4).",
    )
    p.add_argument(
        "--src-nodata",
        type=float,
        default=None,
        help="Optional source nodata override. Default: read from input.",
    )
    p.add_argument(
        "--dst-nodata",
        type=float,
        default=0.0,
        help="Destination nodata value (default: 0.0).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = resolve_path(args.input or default_input_for_year(args.year))
    output_path = resolve_path(args.output or default_output_for_year(args.year))

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if args.overwrite:
            output_path.unlink()
        else:
            raise SystemExit(f"Output already exists (use --overwrite): {output_path}")

    gdal.SetConfigOption("NUM_THREADS", str(max(1, args.threads)))
    gdal.SetConfigOption("GDAL_NUM_THREADS", str(max(1, args.threads)))
    gdal.UseExceptions()

    src_ds = gdal.Open(str(input_path), gdal.GA_ReadOnly)
    if src_ds is None:
        raise SystemExit(f"Failed to open input: {input_path}")

    band1 = src_ds.GetRasterBand(1)
    detected_src_nodata = band1.GetNoDataValue()
    src_nodata = detected_src_nodata if args.src_nodata is None else args.src_nodata

    resample_map = {
        "near": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "cubicspline": gdal.GRA_CubicSpline,
        "lanczos": gdal.GRA_Lanczos,
        "average": gdal.GRA_Average,
    }

    creation_options = [
        "COMPRESS=ZSTD",
        "TILED=YES",
        "BLOCKXSIZE=512",
        "BLOCKYSIZE=512",
        "PREDICTOR=3",
        "BIGTIFF=IF_SAFER",
        "INTERLEAVE=PIXEL",
    ]

    warp_opts = gdal.WarpOptions(
        dstSRS=args.dst_crs,
        xRes=args.res,
        yRes=args.res,
        targetAlignedPixels=True,
        resampleAlg=resample_map[args.resampling],
        srcNodata=src_nodata,
        dstNodata=args.dst_nodata,
        outputType=gdal.GDT_Float32,
        multithread=True,
        creationOptions=creation_options,
    )

    log("Preparing AlphaEarth raster for inference")
    log(f"year       : {args.year}")
    log(f"input      : {input_path}")
    log(f"output     : {output_path}")
    log(f"dst CRS    : {args.dst_crs}")
    log(f"resolution : {args.res} m")
    log(f"resampling : {args.resampling}")
    log(f"threads    : {args.threads}")
    log(f"src nodata : {src_nodata}")
    log(f"dst nodata : {args.dst_nodata}")
    log("output type: Float32")

    dst_ds = gdal.Warp(destNameOrDestDS=str(output_path), srcDSOrSrcDSTab=src_ds, options=warp_opts)
    if dst_ds is None:
        raise SystemExit("gdal.Warp failed (no output produced).")

    dst_ds.FlushCache()
    dst_ds = None
    src_ds = None

    log(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
