#!/usr/bin/env python3
"""
Step 1 — Reproject AlphaEarth (EPSG:4326) -> UTM 46N (EPSG:32646) at 10 m
with pixel-grid alignment (-tap) for best label/feature matching later.

Uses GDAL Warp (Python bindings), with controlled CPU usage (no "ALL_CPUS").

Example:
  python scripts/features/warp_alphaearth_to_utm10.py \
    --input data/processed/features/bd_coastal_alphaearth_2023_clipped.tif \
    --output data/processed/features/bd_coastal_alphaearth_2023_utm46.tif
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from osgeo import gdal


def log(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Warp AlphaEarth embeddings to EPSG:32646 @ 10m, aligned grid (tap)."
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input AlphaEarth GeoTIFF (likely EPSG:4326).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output warped GeoTIFF (EPSG:32646).",
    )
    p.add_argument(
        "--dst-crs",
        type=str,
        default="EPSG:32646",
        help="Destination CRS (default: EPSG:32646).",
    )
    p.add_argument(
        "--res",
        type=float,
        default=10.0,
        help="Output pixel size in meters (default: 10).",
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
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Output data type (default: float32).",
    )
    p.add_argument(
        "--src-nodata",
        type=float,
        default=None,
        help="Source nodata value (default: read from dataset if present).",
    )
    p.add_argument(
        "--dst-nodata",
        type=float,
        default=None,
        help="Destination nodata value (default: use source nodata if available, else 0).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it exists.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_path = args.input
    out_path = args.output

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if args.overwrite:
            out_path.unlink()
        else:
            raise SystemExit(f"Output already exists (use --overwrite): {out_path}")

    gdal.SetConfigOption("NUM_THREADS", str(max(1, args.threads)))
    gdal.SetConfigOption("GDAL_NUM_THREADS", str(max(1, args.threads)))

    gdal.UseExceptions()

    src_ds = gdal.Open(str(in_path), gdal.GA_ReadOnly)
    if src_ds is None:
        raise SystemExit(f"Failed to open: {in_path}")

    band1 = src_ds.GetRasterBand(1)
    detected_src_nodata = band1.GetNoDataValue()
    src_nodata = detected_src_nodata if args.src_nodata is None else args.src_nodata

    if args.dst_nodata is not None:
        dst_nodata = args.dst_nodata
    else:
        dst_nodata = src_nodata if src_nodata is not None else 0.0

    out_dtype = gdal.GDT_Float32 if args.dtype == "float32" else gdal.GDT_Float64

    resample_map = {
        "near": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "cubicspline": gdal.GRA_CubicSpline,
        "lanczos": gdal.GRA_Lanczos,
        "average": gdal.GRA_Average,
    }
    resample_alg = resample_map[args.resampling]

    creation_options = [
        "COMPRESS=ZSTD",
        "TILED=YES",
        "BLOCKXSIZE=512",
        "BLOCKYSIZE=512",
        "PREDICTOR=2",
        "BIGTIFF=IF_SAFER",
    ]

    warp_opts = gdal.WarpOptions(
        dstSRS=args.dst_crs,
        xRes=args.res,
        yRes=args.res,
        targetAlignedPixels=True,
        resampleAlg=resample_alg,
        srcNodata=src_nodata,
        dstNodata=dst_nodata,
        outputType=out_dtype,
        multithread=True,
        creationOptions=creation_options,
    )

    log("Warping AlphaEarth to UTM grid for training alignment")
    log(f"input      : {in_path}")
    log(f"output     : {out_path}")
    log(f"dst CRS    : {args.dst_crs}")
    log(f"resolution : {args.res} m")
    log(f"resampling : {args.resampling}")
    log(f"threads    : {args.threads}")
    log(f"dtype      : {args.dtype}")
    log(f"src nodata : {src_nodata}")
    log(f"dst nodata : {dst_nodata}")

    dst_ds = gdal.Warp(destNameOrDestDS=str(out_path), srcDSOrSrcDSTab=src_ds, options=warp_opts)
    if dst_ds is None:
        raise SystemExit("gdal.Warp failed (no output produced).")

    dst_ds.FlushCache()
    dst_ds = None
    src_ds = None

    log(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
