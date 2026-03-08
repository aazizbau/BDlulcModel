#!/usr/bin/env python3
"""
Create AWEI(sh) GeoTIFF from Sentinel-2 coastal solid rasters.

AWEI(sh) = Blue + 2.5 * Green - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2

Expected default input files for --year 2017:
  data/interim/S2_2017_B2_10m_utm46_bdcoastal_solid.tif
  data/interim/S2_2017_B3_10m_utm46_bdcoastal_solid.tif
  data/interim/S2_2017_B8_10m_utm46_bdcoastal_solid.tif
  data/interim/S2_2017_B11_10m_utm46_bdcoastal_solid.tif
  data/interim/S2_2017_B12_10m_utm46_bdcoastal_solid.tif

Expected default output file:
  data/interim/bdcoastal_solid_2017_utm46_awei_sh.tif

Example runs:
python scripts/s2_indices/make_awei_sh_image.py --year 2017

python scripts/s2_indices/make_awei_sh_image.py \
  --year 2017 \
  --crs EPSG:32646 \
  --output data/interim/bdcoastal_solid_2017_utm46_awei_sh.tif

python scripts/s2_indices/make_awei_sh_image.py \
  --year 2017 \
  --blue data/interim/S2_2017_B2_10m_utm46_bdcoastal_solid.tif \
  --green data/interim/S2_2017_B3_10m_utm46_bdcoastal_solid.tif \
  --nir data/interim/S2_2017_B8_10m_utm46_bdcoastal_solid.tif \
  --swir1 data/interim/S2_2017_B11_10m_utm46_bdcoastal_solid.tif \
  --swir2 data/interim/S2_2017_B12_10m_utm46_bdcoastal_solid.tif \
  --output data/interim/bdcoastal_solid_2017_utm46_awei_sh.tif
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "interim"
DEFAULT_CRS = "EPSG:32646"
DEFAULT_NODATA_IN = 65535
DEFAULT_NODATA_OUT = -9999.0
JST = timezone(timedelta(hours=9))


def ts() -> str:
    return datetime.now(JST).strftime("[%Y-%m-%dT%H:%M:%S%z]")


def iso_now_jst() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{ts()} {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make AWEI(sh) image from Sentinel-2 B2, B3, B8, B11, and B12 rasters."
    )
    parser.add_argument("--year", type=int, required=True, help="Year, e.g. 2017.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f'Input directory (default: "{DEFAULT_INPUT_DIR}").',
    )
    parser.add_argument(
        "--crs",
        type=str,
        default=DEFAULT_CRS,
        help=f'Expected CRS (default: "{DEFAULT_CRS}").',
    )
    parser.add_argument("--blue", type=Path, default=None, help="Optional custom B2 path.")
    parser.add_argument("--green", type=Path, default=None, help="Optional custom B3 path.")
    parser.add_argument("--nir", type=Path, default=None, help="Optional custom B8 path.")
    parser.add_argument("--swir1", type=Path, default=None, help="Optional custom B11 path.")
    parser.add_argument("--swir2", type=Path, default=None, help="Optional custom B12 path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output AWEI(sh) path.")
    parser.add_argument(
        "--input-nodata",
        type=int,
        default=DEFAULT_NODATA_IN,
        help=f"Fallback input nodata (default: {DEFAULT_NODATA_IN}).",
    )
    parser.add_argument(
        "--output-nodata",
        type=float,
        default=DEFAULT_NODATA_OUT,
        help=f"Output nodata (default: {DEFAULT_NODATA_OUT}).",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path]:
    blue_path = args.blue or (args.input_dir / f"S2_{args.year}_B2_10m_utm46_bdcoastal_solid.tif")
    green_path = args.green or (args.input_dir / f"S2_{args.year}_B3_10m_utm46_bdcoastal_solid.tif")
    nir_path = args.nir or (args.input_dir / f"S2_{args.year}_B8_10m_utm46_bdcoastal_solid.tif")
    swir1_path = args.swir1 or (args.input_dir / f"S2_{args.year}_B11_10m_utm46_bdcoastal_solid.tif")
    swir2_path = args.swir2 or (args.input_dir / f"S2_{args.year}_B12_10m_utm46_bdcoastal_solid.tif")
    out_path = args.output or (args.input_dir / f"bdcoastal_solid_{args.year}_utm46_awei_sh.tif")
    return blue_path, green_path, nir_path, swir1_path, swir2_path, out_path


def same_transform(a, b, tol: float = 1e-9) -> bool:
    return (
        abs(a.a - b.a) < tol
        and abs(a.b - b.b) < tol
        and abs(a.c - b.c) < tol
        and abs(a.d - b.d) < tol
        and abs(a.e - b.e) < tol
        and abs(a.f - b.f) < tol
    )


def compute_awei_sh(
    blue: np.ndarray,
    green: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    swir2: np.ndarray,
    nodata_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute AWEI(sh) (reflectance-based)."""
    blue = blue.astype("float32") / 10000.0
    green = green.astype("float32") / 10000.0
    nir = nir.astype("float32") / 10000.0
    swir1 = swir1.astype("float32") / 10000.0
    swir2 = swir2.astype("float32") / 10000.0

    awei = blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2
    mask = nodata_mask if nodata_mask is not None else np.zeros_like(awei, dtype=bool)
    awei[mask] = np.nan
    return awei.astype("float32")


def compute_awei_sh_block(
    blue: np.ndarray,
    green: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    swir2: np.ndarray,
    blue_nodata: float | int | None,
    green_nodata: float | int | None,
    nir_nodata: float | int | None,
    swir1_nodata: float | int | None,
    swir2_nodata: float | int | None,
    output_nodata: float,
) -> tuple[np.ndarray, int]:
    nodata_mask = np.zeros(blue.shape, dtype=bool)

    if blue_nodata is not None:
        nodata_mask |= blue == blue_nodata
    if green_nodata is not None:
        nodata_mask |= green == green_nodata
    if nir_nodata is not None:
        nodata_mask |= nir == nir_nodata
    if swir1_nodata is not None:
        nodata_mask |= swir1 == swir1_nodata
    if swir2_nodata is not None:
        nodata_mask |= swir2 == swir2_nodata

    awei = compute_awei_sh(
        blue=blue,
        green=green,
        nir=nir,
        swir1=swir1,
        swir2=swir2,
        nodata_mask=nodata_mask,
    )

    valid = np.isfinite(awei)
    valid_count = int(valid.sum())

    out = np.full(awei.shape, output_nodata, dtype=np.float32)
    out[valid] = awei[valid]
    return out, valid_count


def compute_raster_stats_blockwise(
    raster_path: Path,
    nodata_value: float | int | None,
) -> dict[str, str]:
    count_valid = 0
    sum_valid = 0.0
    sumsq_valid = 0.0
    min_val = None
    max_val = None

    with rasterio.open(raster_path) as src:
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window)

            valid = np.isfinite(arr)
            if nodata_value is not None:
                valid &= arr != nodata_value

            if not np.any(valid):
                continue

            vals = arr[valid].astype(np.float64, copy=False)

            block_min = float(vals.min())
            block_max = float(vals.max())

            if min_val is None or block_min < min_val:
                min_val = block_min
            if max_val is None or block_max > max_val:
                max_val = block_max

            count_valid += vals.size
            sum_valid += float(vals.sum())
            sumsq_valid += float(np.square(vals, dtype=np.float64).sum())

    if count_valid == 0:
        return {"min": "None", "max": "None", "mean": "None", "std": "None"}

    mean_val = sum_valid / count_valid
    variance = max(0.0, (sumsq_valid / count_valid) - (mean_val * mean_val))
    std_val = variance ** 0.5

    return {
        "min": f"{min_val:.10f}",
        "max": f"{max_val:.10f}",
        "mean": f"{mean_val:.10f}",
        "std": f"{std_val:.10f}",
    }


def print_output_summary(out_path: Path) -> None:
    with rasterio.open(out_path) as src:
        tags = src.tags()
        transform = src.transform

        log("Output validation:")
        log(f"  Path       : {out_path}")
        log(f"  CRS        : {src.crs.to_string() if src.crs else 'None'}")
        log(f"  Raster size: {src.width} x {src.height}")
        log(f"  Pixel size : ({transform.a}, {transform.e})")
        log(f"  Nodata     : {src.nodata}")
        log(f"  Dtype      : {src.dtypes[0]}")

        if "AWEI_SH_MIN" in tags and "AWEI_SH_MAX" in tags:
            log(f"  Min/Max    : {tags['AWEI_SH_MIN']} / {tags['AWEI_SH_MAX']}")
        if "AWEI_SH_MEAN" in tags and "AWEI_SH_STD" in tags:
            log(f"  Mean/Std   : {tags['AWEI_SH_MEAN']} / {tags['AWEI_SH_STD']}")

        log("  Embedded metadata tags:")
        for k in sorted(tags):
            log(f"    {k}={tags[k]}")


def main() -> None:
    args = parse_args()
    args.input_dir = resolve_path(args.input_dir)
    if args.blue is not None:
        args.blue = resolve_path(args.blue)
    if args.green is not None:
        args.green = resolve_path(args.green)
    if args.nir is not None:
        args.nir = resolve_path(args.nir)
    if args.swir1 is not None:
        args.swir1 = resolve_path(args.swir1)
    if args.swir2 is not None:
        args.swir2 = resolve_path(args.swir2)
    if args.output is not None:
        args.output = resolve_path(args.output)

    blue_path, green_path, nir_path, swir1_path, swir2_path, out_path = resolve_paths(args)

    log("Starting AWEI(sh) creation")
    log(f"Year         : {args.year}")
    log(f"Blue (B2)    : {blue_path}")
    log(f"Green (B3)   : {green_path}")
    log(f"NIR (B8)     : {nir_path}")
    log(f"SWIR1 (B11)  : {swir1_path}")
    log(f"SWIR2 (B12)  : {swir2_path}")
    log(f"Output       : {out_path}")
    log(f"Expected CRS : {args.crs}")
    log(f"Input nodata : {args.input_nodata}")
    log(f"Output nodata: {args.output_nodata}")

    for p, label in [
        (blue_path, "Blue"),
        (green_path, "Green"),
        (nir_path, "NIR"),
        (swir1_path, "SWIR1"),
        (swir2_path, "SWIR2"),
    ]:
        if not p.exists():
            raise SystemExit(f"ERROR: {label} band file not found: {p}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_valid_pixels = 0

    with (
        rasterio.open(blue_path) as blue_src,
        rasterio.open(green_path) as green_src,
        rasterio.open(nir_path) as nir_src,
        rasterio.open(swir1_path) as swir1_src,
        rasterio.open(swir2_path) as swir2_src,
    ):
        if any(src.count != 1 for src in [blue_src, green_src, nir_src, swir1_src, swir2_src]):
            raise SystemExit("ERROR: Expected single-band rasters for B2, B3, B8, B11, and B12.")

        width = blue_src.width
        height = blue_src.height
        for label, src in [
            ("B3", green_src),
            ("B8", nir_src),
            ("B11", swir1_src),
            ("B12", swir2_src),
        ]:
            if src.width != width or src.height != height:
                raise SystemExit(
                    f"ERROR: Raster size mismatch. "
                    f"B2={width}x{height}, {label}={src.width}x{src.height}"
                )

        for label, src in [("B3", green_src), ("B8", nir_src), ("B11", swir1_src), ("B12", swir2_src)]:
            if blue_src.crs != src.crs:
                raise SystemExit(f"ERROR: CRS mismatch. B2={blue_src.crs}, {label}={src.crs}")

        if blue_src.crs is None:
            raise SystemExit("ERROR: Input rasters have no CRS.")

        if blue_src.crs.to_string() != args.crs:
            raise SystemExit(
                f"ERROR: Input CRS is {blue_src.crs.to_string()} but expected {args.crs}."
            )

        for label, src in [("B3", green_src), ("B8", nir_src), ("B11", swir1_src), ("B12", swir2_src)]:
            if not same_transform(blue_src.transform, src.transform):
                raise SystemExit(f"ERROR: B2 and {label} transforms differ. Align rasters first.")

        blue_nodata = blue_src.nodata if blue_src.nodata is not None else args.input_nodata
        green_nodata = green_src.nodata if green_src.nodata is not None else args.input_nodata
        nir_nodata = nir_src.nodata if nir_src.nodata is not None else args.input_nodata
        swir1_nodata = swir1_src.nodata if swir1_src.nodata is not None else args.input_nodata
        swir2_nodata = swir2_src.nodata if swir2_src.nodata is not None else args.input_nodata

        profile = blue_src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=args.output_nodata,
            compress="ZSTD",
            predictor=3,
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )

        log("Writing AWEI(sh) raster by blocks...")
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in blue_src.block_windows(1):
                blue = blue_src.read(1, window=window)
                green = green_src.read(1, window=window)
                nir = nir_src.read(1, window=window)
                swir1 = swir1_src.read(1, window=window)
                swir2 = swir2_src.read(1, window=window)

                awei, valid_count = compute_awei_sh_block(
                    blue=blue,
                    green=green,
                    nir=nir,
                    swir1=swir1,
                    swir2=swir2,
                    blue_nodata=blue_nodata,
                    green_nodata=green_nodata,
                    nir_nodata=nir_nodata,
                    swir1_nodata=swir1_nodata,
                    swir2_nodata=swir2_nodata,
                    output_nodata=args.output_nodata,
                )
                total_valid_pixels += valid_count
                dst.write(awei, 1, window=window)

        log("Computing AWEI(sh) statistics block by block...")
        awei_stats = compute_raster_stats_blockwise(
            out_path,
            nodata_value=args.output_nodata,
        )

        with rasterio.open(out_path, "r+") as dst:
            total_pixels = dst.width * dst.height
            valid_pixels = total_valid_pixels
            nodata_pixels = int(total_pixels - valid_pixels)

            dst.update_tags(
                AREA_OR_POINT="Area",
                INDEX_NAME="AWEI_SH",
                INDEX_FORMULA="B2 + 2.5*B3 - 1.5*(B8 + B11) - 0.25*B12",
                INDEX_DESCRIPTION="Automated Water Extraction Index (shadow)",
                YEAR=str(args.year),
                INPUT_BLUE=str(blue_path),
                INPUT_GREEN=str(green_path),
                INPUT_NIR=str(nir_path),
                INPUT_SWIR1=str(swir1_path),
                INPUT_SWIR2=str(swir2_path),
                INPUT_BLUE_BAND="B2",
                INPUT_GREEN_BAND="B3",
                INPUT_NIR_BAND="B8",
                INPUT_SWIR1_BAND="B11",
                INPUT_SWIR2_BAND="B12",
                INPUT_BLUE_DTYPE=blue_src.dtypes[0],
                INPUT_GREEN_DTYPE=green_src.dtypes[0],
                INPUT_NIR_DTYPE=nir_src.dtypes[0],
                INPUT_SWIR1_DTYPE=swir1_src.dtypes[0],
                INPUT_SWIR2_DTYPE=swir2_src.dtypes[0],
                INPUT_BLUE_NODATA=str(blue_nodata),
                INPUT_GREEN_NODATA=str(green_nodata),
                INPUT_NIR_NODATA=str(nir_nodata),
                INPUT_SWIR1_NODATA=str(swir1_nodata),
                INPUT_SWIR2_NODATA=str(swir2_nodata),
                REFLECTANCE_SCALE="0.0001",
                EXPECTED_CRS=args.crs,
                OUTPUT_NODATA=str(args.output_nodata),
                CREATED_AT_JST=iso_now_jst(),
                SCRIPT_NAME=Path(sys.argv[0]).name,
                COMMAND=" ".join(sys.argv),
                WIDTH=str(dst.width),
                HEIGHT=str(dst.height),
                COUNT=str(dst.count),
                DTYPE=dst.dtypes[0],
                PIXEL_SIZE_X=str(dst.transform.a),
                PIXEL_SIZE_Y=str(dst.transform.e),
                ORIGIN_X=str(dst.transform.c),
                ORIGIN_Y=str(dst.transform.f),
                VALID_PIXEL_COUNT=str(valid_pixels),
                NODATA_PIXEL_COUNT=str(nodata_pixels),
                VALID_FRACTION=f"{valid_pixels / total_pixels:.10f}" if total_pixels > 0 else "None",
                AWEI_SH_MIN=awei_stats["min"],
                AWEI_SH_MAX=awei_stats["max"],
                AWEI_SH_MEAN=awei_stats["mean"],
                AWEI_SH_STD=awei_stats["std"],
            )

        log(f"AWEI(sh) raster written successfully (valid pixels written: {total_valid_pixels})")

    print_output_summary(out_path)
    log("Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user")
        sys.exit(130)
