#!/usr/bin/env python3
"""
Create MSAVI GeoTIFF from Sentinel-2 coastal solid rasters.

MSAVI = (2 * NIR + 1 - sqrt((2 * NIR + 1)^2 - 8 * (NIR - Red))) / 2

Expected default input files for --year 2017:
  data/interim/S2_2017_B4_10m_utm46_bdcoastal_solid.tif
  data/interim/S2_2017_B8_10m_utm46_bdcoastal_solid.tif

Expected default output file:
  data/interim/bdcoastal_solid_2017_utm46_msavi.tif

Example runs:
python scripts/s2_indices/make_msavi_image.py --year 2017

python scripts/s2_indices/make_msavi_image.py \
  --year 2017 \
  --crs EPSG:32646 \
  --output data/interim/bdcoastal_solid_2017_utm46_msavi.tif

python scripts/s2_indices/make_msavi_image.py \
  --year 2017 \
  --red data/interim/S2_2017_B4_10m_utm46_bdcoastal_solid.tif \
  --nir data/interim/S2_2017_B8_10m_utm46_bdcoastal_solid.tif \
  --output data/interim/bdcoastal_solid_2017_utm46_msavi.tif

Reproduction and AOI adaptation
-------------------------------
Workflow role: Calculate Sentinel-2 spectral indices from aligned reflectance bands.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--year``, ``--input-dir``, ``--crs``, ``--red``, ``--nir``, ``--output``, ``--input-nodata``, ``--output-nodata``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace band paths with aligned reflectance rasters for the target AOI and keep nodata masks, grid geometry, and scale factors consistent.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
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
        description="Make MSAVI image from Sentinel-2 B4 (red) and B8 (NIR) rasters."
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
    parser.add_argument(
        "--red",
        type=Path,
        default=None,
        help="Optional custom B4 path.",
    )
    parser.add_argument(
        "--nir",
        type=Path,
        default=None,
        help="Optional custom B8 path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output MSAVI path.",
    )
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


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    red_path = args.red or (args.input_dir / f"S2_{args.year}_B4_10m_utm46_bdcoastal_solid.tif")
    nir_path = args.nir or (args.input_dir / f"S2_{args.year}_B8_10m_utm46_bdcoastal_solid.tif")
    out_path = args.output or (args.input_dir / f"bdcoastal_solid_{args.year}_utm46_msavi.tif")
    return red_path, nir_path, out_path


def same_transform(a, b, tol: float = 1e-9) -> bool:
    return (
        abs(a.a - b.a) < tol
        and abs(a.b - b.b) < tol
        and abs(a.c - b.c) < tol
        and abs(a.d - b.d) < tol
        and abs(a.e - b.e) < tol
        and abs(a.f - b.f) < tol
    )


def compute_msavi(
    red: np.ndarray,
    nir: np.ndarray,
    nodata_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute MSAVI (reflectance-based)."""
    red = red.astype("float32") / 10000.0
    nir = nir.astype("float32") / 10000.0

    term = (2.0 * nir + 1.0)
    discriminant = term * term - 8.0 * (nir - red)
    discriminant = np.maximum(discriminant, 0.0)

    msavi = np.empty_like(nir, dtype="float32")
    mask = nodata_mask if nodata_mask is not None else np.zeros_like(nir, dtype=bool)
    valid = ~mask

    msavi[valid] = (term[valid] - np.sqrt(discriminant[valid])) / 2.0
    msavi[mask] = np.nan
    return msavi


def compute_msavi_block(
    red: np.ndarray,
    nir: np.ndarray,
    red_nodata: float | int | None,
    nir_nodata: float | int | None,
    output_nodata: float,
) -> tuple[np.ndarray, int]:
    nodata_mask = np.zeros(red.shape, dtype=bool)

    if red_nodata is not None:
        nodata_mask |= red == red_nodata
    if nir_nodata is not None:
        nodata_mask |= nir == nir_nodata

    msavi = compute_msavi(
        red=red,
        nir=nir,
        nodata_mask=nodata_mask,
    )

    valid = np.isfinite(msavi)
    valid_count = int(valid.sum())

    out = np.full(msavi.shape, output_nodata, dtype=np.float32)
    out[valid] = msavi[valid]
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
        return {
            "min": "None",
            "max": "None",
            "mean": "None",
            "std": "None",
        }

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

        if "MSAVI_MIN" in tags and "MSAVI_MAX" in tags:
            log(f"  Min/Max    : {tags['MSAVI_MIN']} / {tags['MSAVI_MAX']}")
        if "MSAVI_MEAN" in tags and "MSAVI_STD" in tags:
            log(f"  Mean/Std   : {tags['MSAVI_MEAN']} / {tags['MSAVI_STD']}")

        log("  Embedded metadata tags:")
        for k in sorted(tags):
            log(f"    {k}={tags[k]}")


def main() -> None:
    args = parse_args()
    args.input_dir = resolve_path(args.input_dir)
    if args.red is not None:
        args.red = resolve_path(args.red)
    if args.nir is not None:
        args.nir = resolve_path(args.nir)
    if args.output is not None:
        args.output = resolve_path(args.output)

    red_path, nir_path, out_path = resolve_paths(args)

    log("Starting MSAVI creation")
    log(f"Year         : {args.year}")
    log(f"Red (B4)     : {red_path}")
    log(f"NIR (B8)     : {nir_path}")
    log(f"Output       : {out_path}")
    log(f"Expected CRS : {args.crs}")
    log(f"Input nodata : {args.input_nodata}")
    log(f"Output nodata: {args.output_nodata}")

    if not red_path.exists():
        raise SystemExit(f"ERROR: Red band file not found: {red_path}")
    if not nir_path.exists():
        raise SystemExit(f"ERROR: NIR band file not found: {nir_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_valid_pixels = 0

    with rasterio.open(red_path) as red_src, rasterio.open(nir_path) as nir_src:
        if red_src.count != 1 or nir_src.count != 1:
            raise SystemExit("ERROR: Expected single-band rasters for B4 and B8.")

        if red_src.width != nir_src.width or red_src.height != nir_src.height:
            raise SystemExit(
                f"ERROR: Raster size mismatch. "
                f"B4={red_src.width}x{red_src.height}, "
                f"B8={nir_src.width}x{nir_src.height}"
            )

        if red_src.crs != nir_src.crs:
            raise SystemExit(f"ERROR: CRS mismatch. B4={red_src.crs}, B8={nir_src.crs}")

        if red_src.crs is None:
            raise SystemExit("ERROR: Input rasters have no CRS.")

        if red_src.crs.to_string() != args.crs:
            raise SystemExit(
                f"ERROR: Input CRS is {red_src.crs.to_string()} but expected {args.crs}."
            )

        if not same_transform(red_src.transform, nir_src.transform):
            raise SystemExit("ERROR: B4 and B8 transforms differ. Align rasters first.")

        red_nodata = red_src.nodata if red_src.nodata is not None else args.input_nodata
        nir_nodata = nir_src.nodata if nir_src.nodata is not None else args.input_nodata

        profile = red_src.profile.copy()
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

        log("Writing MSAVI raster by blocks...")
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in red_src.block_windows(1):
                red = red_src.read(1, window=window)
                nir = nir_src.read(1, window=window)

                msavi, valid_count = compute_msavi_block(
                    red=red,
                    nir=nir,
                    red_nodata=red_nodata,
                    nir_nodata=nir_nodata,
                    output_nodata=args.output_nodata,
                )
                total_valid_pixels += valid_count
                dst.write(msavi, 1, window=window)

        log("Computing MSAVI statistics block by block...")
        msavi_stats = compute_raster_stats_blockwise(
            out_path,
            nodata_value=args.output_nodata,
        )

        with rasterio.open(out_path, "r+") as dst:
            total_pixels = dst.width * dst.height
            valid_pixels = total_valid_pixels
            nodata_pixels = int(total_pixels - valid_pixels)

            dst.update_tags(
                AREA_OR_POINT="Area",
                INDEX_NAME="MSAVI",
                INDEX_FORMULA="(2 * B8 + 1 - sqrt((2 * B8 + 1)^2 - 8 * (B8 - B4))) / 2",
                INDEX_DESCRIPTION="Modified Soil-Adjusted Vegetation Index",
                YEAR=str(args.year),
                INPUT_RED=str(red_path),
                INPUT_NIR=str(nir_path),
                INPUT_RED_BAND="B4",
                INPUT_NIR_BAND="B8",
                INPUT_RED_DTYPE=red_src.dtypes[0],
                INPUT_NIR_DTYPE=nir_src.dtypes[0],
                INPUT_RED_NODATA=str(red_nodata),
                INPUT_NIR_NODATA=str(nir_nodata),
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
                MSAVI_MIN=msavi_stats["min"],
                MSAVI_MAX=msavi_stats["max"],
                MSAVI_MEAN=msavi_stats["mean"],
                MSAVI_STD=msavi_stats["std"],
            )

        log(f"MSAVI raster written successfully (valid pixels written: {total_valid_pixels})")

    print_output_summary(out_path)
    log("Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user")
        sys.exit(130)
