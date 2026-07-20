#!/usr/bin/env python3
"""
Create NDBI GeoTIFF from Sentinel-2 coastal solid rasters.

NDBI = (SWIR - NIR) / (SWIR + NIR)

Expected default input files for --year 2017:
  data/interim/S2_2017_B11_10m_utm46_bdcoastal_solid.tif
  data/interim/S2_2017_B8_10m_utm46_bdcoastal_solid.tif

Expected default output file:
  data/interim/bdcoastal_solid_2017_utm46_ndbi.tif

Example runs:
python scripts/s2_indices/make_ndbi_image.py --year 2017

python scripts/s2_indices/make_ndbi_image.py \
  --year 2017 \
  --crs EPSG:32646 \
  --output data/interim/bdcoastal_solid_2017_utm46_ndbi.tif

python scripts/s2_indices/make_ndbi_image.py \
  --year 2017 \
  --swir data/interim/S2_2017_B11_10m_utm46_bdcoastal_solid.tif \
  --nir data/interim/S2_2017_B8_10m_utm46_bdcoastal_solid.tif \
  --output data/interim/bdcoastal_solid_2017_utm46_ndbi.tif

Reproduction and AOI adaptation
-------------------------------
Workflow role: Calculate Sentinel-2 spectral indices from aligned reflectance bands.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--year``, ``--input-dir``, ``--crs``, ``--swir``, ``--nir``, ``--output``, ``--input-nodata``, ``--output-nodata``. Run the ``--help`` command below for required values, defaults, and accepted choices.
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
        description="Make NDBI image from Sentinel-2 B11 (SWIR) and B8 (NIR) rasters."
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
        "--swir",
        type=Path,
        default=None,
        help="Optional custom B11 path.",
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
        help="Optional output NDBI path.",
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
    swir_path = args.swir or (args.input_dir / f"S2_{args.year}_B11_10m_utm46_bdcoastal_solid.tif")
    nir_path = args.nir or (args.input_dir / f"S2_{args.year}_B8_10m_utm46_bdcoastal_solid.tif")
    out_path = args.output or (args.input_dir / f"bdcoastal_solid_{args.year}_utm46_ndbi.tif")
    return swir_path, nir_path, out_path


def same_transform(a, b, tol: float = 1e-9) -> bool:
    return (
        abs(a.a - b.a) < tol
        and abs(a.b - b.b) < tol
        and abs(a.c - b.c) < tol
        and abs(a.d - b.d) < tol
        and abs(a.e - b.e) < tol
        and abs(a.f - b.f) < tol
    )


def compute_ndbi(
    nir: np.ndarray,
    swir: np.ndarray,
    nodata_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute NDBI (reflectance-based)."""
    nir = nir.astype("float32") / 10000.0
    swir = swir.astype("float32") / 10000.0

    denom = swir + nir
    zero_denom = denom == 0
    mask = zero_denom if nodata_mask is None else (zero_denom | nodata_mask)

    ndbi = np.empty_like(denom, dtype="float32")
    valid = ~mask
    ndbi[valid] = (swir[valid] - nir[valid]) / denom[valid]
    ndbi[mask] = np.nan
    return ndbi


def compute_ndbi_block(
    nir: np.ndarray,
    swir: np.ndarray,
    nir_nodata: float | int | None,
    swir_nodata: float | int | None,
    output_nodata: float,
) -> tuple[np.ndarray, int]:
    nodata_mask = np.zeros(nir.shape, dtype=bool)

    if nir_nodata is not None:
        nodata_mask |= nir == nir_nodata
    if swir_nodata is not None:
        nodata_mask |= swir == swir_nodata

    ndbi = compute_ndbi(
        nir=nir,
        swir=swir,
        nodata_mask=nodata_mask,
    )

    valid = np.isfinite(ndbi)
    valid_count = int(valid.sum())

    out = np.full(ndbi.shape, output_nodata, dtype=np.float32)
    out[valid] = ndbi[valid]
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

        if "NDBI_MIN" in tags and "NDBI_MAX" in tags:
            log(f"  Min/Max    : {tags['NDBI_MIN']} / {tags['NDBI_MAX']}")
        if "NDBI_MEAN" in tags and "NDBI_STD" in tags:
            log(f"  Mean/Std   : {tags['NDBI_MEAN']} / {tags['NDBI_STD']}")

        log("  Embedded metadata tags:")
        for k in sorted(tags):
            log(f"    {k}={tags[k]}")


def main() -> None:
    args = parse_args()
    args.input_dir = resolve_path(args.input_dir)
    if args.swir is not None:
        args.swir = resolve_path(args.swir)
    if args.nir is not None:
        args.nir = resolve_path(args.nir)
    if args.output is not None:
        args.output = resolve_path(args.output)

    swir_path, nir_path, out_path = resolve_paths(args)

    log("Starting NDBI creation")
    log(f"Year         : {args.year}")
    log(f"SWIR (B11)   : {swir_path}")
    log(f"NIR (B8)     : {nir_path}")
    log(f"Output       : {out_path}")
    log(f"Expected CRS : {args.crs}")
    log(f"Input nodata : {args.input_nodata}")
    log(f"Output nodata: {args.output_nodata}")

    if not swir_path.exists():
        raise SystemExit(f"ERROR: SWIR band file not found: {swir_path}")
    if not nir_path.exists():
        raise SystemExit(f"ERROR: NIR band file not found: {nir_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_valid_pixels = 0

    with rasterio.open(swir_path) as swir_src, rasterio.open(nir_path) as nir_src:
        if swir_src.count != 1 or nir_src.count != 1:
            raise SystemExit("ERROR: Expected single-band rasters for B11 and B8.")

        if swir_src.width != nir_src.width or swir_src.height != nir_src.height:
            raise SystemExit(
                f"ERROR: Raster size mismatch. "
                f"B11={swir_src.width}x{swir_src.height}, "
                f"B8={nir_src.width}x{nir_src.height}"
            )

        if swir_src.crs != nir_src.crs:
            raise SystemExit(f"ERROR: CRS mismatch. B11={swir_src.crs}, B8={nir_src.crs}")

        if swir_src.crs is None:
            raise SystemExit("ERROR: Input rasters have no CRS.")

        if swir_src.crs.to_string() != args.crs:
            raise SystemExit(
                f"ERROR: Input CRS is {swir_src.crs.to_string()} but expected {args.crs}."
            )

        if not same_transform(swir_src.transform, nir_src.transform):
            raise SystemExit("ERROR: B11 and B8 transforms differ. Align rasters first.")

        swir_nodata = swir_src.nodata if swir_src.nodata is not None else args.input_nodata
        nir_nodata = nir_src.nodata if nir_src.nodata is not None else args.input_nodata

        profile = swir_src.profile.copy()
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

        log("Writing NDBI raster by blocks...")
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in swir_src.block_windows(1):
                swir = swir_src.read(1, window=window)
                nir = nir_src.read(1, window=window)

                ndbi, valid_count = compute_ndbi_block(
                    nir=nir,
                    swir=swir,
                    nir_nodata=nir_nodata,
                    swir_nodata=swir_nodata,
                    output_nodata=args.output_nodata,
                )
                total_valid_pixels += valid_count
                dst.write(ndbi, 1, window=window)

        log("Computing NDBI statistics block by block...")
        ndbi_stats = compute_raster_stats_blockwise(
            out_path,
            nodata_value=args.output_nodata,
        )

        with rasterio.open(out_path, "r+") as dst:
            total_pixels = dst.width * dst.height
            valid_pixels = total_valid_pixels
            nodata_pixels = int(total_pixels - valid_pixels)

            dst.update_tags(
                AREA_OR_POINT="Area",
                INDEX_NAME="NDBI",
                INDEX_FORMULA="(B11 - B8) / (B11 + B8)",
                INDEX_DESCRIPTION="Normalized Difference Built-up Index",
                YEAR=str(args.year),
                INPUT_SWIR=str(swir_path),
                INPUT_NIR=str(nir_path),
                INPUT_SWIR_BAND="B11",
                INPUT_NIR_BAND="B8",
                INPUT_SWIR_DTYPE=swir_src.dtypes[0],
                INPUT_NIR_DTYPE=nir_src.dtypes[0],
                INPUT_SWIR_NODATA=str(swir_nodata),
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
                NDBI_MIN=ndbi_stats["min"],
                NDBI_MAX=ndbi_stats["max"],
                NDBI_MEAN=ndbi_stats["mean"],
                NDBI_STD=ndbi_stats["std"],
            )

        log(f"NDBI raster written successfully (valid pixels written: {total_valid_pixels})")

    print_output_summary(out_path)
    log("Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user")
        sys.exit(130)
