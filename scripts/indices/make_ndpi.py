"""
Compute NDPI (Normalized Difference Pond Index) from Sentinel-2 rasters.

NDPI = (SWIR1 - Green) / (SWIR1 + Green)

Example:
    python scripts/indices/make_ndpi.py --year 2017
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio


DEFAULT_BASE = Path(os.environ.get("BD_COASTAL_TILES_DIR", "data/raw/bd_coastal_tiles"))
DEFAULT_OUTPUT_CRS = "EPSG:32646"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute NDPI from B11/B03 coastal mosaics.")
    parser.add_argument("--year", type=int, default=2017, help="Year folder under the coastal tiles root.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE,
        help=f"Base directory for coastal mosaics (default: {DEFAULT_BASE}).",
    )
    parser.add_argument(
        "--output-crs",
        type=str,
        default=DEFAULT_OUTPUT_CRS,
        help=f"CRS for the output NDPI (default: {DEFAULT_OUTPUT_CRS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NDPI GeoTIFF path. Default: <base>/<year>/ndpi_<year>.tif",
    )
    return parser.parse_args()


def compute_ndpi(
    swir: np.ndarray,
    green: np.ndarray,
    nodata_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute NDPI (reflectance-based)."""
    swir = swir.astype("float32") / 10000.0
    green = green.astype("float32") / 10000.0

    denom = swir + green
    zero_denom = denom == 0
    mask = zero_denom if nodata_mask is None else (zero_denom | nodata_mask)

    ndpi = np.empty_like(swir, dtype="float32")
    valid = ~mask
    ndpi[valid] = (swir[valid] - green[valid]) / denom[valid]
    ndpi[mask] = np.nan
    return ndpi


def resolve_band_paths(base_dir: Path, year: int) -> Tuple[Path, Path]:
    year_dir = base_dir / str(year)
    if not year_dir.exists():
        raise SystemExit(f"Year folder not found: {year_dir}")

    swir_path = year_dir / f"coastal_{year}_10_B11_solid.tif"
    green_path = year_dir / f"coastal_{year}_10_B03_solid.tif"
    return swir_path, green_path


def main() -> None:
    args = parse_args()

    swir_path, green_path = resolve_band_paths(args.base_dir, args.year)

    if not swir_path.exists():
        raise SystemExit(f"SWIR band not found: {swir_path}")
    if not green_path.exists():
        raise SystemExit(f"Green band not found: {green_path}")

    out_path = args.output or (args.base_dir / str(args.year) / f"ndpi_{args.year}.tif")

    with rasterio.open(swir_path) as swir_src, rasterio.open(green_path) as green_src:
        if swir_src.crs != green_src.crs or swir_src.transform != green_src.transform:
            raise SystemExit("SWIR and Green rasters differ in CRS or transform; align them first.")

        profile = swir_src.profile.copy()
        profile.update(
            {
                "count": 1,
                "dtype": "float32",
                "nodata": np.nan,
                "crs": args.output_crs,
            }
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in swir_src.block_windows(1):
                swir = swir_src.read(1, window=window)
                green = green_src.read(1, window=window)

                nodata_mask = None
                if swir_src.nodata is not None:
                    nodata_mask = swir == swir_src.nodata

                ndpi_block = compute_ndpi(swir, green, nodata_mask)
                dst.write(ndpi_block, 1, window=window)

    print(f"Saved NDPI to {out_path} (year {args.year})")


if __name__ == "__main__":
    main()
