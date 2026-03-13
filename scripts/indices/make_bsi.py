"""
Compute BSI (Bare Soil Index) from Sentinel-2 rasters.

BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))

Example:
    python scripts/indices/make_bsi.py --year 2017
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
    parser = argparse.ArgumentParser(description="Compute BSI from Sentinel-2 coastal mosaics.")
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
        help=f"CRS for the output BSI (default: {DEFAULT_OUTPUT_CRS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output BSI GeoTIFF path. Default: <base>/<year>/bsi_<year>.tif",
    )
    return parser.parse_args()


def compute_bsi(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
    nodata_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute BSI (reflectance-based)."""
    blue = blue.astype("float32") / 10000.0
    red = red.astype("float32") / 10000.0
    nir = nir.astype("float32") / 10000.0
    swir = swir.astype("float32") / 10000.0

    num = (swir + red) - (nir + blue)
    denom = (swir + red) + (nir + blue)

    zero_denom = denom == 0
    mask = zero_denom if nodata_mask is None else (zero_denom | nodata_mask)

    bsi = np.empty_like(denom, dtype="float32")
    valid = ~mask
    bsi[valid] = num[valid] / denom[valid]
    bsi[mask] = np.nan
    return bsi


def resolve_band_paths(base_dir: Path, year: int) -> Tuple[Path, Path, Path, Path]:
    year_dir = base_dir / str(year)
    if not year_dir.exists():
        raise SystemExit(f"Year folder not found: {year_dir}")

    blue_path = year_dir / f"coastal_{year}_10_B02_solid.tif"
    red_path = year_dir / f"coastal_{year}_10_B04_solid.tif"
    nir_path = year_dir / f"coastal_{year}_10_B08_solid.tif"
    swir_path = year_dir / f"coastal_{year}_10_B11_solid.tif"
    return blue_path, red_path, nir_path, swir_path


def main() -> None:
    args = parse_args()

    blue_path, red_path, nir_path, swir_path = resolve_band_paths(args.base_dir, args.year)

    if not blue_path.exists():
        raise SystemExit(f"Blue band not found: {blue_path}")
    if not red_path.exists():
        raise SystemExit(f"Red band not found: {red_path}")
    if not nir_path.exists():
        raise SystemExit(f"NIR band not found: {nir_path}")
    if not swir_path.exists():
        raise SystemExit(f"SWIR band not found: {swir_path}")

    out_path = args.output or (args.base_dir / str(args.year) / f"bsi_{args.year}.tif")

    with (
        rasterio.open(blue_path) as blue_src,
        rasterio.open(red_path) as red_src,
        rasterio.open(nir_path) as nir_src,
        rasterio.open(swir_path) as swir_src,
    ):
        if not (
            blue_src.crs == red_src.crs == nir_src.crs == swir_src.crs
            and blue_src.transform == red_src.transform == nir_src.transform == swir_src.transform
        ):
            raise SystemExit("Input rasters differ in CRS or transform; align them first.")

        profile = blue_src.profile.copy()
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
            for _, window in blue_src.block_windows(1):
                blue = blue_src.read(1, window=window)
                red = red_src.read(1, window=window)
                nir = nir_src.read(1, window=window)
                swir = swir_src.read(1, window=window)

                nodata_mask = None
                if blue_src.nodata is not None:
                    nodata_mask = blue == blue_src.nodata

                bsi_block = compute_bsi(blue, red, nir, swir, nodata_mask)
                dst.write(bsi_block, 1, window=window)

    print(f"Saved BSI to {out_path} (year {args.year})")


if __name__ == "__main__":
    main()
