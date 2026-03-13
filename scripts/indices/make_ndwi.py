"""
Compute NDWI (McFeeters) from Sentinel-2 rasters.

NDWI = (Green - NIR) / (Green + NIR)

Example:
    python scripts/indices/make_ndwi.py --year 2017
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
    parser = argparse.ArgumentParser(description="Compute NDWI from B03/B08 coastal mosaics.")
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
        help=f"CRS for the output NDWI (default: {DEFAULT_OUTPUT_CRS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NDWI GeoTIFF path. Default: <base>/<year>/ndwi_<year>.tif",
    )
    return parser.parse_args()


def compute_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    nodata_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute NDWI (McFeeters), reflectance-based."""
    green = green.astype("float32") / 10000.0
    nir = nir.astype("float32") / 10000.0

    denom = green + nir
    zero_denom = denom == 0
    mask = zero_denom if nodata_mask is None else (zero_denom | nodata_mask)

    ndwi = np.empty_like(green, dtype="float32")
    valid = ~mask
    ndwi[valid] = (green[valid] - nir[valid]) / denom[valid]
    ndwi[mask] = np.nan
    return ndwi


def resolve_band_paths(base_dir: Path, year: int) -> Tuple[Path, Path]:
    year_dir = base_dir / str(year)
    if not year_dir.exists():
        raise SystemExit(f"Year folder not found: {year_dir}")

    green_path = year_dir / f"coastal_{year}_10_B03_solid.tif"
    nir_path = year_dir / f"coastal_{year}_10_B08_solid.tif"
    return green_path, nir_path


def main() -> None:
    args = parse_args()

    green_path, nir_path = resolve_band_paths(args.base_dir, args.year)

    if not green_path.exists():
        raise SystemExit(f"Green band not found: {green_path}")
    if not nir_path.exists():
        raise SystemExit(f"NIR band not found: {nir_path}")

    out_path = args.output or (args.base_dir / str(args.year) / f"ndwi_{args.year}.tif")

    with rasterio.open(green_path) as green_src, rasterio.open(nir_path) as nir_src:
        if green_src.crs != nir_src.crs or green_src.transform != nir_src.transform:
            raise SystemExit("Green and NIR rasters differ in CRS or transform; align them first.")

        profile = green_src.profile.copy()
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
            for _, window in green_src.block_windows(1):
                green = green_src.read(1, window=window)
                nir = nir_src.read(1, window=window)

                nodata_mask = None
                if green_src.nodata is not None:
                    nodata_mask = green == green_src.nodata

                ndwi_block = compute_ndwi(green, nir, nodata_mask)
                dst.write(ndwi_block, 1, window=window)

    print(f"Saved NDWI to {out_path} (year {args.year})")


if __name__ == "__main__":
    main()
