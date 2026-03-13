"""
Compute NDMI (Normalized Difference Moisture Index) from Sentinel-2 rasters.

NDMI = (NIR - SWIR1) / (NIR + SWIR1)

Example:
    python scripts/indices/make_ndmi.py --year 2017
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
    parser = argparse.ArgumentParser(description="Compute NDMI from B08/B11 coastal mosaics.")
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
        help=f"CRS for the output NDMI (default: {DEFAULT_OUTPUT_CRS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NDMI GeoTIFF path. Default: <base>/<year>/ndmi_<year>.tif",
    )
    return parser.parse_args()


def compute_ndmi(
    nir: np.ndarray,
    swir: np.ndarray,
    nodata_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute NDMI (reflectance-based)."""
    nir = nir.astype("float32") / 10000.0
    swir = swir.astype("float32") / 10000.0

    denom = nir + swir
    zero_denom = denom == 0
    mask = zero_denom if nodata_mask is None else (zero_denom | nodata_mask)

    ndmi = np.empty_like(nir, dtype="float32")
    valid = ~mask
    ndmi[valid] = (nir[valid] - swir[valid]) / denom[valid]
    ndmi[mask] = np.nan
    return ndmi


def resolve_band_paths(base_dir: Path, year: int) -> Tuple[Path, Path]:
    year_dir = base_dir / str(year)
    if not year_dir.exists():
        raise SystemExit(f"Year folder not found: {year_dir}")

    nir_path = year_dir / f"coastal_{year}_10_B08_solid.tif"
    swir_path = year_dir / f"coastal_{year}_10_B11_solid.tif"
    return nir_path, swir_path


def main() -> None:
    args = parse_args()

    nir_path, swir_path = resolve_band_paths(args.base_dir, args.year)

    if not nir_path.exists():
        raise SystemExit(f"NIR band not found: {nir_path}")
    if not swir_path.exists():
        raise SystemExit(f"SWIR band not found: {swir_path}")

    out_path = args.output or (args.base_dir / str(args.year) / f"ndmi_{args.year}.tif")

    with rasterio.open(nir_path) as nir_src, rasterio.open(swir_path) as swir_src:
        if nir_src.crs != swir_src.crs or nir_src.transform != swir_src.transform:
            raise SystemExit("NIR and SWIR rasters differ in CRS or transform; align them first.")

        profile = nir_src.profile.copy()
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
            for _, window in nir_src.block_windows(1):
                nir = nir_src.read(1, window=window)
                swir = swir_src.read(1, window=window)

                nodata_mask = None
                if nir_src.nodata is not None:
                    nodata_mask = nir == nir_src.nodata

                ndmi_block = compute_ndmi(nir, swir, nodata_mask)
                dst.write(ndmi_block, 1, window=window)

    print(f"Saved NDMI to {out_path} (year {args.year})")


if __name__ == "__main__":
    main()
