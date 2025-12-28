"""
Compute NIRv (Near-Infrared Reflectance of Vegetation) from Sentinel-2 rasters.

NIRv = NDVI * NIR

Example:
    python scripts/indices/make_nirv.py --year 2017
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio


DEFAULT_BASE = Path("/media/abdul-aziz/345E19F75E19B29A/bd_coastal_tiles")
DEFAULT_OUTPUT_CRS = "EPSG:32646"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute NIRv from B04/B08 coastal mosaics.")
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
        help=f"CRS for the output NIRv (default: {DEFAULT_OUTPUT_CRS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NIRv GeoTIFF path. Default: <base>/<year>/nirv_<year>.tif",
    )
    return parser.parse_args()


def compute_nirv(
    red: np.ndarray,
    nir: np.ndarray,
    nodata_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute NIRv = NDVI * NIR (reflectance-based)."""

    # ------------------------------------------------------------------
    # FIX: convert Sentinel-2 scaled reflectance (0–10000) → 0–1
    # ------------------------------------------------------------------
    red = red.astype("float32") / 10000.0
    nir = nir.astype("float32") / 10000.0
    # ------------------------------------------------------------------

    denom = nir + red
    zero_denom = denom == 0
    mask = zero_denom if nodata_mask is None else (zero_denom | nodata_mask)

    nirv = np.empty_like(nir, dtype="float32")
    valid = ~mask

    ndvi = np.zeros_like(nir, dtype="float32")
    ndvi[valid] = (nir[valid] - red[valid]) / denom[valid]

    nirv[valid] = ndvi[valid] * nir[valid]
    nirv[mask] = np.nan
    return nirv


def resolve_band_paths(base_dir: Path, year: int) -> Tuple[Path, Path]:
    year_dir = base_dir / str(year)
    if not year_dir.exists():
        raise SystemExit(f"Year folder not found: {year_dir}")

    red_path = year_dir / f"coastal_{year}_10_B04_solid.tif"
    nir_path = year_dir / f"coastal_{year}_10_B08_solid.tif"
    return red_path, nir_path


def main() -> None:
    args = parse_args()

    red_path, nir_path = resolve_band_paths(args.base_dir, args.year)

    if not red_path.exists():
        raise SystemExit(f"Red band not found: {red_path}")
    if not nir_path.exists():
        raise SystemExit(f"NIR band not found: {nir_path}")

    out_path = args.output or (args.base_dir / str(args.year) / f"nirv_{args.year}.tif")

    with rasterio.open(red_path) as red_src, rasterio.open(nir_path) as nir_src:
        if red_src.crs != nir_src.crs or red_src.transform != nir_src.transform:
            raise SystemExit("Red and NIR rasters differ in CRS or transform; align them first.")

        profile = red_src.profile.copy()
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
            for _, window in red_src.block_windows(1):
                red = red_src.read(1, window=window)
                nir = nir_src.read(1, window=window)

                nodata_mask = None
                if red_src.nodata is not None:
                    nodata_mask = red == red_src.nodata

                nirv_block = compute_nirv(red, nir, nodata_mask)
                dst.write(nirv_block, 1, window=window)

    print(f"Saved NIRv to {out_path} (year {args.year})")


if __name__ == "__main__":
    main()