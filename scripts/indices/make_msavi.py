"""
Compute MSAVI (Modified Soil Adjusted Vegetation Index) from Sentinel-2 rasters.

MSAVI = (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red))) / 2

Example:
    python scripts/indices/make_msavi.py --year 2017
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
    parser = argparse.ArgumentParser(description="Compute MSAVI from B04/B08 coastal mosaics.")
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
        help=f"CRS for the output MSAVI (default: {DEFAULT_OUTPUT_CRS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MSAVI GeoTIFF path. Default: <base>/<year>/msavi_<year>.tif",
    )
    return parser.parse_args()


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

    out_path = args.output or (args.base_dir / str(args.year) / f"msavi_{args.year}.tif")

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

                msavi_block = compute_msavi(red, nir, nodata_mask)
                dst.write(msavi_block, 1, window=window)

    print(f"Saved MSAVI to {out_path} (year {args.year})")


if __name__ == "__main__":
    main()
