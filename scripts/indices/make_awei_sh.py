"""
Compute AWEI(sh) (Automated Water Extraction Index – shadow) from Sentinel-2 rasters.

AWEI_sh = Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2

Example:
    python scripts/indices/make_awei_sh.py --year 2017
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
    parser = argparse.ArgumentParser(description="Compute AWEI(sh) from Sentinel-2 coastal mosaics.")
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
        help=f"CRS for the output AWEI(sh) (default: {DEFAULT_OUTPUT_CRS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output AWEI(sh) GeoTIFF path. Default: <base>/<year>/awei_sh_<year>.tif",
    )
    return parser.parse_args()


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


def resolve_band_paths(base_dir: Path, year: int) -> Tuple[Path, Path, Path, Path, Path]:
    year_dir = base_dir / str(year)
    if not year_dir.exists():
        raise SystemExit(f"Year folder not found: {year_dir}")

    blue_path = year_dir / f"coastal_{year}_10_B02_solid.tif"
    green_path = year_dir / f"coastal_{year}_10_B03_solid.tif"
    nir_path = year_dir / f"coastal_{year}_10_B08_solid.tif"
    swir1_path = year_dir / f"coastal_{year}_10_B11_solid.tif"
    swir2_path = year_dir / f"coastal_{year}_10_B12_solid.tif"
    return blue_path, green_path, nir_path, swir1_path, swir2_path


def main() -> None:
    args = parse_args()

    blue_path, green_path, nir_path, swir1_path, swir2_path = resolve_band_paths(
        args.base_dir, args.year
    )

    for path, name in [
        (blue_path, "Blue"),
        (green_path, "Green"),
        (nir_path, "NIR"),
        (swir1_path, "SWIR1"),
        (swir2_path, "SWIR2"),
    ]:
        if not path.exists():
            raise SystemExit(f"{name} band not found: {path}")

    out_path = args.output or (args.base_dir / str(args.year) / f"awei_sh_{args.year}.tif")

    with (
        rasterio.open(blue_path) as blue_src,
        rasterio.open(green_path) as green_src,
        rasterio.open(nir_path) as nir_src,
        rasterio.open(swir1_path) as swir1_src,
        rasterio.open(swir2_path) as swir2_src,
    ):
        if not (
            blue_src.crs == green_src.crs == nir_src.crs == swir1_src.crs == swir2_src.crs
            and blue_src.transform == green_src.transform == nir_src.transform
            == swir1_src.transform == swir2_src.transform
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
                green = green_src.read(1, window=window)
                nir = nir_src.read(1, window=window)
                swir1 = swir1_src.read(1, window=window)
                swir2 = swir2_src.read(1, window=window)

                nodata_mask = None
                if blue_src.nodata is not None:
                    nodata_mask = blue == blue_src.nodata

                awei_block = compute_awei_sh(blue, green, nir, swir1, swir2, nodata_mask)
                dst.write(awei_block, 1, window=window)

    print(f"Saved AWEI(sh) to {out_path} (year {args.year})")


if __name__ == "__main__":
    main()
