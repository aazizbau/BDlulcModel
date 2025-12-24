"""
Compute NDVI from Sentinel-2 coastal mosaics (B04 red and B08 NIR) and write a GeoTIFF.
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
    parser = argparse.ArgumentParser(description="Compute NDVI from B04/B08 coastal mosaics.")
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
        help=f"CRS for the output NDVI (default: {DEFAULT_OUTPUT_CRS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NDVI GeoTIFF path. Default: <base>/<year>/ndvi_<year>.tif",
    )
    return parser.parse_args()


def resolve_band_paths(base_dir: Path, year: int) -> Tuple[Path, Path]:
    year_dir = base_dir / str(year)
    if not year_dir.exists():
        raise SystemExit(f"Year folder not found: {year_dir}")

    red_path = year_dir / f"coastal_{year}_10_B04_solid.tif"
    nir_path = year_dir / f"coastal_{year}_10_B08_solid.tif"
    return red_path, nir_path


def compute_ndvi(red: np.ndarray, nir: np.ndarray, nodata_mask: np.ndarray | None) -> np.ndarray:
    red = red.astype("float32")
    nir = nir.astype("float32")
    denom = nir + red
    zero_denom = denom == 0
    mask = zero_denom if nodata_mask is None else (zero_denom | nodata_mask)
    ndvi = np.empty_like(red, dtype="float32")
    valid = ~mask
    ndvi[valid] = (nir[valid] - red[valid]) / denom[valid]
    ndvi[mask] = np.nan
    return ndvi


def main() -> None:
    args = parse_args()
    red_path, nir_path = resolve_band_paths(args.base_dir, args.year)

    if not red_path.exists():
        raise SystemExit(f"Red band not found: {red_path}")
    if not nir_path.exists():
        raise SystemExit(f"NIR band not found: {nir_path}")

    out_path = args.output or (args.base_dir / str(args.year) / f"ndvi_{args.year}.tif")

    with rasterio.open(red_path) as red_src, rasterio.open(nir_path) as nir_src:
        if red_src.crs != nir_src.crs or red_src.transform != nir_src.transform:
            raise SystemExit("Red and NIR rasters differ in CRS or transform; align them first.")
        if args.output_crs and red_src.crs and red_src.crs.to_string() != args.output_crs:
            raise SystemExit(
                f"Input CRS is {red_src.crs.to_string()} but --output-crs is {args.output_crs}. "
                "Reproject inputs first or set --output-crs to match."
            )

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
                ndvi_block = compute_ndvi(red, nir, nodata_mask)
                dst.write(ndvi_block.astype("float32"), 1, window=window)

    print(f"Saved NDVI to {out_path} (year {args.year})")


if __name__ == "__main__":
    main()
