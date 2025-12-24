"""
Compute EVI from Sentinel-2 coastal mosaics (B02 blue, B04 red, B08 NIR) and write a GeoTIFF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio


DEFAULT_BASE = Path("/media/abdul-aziz/345E19F75E19B29A/bd_coastal_tiles")
DEFAULT_OUTPUT_CRS = "EPSG:32646"
DEFAULT_SCALE = 1 / 10000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute EVI from coastal B02/B04/B08 mosaics.")
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
        help=f"CRS for the output EVI (default: {DEFAULT_OUTPUT_CRS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output EVI GeoTIFF path. Default: <base>/<year>/evi_<year>.tif",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help="Scale factor to convert DN to reflectance (default: 1/10000).",
    )
    return parser.parse_args()


def resolve_band_paths(base_dir: Path, year: int) -> tuple[Path, Path, Path]:
    year_dir = base_dir / str(year)
    if not year_dir.exists():
        raise SystemExit(f"Year folder not found: {year_dir}")

    blue_path = year_dir / f"coastal_{year}_10_B02_solid.tif"
    red_path = year_dir / f"coastal_{year}_10_B04_solid.tif"
    nir_path = year_dir / f"coastal_{year}_10_B08_solid.tif"
    return blue_path, red_path, nir_path


def compute_evi(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    nodata_mask: np.ndarray | None,
    scale: float,
) -> np.ndarray:
    blue = blue.astype("float32") * scale
    red = red.astype("float32") * scale
    nir = nir.astype("float32") * scale

    invalid_ref = (blue < 0) | (red < 0) | (nir < 0) | (blue > 2) | (red > 2) | (nir > 2)
    denom = nir + 6 * red - 7.5 * blue + 1
    bad_denom = np.isclose(denom, 0, atol=1e-6) | (denom <= 0)
    mask = invalid_ref | bad_denom
    if nodata_mask is not None:
        mask |= nodata_mask

    evi = np.empty_like(red, dtype="float32")
    valid = ~mask
    evi[valid] = 2.5 * ((nir[valid] - red[valid]) / denom[valid])
    implausible = (evi < -2) | (evi > 2)
    evi[implausible] = np.nan
    evi[mask] = np.nan
    return evi


def main() -> None:
    args = parse_args()
    blue_path, red_path, nir_path = resolve_band_paths(args.base_dir, args.year)

    if not blue_path.exists():
        raise SystemExit(f"Blue band not found: {blue_path}")
    if not red_path.exists():
        raise SystemExit(f"Red band not found: {red_path}")
    if not nir_path.exists():
        raise SystemExit(f"NIR band not found: {nir_path}")

    out_path = args.output or (args.base_dir / str(args.year) / f"evi_{args.year}.tif")

    with (
        rasterio.open(blue_path) as blue_src,
        rasterio.open(red_path) as red_src,
        rasterio.open(nir_path) as nir_src,
    ):
        if not (
            blue_src.crs == red_src.crs == nir_src.crs
            and blue_src.transform == red_src.transform == nir_src.transform
        ):
            raise SystemExit("Input rasters differ in CRS or transform; align them first.")
        if args.output_crs and blue_src.crs and blue_src.crs.to_string() != args.output_crs:
            raise SystemExit(
                f"Input CRS is {blue_src.crs.to_string()} but --output-crs is {args.output_crs}. "
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
                blue = blue_src.read(1, window=window)
                red = red_src.read(1, window=window)
                nir = nir_src.read(1, window=window)

                masks = []
                for band, nod in (
                    (blue, blue_src.nodata),
                    (red, red_src.nodata),
                    (nir, nir_src.nodata),
                ):
                    if nod is not None:
                        masks.append(band == nod)
                    masks.append((band <= 0) | (band > 12000))
                nodata_mask = np.logical_or.reduce(masks) if masks else None

                evi_block = compute_evi(blue, red, nir, nodata_mask, scale=args.scale)
                dst.write(evi_block.astype("float32"), 1, window=window)

    print(f"Saved EVI to {out_path} (year {args.year}, scale {args.scale})")


if __name__ == "__main__":
    main()
