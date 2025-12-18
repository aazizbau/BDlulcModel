"""
Download only missing AlphaEarth tiles for the Bangladesh coastal AOI.

Usage:
    python scripts/gee/download_missing_alphaearth_tiles.py \
        --year 2024 \
        --output data/raw/embeddings/bd_coastal_alphaearth_2024.tif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import ee
import geemap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gee import download_alphaearth_embeddings as downloader


def list_missing_tiles(
    parent: Path,
    stem: str,
    suffix: str,
    tiles: Iterable[Tuple[int, int, ee.Geometry]],
) -> list[Tuple[int, int, ee.Geometry, Path]]:
    """Return tiles that are absent on disk."""
    missing = []
    for row, col, region in tiles:
        tile_path = parent / f"{stem}_r{row:02d}_c{col:02d}{suffix}"
        if not tile_path.exists():
            missing.append((row, col, region, tile_path))
    return missing


def download_missing_tiles(
    year: int,
    output: Path,
    crs: str,
    scale: int,
    tile_width_km: float,
    tile_height_km: float,
    overlap_km: float,
    project: str | None,
) -> None:
    """Download AlphaEarth embeddings only for tiles that are missing."""
    downloader.initialize_earth_engine(project=project)
    geometry = downloader.create_bd_coastal_geometry()
    image = downloader.build_embeddings_image(year, geometry)

    parent, stem, suffix = downloader.resolve_output_template(output)
    parent.mkdir(parents=True, exist_ok=True)

    tiles_iter = downloader.iterate_tiles(
        downloader.BD_COASTAL_BBOX, geometry, tile_width_km, tile_height_km, overlap_km
    )
    missing_tiles = list_missing_tiles(parent, stem, suffix, tiles_iter)

    if not missing_tiles:
        print("No missing tiles detected; nothing to download.")
        return

    total = len(missing_tiles)
    for idx, (row, col, region, tile_path) in enumerate(missing_tiles, start=1):
        print(
            f"[{idx}/{total}] Downloading missing tile r{row:02d} c{col:02d} -> {tile_path}"
        )
        geemap.ee_export_image(
            image.clip(region),
            filename=str(tile_path),
            region=region,
            crs=crs,
            scale=scale,
        )
    print("Missing tile downloads complete.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download missing AlphaEarth tiles.")
    parser.add_argument("--year", type=int, default=2024, help="Acquisition year.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/embeddings/bd_coastal_alphaearth_2024.tif"),
        help="Base output path used for previous downloads.",
    )
    parser.add_argument("--crs", type=str, default="EPSG:4326", help="Export CRS.")
    parser.add_argument("--scale", type=int, default=10, help="Pixel resolution (m).")
    parser.add_argument(
        "--tile-width-km",
        type=float,
        default=2.5,
        help="Tile width in kilometers.",
    )
    parser.add_argument(
        "--tile-height-km",
        type=float,
        default=2.5,
        help="Tile height in kilometers.",
    )
    parser.add_argument(
        "--tile-overlap-km",
        type=float,
        default=0.5,
        help="Tile overlap in kilometers.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Optional Earth Engine project ID.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    download_missing_tiles(
        year=args.year,
        output=args.output,
        crs=args.crs,
        scale=args.scale,
        tile_width_km=args.tile_width_km,
        tile_height_km=args.tile_height_km,
        overlap_km=args.tile_overlap_km,
        project=args.project,
    )


if __name__ == "__main__":
    main()
