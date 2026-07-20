"""
Download only missing natural forest tiles for the Bangladesh coastal AOI.

Usage:
    python scripts/forest/download_missing_naturalforest_tiles.py \
        --output data/raw/forest/bd_coastal_naturalforest_2020.tif

Reproduction and AOI adaptation
-------------------------------
Workflow role: Download, validate, and mosaic natural-forest reference data.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--output``, ``--crs``, ``--scale``, ``--tile-width-km``, ``--tile-height-km``, ``--tile-overlap-km``, ``--project``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace Earth Engine project/AOI settings and local tile directories, then verify complete tile coverage before mosaicking.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import ee
import geemap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.forest import download_naturalforest as downloader

GEE_PROJECT_ENV = "GEE_PROJECT_ID"


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
    output: Path,
    crs: str,
    scale: int,
    tile_width_km: float,
    tile_height_km: float,
    overlap_km: float,
    project: str | None,
) -> None:
    """Download natural forest only for tiles that are missing."""
    downloader.initialize_earth_engine(project=project)
    geometry = downloader.create_bd_coastal_geometry()
    image = downloader.build_natural_forest_image(geometry)

    parent, stem, suffix = downloader.resolve_output_template(output)
    parent.mkdir(parents=True, exist_ok=True)

    tiles_iter = downloader.iterate_tiles(
        downloader.BD_COASTAL_BBOX,
        geometry,
        tile_width_km,
        tile_height_km,
        overlap_km,
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
    parser = argparse.ArgumentParser(description="Download missing natural forest tiles.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/forest/bd_coastal_naturalforest_2020.tif"),
        help="Base output path used for previous downloads.",
    )
    parser.add_argument("--crs", type=str, default="EPSG:4326", help="Export CRS.")
    parser.add_argument("--scale", type=int, default=10, help="Pixel resolution (m).")
    parser.add_argument(
        "--tile-width-km",
        type=float,
        default=10,
        help="Tile width in kilometers.",
    )
    parser.add_argument(
        "--tile-height-km",
        type=float,
        default=10,
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
        default=os.environ.get(GEE_PROJECT_ENV),
        help=f'Optional Earth Engine project ID (default: env "{GEE_PROJECT_ENV}").',
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    download_missing_tiles(
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
