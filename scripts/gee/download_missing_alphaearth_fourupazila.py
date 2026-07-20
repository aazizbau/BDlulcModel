"""
Download missing AlphaEarth embeddings tiles for the Four Upazila region.

Usage:
    export GEE_PROJECT_ID="your-ee-project-id"

    python scripts/gee/download_missing_alphaearth_fourupazila.py \
        --year 2023 \
        --project "${GEE_PROJECT_ID}" \
        --output data/raw/embeddings/fourupazila/bd_coastal_fourupazila_alphaearth_2023.tif \
        --crs EPSG:32646

Reproduction and AOI adaptation
-------------------------------
Workflow role: Acquire or prepare Earth Engine products such as AlphaEarth, Dynamic World, or ALOS DSM data.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--year``, ``--output``, ``--crs``, ``--scale``, ``--project``, ``--tile-width-km``, ``--tile-height-km``, ``--tile-overlap-km``, ``--dry-run``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace the Earth Engine project, AOI vector or bounds, year, scale, CRS, and export directory. Authenticate the account that owns the target project first.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geemap

from scripts.gee import download_alphaearth_fourupazila as downloader

GEE_PROJECT_ENV = "GEE_PROJECT_ID"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download missing AlphaEarth embeddings for the Four Upazila AOI."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Acquisition year for the embeddings (default: 2023).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/raw/embeddings/fourupazila/bd_coastal_fourupazila_alphaearth_2023.tif"
        ),
        help=(
            "Base output path. Each tile appends _rXX_cYY before the file suffix "
            "(default: data/raw/embeddings/fourupazila/bd_coastal_fourupazila_alphaearth_2023.tif)."
        ),
    )
    parser.add_argument(
        "--crs",
        type=str,
        default="EPSG:32646",
        help="CRS for the export (default: EPSG:32646).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=10,
        help="Pixel resolution in meters for export (default: 10m).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=os.environ.get(GEE_PROJECT_ENV),
        help=f'Optional Earth Engine project ID for initialization (default: env "{GEE_PROJECT_ENV}").',
    )
    parser.add_argument(
        "--tile-width-km",
        type=float,
        default=2.5,
        help="Tile width in kilometers (default: 2.5 km).",
    )
    parser.add_argument(
        "--tile-height-km",
        type=float,
        default=2.5,
        help="Tile height in kilometers (default: 2.5 km).",
    )
    parser.add_argument(
        "--tile-overlap-km",
        type=float,
        default=0.5,
        help="Overlap between adjacent tiles in kilometers (default: 0.5 km).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print missing tiles without downloading.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    downloader.initialize_earth_engine(project=args.project)

    geometry = downloader.create_fourupazila_geometry()
    embeddings_image = downloader.build_embeddings_image(args.year, geometry)

    tiles = list(
        downloader.iterate_tiles(
            downloader.FOURUPAZILA_BBOX,
            geometry,
            args.tile_width_km,
            args.tile_height_km,
            args.tile_overlap_km,
        )
    )
    if not tiles:
        raise SystemExit("No tiles generated for the provided geometry.")

    parent, stem, suffix = downloader.resolve_output_template(args.output)
    parent.mkdir(parents=True, exist_ok=True)

    missing = []
    for row, col, tile_region in tiles:
        tile_path = parent / f"{stem}_r{row:02d}_c{col:02d}{suffix}"
        if not tile_path.exists():
            missing.append((row, col, tile_region, tile_path))

    if not missing:
        print("No missing tiles found.")
        return

    print(f"Missing tiles: {len(missing)} / {len(tiles)}")
    for idx, (row, col, tile_region, tile_path) in enumerate(missing, start=1):
        print(f"[{idx}/{len(missing)}] row={row} col={col} -> {tile_path}")
        if args.dry_run:
            continue
        geemap.ee_export_image(
            embeddings_image.clip(tile_region),
            filename=str(tile_path),
            region=tile_region,
            crs=args.crs,
            scale=args.scale,
        )

    if args.dry_run:
        print("Dry run complete. No tiles were downloaded.")
    else:
        print("Missing tiles downloaded.")


if __name__ == "__main__":
    main()
