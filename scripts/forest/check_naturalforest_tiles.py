"""
Verify natural forest tile downloads for the Bangladesh coastal AOI.

Usage:
    python scripts/forest/check_naturalforest_tiles.py \
        --output data/raw/forest/bd_coastal_naturalforest_2020.tif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.forest.download_naturalforest import (
    AOI_NAME,
    BD_COASTAL_BBOX,
    create_bd_coastal_geometry,
    initialize_earth_engine,
    iterate_tiles,
    resolve_output_template,
)


def check_tiles(
    output: Path,
    tile_width_km: float,
    tile_height_km: float,
    overlap_km: float,
    project: str | None,
) -> None:
    """Check for missing natural forest tiles against on-disk outputs."""
    initialize_earth_engine(project=project)
    geometry = create_bd_coastal_geometry()

    parent, stem, suffix = resolve_output_template(output)
    parent.mkdir(parents=True, exist_ok=True)

    missing: list[Path] = []
    total = 0
    for row, col, _ in iterate_tiles(
        BD_COASTAL_BBOX,
        geometry,
        tile_width_km,
        tile_height_km,
        overlap_km,
    ):
        total += 1
        tile_path = parent / f"{stem}_r{row:02d}_c{col:02d}{suffix}"
        if not tile_path.exists():
            missing.append(tile_path)

    print(f"AOI: {AOI_NAME}")
    print(f"Expected tiles: {total}")
    print(f"Found tiles: {total - len(missing)}")
    if missing:
        print(f"Missing tiles ({len(missing)}):")
        for path in missing:
            print(f"  {path}")
    else:
        print("All tiles present.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Check natural forest tile downloads for {AOI_NAME}."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/forest/bd_coastal_naturalforest_2020.tif"),
        help="Base output path used for downloads.",
    )
    parser.add_argument(
        "--tile-width-km",
        type=float,
        default=10,
        help="Tile width in kilometers (default: 10).",
    )
    parser.add_argument(
        "--tile-height-km",
        type=float,
        default=10,
        help="Tile height in kilometers (default: 10).",
    )
    parser.add_argument(
        "--tile-overlap-km",
        type=float,
        default=0.5,
        help="Overlap between tiles in kilometers (default: 0.5).",
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
    check_tiles(
        output=args.output,
        tile_width_km=args.tile_width_km,
        tile_height_km=args.tile_height_km,
        overlap_km=args.tile_overlap_km,
        project=args.project,
    )


if __name__ == "__main__":
    main()
