#!/usr/bin/env python3
"""
Create a dissolved outer-boundary GeoPackage from an input vector file.

The output contains one dissolved geometry and a name field with the value
"mapura" by default.

Complete example run:
    python scripts/upazila_map/dissolved_for_outerboundary.py \
        --input-vector assets/maps/manpura_landuse.gpkg \
        --output-epsg 4326 \
        --output-vector assets/maps/manpura_dissolved.gpkg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def remove_polygon_holes(geometry):
    """
    Retain only polygon exterior rings.

    This removes holes inside polygons while preserving separate polygon parts.
    """
    if geometry is None or geometry.is_empty:
        return geometry

    if isinstance(geometry, Polygon):
        return Polygon(geometry.exterior)

    if isinstance(geometry, MultiPolygon):
        return MultiPolygon(
            [Polygon(polygon.exterior) for polygon in geometry.geoms]
        )

    return geometry


def create_dissolved_boundary(
    input_vector: Path,
    output_vector: Path,
    output_epsg: int,
    name: str,
    layer: str | None = None,
    remove_holes: bool = False,
) -> None:
    """Read, dissolve, reproject, and export the input vector data."""

    if not input_vector.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_vector}")

    gdf = gpd.read_file(input_vector, layer=layer)

    if gdf.empty:
        raise ValueError(f"No features were found in: {input_vector}")

    if gdf.crs is None:
        print(
            "Warning: the input has no CRS metadata. "
            "Assigning EPSG:32646 before reprojection.",
            file=sys.stderr,
        )
        gdf = gdf.set_crs("EPSG:32646")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError("The input contains no valid, non-empty geometries.")

    invalid_count = int((~gdf.geometry.is_valid).sum())
    if invalid_count:
        print(f"Repairing {invalid_count} invalid geometries...")
        gdf["geometry"] = gdf.geometry.make_valid()

    dissolved_geometry = gdf.geometry.union_all()

    if remove_holes:
        dissolved_geometry = remove_polygon_holes(dissolved_geometry)

    output_gdf = gpd.GeoDataFrame(
        {"name": [name]},
        geometry=[dissolved_geometry],
        crs=gdf.crs,
    )

    output_gdf = output_gdf.to_crs(f"EPSG:{output_epsg}")

    output_vector.parent.mkdir(parents=True, exist_ok=True)

    if output_vector.exists():
        output_vector.unlink()

    output_gdf.to_file(
        output_vector,
        layer=f"{name}_boundary",
        driver="GPKG",
        index=False,
    )

    print(f"Output saved to: {output_vector}")
    print(f"Output CRS: {output_gdf.crs}")
    print(f"Feature count: {len(output_gdf)}")
    print(f"Name field: {name}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dissolve all geometries in a vector dataset and export one "
            "outer-boundary feature."
        )
    )

    parser.add_argument(
        "--input-vector",
        type=Path,
        default=Path("assets/maps/manpura_landuse.gpkg"),
        help="Path to the input vector file.",
    )

    parser.add_argument(
        "--output-epsg",
        type=int,
        default=4326,
        help="EPSG code for the output CRS (default: 4326).",
    )

    parser.add_argument(
        "--output-vector",
        type=Path,
        default=Path("assets/maps/manpura_dissolved.gpkg"),
        help="Output GeoPackage path.",
    )

    parser.add_argument(
        "--name",
        default="mapura",
        help='Value written to the output "name" field (default: mapura).',
    )

    parser.add_argument(
        "--layer",
        default=None,
        help="Input layer name. Omit it when the GeoPackage has only one layer.",
    )

    parser.add_argument(
        "--remove-holes",
        action="store_true",
        help="Remove interior polygon holes and retain only exterior rings.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    try:
        create_dissolved_boundary(
            input_vector=resolve_path(args.input_vector),
            output_vector=resolve_path(args.output_vector),
            output_epsg=args.output_epsg,
            name=args.name,
            layer=args.layer,
            remove_holes=args.remove_holes,
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
