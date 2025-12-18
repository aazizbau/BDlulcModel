"""
Download Google AlphaEarth satellite embeddings for the Bangladesh coastal region (tiled).

Usage:
    python scripts/gee/download_alphaearth_embeddings.py \
        --year 2024 \
        --project ee-your-project-id \
        --output data/raw/embeddings/bd_coastal_alphaearth_2024.tif \
        --tile-width-km 30 --tile-height-km 30
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import ee
import geemap


AOI_NAME = "BD_COASTAL_DISREI"
BD_COASTAL_BBOX = [
    [88.4663, 23.5885],  # upper left
    [88.6038, 20.2039],  # lower left
    [92.8495, 20.2278],  # lower right
    [92.6043, 23.7499],  # upper right
    [88.4663, 23.5885],  # close polygon
]


def initialize_earth_engine(project: str | None = None) -> None:
    """Authenticate and initialize Google Earth Engine."""
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except ee.EEException:
        print("Authenticating to Google Earth Engine ...")
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    else:
        print("Google Earth Engine initialized.")


def create_bd_coastal_geometry() -> ee.Geometry:
    """Return a polygon geometry covering the Bangladesh coastal bounding box."""
    return ee.Geometry.Polygon([BD_COASTAL_BBOX])


def build_embeddings_image(year: int, geometry: ee.Geometry) -> ee.Image:
    """Return the AlphaEarth mosaic for the requested year and geometry."""
    collection = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filter(ee.Filter.date(f"{year}-01-01", f"{year + 1}-01-01"))
        .filter(ee.Filter.bounds(geometry))
    )

    size = collection.size().getInfo()
    if size == 0:
        raise RuntimeError("No AlphaEarth embeddings found for the supplied parameters.")

    print(f"Found {size} AlphaEarth tiles. Mosaicking into a single image ...")
    return collection.mosaic()


def km_to_deg_lat(kilometers: float) -> float:
    """Convert kilometers to degrees latitude."""
    return kilometers / 111.32


def km_to_deg_lon(kilometers: float, reference_lat: float) -> float:
    """Convert kilometers to degrees longitude at the provided latitude."""
    cos_lat = math.cos(math.radians(reference_lat))
    if cos_lat == 0:
        raise ValueError("Cannot compute longitude degrees at the poles.")
    return kilometers / (111.32 * cos_lat)


def get_bounds_from_polygon(polygon: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """Return bounding coordinates (min_lon, min_lat, max_lon, max_lat)."""
    lons = [pt[0] for pt in polygon]
    lats = [pt[1] for pt in polygon]
    return min(lons), min(lats), max(lons), max(lats)


def iterate_tiles(
    polygon: Sequence[Sequence[float]],
    geometry: ee.Geometry,
    tile_width_km: float,
    tile_height_km: float,
    overlap_km: float,
) -> Iterable[Tuple[int, int, ee.Geometry]]:
    """Yield tiled sub-geometries covering the target region."""
    min_lon, min_lat, max_lon, max_lat = get_bounds_from_polygon(polygon)
    ref_lat = 0.5 * (min_lat + max_lat)

    tile_width_deg = km_to_deg_lon(tile_width_km, ref_lat)
    tile_height_deg = km_to_deg_lat(tile_height_km)
    overlap_lon_deg = km_to_deg_lon(overlap_km, ref_lat)
    overlap_lat_deg = km_to_deg_lat(overlap_km)

    if tile_width_deg <= 0 or tile_height_deg <= 0:
        raise ValueError("Tile width/height must be greater than zero.")

    step_lon = tile_width_deg - overlap_lon_deg
    step_lat = tile_height_deg - overlap_lat_deg
    if step_lon <= 0 or step_lat <= 0:
        raise ValueError("Tile overlap must be smaller than the tile dimensions.")

    row = 0
    lat_start = min_lat
    while lat_start < max_lat:
        lat_end = min(lat_start + tile_height_deg, max_lat)
        col = 0
        lon_start = min_lon
        while lon_start < max_lon:
            lon_end = min(lon_start + tile_width_deg, max_lon)
            rect = ee.Geometry.Rectangle(
                [lon_start, lat_start, lon_end, lat_end],
                geodesic=False,
            )
            tile_region = rect.intersection(geometry, ee.ErrorMargin(1))
            area = tile_region.area(maxError=1).getInfo()
            if area and area > 0:
                yield row, col, tile_region
            lon_start += step_lon
            col += 1
        lat_start += step_lat
        row += 1


def resolve_output_template(base: Path) -> Tuple[Path, str, str]:
    """Return parent directory, filename stem, and suffix for tile exports."""
    if base.suffix:
        return base.parent, base.stem, base.suffix
    return base, "tile", ".tif"


def export_tiled_embeddings(
    image: ee.Image,
    polygon: Sequence[Sequence[float]],
    geometry: ee.Geometry,
    output: Path,
    crs: str,
    scale: int,
    tile_width_km: float,
    tile_height_km: float,
    overlap_km: float,
) -> None:
    """Export the embeddings image to disk using a tiled strategy."""
    tiles = list(iterate_tiles(polygon, geometry, tile_width_km, tile_height_km, overlap_km))
    if not tiles:
        raise RuntimeError("No tiles generated for the provided geometry.")

    parent, stem, suffix = resolve_output_template(output)
    parent.mkdir(parents=True, exist_ok=True)

    for idx, (row, col, tile_region) in enumerate(tiles, start=1):
        tile_path = parent / f"{stem}_r{row:02d}_c{col:02d}{suffix}"
        print(f"[{idx}/{len(tiles)}] Exporting tile row={row} col={col} -> {tile_path} ...")
        geemap.ee_export_image(
            image.clip(tile_region),
            filename=str(tile_path),
            region=tile_region,
            crs=crs,
            scale=scale,
        )
    print("All tiles downloaded.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Download AlphaEarth embeddings for {AOI_NAME}."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Acquisition year for the embeddings (default: 2024).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/embeddings/bd_coastal_alphaearth_2024.tif"),
        help=(
            "Base output path. Each tile appends _rXX_cYY before the file suffix "
            "(default: data/raw/embeddings/bd_coastal_alphaearth_2024.tif)."
        ),
    )
    parser.add_argument(
        "--crs",
        type=str,
        default="EPSG:4326",
        help="CRS for the export (default: EPSG:4326).",
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
        default=None,
        help="Optional Earth Engine project ID for initialization.",
    )
    parser.add_argument(
        "--tile-width-km",
        type=float,
        default=2.5,
        help="Tile width in kilometers (default: 30 km).",
    )
    parser.add_argument(
        "--tile-height-km",
        type=float,
        default=2.5,
        help="Tile height in kilometers (default: 30 km).",
    )
    parser.add_argument(
        "--tile-overlap-km",
        type=float,
        default=0.5,
        help="Overlap between adjacent tiles in kilometers (default: 2 km).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    initialize_earth_engine(project=args.project)
    geometry = create_bd_coastal_geometry()
    embeddings_image = build_embeddings_image(args.year, geometry)
    export_tiled_embeddings(
        embeddings_image,
        BD_COASTAL_BBOX,
        geometry,
        output=args.output,
        crs=args.crs,
        scale=args.scale,
        tile_width_km=args.tile_width_km,
        tile_height_km=args.tile_height_km,
        overlap_km=args.tile_overlap_km,
    )


if __name__ == "__main__":
    main()
