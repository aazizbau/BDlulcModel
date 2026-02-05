#!/usr/bin/env python3
"""
Rasterize 10-class Upazila GPKG into a 10m label GeoTIFF (EPSG:32646).

This v3 keeps your original behavior, but adds optional knobs:
- --all-touched (default False) if you ever want more inclusive burning
- same outputs and defaults as v2

Example:
  python scripts/labels/make_training_label_v3.py --upazila betagi
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

UPAZILA_GPKG = {
    "manpura": Path("assets/maps/manpura_10class.gpkg"),
    "betagi": Path("assets/maps/betagi_10class.gpkg"),
    "amtali": Path("assets/maps/amtali_10class.gpkg"),
    "bamna": Path("assets/maps/bamna_10class.gpkg"),
}

DEFAULT_OUT_DIR = Path("assets/training_labels")
DEFAULT_RES = 10.0
DEFAULT_CRS = "EPSG:32646"

LABEL_FIELD = "class10_id"
NAME_FIELD = "class10_name"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rasterize training labels (10-class) from upazila GPKG.")
    p.add_argument("--upazila", required=True, choices=sorted(UPAZILA_GPKG.keys()))
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--resolution", type=float, default=DEFAULT_RES)
    p.add_argument("--crs", default=DEFAULT_CRS)
    p.add_argument("--nodata", type=int, default=0)
    p.add_argument(
        "--all-touched",
        action="store_true",
        help="If set, rasterize with all_touched=True (default False).",
    )
    return p.parse_args()


def aligned_bounds(minx: float, miny: float, maxx: float, maxy: float, res: float) -> tuple[float, float, float, float]:
    minx_a = np.floor(minx / res) * res
    miny_a = np.floor(miny / res) * res
    maxx_a = np.ceil(maxx / res) * res
    maxy_a = np.ceil(maxy / res) * res
    return float(minx_a), float(miny_a), float(maxx_a), float(maxy_a)


def main() -> None:
    args = parse_args()

    gpkg_path = UPAZILA_GPKG[args.upazila]
    if not gpkg_path.exists():
        raise SystemExit(f"GPKG not found: {gpkg_path}")

    gdf = gpd.read_file(gpkg_path)
    if gdf.empty:
        raise SystemExit(f"No features found in: {gpkg_path}")

    if gdf.crs is None:
        raise SystemExit(f"{gpkg_path} has no CRS. Expected {args.crs}.")
    if str(gdf.crs).upper() != args.crs.upper():
        gdf = gdf.to_crs(args.crs)

    missing = [c for c in [LABEL_FIELD, NAME_FIELD] if c not in gdf.columns]
    if missing:
        raise SystemExit(f"Missing fields {missing} in {gpkg_path}. Columns={list(gdf.columns)}")

    gdf = gdf[[LABEL_FIELD, NAME_FIELD, "geometry"]].copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.buffer(0)

    ids = gdf[LABEL_FIELD].to_numpy()
    bad = (~np.isfinite(ids)) | (ids < 1) | (ids > 10)
    if np.any(bad):
        bad_vals = np.unique(ids[bad])
        raise SystemExit(f"Invalid class ids in '{LABEL_FIELD}', expected 1..10. Bad: {bad_vals[:20]}")

    present = (
        gdf[[LABEL_FIELD, NAME_FIELD]]
        .drop_duplicates()
        .sort_values(LABEL_FIELD)
        .to_records(index=False)
        .tolist()
    )
    print(f"[INFO] {args.upazila}: classes present (id -> name):")
    for cid, cname in present:
        print(f"  {int(cid):2d} -> {cname}")

    minx, miny, maxx, maxy = aligned_bounds(*gdf.total_bounds, args.resolution)
    width = int(round((maxx - minx) / args.resolution))
    height = int(round((maxy - miny) / args.resolution))
    if width <= 0 or height <= 0:
        raise SystemExit("Computed non-positive raster dims. Check bounds/CRS.")

    transform = from_origin(minx, maxy, args.resolution, args.resolution)
    shapes = ((geom, int(val)) for geom, val in zip(gdf.geometry, gdf[LABEL_FIELD]))

    label_raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=int(args.nodata),
        dtype="uint8",
        all_touched=bool(args.all_touched),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.upazila}_label_10class_{int(args.resolution)}m.tif"

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": args.crs,
        "transform": transform,
        "nodata": int(args.nodata),
        "compress": "ZSTD",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(label_raster, 1)

    unique, counts = np.unique(label_raster, return_counts=True)
    summary = {int(k): int(v) for k, v in zip(unique, counts)}

    print(f"\nSaved labels: {out_path}")
    print(f"CRS: {args.crs} | res={args.resolution}m | shape={height}x{width} | nodata={args.nodata}")
    print(f"Pixel counts (including nodata): {summary}")


if __name__ == "__main__":
    main()
