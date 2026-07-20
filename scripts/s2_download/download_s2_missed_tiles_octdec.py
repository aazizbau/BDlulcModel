#!/usr/bin/env python3
"""Reproduction and AOI adaptation
-------------------------------
Workflow role: Download cloud-screened Sentinel-2 seasonal imagery from Earth Engine.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--year``, ``--band``, ``--aoi``, ``--project``, ``--outdir``, ``--tile-deg``, ``--scale``, ``--crs``, ``--cloud-threshold``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace the Earth Engine project, AOI geometry, years, cloud thresholds, seasonal window, export scale/CRS, and destination directory.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.

Reproducible invocation
~~~~~~~~~~~~~~~~~~~~~~~
Inspect the complete interface before supplying AOI-specific paths::

    python scripts/s2_download/download_s2_missed_tiles_octdec.py --help
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import ee
from tqdm import tqdm

# Connect to your existing script's directory
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import the core functions directly from your Gemini master script
from download_s2_octdec import (
    initialize_earth_engine, load_aoi, bounds_from_polygon,
    _normalize_band, build_composite, make_tiles, get_download_url, download,
    NODATA, BAND_20M, SCALE_DEFAULT, CRS_DEFAULT
)

GEE_PROJECT_ENV = "GEE_PROJECT_ID"

def run(cmd):
    subprocess.run(cmd, check=True)

def main():
    p = argparse.ArgumentParser(description="Heal GEE User Memory Limit Exceeded tiles.")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--band", type=str, required=True)
    p.add_argument("--aoi", type=Path, default=Path("configs/bd_coastal_aoi.yaml"))
    p.add_argument("--project", type=str, default=os.environ.get(GEE_PROJECT_ENV))
    p.add_argument("--outdir", type=Path, default=Path("data/raw/sentinel_gemini"))
    p.add_argument("--tile-deg", type=float, default=0.125)
    p.add_argument("--scale", type=float, default=10.0)
    p.add_argument("--crs", type=str, default=CRS_DEFAULT)
    p.add_argument("--cloud-threshold", type=float, default=0.60)
    args = p.parse_args()

    if not args.project:
        raise SystemExit(
            f'Missing GEE project ID. Set --project or export {GEE_PROJECT_ENV}="your-ee-project-id".'
        )

    initialize_earth_engine(project=args.project)

    aoi = load_aoi(args.aoi)
    roi = aoi.to_ee_geometry()
    min_lat, max_lat, min_lon, max_lon = bounds_from_polygon(aoi.bbox_polygon())

    band_norm = _normalize_band(args.band)
    is_20m = band_norm in BAND_20M
    download_scale = 20.0 if is_20m else float(args.scale)

    out_base = args.outdir / aoi.name / f"S2_{args.year}_octdec_{band_norm}"
    tiles_dir = out_base / "tiles"
    vrt = out_base.with_suffix(".vrt")
    tif = out_base.with_name(out_base.name + f"_mosaic_{int(download_scale)}m_lzw.tif")

    print(f"\n--- Healing GEE Memory Limits for {args.year} {band_norm} ---")

    # Rebuild the theoretical tile grid
    tiles = make_tiles(min_lat, max_lat, min_lon, max_lon, args.tile_deg, args.crs)

    # Detect which tiles are missing (failed to download)
    missing_tiles = []
    for t in tiles:
        fname = tiles_dir / f"{aoi.name}_{args.year}_{band_norm}_{t.tag()}.tif"
        # Check if quadrants already exist from a partially completed heal run
        q_files = list(tiles_dir.glob(f"{aoi.name}_{args.year}_{band_norm}_{t.tag()}_q*.tif"))
        if not fname.exists() and len(q_files) < 4:
            missing_tiles.append(t)

    if not missing_tiles:
        print("No missing tiles detected. The mosaic is already complete!")
        return

    print(f"Detected {len(missing_tiles)} missing tiles. Applying 4x subdivision strategy...")
    image = build_composite(args.year, args.band, roi, args.cloud_threshold)

    for t in tqdm(missing_tiles, desc="Downloading memory-safe quadrants"):
        mid_lon = (t.min_lon + t.max_lon) / 2.0
        mid_lat = (t.min_lat + t.max_lat) / 2.0

        # Subdivide the failing block into 4 smaller, memory-safe quadrants
        quadrants = [
            ("q1", t.min_lon, mid_lat, mid_lon, t.max_lat),
            ("q2", mid_lon, mid_lat, t.max_lon, t.max_lat),
            ("q3", t.min_lon, t.min_lat, mid_lon, mid_lat),
            ("q4", mid_lon, t.min_lat, t.max_lon, mid_lat),
        ]

        for q_tag, mn_lon, mn_lat, mx_lon, mx_lat in quadrants:
            sub_geom = ee.Geometry.Rectangle([mn_lon, mn_lat, mx_lon, mx_lat], proj=t.crs, geodesic=False)
            sub_fname = tiles_dir / f"{aoi.name}_{args.year}_{band_norm}_{t.tag()}_{q_tag}.tif"

            if sub_fname.exists():
                continue

            url = get_download_url(
                image,
                {"region": sub_geom, "scale": download_scale, "crs": args.crs, "filePerBand": False, "format": "GEO_TIFF"}
            )
            download(url, sub_fname)

    # --- Rebuild Mosaic ---
    paths = list(tiles_dir.glob("*.tif"))
    print("\nRebuilding VRT with healed quadrants...")
    run(["gdalbuildvrt", "-overwrite", str(vrt), *map(str, sorted(paths))])

    print(f"Building final complete mosaic GeoTIFF at {download_scale}m...")
    run([
        "gdal_translate", str(vrt), str(tif),
        "-a_nodata", str(NODATA),
        "-co", "COMPRESS=LZW", "-co", "BIGTIFF=YES", "-co", "TILED=YES",
    ])

    # Ensure 20m -> 10m logic triggers correctly if repairing B11/B12
    if is_20m and float(args.scale) == 10.0:
        resampled_tif = out_base.with_name(out_base.name + f"_resampled_{int(args.scale)}m_lzw.tif")
        if resampled_tif.exists(): resampled_tif.unlink()
        print(f"Upscaling native 20m export to exactly {args.scale}m...")
        run([
            "gdal_translate", "-r", "bilinear", "-outsize", "200%", "200%",
            "-a_nodata", str(NODATA), "-co", "COMPRESS=LZW", "-co", "BIGTIFF=YES", "-co", "TILED=YES",
            str(tif), str(resampled_tif)
        ])
        print(f"Final Healed {args.scale}m Mosaic: {resampled_tif}")
    else:
        print(f"Final Healed {args.scale}m Mosaic: {tif}")

if __name__ == "__main__":
    main()
