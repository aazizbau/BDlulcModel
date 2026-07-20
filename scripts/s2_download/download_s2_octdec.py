#!/usr/bin/env python3
"""
Fresh-download CLI Sentinel-2 Oct-Dec composite (Cloud Score+ masked, tiled),
with unmask->65535, nodata handling, native scale exporting, and local resampling.

One-pass robust default: tile_deg = 0.125

Reproduction and AOI adaptation
-------------------------------
Workflow role: Download cloud-screened Sentinel-2 seasonal imagery from Earth Engine.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--year``, ``--band``, ``--aoi``, ``--project``, ``--outdir``, ``--resume``, ``--tile-deg``, ``--scale``, ``--crs``, ``--cloud-threshold``. Run the ``--help`` command below for required values, defaults, and accepted choices.
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

    python scripts/s2_download/download_s2_octdec.py --help
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import ee
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gee.aoi import load_aoi
from src.gee.earth_engine import initialize_earth_engine

# -----------------------------
# CONFIG
# -----------------------------

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CSPLUS_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"

QA_BAND = "cs_cdf"
CLEAR_THRESHOLD = 0.60

SCALE_DEFAULT = 10
CRS_DEFAULT = "EPSG:4326"

# Robust default: smaller tiles
TILE_DEG_DEFAULT = 0.125

MAX_RETRIES = 6
RETRY_SLEEP = 10

NODATA = 65535  # uint16-safe nodata fill
GEE_PROJECT_ENV = "GEE_PROJECT_ID"

BAND_20M = {"B5", "B6", "B7", "B8A", "B11", "B12"}

# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Sentinel-2 composite with native resolution handling and robust masking."
    )
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--band", type=str, required=True)
    p.add_argument("--aoi", type=Path, default=Path("configs/bd_coastal_aoi.yaml"))
    p.add_argument("--project", type=str, default=os.environ.get(GEE_PROJECT_ENV))
    p.add_argument("--outdir", type=Path, default=Path("data/raw/sentinel_gemini"))
    p.add_argument("--resume", action="store_true", help="Resume by reusing existing tiles")

    p.add_argument("--tile-deg", type=float, default=TILE_DEG_DEFAULT, help="Tile size in degrees")
    p.add_argument("--scale", type=float, default=SCALE_DEFAULT, help="Target pixel scale in meters (default: 10)")
    p.add_argument("--crs", type=str, default=CRS_DEFAULT, help="Output CRS (default: EPSG:4326)")
    p.add_argument("--cloud-threshold", type=float, default=CLEAR_THRESHOLD, help="CS+ threshold")

    return p.parse_args(argv)

# -----------------------------
# EE processing
# -----------------------------

def process_image_mask(img: ee.Image, cloud_threshold: float) -> ee.Image:
    # 1. Edge mask style (B8A + B9)
    img = img.updateMask(img.select("B8A").mask().updateMask(img.select("B9").mask()))

    # 2. Robust Cloud Masking (Fix for missing 2017 tiles)
    qa = img.select(QA_BAND)
    # Unmask missing QA values to -1.
    # This prevents dropping an entire tile if Cloud Score+ skipped the granule.
    qa_unmasked = qa.unmask(-1)

    # Keep pixels if cloud score is >= threshold OR if cloud score is missing (-1)
    valid_mask = qa_unmasked.gte(cloud_threshold).Or(qa_unmasked.eq(-1))

    return img.updateMask(valid_mask)


def build_collection(start: str, end: str, roi: ee.Geometry, cloud_threshold: float) -> ee.ImageCollection:
    s2 = ee.ImageCollection(S2_COLLECTION)
    cs = ee.ImageCollection(CSPLUS_COLLECTION)

    linked = s2.linkCollection(cs, [QA_BAND]).filterBounds(roi).filterDate(start, end)

    return linked.map(lambda img: process_image_mask(img, cloud_threshold))


def _normalize_band(band: str) -> str:
    band = band.strip()
    if band.upper().startswith("B") and band[1:].isdigit():
        return f"B{int(band[1:])}"
    return band


def build_composite(year: int, band: str, roi: ee.Geometry, cloud_threshold: float) -> ee.Image:
    # 1. Primary Window (Oct 1 - Dec 31)
    ic_primary = build_collection(f"{year}-10-01", f"{year + 1}-01-01", roi, cloud_threshold)
    comp = ic_primary.select([_normalize_band(band)]).median().clip(roi)

    # 2. Thesis-Safe Gap Fill (Only for 2017)
    if year == 2017:
        print("\n[GEE LOGIC] Applying Intra-Year Dry Season Gap-Fill for 2017 missing tiles...")
        # Backup Window: Early 2017 Dry Season (Jan 1 - Apr 30)
        ic_backup = build_collection(f"{year}-01-01", f"{year}-05-01", roi, cloud_threshold)
        comp_backup = ic_backup.select([_normalize_band(band)]).median().clip(roi)

        # Patch the holes in the primary composite with the backup composite
        comp = comp.unmask(comp_backup)

    # 3. Fill any remaining truly empty pixels (ocean/out of bounds)
    comp = comp.unmask(NODATA)

    return comp.toUint16()

# -----------------------------
# AOI bounds
# -----------------------------

def bounds_from_polygon(polygon: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    lons = [pt[0] for pt in polygon]
    lats = [pt[1] for pt in polygon]
    return min(lats), max(lats), min(lons), max(lons)

# -----------------------------
# Tiling
# -----------------------------

@dataclass
class Tile:
    r: int
    c: int
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    crs: str

    def geom(self) -> ee.Geometry:
        return ee.Geometry.Rectangle(
            [self.min_lon, self.min_lat, self.max_lon, self.max_lat],
            proj=self.crs,
            geodesic=False,
        )

    def tag(self) -> str:
        return f"r{self.r:03d}_c{self.c:03d}"


def make_tiles(min_lat: float, max_lat: float, min_lon: float, max_lon: float, step: float, crs: str) -> List[Tile]:
    tiles: List[Tile] = []
    rows = math.ceil((max_lat - min_lat) / step)
    cols = math.ceil((max_lon - min_lon) / step)

    for r in range(rows):
        for c in range(cols):
            tiles.append(
                Tile(
                    r=r,
                    c=c,
                    min_lon=min_lon + c * step,
                    min_lat=min_lat + r * step,
                    max_lon=min(min_lon + (c + 1) * step, max_lon),
                    max_lat=min(min_lat + (r + 1) * step, max_lat),
                    crs=crs,
                )
            )
    return tiles

# -----------------------------
# Download helpers
# -----------------------------

def download(url: str, out: Path) -> bool:
    tmp = out.with_suffix(out.suffix + ".part")
    last_err: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                if r.status_code >= 400:
                    body = ""
                    try:
                        body = r.text[:2000]
                    except Exception:
                        body = "<no body>"
                    raise RuntimeError(f"HTTP {r.status_code} for {out}\n{body}")

                with open(tmp, "wb") as f:
                    for ch in r.iter_content(1024 * 1024):
                        if ch:
                            f.write(ch)

            os.replace(tmp, out)
            return True

        except Exception as exc:
            last_err = exc
            time.sleep(RETRY_SLEEP * attempt)

    print(f"Download failed after {MAX_RETRIES} attempts: {out} | {last_err}")
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass
    return False


def get_download_url(image: ee.Image, request: dict) -> str:
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return image.getDownloadURL(request)
        except Exception as exc:
            last_err = exc
            time.sleep(RETRY_SLEEP * attempt)
    raise RuntimeError(f"Failed to get download URL after {MAX_RETRIES} attempts: {last_err}")


def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def write_sidecar_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

# -----------------------------
# MAIN
# -----------------------------

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    initialize_earth_engine(project=args.project)

    aoi = load_aoi(args.aoi)
    roi = aoi.to_ee_geometry()

    min_lat, max_lat, min_lon, max_lon = bounds_from_polygon(aoi.bbox_polygon())

    band_norm = _normalize_band(args.band)
    is_20m = band_norm in BAND_20M

    # CRITICAL: Dynamically set the GEE download scale
    download_scale = 20.0 if is_20m else float(args.scale)

    out_base = args.outdir / aoi.name / f"S2_{args.year}_octdec_{band_norm}"
    tiles_dir = out_base / "tiles"
    vrt = out_base.with_suffix(".vrt")
    tif = out_base.with_name(out_base.name + f"_mosaic_{int(download_scale)}m_lzw.tif")
    sidecar = out_base.with_name(out_base.name + "_meta.json")
    failed_txt = out_base.with_name(out_base.name + "_failed_tiles.txt")

    if args.resume:
        for p in [vrt, tif, sidecar, failed_txt]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        tiles_dir.mkdir(parents=True, exist_ok=True)
    else:
        if tiles_dir.exists():
            shutil.rmtree(tiles_dir, ignore_errors=True)
        for p in [vrt, tif, sidecar, failed_txt]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        tiles_dir.mkdir(parents=True, exist_ok=True)

    print(f"AOI: {aoi.name}")
    print(f"Year/Band: {args.year} / {band_norm}")
    print(f"Native 20m Band: {is_20m} -> Fetching from GEE at {download_scale}m")

    image = build_composite(
        args.year,
        args.band,
        roi,
        args.cloud_threshold,
    )

    write_sidecar_json(
        sidecar,
        {
            "aoi": aoi.name,
            "year": args.year,
            "band": band_norm,
            "collection": S2_COLLECTION,
            "export": {
                "crs": args.crs,
                "target_scale": args.scale,
                "downloaded_scale": download_scale,
                "tile_deg": args.tile_deg,
                "dtype": "uint16",
                "nodata": NODATA,
            },
        },
    )

    tiles = make_tiles(min_lat, max_lat, min_lon, max_lon, args.tile_deg, args.crs)
    print(f"Total tiles to download: {len(tiles)}")

    paths: List[Path] = []
    failed: List[Path] = []

    for tile in tqdm(tiles, desc="Downloading tiles"):
        fname = tiles_dir / f"{aoi.name}_{args.year}_{band_norm}_{tile.tag()}.tif"

        if args.resume and fname.exists():
            paths.append(fname)
            continue

        url = get_download_url(
            image,
            {
                "region": tile.geom(),
                "scale": download_scale,
                "crs": args.crs,
                "filePerBand": False,
                "format": "GEO_TIFF",
            },
        )

        ok = download(url, fname)
        if ok:
            paths.append(fname)
        else:
            failed.append(fname)

    paths = [p for p in paths if p.exists()]

    if failed:
        print(f"\nWARNING: {len(failed)} tiles failed.")
        with open(failed_txt, "w", encoding="utf-8") as f:
            for p in failed:
                f.write(str(p) + "\n")

    if not paths:
        print("No tiles downloaded successfully; skipping VRT/mosaic build.")
        return

    print("\nBuilding VRT...")
    run(["gdalbuildvrt", "-overwrite", str(vrt), *map(str, sorted(paths))])

    print(f"Building mosaic GeoTIFF at {download_scale}m...")
    run([
        "gdal_translate", str(vrt), str(tif),
        "-a_nodata", str(NODATA),
        "-co", "COMPRESS=LZW", "-co", "BIGTIFF=YES", "-co", "TILED=YES",
    ])

    if is_20m and float(args.scale) == 10.0:
        resampled_tif = out_base.with_name(out_base.name + f"_resampled_{int(args.scale)}m_lzw.tif")

        if resampled_tif.exists():
            resampled_tif.unlink()

        print(f"\n--- Local Resampling ---")
        print(f"Upscaling native 20m export to exactly {args.scale}m using gdal_translate...")

        run([
            "gdal_translate",
            "-r", "bilinear",
            "-outsize", "200%", "200%",
            "-a_nodata", str(NODATA),
            "-co", "COMPRESS=LZW",
            "-co", "BIGTIFF=YES",
            "-co", "TILED=YES",
            str(tif), str(resampled_tif)
        ])
        print(f"Final {args.scale}m Native-Aligned Mosaic: {resampled_tif}")
    else:
        print(f"Final {args.scale}m Mosaic : {tif}")

    print("\nDONE")

if __name__ == "__main__":
    main()
