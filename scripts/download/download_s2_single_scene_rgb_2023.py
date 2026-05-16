#!/usr/bin/env python3
"""
Download one cloud-masked single-date Sentinel-2 RGB image for a small AOI.

Purpose:
    Image (A): Sentinel-2 RGB image with cloud-masked / missing pixels,
    without median composite generation.

Example:
    python scripts/download/download_s2_single_scene_rgb_2023.py \
        --year 2023 \
        --bbox 89.88 23.78 89.96 23.84 \
        --project YOUR_GEE_PROJECT_ID \
        --out data/raw/sentinel_single_scene/s2_2023_single_scene_rgb.tif

Notes:
    --bbox order is:
        min_lon min_lat max_lon max_lat
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import ee
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CSPLUS_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"

QA_BAND = "cs_cdf"
CLEAR_THRESHOLD_DEFAULT = 0.60
NODATA = 65535

MAX_RETRIES = 6
RETRY_SLEEP = 10


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def jst_now() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S UTC+09:00")


def log(msg: str) -> None:
    print(f"[{jst_now()}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download one single-date cloud-masked Sentinel-2 RGB image."
    )

    p.add_argument("--year", type=int, default=2023)
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Small map-area bounding box in EPSG:4326.",
    )
    p.add_argument(
        "--project",
        type=str,
        default=os.environ.get("GEE_PROJECT_ID"),
        help="Google Earth Engine project ID.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/sentinel_single_scene/s2_2023_single_scene_rgb.tif"),
    )
    p.add_argument(
        "--start-month-day",
        type=str,
        default="10-01",
        help="Start month-day for image search. Default: 10-01.",
    )
    p.add_argument(
        "--end-month-day",
        type=str,
        default="12-31",
        help="End month-day for image search. Default: 12-31.",
    )
    p.add_argument(
        "--cloud-threshold",
        type=float,
        default=CLEAR_THRESHOLD_DEFAULT,
        help="Cloud Score+ clear-sky threshold.",
    )
    p.add_argument(
        "--max-cloud-percent",
        type=float,
        default=80.0,
        help="Maximum Sentinel-2 CLOUDY_PIXEL_PERCENTAGE for candidate scenes.",
    )
    p.add_argument("--scale", type=float, default=10.0)
    p.add_argument("--crs", type=str, default="EPSG:4326")

    return p.parse_args()


def initialize_ee(project: str | None) -> None:
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        log("Earth Engine initialized.")
    except Exception:
        log("Earth Engine authentication required...")
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        log("Earth Engine initialized after authentication.")


def mask_s2_cloud_score_plus(img: ee.Image, threshold: float) -> ee.Image:
    """
    Apply Cloud Score+ clear-sky masking.

    Missing Cloud Score+ values are treated as invalid here because the purpose
    is to create a single-scene image where missing/cloud-masked pixels remain visible.
    """
    qa = img.select(QA_BAND)
    clear_mask = qa.gte(threshold)

    return (
        img.updateMask(clear_mask)
        .select(["B4", "B3", "B2"])
        .unmask(NODATA)
        .toUint16()
    )


def build_single_scene(year: int, bbox: list[float], threshold: float, max_cloud_percent: float) -> ee.Image:
    min_lon, min_lat, max_lon, max_lat = bbox
    roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat], geodesic=False)

    start = f"{year}-10-01"
    end = f"{year + 1}-01-01"

    s2 = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_percent))
    )

    cs = ee.ImageCollection(CSPLUS_COLLECTION)

    linked = s2.linkCollection(cs, [QA_BAND])

    # Select the least cloudy Sentinel-2 scene in the target dry-season window.
    selected = ee.Image(linked.sort("CLOUDY_PIXEL_PERCENTAGE").first())

    selected_info = selected.toDictionary(
        ["system:index", "system:time_start", "CLOUDY_PIXEL_PERCENTAGE", "MGRS_TILE"]
    ).getInfo()

    log("Selected Sentinel-2 scene:")
    log(json.dumps(selected_info, indent=2))

    rgb = mask_s2_cloud_score_plus(selected, threshold).clip(roi)

    return rgb.set({
        "selected_scene": selected_info.get("system:index"),
        "cloud_threshold": threshold,
        "year": year,
        "purpose": "single_scene_cloud_masked_rgb_without_median_composite",
    })


def get_download_url(image: ee.Image, request: dict) -> str:
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return image.getDownloadURL(request)
        except Exception as exc:
            last_err = exc
            log(f"Attempt {attempt} failed to get download URL: {exc}")
            time.sleep(RETRY_SLEEP * attempt)

    raise RuntimeError(f"Failed to get download URL: {last_err}")


def download_file(url: str, out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_zip.with_suffix(out_zip.suffix + ".part")

    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1000]}")

                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)

            tmp.replace(out_zip)
            return

        except Exception as exc:
            last_err = exc
            log(f"Download attempt {attempt} failed: {exc}")
            time.sleep(RETRY_SLEEP * attempt)

    raise RuntimeError(f"Download failed: {last_err}")


def extract_geotiff(downloaded_zip: Path, final_out: Path) -> None:
    extract_dir = downloaded_zip.with_suffix("")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(downloaded_zip, "r") as z:
        z.extractall(extract_dir)

    tif_files = sorted(extract_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No GeoTIFF found inside {downloaded_zip}")

    final_out.parent.mkdir(parents=True, exist_ok=True)
    tif_files[0].replace(final_out)

    log(f"Saved GeoTIFF: {final_out}")


def main() -> None:
    args = parse_args()
    args.out = resolve_path(args.out)

    initialize_ee(args.project)

    min_lon, min_lat, max_lon, max_lat = args.bbox
    roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat], geodesic=False)

    image = build_single_scene(
        year=args.year,
        bbox=args.bbox,
        threshold=args.cloud_threshold,
        max_cloud_percent=args.max_cloud_percent,
    )

    request = {
        "region": roi,
        "scale": args.scale,
        "crs": args.crs,
        "filePerBand": False,
        "format": "GEO_TIFF",
    }

    log("Requesting Earth Engine download URL...")
    url = get_download_url(image, request)

    zip_path = args.out.with_suffix(".zip")

    log(f"Downloading to: {zip_path}")
    download_file(url, zip_path)

    log("Extracting GeoTIFF...")
    extract_geotiff(zip_path, args.out)

    log("DONE")


if __name__ == "__main__":
    main()
