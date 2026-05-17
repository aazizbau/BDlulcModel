#!/usr/bin/env python3
"""
Download a SINGLE-DATE cloud-masked Sentinel-2 RGB image for a small AOI.

Purpose:
    Image (A): Sentinel-2 RGB image from one observation date,
    with cloud-masked / missing pixels visible as nodata/white.

This script is designed for comparison with:
    Image (B): Oct-Dec median composite Sentinel-2 RGB image.

Important:
    This script does NOT select the least-cloudy image from the whole Oct-Dec period.
    Instead, it selects the Sentinel-2 scene closest to a user-specified date.
    This helps Image (A) show realistic single-day missing/cloud-masked pixels.

Example run using the bbox from the map:
    python scripts/download_single_scene/download_s2_single_date_rgb_2023.py \
        --date 2023-10-18 \
        --bbox 90.20 22.30 90.283333 22.366667 \
        --project YOUR_GEE_PROJECT_ID \
        --cloud-threshold 0.80 \
        --out data/raw/sentinel_single_scene/s2_2023_10_18_single_date_rgb.tif

BBOX order:
    min_lon min_lat max_lon max_lat

Approximate bbox explanation:
    90°12′E = 90.20
    90°17′E = 90.283333
    22°18′N = 22.30
    22°22′N = 22.366667
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import ee
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CSPLUS_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"

QA_BAND = "cs_cdf"

# Higher value = stricter cloud masking = more missing pixels.
CLEAR_THRESHOLD_DEFAULT = 0.80

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
        description="Download a single-date cloud-masked Sentinel-2 RGB image."
    )

    p.add_argument(
        "--date",
        type=str,
        required=True,
        help="Target date in YYYY-MM-DD format, for example 2023-10-18.",
    )

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
        default=Path("data/raw/sentinel_single_scene/s2_2023_single_date_rgb.tif"),
        help="Output GeoTIFF path.",
    )

    p.add_argument(
        "--cloud-threshold",
        type=float,
        default=CLEAR_THRESHOLD_DEFAULT,
        help=(
            "Cloud Score+ clear-sky threshold. "
            "Higher values mask more pixels. Recommended: 0.75 to 0.90."
        ),
    )

    p.add_argument(
        "--search-days",
        type=int,
        default=3,
        help=(
            "Search window around the target date. "
            "For example, 3 means target date +/- 3 days."
        ),
    )

    p.add_argument(
        "--max-cloud-percent",
        type=float,
        default=100.0,
        help=(
            "Maximum Sentinel-2 CLOUDY_PIXEL_PERCENTAGE allowed. "
            "Use 100 to allow cloudy scenes for demonstration."
        ),
    )

    p.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="Output pixel scale in meters.",
    )

    p.add_argument(
        "--crs",
        type=str,
        default="EPSG:4326",
        help="Output CRS. Default is EPSG:4326 for easy plotting with lon/lat axes.",
    )

    p.add_argument(
        "--list-scenes-only",
        action="store_true",
        help="Only print available scenes near the target date and do not download.",
    )

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
    Apply strict Cloud Score+ clear-sky masking.

    Important:
        Missing Cloud Score+ pixels are treated as invalid/masked.
        This is intentional for Image (A), because the goal is to show
        missing/cloud-masked pixels from a single observation.
    """
    qa = img.select(QA_BAND)

    # Do NOT unmask missing CS+ values to valid pixels.
    # Only pixels with cs_cdf >= threshold are retained.
    clear_mask = qa.gte(threshold)

    rgb = (
        img.updateMask(clear_mask)
        .select(["B4", "B3", "B2"])
        .unmask(NODATA)
        .toUint16()
    )

    return rgb


def get_candidate_collection(
    target_date: str,
    bbox: list[float],
    search_days: int,
    max_cloud_percent: float,
) -> tuple[ee.ImageCollection, ee.Geometry, str, str, int]:
    min_lon, min_lat, max_lon, max_lat = bbox
    roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat], geodesic=False)

    dt = datetime.strptime(target_date, "%Y-%m-%d")
    start_dt = dt - timedelta(days=search_days)
    end_dt = dt + timedelta(days=search_days + 1)

    start = start_dt.strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")

    target_millis = int(dt.timestamp() * 1000)

    log(f"Searching Sentinel-2 scenes from {start} to {end}")
    log(f"Target date: {target_date}")

    s2 = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_percent))
    )

    cs = ee.ImageCollection(CSPLUS_COLLECTION)

    linked = s2.linkCollection(cs, [QA_BAND])

    def add_time_diff(img: ee.Image) -> ee.Image:
        diff = ee.Number(img.get("system:time_start")).subtract(target_millis).abs()
        return img.set("time_diff_abs", diff)

    linked = linked.map(add_time_diff)

    return linked, roi, start, end, target_millis


def list_candidate_scenes(collection: ee.ImageCollection) -> None:
    info = (
        collection.sort("time_diff_abs")
        .limit(20)
        .aggregate_array("system:index")
        .getInfo()
    )

    count = collection.size().getInfo()
    log(f"Number of candidate scenes: {count}")

    if count == 0:
        log("No candidate scenes found.")
        return

    log("Candidate scene IDs:")
    for scene_id in info:
        log(f"  {scene_id}")


def build_single_date_image(
    target_date: str,
    bbox: list[float],
    cloud_threshold: float,
    search_days: int,
    max_cloud_percent: float,
) -> tuple[ee.Image, ee.Geometry]:
    collection, roi, start, end, target_millis = get_candidate_collection(
        target_date=target_date,
        bbox=bbox,
        search_days=search_days,
        max_cloud_percent=max_cloud_percent,
    )

    count = collection.size().getInfo()
    if count == 0:
        raise RuntimeError(
            f"No Sentinel-2 scene found from {start} to {end}. "
            f"Try increasing --search-days."
        )

    selected = ee.Image(collection.sort("time_diff_abs").first())
    selected_date = ee.Date(selected.get("system:time_start"))

    selected_info = selected.toDictionary(
        [
            "system:index",
            "system:time_start",
            "CLOUDY_PIXEL_PERCENTAGE",
            "MGRS_TILE",
            "time_diff_abs",
        ]
    ).getInfo()

    log("Selected Sentinel-2 scene closest to target date:")
    log(json.dumps(selected_info, indent=2))

    # Mosaic same-day granules only.
    # This keeps the image as a single-date observation, but avoids narrow-strip
    # output when the bbox crosses Sentinel-2 MGRS tile boundaries.
    same_date_rgb = (
        collection.filterDate(selected_date, selected_date.advance(1, "day"))
        .filterBounds(roi)
        .map(lambda img: mask_s2_cloud_score_plus(img, cloud_threshold))
        .mosaic()
        .clip(roi)
        .unmask(NODATA)
        .toUint16()
    )

    same_date_rgb = same_date_rgb.set(
        {
            "target_date": target_date,
            "selected_scene": selected_info.get("system:index"),
            "cloud_threshold": cloud_threshold,
            "nodata": NODATA,
            "purpose": "single_date_cloud_masked_rgb_without_median_composite",
        }
    )

    return same_date_rgb, roi


def get_download_url(image: ee.Image, request: dict) -> str:
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return image.getDownloadURL(request)
        except Exception as exc:
            last_err = exc
            log(f"Attempt {attempt} failed to get download URL: {exc}")
            time.sleep(RETRY_SLEEP * attempt)

    raise RuntimeError(f"Failed to get download URL after {MAX_RETRIES} attempts: {last_err}")


def download_file(url: str, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(out_file.suffix + ".part")

    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                if r.status_code >= 400:
                    body = r.text[:1000] if hasattr(r, "text") else ""
                    raise RuntimeError(f"HTTP {r.status_code}: {body}")

                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)

            tmp.replace(out_file)
            return

        except Exception as exc:
            last_err = exc
            log(f"Download attempt {attempt} failed: {exc}")
            time.sleep(RETRY_SLEEP * attempt)

    raise RuntimeError(f"Download failed after {MAX_RETRIES} attempts: {last_err}")


def extract_geotiff(downloaded_file: Path, final_out: Path) -> None:
    final_out.parent.mkdir(parents=True, exist_ok=True)

    if not zipfile.is_zipfile(downloaded_file):
        downloaded_file.replace(final_out)
        log(f"Saved GeoTIFF: {final_out}")
        return

    extract_dir = downloaded_file.with_suffix("")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(downloaded_file, "r") as z:
        z.extractall(extract_dir)

    tif_files = sorted(extract_dir.glob("*.tif"))

    if not tif_files:
        raise FileNotFoundError(f"No GeoTIFF found inside {downloaded_file}")

    tif_files[0].replace(final_out)

    log(f"Saved GeoTIFF: {final_out}")


def main() -> None:
    args = parse_args()
    args.out = resolve_path(args.out)

    initialize_ee(args.project)

    collection, _, _, _, _ = get_candidate_collection(
        target_date=args.date,
        bbox=list(args.bbox),
        search_days=args.search_days,
        max_cloud_percent=args.max_cloud_percent,
    )

    if args.list_scenes_only:
        list_candidate_scenes(collection)
        return

    image, roi = build_single_date_image(
        target_date=args.date,
        bbox=list(args.bbox),
        cloud_threshold=args.cloud_threshold,
        search_days=args.search_days,
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
