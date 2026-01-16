"""
Window-streamed clipping of AlphaEarth embedding mosaic to coastal Bangladesh AOI.

Features:
- Streaming / block-based clipping (safe for TB-scale rasters)
- Optional reprojection: EPSG:4326 (default) or EPSG:32646
- Automatic GDAL cache scaling based on free RAM (if psutil installed)
- Progress + ETA logging (clip + reprojection)

Usage:
    python scripts/gee/clip_alphaearth_mosaic.py

    python scripts/gee/clip_alphaearth_mosaic.py --crs EPSG:32646

    python scripts/gee/clip_alphaearth_mosaic.py \
        --input data/interim/custom_alphaearth.tif \
        --output data/processed/features/custom_clipped.tif \
        --cache-mb 8192
"""

from __future__ import annotations

import argparse
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import fiona
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from rasterio.transform import array_bounds
from rasterio.errors import WindowError
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, transform_geom


# -------------------------
# Defaults
# -------------------------

DEFAULT_INPUT = Path("data/interim/bd_coastal_alphaearth_2024_mosaic.tif")
DEFAULT_OUTPUT = Path("data/processed/features/bd_coastal_alphaearth_2024_clipped.tif")
DEFAULT_VECTOR = Path("/media/abdul-aziz/sdb7/masters_research/bd_coastal_map/bd_coastal_map_solid_gp.gpkg")
DEFAULT_CACHE_MB = 4096


# -------------------------
# Utilities
# -------------------------

def log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] {msg}", flush=True)


def try_available_mb() -> int | None:
    """Return available RAM in MB, or None if psutil is unavailable."""
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available / (1024 * 1024))
    except Exception:
        return None


def auto_scale_cache_mb(requested_mb: int, *, floor_mb: int = 512, cap_mb: int = 16384) -> int:
    """
    Scale GDAL cache based on free RAM:
    - Use up to ~70% of available memory
    - Never below floor_mb
    - Never above cap_mb
    - Never above requested_mb (requested acts like a cap)
    """
    avail = try_available_mb()
    if avail is None:
        return max(floor_mb, min(requested_mb, cap_mb))

    safe_target = int(avail * 0.70)
    scaled = min(requested_mb, safe_target, cap_mb)
    return max(floor_mb, scaled)


def read_geometries(vector_path: Path) -> List[dict]:
    with fiona.open(vector_path) as src:
        return [feat["geometry"] for feat in src]


def human_eta(seconds: float) -> str:
    if seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def iter_blocks_in_window(ds: rasterio.DatasetReader, win: Window) -> Iterable[Window]:
    """
    Yield dataset block windows that intersect the given window.
    We iterate block windows (band 1) and filter by intersection.
    """
    for _, bw in ds.block_windows(1):
        try:
            bw.intersection(win)
        except WindowError:
            continue
        else:
            yield bw


def count_blocks_in_window(ds: rasterio.DatasetReader, win: Window) -> int:
    c = 0
    for _, bw in ds.block_windows(1):
        try:
            bw.intersection(win)
        except WindowError:
            continue
        else:
            c += 1
    return c


def write_meta_like(
    src_meta: dict,
    *,
    height: int,
    width: int,
    transform,
    crs,
    nodata,
    compress: str = "ZSTD",
    block_size: int = 512,
) -> dict:
    meta = src_meta.copy()
    meta.update(
        {
            "height": height,
            "width": width,
            "transform": transform,
            "crs": crs,
            "nodata": nodata,
            "tiled": True,
            "interleave": "band",
            "blockxsize": block_size,
            "blockysize": block_size,
            "BIGTIFF": "IF_SAFER",
            "compress": compress,  # rasterio uses lowercase 'compress'
        }
    )
    return meta


# -------------------------
# Streaming clip (no reprojection)
# -------------------------

def stream_clip_same_crs(
    src_path: Path,
    vector_path: Path,
    out_path: Path,
    *,
    cache_mb: int,
    progress_every_s: float = 5.0,
) -> None:
    """
    Stream-clip raster to AOI in source CRS, writing only the cropped extent.
    Reads and writes by blocks, applying a geometry mask per block.
    """
    geoms = read_geometries(vector_path)

    env = {
        # IMPORTANT: rasterio requires correct types: GDAL_CACHEMAX must be int
        "GDAL_CACHEMAX": int(cache_mb),
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "RASTERIO_NUM_THREADS": "ALL_CPUS",
        "GDAL_DISABLE_READDIR_ON_OPEN": "TRUE",
    }

    with rasterio.Env(**env):
        with rasterio.open(src_path) as src:
            if src.crs is None:
                raise SystemExit("Input raster has no CRS. Cannot clip safely.")

            # Reproject AOI geometries into source CRS if needed (vector often EPSG:4326)
            # Fiona returns geometries as-is; we don't know their CRS from here reliably.
            # Best effort: assume vector is EPSG:4326 unless you store otherwise.
            # If your vector CRS differs, pass a vector already in EPSG:4326 or same as raster.
            # We'll attempt a safe transform from EPSG:4326 -> src.crs.
            src_crs = src.crs

            try:
                geoms_src = [transform_geom("EPSG:4326", src_crs, g, precision=6) for g in geoms]
            except Exception:
                # If transform fails, assume vector is already in src CRS
                geoms_src = geoms

            # Compute a tight window around AOI in source pixel coords
            aoi_win = geometry_window(src, geoms_src, pad_x=0, pad_y=0, north_up=True, rotated=False)

            # Clip window to raster bounds
            aoi_win = aoi_win.intersection(Window(0, 0, src.width, src.height))

            # Output transform is window transform
            out_transform = rasterio.windows.transform(aoi_win, src.transform)

            out_h = int(aoi_win.height)
            out_w = int(aoi_win.width)

            nodata = src.nodata
            if nodata is None:
                nodata = 0  # safe fallback for embeddings

            out_meta = write_meta_like(
                src.meta,
                height=out_h,
                width=out_w,
                transform=out_transform,
                crs=src.crs,
                nodata=nodata,
            )

            out_path.parent.mkdir(parents=True, exist_ok=True)

            total_blocks = count_blocks_in_window(src, aoi_win)
            log(f"Streaming clip window: {out_w} x {out_h} px, blocks={total_blocks}")

            t0 = time.time()
            last_log = t0
            done = 0

            with rasterio.open(out_path, "w", **out_meta) as dst:
                for bw in iter_blocks_in_window(src, aoi_win):
                    # Intersect block window with AOI window, read only intersection
                    try:
                        iw = bw.intersection(aoi_win)
                    except WindowError:
                        continue

                    data = src.read(window=iw)  # shape (bands, h, w)

                    # Build mask for this block intersection area
                    block_transform = rasterio.windows.transform(iw, src.transform)
                    m = geometry_mask(
                        geoms_src,
                        transform=block_transform,
                        invert=True,          # True inside geometry
                        out_shape=(int(iw.height), int(iw.width)),
                        all_touched=False,
                    )

                    # Apply nodata outside AOI
                    # data is (bands, h, w); mask is (h, w)
                    if data.dtype.kind in ("f", "c"):
                        fill = float(nodata)
                    else:
                        fill = int(nodata)

                    # Where mask is False (outside), fill with nodata
                    data[:, ~m] = fill

                    # Destination window is relative to output (subtract AOI origin)
                    dst_win = Window(
                        col_off=int(iw.col_off - aoi_win.col_off),
                        row_off=int(iw.row_off - aoi_win.row_off),
                        width=int(iw.width),
                        height=int(iw.height),
                    )
                    dst.write(data, window=dst_win)

                    done += 1
                    now = time.time()
                    if (now - last_log) >= progress_every_s or done == total_blocks:
                        elapsed = now - t0
                        rate = done / elapsed if elapsed > 0 else 0.0
                        remain = (total_blocks - done) / rate if rate > 0 else float("inf")
                        log(f"[clip] {done}/{total_blocks} blocks | {rate:.2f} blk/s | ETA {human_eta(remain)}")
                        last_log = now


# -------------------------
# Streaming reprojection (block-based)
# -------------------------

def stream_reproject(
    src_path: Path,
    dst_path: Path,
    dst_crs: str,
    *,
    cache_mb: int,
    progress_every_s: float = 5.0,
) -> None:
    """
    Stream reproject src raster to dst_crs by iterating over destination blocks.
    """
    env = {
        "GDAL_CACHEMAX": int(cache_mb),
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "RASTERIO_NUM_THREADS": "ALL_CPUS",
        "GDAL_DISABLE_READDIR_ON_OPEN": "TRUE",
    }

    with rasterio.Env(**env):
        with rasterio.open(src_path) as src:
            if src.crs is None:
                raise SystemExit("Source raster has no CRS. Cannot reproject safely.")

            src_bounds = array_bounds(src.height, src.width, src.transform)
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,
                dst_crs,
                src.width,
                src.height,
                *src_bounds,
            )

            nodata = src.nodata
            if nodata is None:
                nodata = 0

            dst_meta = write_meta_like(
                src.meta,
                height=dst_height,
                width=dst_width,
                transform=dst_transform,
                crs=dst_crs,
                nodata=nodata,
            )

            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Create destination
            with rasterio.open(dst_path, "w", **dst_meta) as dst:
                # Count dst blocks
                total_blocks = 0
                for _, bw in dst.block_windows(1):
                    total_blocks += 1

                t0 = time.time()
                last_log = t0
                done = 0

                for _, bw in dst.block_windows(1):
                    # Reproject only this destination window
                    for b in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, b),
                            destination=rasterio.band(dst, b),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            dst_window=bw,
                            resampling=Resampling.nearest,  # embeddings must not interpolate
                            src_nodata=nodata,
                            dst_nodata=nodata,
                        )

                    done += 1
                    now = time.time()
                    if (now - last_log) >= progress_every_s or done == total_blocks:
                        elapsed = now - t0
                        rate = done / elapsed if elapsed > 0 else 0.0
                        remain = (total_blocks - done) / rate if rate > 0 else float("inf")
                        log(f"[reproj] {done}/{total_blocks} blocks | {rate:.2f} blk/s | ETA {human_eta(remain)}")
                        last_log = now


# -------------------------
# Orchestration
# -------------------------

def clip_alphaearth(
    input_tif: Path,
    output_tif: Path,
    vector_path: Path,
    dst_crs: str,
    cache_mb_requested: int,
    progress_every_s: float,
) -> None:
    # Auto-scale cache based on free RAM
    cache_mb = auto_scale_cache_mb(cache_mb_requested)
    avail = try_available_mb()
    if avail is None:
        log(f"Cache MB : requested {cache_mb_requested} (psutil not available; using {cache_mb})")
    else:
        log(f"Cache MB : requested {cache_mb_requested}, available {avail}, using {cache_mb}")

    if not input_tif.exists():
        raise SystemExit(f"Input raster not found: {input_tif}")
    if not vector_path.exists():
        raise SystemExit(f"Vector AOI not found: {vector_path}")

    # Step 1: stream clip in source CRS
    tmp_clip = output_tif.with_suffix("").as_posix() + "_tmp_clip_src.tif"
    tmp_clip_path = Path(tmp_clip)

    log("Step 1/2: Streaming clip (source CRS)")
    stream_clip_same_crs(
        src_path=input_tif,
        vector_path=vector_path,
        out_path=tmp_clip_path,
        cache_mb=cache_mb,
        progress_every_s=progress_every_s,
    )

    # Step 2: optional reprojection
    with rasterio.open(tmp_clip_path) as clipped_src:
        src_crs = clipped_src.crs.to_string() if clipped_src.crs else None

    if src_crs is None:
        raise SystemExit("Temporary clipped raster has no CRS. Something went wrong.")

    if dst_crs != src_crs:
        log("Step 2/2: Streaming reprojection to requested CRS")
        stream_reproject(
            src_path=tmp_clip_path,
            dst_path=output_tif,
            dst_crs=dst_crs,
            cache_mb=cache_mb,
            progress_every_s=progress_every_s,
        )
        # remove temp
        try:
            tmp_clip_path.unlink()
        except Exception:
            pass
    else:
        # No reprojection; move temp to final output (atomic-ish)
        output_tif.parent.mkdir(parents=True, exist_ok=True)
        if output_tif.exists():
            output_tif.unlink()
        tmp_clip_path.rename(output_tif)

    log(f"Saved clipped AlphaEarth embeddings → {output_tif}")


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Window-streamed clip of AlphaEarth embeddings")
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input AlphaEarth mosaic (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output clipped GeoTIFF (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--vector",
        type=Path,
        default=DEFAULT_VECTOR,
        help=f"Coastal AOI vector (default: {DEFAULT_VECTOR})",
    )
    p.add_argument(
        "--crs",
        default="EPSG:4326",
        choices=["EPSG:4326", "EPSG:32646"],
        help="Output CRS (default: EPSG:4326)",
    )
    p.add_argument(
        "--cache-mb",
        type=int,
        default=DEFAULT_CACHE_MB,
        help=(
            "Requested GDAL cache size in MB (default: 4096). "
            "Script auto-scales down if free RAM is lower."
        ),
    )
    p.add_argument(
        "--progress-every-s",
        type=float,
        default=5.0,
        help="Progress/ETA logging interval in seconds (default: 5).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log("Starting AlphaEarth mosaic clipping")
    log(f"Input    : {args.input}")
    log(f"Output   : {args.output}")
    log(f"CRS      : {args.crs}")

    clip_alphaearth(
        input_tif=args.input,
        output_tif=args.output,
        vector_path=args.vector,
        dst_crs=args.crs,
        cache_mb_requested=max(256, int(args.cache_mb)),
        progress_every_s=max(1.0, float(args.progress_every_s)),
    )


if __name__ == "__main__":
    main()
