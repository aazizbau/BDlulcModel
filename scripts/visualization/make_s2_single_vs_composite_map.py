#!/usr/bin/env python3
"""
Create a two-panel Sentinel-2 figure:

    Image (A): single-date cloud-masked Sentinel-2 RGB image
    Image (B): Oct-Dec median composite Sentinel-2 RGB image

Expected composite band paths:
    data/raw/sentinel_gemini/BD_COASTAL_BBOX/S2_2023_octdec_B4_mosaic_10m_lzw.tif
    data/raw/sentinel_gemini/BD_COASTAL_BBOX/S2_2023_octdec_B3_mosaic_10m_lzw.tif
    data/raw/sentinel_gemini/BD_COASTAL_BBOX/S2_2023_octdec_B2_mosaic_10m_lzw.tif

Example:
    python scripts/visualization/make_s2_single_vs_composite_map.py \
        --single-rgb data/raw/sentinel_single_scene/s2_2023_single_scene_rgb.tif \
        --bbox 89.88 23.78 89.96 23.84 \
        --out outputs/figures/compare_single_vs_composite_s2_2023.png

BBOX order:
    min_lon min_lat max_lon max_lat
"""

from __future__ import annotations

import argparse
import io
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from PIL import Image
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

try:
    import cairosvg

    HAVE_SVG = True
except Exception:
    HAVE_SVG = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NORTH_ARROW = Path("assets/maps/NorthArrow.svg")
NODATA_DEFAULT = 65535


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def jst_now() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S UTC+09:00")


def log(msg: str) -> None:
    print(f"[{jst_now()}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make two-panel Sentinel-2 single-scene vs Oct-Dec composite RGB map."
    )

    p.add_argument(
        "--single-rgb",
        type=Path,
        required=True,
        help="Single-date cloud-masked RGB GeoTIFF. Expected band order: B4, B3, B2.",
    )

    p.add_argument(
        "--comp-dir",
        type=Path,
        default=Path("data/raw/sentinel_gemini/BD_COASTAL_BBOX"),
        help="Directory containing Sentinel-2 Oct-Dec composite bands.",
    )

    p.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Composite year. Default: 2023.",
    )

    p.add_argument(
        "--comp-red",
        type=Path,
        default=None,
        help="Optional path to composite B4 GeoTIFF. If omitted, built from --comp-dir and --year.",
    )
    p.add_argument(
        "--comp-green",
        type=Path,
        default=None,
        help="Optional path to composite B3 GeoTIFF. If omitted, built from --comp-dir and --year.",
    )
    p.add_argument(
        "--comp-blue",
        type=Path,
        default=None,
        help="Optional path to composite B2 GeoTIFF. If omitted, built from --comp-dir and --year.",
    )

    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Map bounding box in EPSG:4326.",
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/figures/figure_3_1_single_vs_composite_s2_2023.png"),
    )

    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--nodata", type=float, default=NODATA_DEFAULT)

    p.add_argument("--title-a", type=str, default="Image (A)")
    p.add_argument("--title-b", type=str, default="Image (B)")

    p.add_argument("--north-arrow", type=Path, default=DEFAULT_NORTH_ARROW, help="North arrow SVG path.")

    return p.parse_args()


def resolve_composite_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    comp_dir = resolve_path(args.comp_dir)
    red = resolve_path(args.comp_red) if args.comp_red else comp_dir / f"S2_{args.year}_octdec_B4_mosaic_10m_lzw.tif"
    green = resolve_path(args.comp_green) if args.comp_green else comp_dir / f"S2_{args.year}_octdec_B3_mosaic_10m_lzw.tif"
    blue = resolve_path(args.comp_blue) if args.comp_blue else comp_dir / f"S2_{args.year}_octdec_B2_mosaic_10m_lzw.tif"

    for path in [red, green, blue]:
        if not path.exists():
            raise FileNotFoundError(f"Composite band not found: {path}")

    return red, green, blue


def decimal_degree_to_dm(value: float, axis: str) -> str:
    direction = "E" if axis == "lon" and value >= 0 else ""
    direction = "W" if axis == "lon" and value < 0 else direction
    direction = "N" if axis == "lat" and value >= 0 else direction
    direction = "S" if axis == "lat" and value < 0 else direction

    value_abs = abs(value)
    deg = int(value_abs)
    minutes = int(round((value_abs - deg) * 60))

    if minutes == 60:
        deg += 1
        minutes = 0

    return f"{deg}°{minutes:02d}′{direction}"


def assert_bbox_overlaps_raster(path: Path, bbox4326: list[float]) -> None:
    with rasterio.open(path) as src:
        bounds_src = transform_bounds("EPSG:4326", src.crs, *bbox4326, densify_pts=21)
        req_left, req_bottom, req_right, req_top = bounds_src
        src_left, src_bottom, src_right, src_top = src.bounds

    overlaps = (
        req_left < src_right
        and req_right > src_left
        and req_bottom < src_top
        and req_top > src_bottom
    )
    if not overlaps:
        raise ValueError(
            "Requested --bbox does not overlap the composite raster. "
            f"bbox(EPSG:4326)={bbox4326}; composite bounds({path})="
            f"left={src_left:.6f}, bottom={src_bottom:.6f}, right={src_right:.6f}, top={src_top:.6f}. "
            "Choose a bbox inside the composite coverage or regenerate the composite for this area."
        )


def read_crop_multiband_rgb(
    path: Path,
    bbox4326: list[float],
    nodata: float,
) -> tuple[np.ndarray, tuple[float, float, float, float], str]:
    """
    Read a 3-band RGB GeoTIFF.

    Expected band order:
        band 1 = Red/B4
        band 2 = Green/B3
        band 3 = Blue/B2
    """
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Single RGB image not found: {path}")

    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(
                f"{path} has {src.count} band(s), but at least 3 RGB bands are required."
            )

        bounds_src = transform_bounds("EPSG:4326", src.crs, *bbox4326, densify_pts=21)
        window = from_bounds(*bounds_src, transform=src.transform)

        arr = src.read(
            [1, 2, 3],
            window=window,
            boundless=True,
            fill_value=nodata,
        ).astype("float32")
        extent = window_extent(src, window)

        crs_text = src.crs.to_string()

    arr[arr == nodata] = np.nan
    arr[arr <= 0] = np.nan

    return arr, extent, crs_text


def read_crop_single_band(
    path: Path,
    bbox4326: list[float],
    nodata: float,
) -> tuple[np.ndarray, tuple[float, float, float, float], str]:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")
    assert_bbox_overlaps_raster(path, bbox4326)

    with rasterio.open(path) as src:
        bounds_src = transform_bounds("EPSG:4326", src.crs, *bbox4326, densify_pts=21)
        window = from_bounds(*bounds_src, transform=src.transform)

        arr = src.read(
            1,
            window=window,
            boundless=True,
            fill_value=nodata,
        ).astype("float32")
        extent = window_extent(src, window)

        crs_text = src.crs.to_string()

    arr[arr == nodata] = np.nan
    arr[arr <= 0] = np.nan

    return arr, extent, crs_text


def window_extent(src: rasterio.DatasetReader, window) -> tuple[float, float, float, float]:
    transform = src.window_transform(window)

    left = transform.c
    top = transform.f
    right = left + transform.a * window.width
    bottom = top + transform.e * window.height

    return left, right, bottom, top


def read_crop_composite_rgb(
    red_path: Path,
    green_path: Path,
    blue_path: Path,
    bbox4326: list[float],
    nodata: float,
) -> tuple[np.ndarray, tuple[float, float, float, float], str]:
    r, extent, crs_text = read_crop_single_band(red_path, bbox4326, nodata)
    g, _, _ = read_crop_single_band(green_path, bbox4326, nodata)
    b, _, _ = read_crop_single_band(blue_path, bbox4326, nodata)

    min_h = min(r.shape[0], g.shape[0], b.shape[0])
    min_w = min(r.shape[1], g.shape[1], b.shape[1])

    rgb = np.stack(
        [
            r[:min_h, :min_w],
            g[:min_h, :min_w],
            b[:min_h, :min_w],
        ],
        axis=0,
    )

    return rgb, extent, crs_text


def stretch_rgb(rgb: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """
    Percentile stretch RGB array.

    Input shape:
        3, H, W

    Output shape:
        H, W, 3
    """
    out = np.zeros_like(rgb, dtype="float32")

    for i in range(3):
        band = rgb[i]
        valid = np.isfinite(band)

        if valid.sum() == 0:
            continue

        lo, hi = np.nanpercentile(band, [p_low, p_high])

        if hi <= lo:
            hi = lo + 1

        out[i] = (band - lo) / (hi - lo)

    out = np.clip(out, 0, 1)

    # Missing/cloud-masked pixels are displayed as white.
    missing = ~np.isfinite(rgb).all(axis=0)
    out[:, missing] = 1.0

    return np.moveaxis(out, 0, -1)


def setup_axis(ax, bbox: list[float], show_left_labels: bool = True, show_right_labels: bool = True) -> None:
    min_lon, min_lat, max_lon, max_lat = bbox

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    xticks = [
        min_lon,
        min_lon + (max_lon - min_lon) * 0.5,
        max_lon,
    ]
    yticks = [
        min_lat,
        min_lat + (max_lat - min_lat) * 0.5,
        max_lat,
    ]

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: decimal_degree_to_dm(x, axis="lon"))
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, pos: decimal_degree_to_dm(y, axis="lat"))
    )

    ax.tick_params(
        axis="both",
        labelsize=8,
        direction="inout",
        length=3,
        top=True,
        right=True,
        labeltop=True,
        labelleft=show_left_labels,
        labelright=show_right_labels,
    )

    ax.grid(color="black", linewidth=0.45, alpha=0.55)

    for spine in ax.spines.values():
        spine.set_edgecolor("blue")
        spine.set_linewidth(1.2)

    ax.set_aspect("equal", adjustable="box")


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.03,
        0.95,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(
            facecolor="white",
            edgecolor="lightgray",
            boxstyle="square,pad=0.25",
        ),
        zorder=10,
    )


def add_scalebar(ax, bbox: list[float], length_km: float = 2.0) -> None:
    """
    Add approximate lon/lat scale bar for small map areas.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    center_lat = (min_lat + max_lat) / 2
    km_per_degree_lon = 111.32 * np.cos(np.deg2rad(center_lat))
    length_deg = length_km / km_per_degree_lon

    x0 = max_lon - length_deg - (max_lon - min_lon) * 0.08
    y0 = min_lat + (max_lat - min_lat) * 0.075

    bar_h = (max_lat - min_lat) * 0.018

    ax.add_patch(
        Rectangle(
            (x0, y0),
            length_deg / 2,
            bar_h,
            facecolor="black",
            edgecolor="black",
            linewidth=0.8,
            zorder=8,
        )
    )
    ax.add_patch(
        Rectangle(
            (x0 + length_deg / 2, y0),
            length_deg / 2,
            bar_h,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=8,
        )
    )

    ax.text(
        x0,
        y0 + bar_h * 2.0,
        "0",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="black",
        zorder=9,
    )
    ax.text(
        x0 + length_deg,
        y0 + bar_h * 2.0,
        f"{int(length_km)} km",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="black",
        zorder=9,
    )


def load_svg_as_image(svg_path: Path, target_height_px: int = 220):
    if not svg_path.exists() or not HAVE_SVG:
        return None
    png_bytes = cairosvg.svg2png(url=str(svg_path), output_height=target_height_px)
    return Image.open(io.BytesIO(png_bytes))


def add_north_arrow(ax, svg_path: Path, xy=(0.90, 0.88), zoom=0.20) -> None:
    img = load_svg_as_image(svg_path, target_height_px=220)
    if img is None:
        ax.annotate("N", xy=xy, xycoords="axes fraction", ha="center", va="center", fontsize=13, fontweight="bold", zorder=20)
        ax.annotate(
            "",
            xy=(xy[0], xy[1] - 0.03),
            xytext=(xy[0], xy[1] - 0.13),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black"),
            zorder=20,
        )
        return
    imagebox = OffsetImage(np.asarray(img), zoom=zoom)
    ab = AnnotationBbox(imagebox, xy, xycoords="axes fraction", frameon=False, box_alignment=(0.5, 0.5), zorder=20)
    ax.add_artist(ab)


def main() -> None:
    args = parse_args()
    args.single_rgb = resolve_path(args.single_rgb)
    args.out = resolve_path(args.out)
    north_arrow = resolve_path(args.north_arrow)
    bbox = list(args.bbox)

    comp_red, comp_green, comp_blue = resolve_composite_paths(args)

    log(f"Single RGB image: {args.single_rgb}")
    log(f"Composite red band: {comp_red}")
    log(f"Composite green band: {comp_green}")
    log(f"Composite blue band: {comp_blue}")

    log("Reading single-date RGB image...")
    single_rgb, single_extent, single_crs = read_crop_multiband_rgb(
        args.single_rgb,
        bbox,
        args.nodata,
    )

    log("Reading Oct-Dec composite RGB bands...")
    comp_rgb, comp_extent, comp_crs = read_crop_composite_rgb(
        comp_red,
        comp_green,
        comp_blue,
        bbox,
        args.nodata,
    )

    if single_crs != "EPSG:4326":
        log(f"WARNING: single image CRS is {single_crs}. This script expects EPSG:4326 for map axes.")

    if comp_crs != "EPSG:4326":
        log(f"WARNING: composite CRS is {comp_crs}. This script expects EPSG:4326 for map axes.")

    single_rgb_vis = stretch_rgb(single_rgb)
    comp_rgb_vis = stretch_rgb(comp_rgb)

    fig = plt.figure(figsize=(8.5, 3.7))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        wspace=0.10,
    )

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    ax_a.imshow(single_rgb_vis, extent=single_extent, origin="upper")
    ax_b.imshow(comp_rgb_vis, extent=comp_extent, origin="upper")

    setup_axis(ax_a, bbox, show_left_labels=True, show_right_labels=False)
    setup_axis(ax_b, bbox, show_left_labels=False, show_right_labels=True)

    add_panel_label(ax_a, args.title_a)
    add_panel_label(ax_b, args.title_b)

    add_north_arrow(ax_a, north_arrow)
    add_north_arrow(ax_b, north_arrow)

    add_scalebar(ax_a, bbox, length_km=2)
    add_scalebar(ax_b, bbox, length_km=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    log(f"Saved figure: {args.out}")
    log("DONE")


if __name__ == "__main__":
    main()
