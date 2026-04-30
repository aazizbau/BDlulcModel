#!/usr/bin/env python3
"""
Create a publication-style inferred coastal Bangladesh LULC map with zoom inset.

Inputs
------
- outputs/inference/<year>/lulc_class_<year>.tif
- assets/maps/bd_coastal_zones.gpkg
- assets/maps/sundarbans.gpkg
- assets/maps/NorthArrow.svg
- assets/color_palette_coastal_lulc.json

Output
------
- outputs/figures/bd_coastal_infer_lulc_<year>_zoom.png

Example
-------
python scripts/poster/make_infer_lulc_map_with_zoom.py \
    --year 2017 \
    --seed 42 \
    --zoom-window-km 50 \
    --zoom-inset-x-frac 0.55 \
    --zoom-inset-y-frac 0.50

python scripts/poster/make_infer_lulc_map_with_zoom.py \
    --year 2024 \
    --seed 7 \
    --zoom-window-km 40 \
    --zoom-inset-x-frac 0.60 \
    --zoom-inset-y-frac 0.48
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PIL import Image
from pyproj import CRS, Transformer
from rasterio.windows import Window
from rasterio.windows import from_bounds as win_from_bounds

try:
    import cairosvg

    HAVE_SVG = True
except Exception:
    HAVE_SVG = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZONE_MAP = Path("assets/maps/bd_coastal_zones.gpkg")
DEFAULT_SUNDARBANS_MAP = Path("assets/maps/sundarbans.gpkg")
DEFAULT_NORTH_ARROW = Path("assets/maps/NorthArrow.svg")
DEFAULT_PALETTE = Path("assets/color_palette_coastal_lulc.json")
DEFAULT_INPUT_ROOT = Path("outputs/inference")
DEFAULT_OUTPUT_ROOT = Path("outputs/figures")

FIGSIZE = (11, 9)
FIG_DPI = 300
MAX_DISPLAY_SIZE = 2800
DISPLAY_CHUNK_SIZE = 512
LONGITUDE_LABEL_PAD = 0
TIGHT_LAYOUT_BOTTOM = -0.04
CLASS_NODATA = 0
MAP_TITLE_TEMPLATE = "Bangladesh Coastal LULC {year}"
BAY_LABEL_X_FRAC = 0.25
BAY_LABEL_Y_FRAC = 0.25
SCALEBAR_X_FRAC = 0.02
SCALEBAR_Y_FRAC = 0.03
LEGEND_X_FRAC = 0.42
LEGEND_Y_FRAC = 0.085
LEGEND_FONTSIZE = 10
LEGEND_HANDLE_LENGTH = 1.9
LEGEND_HANDLE_HEIGHT = 1.3
LEGEND_LABEL_SPACING = 0.55
LEGEND_BORDER_PAD = 0.75
LEGEND_BOX_ALPHA = 0.97

ZOOM_WINDOW_KM = 50.0
ZOOM_INSET_X_FRAC = 0.55
ZOOM_INSET_Y_FRAC = 0.50
ZOOM_INSET_W_FRAC = 0.35
ZOOM_INSET_H_FRAC = 0.35
ZOOM_SCALEBAR_X_FRAC = 0.10
ZOOM_SCALEBAR_Y_FRAC = 0.08
ZOOM_CONNECTOR_COLOR = "#D62828"
ZOOM_BOX_LINEWIDTH = 1.5
ZOOM_MAX_RASTER_SIZE = 800

LULC_NAMES = {
    1: "Urban / Institutional Built-up",
    2: "Rural Settlement (Homestead Vegetation)",
    3: "Transport & Coastal Embankments",
    4: "Cropland (All Crop Intensities)",
    5: "Tree-based Agroforestry & Orchard",
    6: "Aquaculture & Inland Ponds",
    7: "Canals & Drainage Network",
    8: "Rivers & Estuarine Channels",
    9: "Mangrove Forest",
    10: "Bare / Exposed Coastal Land",
}

LULC_COLORS = {
    1: "#E66A00",
    2: "#8FBF7A",
    3: "#9C7A5B",
    4: "#FFC636",
    5: "#4F7F3D",
    6: "#00ADA9",
    7: "#7AD9D6",
    8: "#007C91",
    9: "#2F5D50",
    10: "#F3E7CF",
}

ZONE_LABELS = {
    "western": "Western Zone",
    "central": "Central Zone",
    "eastern": "Eastern Zone",
}

ZONE_LABEL_OFFSETS = {
    "western": (0.0, 26000.0),
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_input_path(year: int) -> Path:
    return DEFAULT_INPUT_ROOT / str(year) / f"lulc_class_{year}.tif"


def default_output_path(year: int) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"bd_coastal_infer_lulc_{year}_zoom.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make inferred LULC map with zoom inset for a target year.")
    p.add_argument("--year", type=int, required=True, help="Target year, e.g. 2017 or 2024.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for selecting the zoom box center.")
    p.add_argument("--zoom-window-km", type=float, default=ZOOM_WINDOW_KM, help="Side length of zoom window in km.")
    p.add_argument("--zoom-inset-x-frac", type=float, default=ZOOM_INSET_X_FRAC, help="Zoom inset x position in axes fraction.")
    p.add_argument("--zoom-inset-y-frac", type=float, default=ZOOM_INSET_Y_FRAC, help="Zoom inset y position in axes fraction.")
    p.add_argument("--input", type=Path, default=None, help="Optional inferred LULC GeoTIFF override.")
    p.add_argument("--zone-map", type=Path, default=DEFAULT_ZONE_MAP, help="Coastal zones vector layer.")
    p.add_argument("--sundarbans-map", type=Path, default=DEFAULT_SUNDARBANS_MAP, help="Sundarbans vector layer.")
    p.add_argument("--north-arrow", type=Path, default=DEFAULT_NORTH_ARROW, help="North arrow SVG path.")
    p.add_argument("--palette", type=Path, default=DEFAULT_PALETTE, help="Palette JSON path.")
    p.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    return p.parse_args()


def load_palette(path: Path) -> dict:
    return json.loads(path.read_text())


def decimal_to_dm(value: float, kind: str) -> str:
    hemi = "E" if kind == "lon" and value >= 0 else "W" if kind == "lon" else "N" if value >= 0 else "S"
    deg_abs = abs(value)
    d = int(deg_abs)
    m = int(round((deg_abs - d) * 60))
    if m == 60:
        d += 1
        m = 0
    return f"{d}°{m:02d}′{hemi}"


def decimal_to_dms(value: float, kind: str) -> str:
    hemi = "E" if kind == "lon" and value >= 0 else "W" if kind == "lon" else "N" if value >= 0 else "S"
    deg_abs = abs(value)
    d = int(deg_abs)
    minutes_total = (deg_abs - d) * 60.0
    m = int(minutes_total)
    s = int(round((minutes_total - m) * 60.0))
    if s == 60:
        m += 1
        s = 0
    if m == 60:
        d += 1
        m = 0
    return f"{d}°{m:02d}′{s:02d}″{hemi}"


def km_to_lon_degrees(km: float, lat_deg: float) -> float:
    cos_lat = np.cos(np.deg2rad(lat_deg))
    if abs(cos_lat) < 1e-8:
        cos_lat = 1e-8
    return km / (111.320 * cos_lat)


def km_to_lat_degrees(km: float) -> float:
    return km / 110.574


def add_graticule(ax, color: str, src_crs) -> None:
    src_crs = CRS.from_user_input(src_crs)
    dst_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    def fmt_x(x, _pos=None):
        y_mid = 0.5 * sum(ax.get_ylim())
        lon, _ = transformer.transform(x, y_mid)
        return decimal_to_dm(lon, "lon")

    def fmt_y(y, _pos=None):
        x_mid = 0.5 * sum(ax.get_xlim())
        _, lat = transformer.transform(x_mid, y)
        return decimal_to_dm(lat, "lat")

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_x))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, color=color, linestyle="--", linewidth=0.6, alpha=0.24, zorder=0)
    ax.tick_params(axis="both", labelsize=10, direction="out", top=True, right=True, labeltop=False, labelright=False)
    for label in ax.get_yticklabels():
        label.set_rotation(90)
        label.set_va("center")
        label.set_ha("center")


def apply_zoom_graticule_dms(ax, src_crs) -> None:
    src_crs = CRS.from_user_input(src_crs)
    dst_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    def fmt_x(x, _pos=None):
        y_mid = 0.5 * sum(ax.get_ylim())
        lon, _ = transformer.transform(x, y_mid)
        return decimal_to_dms(lon, "lon")

    def fmt_y(y, _pos=None):
        x_mid = 0.5 * sum(ax.get_xlim())
        _, lat = transformer.transform(x_mid, y)
        return decimal_to_dms(lat, "lat")

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_x))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))


def add_scalebar_2step(ax, length_km=150, location=(0.38, 0.06), fontsize=10, is_geographic=False):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + location[0] * (xlim[1] - xlim[0])
    y0 = ylim[0] + location[1] * (ylim[1] - ylim[0])
    if is_geographic:
        mid_lat = 0.5 * (ylim[0] + ylim[1])
        total_map_units = km_to_lon_degrees(length_km, mid_lat)
    else:
        total_map_units = length_km * 1000.0
    step_map_units = total_map_units / 2.0
    bar_h = 0.014 * (ylim[1] - ylim[0])

    for i, face in enumerate(["black", "white"]):
        rect = Rectangle(
            (x0 + i * step_map_units, y0),
            step_map_units,
            bar_h,
            facecolor=face,
            edgecolor="black",
            linewidth=0.8,
            zorder=10,
        )
        ax.add_patch(rect)

    label_y = y0 + bar_h + 0.012 * (ylim[1] - ylim[0])
    ax.text(x0, label_y, "0", ha="center", va="bottom", fontsize=fontsize, zorder=11)
    ax.text(x0 + step_map_units, label_y, str(int(length_km // 2)), ha="center", va="bottom", fontsize=fontsize, zorder=11)
    ax.text(x0 + 2 * step_map_units, label_y, f"{int(length_km)} km", ha="center", va="bottom", fontsize=fontsize, zorder=11)


def choose_zoom_scalebar_length_km(zoom_window_km: float) -> float:
    nice = np.array([1, 2, 5, 10, 15, 20, 25, 50], dtype=float)
    valid = nice[nice <= zoom_window_km * 0.45]
    if len(valid) == 0:
        return max(0.5, zoom_window_km * 0.2)
    return float(valid[-1])


def add_scalebar_zoom(ax, length_km: float, location=(0.10, 0.08), fontsize=8, is_geographic=False):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + location[0] * (xlim[1] - xlim[0])
    y0 = ylim[0] + location[1] * (ylim[1] - ylim[0])

    if is_geographic:
        mid_lat = 0.5 * (ylim[0] + ylim[1])
        bar_len = km_to_lon_degrees(length_km, mid_lat)
    else:
        bar_len = length_km * 1000.0

    tick_h = 0.020 * (ylim[1] - ylim[0])
    x1 = x0 + bar_len

    ax.plot([x0, x1], [y0, y0], color="black", linewidth=1.6, zorder=20)
    ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color="black", linewidth=1.2, zorder=20)
    ax.plot([x1, x1], [y0 - tick_h, y0 + tick_h], color="black", linewidth=1.2, zorder=20)

    label = f"{int(length_km)} km" if float(length_km).is_integer() else f"{length_km:.1f} km"
    ax.text(
        0.5 * (x0 + x1),
        y0 + 0.035 * (ylim[1] - ylim[0]),
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=1.0),
        zorder=21,
    )


def load_svg_as_image(svg_path: Path, target_height_px: int = 220):
    if not svg_path.exists() or not HAVE_SVG:
        return None
    png_bytes = cairosvg.svg2png(url=str(svg_path), output_height=target_height_px)
    return Image.open(io.BytesIO(png_bytes))


def add_north_arrow(ax, svg_path: Path, xy=(0.92, 0.90), zoom=0.23):
    img = load_svg_as_image(svg_path, target_height_px=220)
    if img is None:
        ax.annotate("N", xy=xy, xycoords="axes fraction", ha="center", va="center", fontsize=16, fontweight="bold", zorder=20)
        ax.annotate("", xy=(xy[0], xy[1] - 0.03), xytext=(xy[0], xy[1] - 0.14), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black"), zorder=20)
        return
    imagebox = OffsetImage(np.asarray(img), zoom=zoom)
    ab = AnnotationBbox(imagebox, xy, xycoords="axes fraction", frameon=False, box_alignment=(0.5, 0.5), zorder=20)
    ax.add_artist(ab)


def set_geographic_aspect(ax, bounds) -> None:
    mean_lat = 0.5 * (bounds.bottom + bounds.top)
    cosv = np.cos(np.deg2rad(mean_lat))
    ax.set_aspect("equal" if abs(cosv) < 1e-8 else 1.0 / cosv)


def read_downsampled_class_raster_windowed(
    ds: rasterio.io.DatasetReader,
    max_size: int,
    chunk_size: int,
) -> np.ndarray:
    src_h = ds.height
    src_w = ds.width

    scale = max(src_h / max_size, src_w / max_size, 1.0)
    dst_h = max(1, int(round(src_h / scale)))
    dst_w = max(1, int(round(src_w / scale)))

    out = np.empty((dst_h, dst_w), dtype=np.uint8)

    for row0 in range(0, dst_h, chunk_size):
        row1 = min(row0 + chunk_size, dst_h)
        for col0 in range(0, dst_w, chunk_size):
            col1 = min(col0 + chunk_size, dst_w)

            src_row0 = int(round(row0 * src_h / dst_h))
            src_row1 = int(round(row1 * src_h / dst_h))
            src_col0 = int(round(col0 * src_w / dst_w))
            src_col1 = int(round(col1 * src_w / dst_w))

            src_row1 = max(src_row0 + 1, min(src_row1, src_h))
            src_col1 = max(src_col0 + 1, min(src_col1, src_w))

            data = ds.read(
                1,
                window=Window(src_col0, src_row0, src_col1 - src_col0, src_row1 - src_row0),
                out_shape=(row1 - row0, col1 - col0),
                resampling=rasterio.enums.Resampling.nearest,
            )
            out[row0:row1, col0:col1] = data.astype(np.uint8, copy=False)

    return out


def read_zoom_rgb(
    ds: rasterio.io.DatasetReader,
    zoom_xmin: float,
    zoom_ymin: float,
    zoom_xmax: float,
    zoom_ymax: float,
    nodata_rgb: tuple[float, float, float],
    max_size: int = ZOOM_MAX_RASTER_SIZE,
) -> np.ndarray | None:
    win = win_from_bounds(zoom_xmin, zoom_ymin, zoom_xmax, zoom_ymax, ds.transform)
    win = win.round_offsets().round_lengths()

    col_off = max(0, int(win.col_off))
    row_off = max(0, int(win.row_off))
    col_end = min(ds.width, col_off + max(1, int(win.width)))
    row_end = min(ds.height, row_off + max(1, int(win.height)))

    if col_end <= col_off or row_end <= row_off:
        return None

    win_clamped = Window(col_off, row_off, col_end - col_off, row_end - row_off)
    src_h = int(win_clamped.height)
    src_w = int(win_clamped.width)
    scale = max(src_h / max_size, src_w / max_size, 1.0)
    dst_h = max(1, int(round(src_h / scale)))
    dst_w = max(1, int(round(src_w / scale)))

    data = ds.read(
        1,
        window=win_clamped,
        out_shape=(dst_h, dst_w),
        resampling=rasterio.enums.Resampling.nearest,
    )
    return class_raster_to_rgb(data, nodata_rgb)


def class_raster_to_rgb(arr: np.ndarray, nodata_rgb: tuple[float, float, float]) -> np.ndarray:
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
    rgb[:] = nodata_rgb
    for class_id, color in LULC_COLORS.items():
        rgb[arr == class_id] = mcolors.to_rgb(color)
    return rgb


def build_zoom_bounds(
    bounds,
    is_geographic: bool,
    seed: int,
    zoom_window_km: float,
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bounds.left, bounds.bottom, bounds.right, bounds.top
    half_km = zoom_window_km / 2.0

    if is_geographic:
        mid_lat = 0.5 * (ymin + ymax)
        half_w = km_to_lon_degrees(half_km, mid_lat)
        half_h = km_to_lat_degrees(half_km)
    else:
        half_w = half_km * 1000.0
        half_h = half_km * 1000.0

    min_cx = xmin + half_w
    max_cx = xmax - half_w
    min_cy = ymin + half_h
    max_cy = ymax - half_h

    rng_x = np.random.default_rng(seed)
    rng_y = np.random.default_rng(seed + 1)
    cx = float(rng_x.uniform(min_cx, max_cx)) if max_cx > min_cx else 0.5 * (xmin + xmax)
    cy = float(rng_y.uniform(min_cy, max_cy)) if max_cy > min_cy else 0.5 * (ymin + ymax)

    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=LULC_COLORS[class_id], edgecolor="#314245", label=LULC_NAMES[class_id])
        for class_id in range(1, 11)
    ]


def main() -> None:
    args = parse_args()
    input_raster = resolve_path(args.input or default_input_path(args.year))
    zone_map = resolve_path(args.zone_map)
    sundarbans_map = resolve_path(args.sundarbans_map)
    north_arrow = resolve_path(args.north_arrow)
    palette_path = resolve_path(args.palette)
    output = resolve_path(args.output or default_output_path(args.year))
    zoom_window_km = args.zoom_window_km
    zoom_inset_x_frac = args.zoom_inset_x_frac
    zoom_inset_y_frac = args.zoom_inset_y_frac

    palette = load_palette(palette_path)
    colors = palette["colors"]

    fig_bg = colors["sand"]
    sea_color = colors["mist_gray"]
    grid_color = colors["deep_slate"]
    zone_edge = "#2b2e07"
    main_text_color = colors["deep_slate"]
    zone_text_color = colors["coral"]
    sundarbans_text_color = colors["deep_slate"]
    bay_text_color = colors["teal_blue"]
    legend_face = "#FFF9EF"
    sea_color_rgb = mcolors.to_rgb(sea_color)

    with rasterio.open(input_raster) as ds:
        if ds.crs is None:
            raise ValueError("Input LULC raster has no CRS.")
        raster_crs = ds.crs
        bounds = ds.bounds
        is_geographic = CRS.from_user_input(raster_crs).is_geographic

        classes = read_downsampled_class_raster_windowed(ds, max_size=MAX_DISPLAY_SIZE, chunk_size=DISPLAY_CHUNK_SIZE)

        zoom_xmin, zoom_ymin, zoom_xmax, zoom_ymax = build_zoom_bounds(bounds, is_geographic, args.seed, zoom_window_km)
        zoom_rgb = read_zoom_rgb(ds, zoom_xmin, zoom_ymin, zoom_xmax, zoom_ymax, nodata_rgb=sea_color_rgb)

    rgb = class_raster_to_rgb(classes, nodata_rgb=sea_color_rgb)

    zones = gpd.read_file(zone_map)
    if zones.empty:
        raise ValueError("Zone map is empty.")
    if zones.crs is None:
        raise ValueError("Zone map has no CRS.")
    zones = zones.to_crs(raster_crs)

    sundarbans = gpd.read_file(sundarbans_map)
    if sundarbans.empty:
        raise ValueError("Sundarbans vector is empty.")
    if sundarbans.crs is None:
        raise ValueError("Sundarbans vector has no CRS.")
    sundarbans = sundarbans.to_crs(raster_crs)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI, facecolor=fig_bg)
    ax.set_facecolor(sea_color)

    ax.imshow(
        rgb,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        interpolation="nearest",
        zorder=1,
    )

    zones.boundary.plot(ax=ax, color=zone_edge, linewidth=1.4, zorder=4)
    sundarbans.boundary.plot(ax=ax, color=zone_edge, linewidth=1.4, zorder=5)

    for _, row in zones.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        zone_key = str(row["zone"]).strip().lower()
        label = ZONE_LABELS.get(zone_key, zone_key.title())
        pt = geom.representative_point()
        dx, dy = ZONE_LABEL_OFFSETS.get(zone_key, (0.0, 0.0))
        txt = ax.text(
            pt.x + dx,
            pt.y + dy,
            label,
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            color=zone_text_color,
            zorder=6,
        )
        txt.set_path_effects([pe.Stroke(linewidth=3, foreground=fig_bg), pe.Normal()])

    for _, row in sundarbans.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        label = str(row["zone"]).strip()
        pt = geom.representative_point()
        txt = ax.text(
            pt.x,
            pt.y,
            label,
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            color=sundarbans_text_color,
            zorder=6,
        )
        txt.set_path_effects([pe.Stroke(linewidth=3, foreground=fig_bg), pe.Normal()])

    bay_x = bounds.left + BAY_LABEL_X_FRAC * (bounds.right - bounds.left)
    bay_y = bounds.bottom + BAY_LABEL_Y_FRAC * (bounds.top - bounds.bottom)
    bay_txt = ax.text(
        bay_x,
        bay_y,
        "Bay of Bengal",
        fontsize=14,
        ha="center",
        va="center",
        color=bay_text_color,
        zorder=5,
    )
    bay_txt.set_path_effects([pe.Stroke(linewidth=4, foreground=fig_bg), pe.Normal()])

    set_geographic_aspect(ax, bounds)
    add_graticule(ax, color=grid_color, src_crs=raster_crs)

    title = MAP_TITLE_TEMPLATE.format(year=args.year)
    ax.set_title(title, fontsize=15, pad=12, color=main_text_color, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=12, color=main_text_color, labelpad=LONGITUDE_LABEL_PAD)
    ax.set_ylabel("Latitude", fontsize=12, color=main_text_color)
    ax.tick_params(axis="both", colors=main_text_color)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    add_north_arrow(ax, north_arrow, xy=(0.92, 0.90), zoom=0.23)
    add_scalebar_2step(ax, length_km=150, location=(SCALEBAR_X_FRAC, SCALEBAR_Y_FRAC), fontsize=10, is_geographic=is_geographic)

    legend = ax.legend(
        handles=legend_handles(),
        loc="lower left",
        bbox_to_anchor=(LEGEND_X_FRAC, LEGEND_Y_FRAC),
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        framealpha=LEGEND_BOX_ALPHA,
        facecolor=legend_face,
        edgecolor=main_text_color,
        ncol=1,
        handlelength=LEGEND_HANDLE_LENGTH,
        handleheight=LEGEND_HANDLE_HEIGHT,
        labelspacing=LEGEND_LABEL_SPACING,
        borderpad=LEGEND_BORDER_PAD,
    )
    legend.set_zorder(12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor(main_text_color)

    # Draw zoom rectangle on main map
    zoom_rect = Rectangle(
        (zoom_xmin, zoom_ymin),
        zoom_xmax - zoom_xmin,
        zoom_ymax - zoom_ymin,
        facecolor="none",
        edgecolor=ZOOM_CONNECTOR_COLOR,
        linewidth=ZOOM_BOX_LINEWIDTH,
        zorder=30,
    )
    ax.add_patch(zoom_rect)

    # Fix layout before reading axis position for inset placement
    plt.tight_layout(rect=(0, TIGHT_LAYOUT_BOTTOM, 1, 1))
    ax_pos = ax.get_position()

    zoom_left = ax_pos.x0 + zoom_inset_x_frac * ax_pos.width
    zoom_bottom = ax_pos.y0 + zoom_inset_y_frac * ax_pos.height
    zoom_width = ZOOM_INSET_W_FRAC * ax_pos.width
    zoom_height = ZOOM_INSET_H_FRAC * ax_pos.height

    ax_zoom = fig.add_axes([zoom_left, zoom_bottom, zoom_width, zoom_height], facecolor=sea_color)

    if zoom_rgb is not None:
        ax_zoom.imshow(
            zoom_rgb,
            extent=(zoom_xmin, zoom_xmax, zoom_ymin, zoom_ymax),
            origin="upper",
            interpolation="nearest",
            zorder=1,
        )

    ax_zoom.set_xlim(zoom_xmin, zoom_xmax)
    ax_zoom.set_ylim(zoom_ymin, zoom_ymax)

    add_graticule(ax_zoom, color=grid_color, src_crs=raster_crs)
    apply_zoom_graticule_dms(ax_zoom, raster_crs)
    ax_zoom.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_zoom.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_zoom.set_xlabel("Longitude", fontsize=8, color=main_text_color, labelpad=1.5)
    ax_zoom.set_ylabel("Latitude", fontsize=8, color=main_text_color, labelpad=1.5)
    ax_zoom.tick_params(axis="both", labelsize=7, colors=main_text_color)
    plt.setp(ax_zoom.get_xticklabels(), rotation=25, ha="right")
    plt.setp(ax_zoom.get_yticklabels(), rotation=0, ha="right", va="center")

    zoom_scalebar_km = choose_zoom_scalebar_length_km(zoom_window_km)
    add_scalebar_zoom(
        ax_zoom,
        length_km=zoom_scalebar_km,
        location=(ZOOM_SCALEBAR_X_FRAC, ZOOM_SCALEBAR_Y_FRAC),
        fontsize=7,
        is_geographic=is_geographic,
    )

    for spine in ax_zoom.spines.values():
        spine.set_linewidth(1.3)
        spine.set_edgecolor(ZOOM_CONNECTOR_COLOR)

    # Connector lines from right edge of zoom rectangle to zoom inset
    con1 = ConnectionPatch(
        xyA=(zoom_xmax, zoom_ymax),
        coordsA=ax.transData,
        xyB=(zoom_xmax, zoom_ymax),
        coordsB=ax_zoom.transData,
        color=ZOOM_CONNECTOR_COLOR,
        linewidth=1.1,
        alpha=0.95,
    )
    con2 = ConnectionPatch(
        xyA=(zoom_xmax, zoom_ymin),
        coordsA=ax.transData,
        xyB=(zoom_xmax, zoom_ymin),
        coordsB=ax_zoom.transData,
        color=ZOOM_CONNECTOR_COLOR,
        linewidth=1.1,
        alpha=0.95,
    )
    fig.add_artist(con1)
    fig.add_artist(con2)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
