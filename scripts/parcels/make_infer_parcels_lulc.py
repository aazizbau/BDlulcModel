#!/usr/bin/env python3
"""
Assign majority inferred LULC class to each parcel for a target upazila and year.

Inputs
------
- outputs/inference/<year>/lulc_class_<year>.tif
- assets/maps/<upazila>_parcels.gpkg
- assets/maps/NorthArrow.svg
- assets/color_palette_coastal_lulc.json

Outputs
-------
- assets/maps/<upazila>_parcels_lulc_<year>.gpkg
- outputs/figures/<upazila>_parcels_lulc_<year>.png

Example
-------
python scripts/parcels/make_infer_parcels_lulc.py \
    --year 2017 \
    --upazila-parcels bamna

python scripts/parcels/make_infer_parcels_lulc.py \
    --year 2017 \
    --upazila-parcels amtali \
    --scalebar-x-frac 0.07 \
    --scalebar-y-frac -0.05 \
    --north-arrow-x-frac 0.97 \
    --north-arrow-y-frac 0.90 \
    --legend-x-frac 1.05 \
    --legend-y-frac -0.06

python scripts/parcels/make_infer_parcels_lulc.py \
    --year 2017 \
    --upazila-parcels betagi \
    --scalebar-x-frac 0.15 \
    --scalebar-y-frac -0.05 \
    --north-arrow-x-frac 1.0 \
    --north-arrow-y-frac 0.90 \
    --legend-x-frac 1.28 \
    --legend-y-frac 0.00
		
python scripts/parcels/make_infer_parcels_lulc.py \
    --year 2017 \
    --upazila-parcels manpura \
    --scalebar-x-frac 0.15 \
    --scalebar-y-frac -0.05 \
    --north-arrow-x-frac 1.3 \
    --north-arrow-y-frac 0.90 \
    --legend-x-frac 1.68 \
    --legend-y-frac -0.05
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
from rasterio.features import geometry_mask
from rasterio.windows import Window
from rasterio.windows import from_bounds
from shapely.geometry import box

try:
    import cairosvg

    HAVE_SVG = True
except Exception:
    HAVE_SVG = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NORTH_ARROW = Path("assets/maps/NorthArrow.svg")
DEFAULT_PALETTE = Path("assets/color_palette_coastal_lulc.json")
DEFAULT_INPUT_ROOT = Path("outputs/inference")
DEFAULT_MAPS_ROOT = Path("assets/maps")
DEFAULT_FIGURES_ROOT = Path("outputs/figures")

UPAZILA_CHOICES = ("bamna", "amtali", "betagi", "manpura")

FIGSIZE = (9.5, 8.5)
FIG_DPI = 450
LONGITUDE_LABEL_PAD = 0
CLASS_NODATA = 0
SCALEBAR_LENGTH_KM = 5
SCALEBAR_X_FRAC = -0.10
SCALEBAR_Y_FRAC = -0.05
NORTH_ARROW_X_FRAC = 1.0
NORTH_ARROW_Y_FRAC = 0.90
LEGEND_X_FRAC = 1.08
LEGEND_Y_FRAC = -0.16
LEGEND_FONTSIZE = 10
LEGEND_HANDLE_LENGTH = 1.6
LEGEND_HANDLE_HEIGHT = 1.2
LEGEND_LABEL_SPACING = 0.45
LEGEND_BORDER_PAD = 0.70
LEGEND_BOX_ALPHA = 0.97
ZOOM_WINDOW_SIZE_M = 500.0
ZOOM_INSET_X_FRAC = 0.54
ZOOM_INSET_Y_FRAC = 0.52
ZOOM_INSET_W_FRAC = 0.28
ZOOM_INSET_H_FRAC = 0.28
ZOOM_SCALEBAR_X_FRAC = 0.10
ZOOM_SCALEBAR_Y_FRAC = 0.08
ZOOM_CONNECTOR_COLOR = "#D62828"
ZOOM_BOX_LINEWIDTH = 1.3

BASE_LEFT = 0.10
BASE_RIGHT = 0.96
BASE_BOTTOM = 0.10
BASE_TOP = 0.95
DECORATION_PAD_FRAC = 0.04

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


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_input_path(year: int) -> Path:
    return DEFAULT_INPUT_ROOT / str(year) / f"lulc_class_{year}.tif"


def default_parcels_path(upazila: str) -> Path:
    return DEFAULT_MAPS_ROOT / f"{upazila}_parcels.gpkg"


def default_output_gpkg(year: int, upazila: str) -> Path:
    return DEFAULT_MAPS_ROOT / f"{upazila}_parcels_lulc_{year}.gpkg"


def default_output_png(year: int, upazila: str) -> Path:
    return DEFAULT_FIGURES_ROOT / f"{upazila}_parcels_lulc_{year}.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assign parcel-level majority inferred LULC for an upazila.")
    p.add_argument("--year", type=int, required=True, help="Target year, e.g. 2017 or 2024.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for selecting the zoom box.")
    p.add_argument(
        "--upazila-parcels",
        required=True,
        choices=UPAZILA_CHOICES,
        help="Upazila parcel layer to use.",
    )
    p.add_argument("--input", type=Path, default=None, help="Optional inferred LULC GeoTIFF override.")
    p.add_argument("--parcels", type=Path, default=None, help="Optional parcel GPKG override.")
    p.add_argument("--north-arrow", type=Path, default=DEFAULT_NORTH_ARROW, help="North arrow SVG path.")
    p.add_argument("--palette", type=Path, default=DEFAULT_PALETTE, help="Palette JSON path.")
    p.add_argument("--output-gpkg", type=Path, default=None, help="Output parcel GPKG path.")
    p.add_argument("--output-png", type=Path, default=None, help="Output figure PNG path.")
    p.add_argument("--scalebar-x-frac", type=float, default=SCALEBAR_X_FRAC, help="Scale bar x position in axes fraction.")
    p.add_argument("--scalebar-y-frac", type=float, default=SCALEBAR_Y_FRAC, help="Scale bar y position in axes fraction.")
    p.add_argument("--north-arrow-x-frac", type=float, default=NORTH_ARROW_X_FRAC, help="North arrow x position in axes fraction.")
    p.add_argument("--north-arrow-y-frac", type=float, default=NORTH_ARROW_Y_FRAC, help="North arrow y position in axes fraction.")
    p.add_argument("--legend-x-frac", type=float, default=LEGEND_X_FRAC, help="Legend x position in axes fraction.")
    p.add_argument("--legend-y-frac", type=float, default=LEGEND_Y_FRAC, help="Legend y position in axes fraction.")
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


def format_scalebar_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def add_scalebar_2step(ax, length_km: float, location=(0.05, 0.04), fontsize=10, is_geographic=False):
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
    bar_h = 0.018 * (ylim[1] - ylim[0])

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

    label_y = y0 + bar_h + 0.015 * (ylim[1] - ylim[0])
    ax.text(x0, label_y, "0", ha="center", va="bottom", fontsize=fontsize, zorder=11)
    ax.text(
        x0 + step_map_units,
        label_y,
        format_scalebar_value(length_km / 2),
        ha="center",
        va="bottom",
        fontsize=fontsize,
        zorder=11,
    )
    ax.text(
        x0 + 2 * step_map_units,
        label_y,
        f"{format_scalebar_value(length_km)} km",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        zorder=11,
    )


def choose_zoom_scalebar_length_m(max_length_m: float) -> float:
    nice = np.array([50, 100, 200, 250], dtype=float)
    valid = nice[nice <= max_length_m]
    if len(valid) == 0:
        return max_length_m
    return float(valid[-1])


def add_scalebar_1step(ax, length_m: float, location=(0.10, 0.08), fontsize=8, is_geographic=False):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + location[0] * (xlim[1] - xlim[0])
    y0 = ylim[0] + location[1] * (ylim[1] - ylim[0])

    if is_geographic:
        mid_lat = 0.5 * (ylim[0] + ylim[1])
        bar_len = km_to_lon_degrees(length_m / 1000.0, mid_lat)
    else:
        bar_len = length_m

    tick_h = 0.020 * (ylim[1] - ylim[0])
    x1 = x0 + bar_len

    ax.plot([x0, x1], [y0, y0], color="black", linewidth=1.6, zorder=20)
    ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color="black", linewidth=1.2, zorder=20)
    ax.plot([x1, x1], [y0 - tick_h, y0 + tick_h], color="black", linewidth=1.2, zorder=20)

    label = f"{int(length_m)} m" if float(length_m).is_integer() else f"{length_m:.0f} m"
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
        ax.annotate(
            "",
            xy=(xy[0], xy[1] - 0.03),
            xytext=(xy[0], xy[1] - 0.14),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black"),
            zorder=20,
        )
        return
    imagebox = OffsetImage(np.asarray(img), zoom=zoom)
    ab = AnnotationBbox(imagebox, xy, xycoords="axes fraction", frameon=False, box_alignment=(0.5, 0.5), zorder=20)
    ax.add_artist(ab)


def set_geographic_aspect(ax, bounds, crs) -> None:
    crs = CRS.from_user_input(crs)
    if crs.is_geographic:
        mean_lat = 0.5 * (bounds[1] + bounds[3])
        cosv = np.cos(np.deg2rad(mean_lat))
        ax.set_aspect("equal" if abs(cosv) < 1e-8 else 1.0 / cosv)
    else:
        ax.set_aspect("equal")


def majority_class_for_geometry(src: rasterio.io.DatasetReader, geom, nodata_value: int) -> tuple[int, int, int, float]:
    window = from_bounds(*geom.bounds, transform=src.transform)
    window = window.round_offsets().round_lengths()

    if window.width <= 0 or window.height <= 0:
        return nodata_value, 0, 0, 0.0

    col_off = max(0, int(window.col_off))
    row_off = max(0, int(window.row_off))
    col_end = min(src.width, int(window.col_off + window.width))
    row_end = min(src.height, int(window.row_off + window.height))

    if col_end <= col_off or row_end <= row_off:
        return nodata_value, 0, 0, 0.0

    window = Window(col_off, row_off, col_end - col_off, row_end - row_off)

    if window.width <= 0 or window.height <= 0:
        return nodata_value, 0, 0, 0.0

    data = src.read(1, window=window)
    if data.size == 0:
        return nodata_value, 0, 0, 0.0

    transform = src.window_transform(window)
    mask = geometry_mask(
        [geom],
        out_shape=data.shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )
    values = data[mask]
    valid = values[values != nodata_value]
    if valid.size == 0:
        return nodata_value, 0, 0, 0.0

    classes, counts = np.unique(valid, return_counts=True)
    max_idx = int(np.argmax(counts))
    majority_class = int(classes[max_idx])
    majority_pixels = int(counts[max_idx])
    total_valid_pixels = int(valid.size)
    majority_fraction = float(majority_pixels / total_valid_pixels)
    return majority_class, majority_pixels, total_valid_pixels, majority_fraction


def legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=LULC_COLORS[class_id], edgecolor="#314245", label=LULC_NAMES[class_id])
        for class_id in range(1, 11)
    ]


def compute_frame_expansion(
    legend_x_frac: float,
    legend_y_frac: float,
    north_arrow_x_frac: float,
    north_arrow_y_frac: float,
    scalebar_x_frac: float,
    scalebar_y_frac: float,
) -> dict[str, float]:
    min_x = min(0.0, legend_x_frac, north_arrow_x_frac, scalebar_x_frac)
    max_x = max(1.0, legend_x_frac, north_arrow_x_frac, scalebar_x_frac)
    min_y = min(0.0, legend_y_frac, north_arrow_y_frac, scalebar_y_frac)
    max_y = max(1.0, legend_y_frac, north_arrow_y_frac, scalebar_y_frac)

    extra_left = max(0.0, -min_x) + DECORATION_PAD_FRAC
    extra_right = max(0.0, max_x - 1.0) + DECORATION_PAD_FRAC
    extra_bottom = max(0.0, -min_y) + DECORATION_PAD_FRAC
    extra_top = max(0.0, max_y - 1.0) + DECORATION_PAD_FRAC

    return {
        "extra_left": extra_left,
        "extra_right": extra_right,
        "extra_bottom": extra_bottom,
        "extra_top": extra_top,
        "width_factor": 1.0 + extra_left + extra_right,
        "height_factor": 1.0 + extra_bottom + extra_top,
    }


def remap_axes_fraction(x: float, y: float, expansion: dict[str, float]) -> tuple[float, float]:
    x_new = (x + expansion["extra_left"]) / expansion["width_factor"]
    y_new = (y + expansion["extra_bottom"]) / expansion["height_factor"]
    return x_new, y_new


def dynamic_figure_size(expansion: dict[str, float]) -> tuple[float, float]:
    return FIGSIZE[0] * expansion["width_factor"], FIGSIZE[1] * expansion["height_factor"]


def dynamic_subplot_kwargs(fig_w: float, fig_h: float) -> dict[str, float]:
    base_left_abs = BASE_LEFT * FIGSIZE[0]
    base_right_abs = (1.0 - BASE_RIGHT) * FIGSIZE[0]
    base_bottom_abs = BASE_BOTTOM * FIGSIZE[1]
    base_top_abs = (1.0 - BASE_TOP) * FIGSIZE[1]

    left = base_left_abs / fig_w
    right = 1.0 - (base_right_abs / fig_w)
    bottom = base_bottom_abs / fig_h
    top = 1.0 - (base_top_abs / fig_h)
    return dict(left=left, right=right, bottom=bottom, top=top)


def expanded_map_bounds(bounds: np.ndarray, expansion: dict[str, float]) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bounds
    width = xmax - xmin
    height = ymax - ymin

    return (
        xmin - width * expansion["extra_left"],
        ymin - height * expansion["extra_bottom"],
        xmax + width * expansion["extra_right"],
        ymax + height * expansion["extra_top"],
    )


def build_zoom_bounds(bounds: np.ndarray, is_geographic: bool, seed: int) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bounds
    half_size_m = 0.5 * ZOOM_WINDOW_SIZE_M

    if is_geographic:
        mid_lat = 0.5 * (ymin + ymax)
        half_w = km_to_lon_degrees(half_size_m / 1000.0, mid_lat)
        half_h = km_to_lat_degrees(half_size_m / 1000.0)
    else:
        half_w = half_size_m
        half_h = half_size_m

    min_cx = xmin + half_w
    max_cx = xmax - half_w
    min_cy = ymin + half_h
    max_cy = ymax - half_h

    if max_cx <= min_cx:
        cx = 0.5 * (xmin + xmax)
    else:
        rng = np.random.default_rng(seed)
        cx = float(rng.uniform(min_cx, max_cx))

    if max_cy <= min_cy:
        cy = 0.5 * (ymin + ymax)
    else:
        rng = np.random.default_rng(seed + 1)
        cy = float(rng.uniform(min_cy, max_cy))

    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def main() -> None:
    args = parse_args()
    input_raster = resolve_path(args.input or default_input_path(args.year))
    parcels_path = resolve_path(args.parcels or default_parcels_path(args.upazila_parcels))
    north_arrow = resolve_path(args.north_arrow)
    palette_path = resolve_path(args.palette)
    output_gpkg = resolve_path(args.output_gpkg or default_output_gpkg(args.year, args.upazila_parcels))
    output_png = resolve_path(args.output_png or default_output_png(args.year, args.upazila_parcels))
    scalebar_x_frac = args.scalebar_x_frac
    scalebar_y_frac = args.scalebar_y_frac
    north_arrow_x_frac = args.north_arrow_x_frac
    north_arrow_y_frac = args.north_arrow_y_frac
    legend_x_frac = args.legend_x_frac
    legend_y_frac = args.legend_y_frac

    palette = load_palette(palette_path)
    colors = palette["colors"]
    fig_bg = colors["sand"]
    grid_color = colors["deep_slate"]
    edge_color = colors["deep_slate"]
    title_color = colors["deep_slate"]
    legend_face = "#FFF9EF"

    parcels = gpd.read_file(parcels_path)
    if parcels.empty:
        raise ValueError("Parcel layer is empty.")
    if parcels.crs is None:
        raise ValueError("Parcel layer has no CRS.")
    parcels = parcels[~parcels.geometry.isna() & ~parcels.geometry.is_empty].copy()
    if parcels.empty:
        raise ValueError("Parcel layer has no valid geometries.")

    with rasterio.open(input_raster) as src:
        if src.crs is None:
            raise ValueError("Input inferred LULC raster has no CRS.")
        raster_crs = src.crs
        is_geographic = CRS.from_user_input(raster_crs).is_geographic
        parcels = parcels.to_crs(raster_crs)

        lulc_classes: list[int] = []
        lulc_names: list[str] = []
        majority_pixels: list[int] = []
        total_valid_pixels: list[int] = []
        majority_fraction: list[float] = []

        for geom in parcels.geometry:
            parcel_class, major_px, total_px, major_frac = majority_class_for_geometry(src, geom, CLASS_NODATA)
            lulc_classes.append(parcel_class)
            lulc_names.append(LULC_NAMES.get(parcel_class, "NoData"))
            majority_pixels.append(major_px)
            total_valid_pixels.append(total_px)
            majority_fraction.append(major_frac)

    parcels["lulc_class"] = lulc_classes
    parcels["lulc_name"] = lulc_names
    parcels["major_px"] = majority_pixels
    parcels["valid_px"] = total_valid_pixels
    parcels["major_frac"] = majority_fraction

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    parcels.to_file(output_gpkg, driver="GPKG")

    bounds = parcels.total_bounds
    expansion = compute_frame_expansion(
        legend_x_frac=legend_x_frac,
        legend_y_frac=legend_y_frac,
        north_arrow_x_frac=north_arrow_x_frac,
        north_arrow_y_frac=north_arrow_y_frac,
        scalebar_x_frac=scalebar_x_frac,
        scalebar_y_frac=scalebar_y_frac,
    )
    fig_w, fig_h = dynamic_figure_size(expansion)
    subplot_kwargs = dynamic_subplot_kwargs(fig_w, fig_h)

    legend_xy = remap_axes_fraction(legend_x_frac, legend_y_frac, expansion)
    north_arrow_xy = remap_axes_fraction(north_arrow_x_frac, north_arrow_y_frac, expansion)
    scalebar_xy = remap_axes_fraction(scalebar_x_frac, scalebar_y_frac, expansion)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=FIG_DPI, facecolor=fig_bg)
    ax.set_facecolor(fig_bg)

    for class_id in range(1, 11):
        subset = parcels[parcels["lulc_class"] == class_id]
        if subset.empty:
            continue
        subset.plot(
            ax=ax,
            facecolor=LULC_COLORS[class_id],
            edgecolor=edge_color,
            linewidth=0.22,
            zorder=2,
        )

    nodata_subset = parcels[parcels["lulc_class"] == CLASS_NODATA]
    if not nodata_subset.empty:
        nodata_subset.plot(
            ax=ax,
            facecolor="#D9D9D9",
            edgecolor=edge_color,
            linewidth=0.22,
            zorder=1,
        )

    parcels.boundary.plot(ax=ax, color="#FFF9EF", linewidth=0.50, zorder=3)
    parcels.boundary.plot(ax=ax, color=edge_color, linewidth=0.20, zorder=4)

    xmin, ymin, xmax, ymax = bounds
    pad_x = 0.04 * (xmax - xmin)
    pad_y = 0.04 * (ymax - ymin)
    base_bounds = np.array([xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y], dtype=float)
    xmin2, ymin2, xmax2, ymax2 = expanded_map_bounds(base_bounds, expansion)
    ax.set_xlim(xmin2, xmax2)
    ax.set_ylim(ymin2, ymax2)

    add_graticule(ax, color=grid_color, src_crs=raster_crs)
    set_geographic_aspect(ax, bounds, raster_crs)

    title = f"{args.upazila_parcels.title()} Parcel LULC {args.year}"
    title_text = ax.set_title(title, fontsize=15, pad=12, color=title_color, fontweight="bold")
    title_text.set_path_effects([pe.Stroke(linewidth=2, foreground=fig_bg), pe.Normal()])
    ax.set_xlabel("Longitude", fontsize=12, color=title_color, labelpad=LONGITUDE_LABEL_PAD)
    ax.set_ylabel("Latitude", fontsize=12, color=title_color)
    ax.tick_params(axis="both", colors=title_color)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    add_north_arrow(ax, north_arrow, xy=north_arrow_xy, zoom=0.23)
    add_scalebar_2step(
        ax,
        length_km=SCALEBAR_LENGTH_KM,
        location=scalebar_xy,
        fontsize=10,
        is_geographic=is_geographic,
    )

    legend = ax.legend(
        handles=legend_handles(),
        loc="lower right",
        bbox_to_anchor=legend_xy,
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        framealpha=LEGEND_BOX_ALPHA,
        facecolor=legend_face,
        edgecolor=title_color,
        ncol=1,
        handlelength=LEGEND_HANDLE_LENGTH,
        handleheight=LEGEND_HANDLE_HEIGHT,
        labelspacing=LEGEND_LABEL_SPACING,
        borderpad=LEGEND_BORDER_PAD,
    )
    legend.set_zorder(12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor(title_color)

    fig.subplots_adjust(**subplot_kwargs)

    zoom_xmin, zoom_ymin, zoom_xmax, zoom_ymax = build_zoom_bounds(bounds, is_geographic, args.seed)
    zoom_extent = (zoom_xmin, zoom_xmax, zoom_ymin, zoom_ymax)
    zoom_bounds = (zoom_xmin, zoom_ymin, zoom_xmax, zoom_ymax)
    zoom_geom = box(zoom_xmin, zoom_ymin, zoom_xmax, zoom_ymax)
    zoom_subset = parcels[parcels.intersects(zoom_geom)].copy()

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

    ax_pos = ax.get_position()
    zoom_left = ax_pos.x0 + ZOOM_INSET_X_FRAC * ax_pos.width
    zoom_bottom = ax_pos.y0 + ZOOM_INSET_Y_FRAC * ax_pos.height
    zoom_width = ZOOM_INSET_W_FRAC * ax_pos.width
    zoom_height = ZOOM_INSET_H_FRAC * ax_pos.height

    ax_zoom = fig.add_axes([zoom_left, zoom_bottom, zoom_width, zoom_height], facecolor=fig_bg)

    for class_id in range(1, 11):
        subset = zoom_subset[zoom_subset["lulc_class"] == class_id]
        if subset.empty:
            continue
        subset.plot(
            ax=ax_zoom,
            facecolor=LULC_COLORS[class_id],
            edgecolor=edge_color,
            linewidth=0.28,
            zorder=2,
        )

    nodata_zoom = zoom_subset[zoom_subset["lulc_class"] == CLASS_NODATA]
    if not nodata_zoom.empty:
        nodata_zoom.plot(
            ax=ax_zoom,
            facecolor="#D9D9D9",
            edgecolor=edge_color,
            linewidth=0.28,
            zorder=1,
        )

    if not zoom_subset.empty:
        zoom_subset.boundary.plot(ax=ax_zoom, color="#FFF9EF", linewidth=0.45, zorder=3)
        zoom_subset.boundary.plot(ax=ax_zoom, color=edge_color, linewidth=0.18, zorder=4)

    ax_zoom.set_xlim(zoom_xmin, zoom_xmax)
    ax_zoom.set_ylim(zoom_ymin, zoom_ymax)
    add_graticule(ax_zoom, color=grid_color, src_crs=raster_crs)
    set_geographic_aspect(ax_zoom, zoom_bounds, raster_crs)
    ax_zoom.set_xlabel("Longitude", fontsize=8, color=title_color, labelpad=1.5)
    ax_zoom.set_ylabel("Latitude", fontsize=8, color=title_color, labelpad=1.5)
    ax_zoom.tick_params(axis="both", labelsize=8, colors=title_color)
    plt.setp(ax_zoom.get_xticklabels(), rotation=25, ha="right")

    zoom_scalebar_length_m = choose_zoom_scalebar_length_m(ZOOM_WINDOW_SIZE_M * 0.45)
    add_scalebar_1step(
        ax_zoom,
        length_m=zoom_scalebar_length_m,
        location=(ZOOM_SCALEBAR_X_FRAC, ZOOM_SCALEBAR_Y_FRAC),
        fontsize=7,
        is_geographic=is_geographic,
    )

    for spine in ax_zoom.spines.values():
        spine.set_linewidth(1.3)
        spine.set_edgecolor(ZOOM_CONNECTOR_COLOR)

    con1 = ConnectionPatch(
        xyA=(zoom_xmax, zoom_ymax),
        coordsA=ax.transData,
        xyB=(zoom_xmin, zoom_ymax),
        coordsB=ax_zoom.transData,
        color=ZOOM_CONNECTOR_COLOR,
        linewidth=1.1,
        alpha=0.95,
    )
    con2 = ConnectionPatch(
        xyA=(zoom_xmax, zoom_ymin),
        coordsA=ax.transData,
        xyB=(zoom_xmin, zoom_ymin),
        coordsB=ax_zoom.transData,
        color=ZOOM_CONNECTOR_COLOR,
        linewidth=1.1,
        alpha=0.95,
    )
    fig.add_artist(con1)
    fig.add_artist(con2)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=FIG_DPI, facecolor=fig_bg, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved GPKG: {output_gpkg}")
    print(f"Saved PNG : {output_png}")


if __name__ == "__main__":
    main()
