#!/usr/bin/env python3
"""
Create a publication-style Sundarbans-only JAXA HRLULC map for a given year.

Inputs
------
- data/processed/jaxa_hrlulc/bd_coastal_jaxa_hrlulc_<year>_clipped.tif
- assets/maps/sundarbans.gpkg
- assets/maps/NorthArrow.svg
- assets/color_palette_coastal_lulc.json

Output
------
- outputs/figures/sundarbans_jaxa_hrlulc_<year>.png

Example
-------
python scripts/visualization/make_jaxa_hrlulc_sundarbans_lulc_map.py --year 2023
python scripts/visualization/make_jaxa_hrlulc_sundarbans_lulc_map.py --year 2023 \
    --buffer-top 1000 --buffer-bottom 10000 --buffer-left 10000 --buffer-right 10000

Complete Example Run
--------------------
python scripts/visualization/make_jaxa_hrlulc_sundarbans_lulc_map.py \
    --year 2023 \
    --add-title \
    --outptut-plot outputs/figures/sundarbans_jaxa_hrlulc_2023.png \
    --buffer-top 1000 \
    --buffer-bottom 10000 \
    --buffer-left 10000 \
    --buffer-right 10000
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
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PIL import Image
from pyproj import CRS, Transformer
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds

try:
    import cairosvg

    HAVE_SVG = True
except Exception:
    HAVE_SVG = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUNDARBANS_MAP = Path("assets/maps/sundarbans.gpkg")
DEFAULT_NORTH_ARROW = Path("assets/maps/NorthArrow.svg")
DEFAULT_PALETTE = Path("assets/color_palette_coastal_lulc.json")
DEFAULT_INPUT_ROOT = Path("data/processed/jaxa_hrlulc")
DEFAULT_OUTPUT_ROOT = Path("outputs/figures")

FIGSIZE = (10, 9)
FIG_DPI = 300
MAX_DISPLAY_SIZE = 2800
DISPLAY_CHUNK_SIZE = 512
LONGITUDE_LABEL_PAD = 0
TIGHT_LAYOUT_BOTTOM = -0.04
CLASS_NODATA = 0
MAP_TITLE_TEMPLATE = "Sundarbans JAXA HRLULC {year}"

# Per-side buffer around the Sundarbans polygon in raster CRS units (metres for UTM).
# Top is kept small so the map does not extend far north into land.
BUFFER_TOP_M    = 1000
BUFFER_BOTTOM_M = 10000
BUFFER_LEFT_M   = 10000
BUFFER_RIGHT_M  = 10000

# Scale bar — 25 km suits the ~100 km Sundarbans extent
SCALEBAR_LENGTH_KM = 25
SCALEBAR_X_FRAC = 0.015
SCALEBAR_Y_FRAC = 0.04

# Bay of Bengal label — bottom strip of the clipped extent
BAY_LABEL_X_FRAC = 0.70
BAY_LABEL_Y_FRAC = 0.23

# North arrow — upper-right corner
NORTH_ARROW_XY = (0.93, 0.91)
NORTH_ARROW_ZOOM = 0.23

LEGEND_FONTSIZE = 9
LEGEND_HANDLE_LENGTH = 1.8
LEGEND_HANDLE_HEIGHT = 1.2
LEGEND_LABEL_SPACING = 0.48
LEGEND_BORDER_PAD = 0.65
LEGEND_BOX_ALPHA = 0.97

JAXA_LULC_COLORS = {
    1:  "#000064",
    2:  "#ff0000",
    3:  "#a12977",
    4:  "#ffc1bf",
    5:  "#42d6ff",
    6:  "#0080ff",
    7:  "#0096a0",
    8:  "#ffff00",
    9:  "#80ff00",
    10: "#56ac00",
    11: "#00ac56",
    12: "#a1556b",
    13: "#9c7ca0",
    14: "#013a24",
    15: "#806400",
}

JAXA_LULC_NAMES = {
    1:  "Water Bodies",
    2:  "Built-up",
    3:  "Solar Panel",
    4:  "Cropland",
    5:  "Single-crop Paddy Field",
    6:  "Multi-crop Paddy Field",
    7:  "Herbaceous Wetland",
    8:  "Grassland",
    9:  "Deciduous Broad-leaved Forest (DBF)",
    10: "Evergreen Broad-leaved Forest (EBF)",
    11: "Evergreen Needle-leaved Forest (ENF)",
    12: "Rubber Tree Plantation",
    13: "Oil Palm Tree Plantation",
    14: "Mangrove Forest",
    15: "Bare",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_input_path(year: int) -> Path:
    return DEFAULT_INPUT_ROOT / f"bd_coastal_jaxa_hrlulc_{year}_clipped.tif"


def default_output_path(year: int) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"sundarbans_jaxa_hrlulc_{year}.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make Sundarbans JAXA HRLULC map for a target year.")
    p.add_argument("--year", type=int, required=True, help="Target year, e.g. 2023.")
    p.add_argument("--input", type=Path, default=None, help="Optional JAXA HRLULC GeoTIFF override.")
    p.add_argument("--sundarbans-map", type=Path, default=DEFAULT_SUNDARBANS_MAP, help="Sundarbans vector layer.")
    p.add_argument("--north-arrow", type=Path, default=DEFAULT_NORTH_ARROW, help="North arrow SVG path.")
    p.add_argument("--palette", type=Path, default=DEFAULT_PALETTE, help="Palette JSON path.")
    p.add_argument("--add-title", action="store_true", help="Show title on top of the plot.")
    p.add_argument(
        "--outptut-plot",
        type=Path,
        default=None,
        help="Output PNG path. Default: outputs/figures/sundarbans_jaxa_hrlulc_<year>.png",
    )
    p.add_argument("--output", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--buffer-top", type=float, default=BUFFER_TOP_M,
                   help="Buffer north of Sundarbans in metres. Default: 1000.")
    p.add_argument("--buffer-bottom", type=float, default=BUFFER_BOTTOM_M,
                   help="Buffer south of Sundarbans in metres (Bay of Bengal). Default: 10000.")
    p.add_argument("--buffer-left", type=float, default=BUFFER_LEFT_M,
                   help="Buffer west of Sundarbans in metres. Default: 10000.")
    p.add_argument("--buffer-right", type=float, default=BUFFER_RIGHT_M,
                   help="Buffer east of Sundarbans in metres. Default: 10000.")
    return p.parse_args()


def load_palette(path: Path) -> dict:
    return json.loads(path.read_text())


def decimal_to_dm(value: float, kind: str) -> str:
    hemi = ("E" if value >= 0 else "W") if kind == "lon" else ("N" if value >= 0 else "S")
    deg_abs = abs(value)
    d = int(deg_abs)
    m = int(round((deg_abs - d) * 60))
    if m == 60:
        d += 1
        m = 0
    return f"{d}°{m:02d}′{hemi}"


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
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, color=color, linestyle="--", linewidth=0.6, alpha=0.24, zorder=0)
    ax.tick_params(axis="both", labelsize=10, direction="out", top=True, right=True,
                   labeltop=False, labelright=False)
    for label in ax.get_yticklabels():
        label.set_rotation(90)
        label.set_va("center")
        label.set_ha("center")


def add_scalebar(
    ax,
    length_km: int = 25,
    location: tuple[float, float] = (0.03, 0.04),
    fontsize: int = 10,
    km_to_crs: float = 1000.0,
) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + location[0] * (xlim[1] - xlim[0])
    y0 = ylim[0] + location[1] * (ylim[1] - ylim[0])
    total_map_units = length_km * km_to_crs
    step_map_units = total_map_units / 2.0
    bar_h = 0.014 * (ylim[1] - ylim[0])
    half_km = length_km // 2

    for i, face in enumerate(["black", "white"]):
        ax.add_patch(Rectangle(
            (x0 + i * step_map_units, y0),
            step_map_units,
            bar_h,
            facecolor=face,
            edgecolor="black",
            linewidth=0.8,
            zorder=10,
        ))

    label_y = y0 + bar_h + 0.012 * (ylim[1] - ylim[0])
    ax.text(x0, label_y, "0", ha="center", va="bottom", fontsize=fontsize, zorder=11)
    ax.text(x0 + step_map_units, label_y, str(half_km), ha="center", va="bottom",
            fontsize=fontsize, zorder=11)
    ax.text(x0 + 2 * step_map_units, label_y, f"{length_km} km", ha="center",
            va="bottom", fontsize=fontsize, zorder=11)


def load_svg_as_image(svg_path: Path, target_height_px: int = 220):
    if not svg_path.exists() or not HAVE_SVG:
        return None
    png_bytes = cairosvg.svg2png(url=str(svg_path), output_height=target_height_px)
    return Image.open(io.BytesIO(png_bytes))


def add_north_arrow(
    ax,
    svg_path: Path,
    xy: tuple[float, float] = (0.93, 0.91),
    zoom: float = 0.23,
) -> None:
    img = load_svg_as_image(svg_path, target_height_px=220)
    if img is None:
        ax.annotate("N", xy=xy, xycoords="axes fraction", ha="center", va="center",
                    fontsize=16, fontweight="bold", zorder=20)
        ax.annotate("", xy=(xy[0], xy[1] - 0.03), xytext=(xy[0], xy[1] - 0.14),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black"), zorder=20)
        return
    imagebox = OffsetImage(np.asarray(img), zoom=zoom)
    ab = AnnotationBbox(imagebox, xy, xycoords="axes fraction", frameon=False,
                        box_alignment=(0.5, 0.5), zorder=20)
    ax.add_artist(ab)


def read_clipped_class_raster(
    ds: rasterio.io.DatasetReader,
    clip_left: float,
    clip_bottom: float,
    clip_right: float,
    clip_top: float,
    max_size: int,
    chunk_size: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Read and nearest-neighbour downsample a spatial sub-window of the raster.

    Returns the pixel array and its actual geographic bounds (left, bottom, right, top).
    """
    win = window_from_bounds(clip_left, clip_bottom, clip_right, clip_top, ds.transform)

    col_off = max(0, int(win.col_off))
    row_off = max(0, int(win.row_off))
    col_end = min(ds.width, int(win.col_off + win.width))
    row_end = min(ds.height, int(win.row_off + win.height))
    win_width = max(1, col_end - col_off)
    win_height = max(1, row_end - row_off)

    window = Window(col_off, row_off, win_width, win_height)
    actual_bounds = ds.window_bounds(window)  # (left, bottom, right, top)

    scale = max(win_height / max_size, win_width / max_size, 1.0)
    dst_h = max(1, int(round(win_height / scale)))
    dst_w = max(1, int(round(win_width / scale)))

    out = np.empty((dst_h, dst_w), dtype=np.int16)
    for row0 in range(0, dst_h, chunk_size):
        row1 = min(row0 + chunk_size, dst_h)
        for col0 in range(0, dst_w, chunk_size):
            col1 = min(col0 + chunk_size, dst_w)
            src_row0 = int(round(row0 * win_height / dst_h))
            src_row1 = int(round(row1 * win_height / dst_h))
            src_col0 = int(round(col0 * win_width / dst_w))
            src_col1 = int(round(col1 * win_width / dst_w))
            src_row1 = max(src_row0 + 1, min(src_row1, win_height))
            src_col1 = max(src_col0 + 1, min(src_col1, win_width))
            tile = ds.read(
                1,
                window=Window(
                    col_off + src_col0, row_off + src_row0,
                    src_col1 - src_col0, src_row1 - src_row0,
                ),
                out_shape=(row1 - row0, col1 - col0),
                resampling=rasterio.enums.Resampling.nearest,
            )
            out[row0:row1, col0:col1] = tile.astype(np.int16, copy=False)

    return out, actual_bounds


def class_raster_to_rgb(arr: np.ndarray, nodata_rgb: tuple[float, float, float]) -> np.ndarray:
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
    rgb[:] = nodata_rgb
    for class_id, color in JAXA_LULC_COLORS.items():
        rgb[arr == class_id] = mcolors.to_rgb(color)
    return rgb


def legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=JAXA_LULC_COLORS[class_id], edgecolor="#314245", label=JAXA_LULC_NAMES[class_id])
        for class_id in range(1, 16)
    ]


def main() -> None:
    args = parse_args()
    input_raster = resolve_path(args.input or default_input_path(args.year))
    sundarbans_map = resolve_path(args.sundarbans_map)
    north_arrow = resolve_path(args.north_arrow)
    palette_path = resolve_path(args.palette)
    output = resolve_path(args.outptut_plot or args.output or default_output_path(args.year))

    palette = load_palette(palette_path)
    colors = palette["colors"]
    fig_bg = colors["sand"]
    sea_color = colors["mist_gray"]
    grid_color = colors["deep_slate"]
    main_text_color = colors["deep_slate"]
    bay_text_color = colors["teal_blue"]
    legend_face = "#FFF9EF"
    boundary_color = "#2b2e07"

    # Load Sundarbans — get mean latitude in WGS84 for aspect correction
    sundarbans_wgs84 = gpd.read_file(sundarbans_map)
    if sundarbans_wgs84.empty:
        raise ValueError("Sundarbans vector is empty.")
    if sundarbans_wgs84.crs is None:
        raise ValueError("Sundarbans vector has no CRS.")
    sb_wgs84_bounds = sundarbans_wgs84.total_bounds  # [minx, miny, maxx, maxy] in degrees
    mean_lat_deg = 0.5 * (sb_wgs84_bounds[1] + sb_wgs84_bounds[3])

    with rasterio.open(input_raster) as ds:
        if ds.crs is None:
            raise ValueError("Input JAXA HRLULC raster has no CRS.")
        raster_crs = ds.crs

        # Reproject Sundarbans to raster CRS and compute clip extent with buffer.
        # If the raster CRS is geographic (degrees), convert metre buffers to degrees.
        sundarbans = sundarbans_wgs84.to_crs(raster_crs)
        sb_bounds = sundarbans.total_bounds  # [minx, miny, maxx, maxy]
        if raster_crs.is_geographic:
            m_per_lat_deg = 111_320.0
            m_per_lon_deg = 111_320.0 * np.cos(np.deg2rad(mean_lat_deg))
            buf_top    = args.buffer_top    / m_per_lat_deg
            buf_bottom = args.buffer_bottom / m_per_lat_deg
            buf_left   = args.buffer_left   / m_per_lon_deg
            buf_right  = args.buffer_right  / m_per_lon_deg
        else:
            buf_top, buf_bottom = args.buffer_top, args.buffer_bottom
            buf_left, buf_right = args.buffer_left, args.buffer_right
        clip_left   = sb_bounds[0] - buf_left
        clip_bottom = sb_bounds[1] - buf_bottom
        clip_right  = sb_bounds[2] + buf_right
        clip_top    = sb_bounds[3] + buf_top

        classes, actual_bounds = read_clipped_class_raster(
            ds,
            clip_left, clip_bottom, clip_right, clip_top,
            max_size=MAX_DISPLAY_SIZE,
            chunk_size=DISPLAY_CHUNK_SIZE,
        )

    left, bottom, right, top = actual_bounds
    rgb = class_raster_to_rgb(classes, nodata_rgb=mcolors.to_rgb(sea_color))

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI, facecolor=fig_bg)
    ax.set_facecolor(sea_color)

    ax.imshow(
        rgb,
        extent=(left, right, bottom, top),
        origin="upper",
        interpolation="nearest",
        zorder=1,
    )

    sundarbans.boundary.plot(ax=ax, color=boundary_color, linewidth=1.4, zorder=4)

    # Sundarbans interior label
    for _, row in sundarbans.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        label = str(row.get("zone", "Sundarbans")).strip()
        pt = geom.representative_point()
        txt = ax.text(
            pt.x, pt.y,
            label,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            color=main_text_color,
            zorder=6,
        )
        txt.set_path_effects([pe.Stroke(linewidth=3, foreground=fig_bg), pe.Normal()])

    # Bay of Bengal label — placed in the southern buffer strip below the polygon
    bay_x = left + BAY_LABEL_X_FRAC * (right - left)
    bay_y = bottom + BAY_LABEL_Y_FRAC * (top - bottom)
    bay_txt = ax.text(
        bay_x, bay_y,
        "Bay of Bengal",
        fontsize=13,
        ha="center",
        va="center",
        style="italic",
        color=bay_text_color,
        zorder=5,
    )
    bay_txt.set_path_effects([pe.Stroke(linewidth=4, foreground=fig_bg), pe.Normal()])

    # Enforce clipped extent — must come after all vector plots
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)

    # Correct aspect for latitude distortion
    cosv = np.cos(np.deg2rad(mean_lat_deg))
    ax.set_aspect(1.0 / cosv if abs(cosv) > 1e-8 else "equal")

    add_graticule(ax, color=grid_color, src_crs=raster_crs)

    if args.add_title:
        title = MAP_TITLE_TEMPLATE.format(year=args.year)
        ax.set_title(title, fontsize=15, pad=12, color=main_text_color, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=12, color=main_text_color, labelpad=LONGITUDE_LABEL_PAD)
    ax.set_ylabel("Latitude", fontsize=12, color=main_text_color)
    ax.tick_params(axis="both", colors=main_text_color)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    add_north_arrow(ax, north_arrow, xy=NORTH_ARROW_XY, zoom=NORTH_ARROW_ZOOM)
    km_to_crs = (1.0 / 111.32) if raster_crs.is_geographic else 1000.0
    add_scalebar(ax, length_km=SCALEBAR_LENGTH_KM,
                 location=(SCALEBAR_X_FRAC, SCALEBAR_Y_FRAC), fontsize=10,
                 km_to_crs=km_to_crs)

    legend = ax.legend(
        handles=legend_handles(),
        loc="lower right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        framealpha=LEGEND_BOX_ALPHA,
        facecolor=legend_face,
        edgecolor=main_text_color,
        ncol=2,
        handlelength=LEGEND_HANDLE_LENGTH,
        handleheight=LEGEND_HANDLE_HEIGHT,
        labelspacing=LEGEND_LABEL_SPACING,
        borderpad=LEGEND_BORDER_PAD,
    )
    legend.set_zorder(12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor(main_text_color)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, TIGHT_LAYOUT_BOTTOM, 1, 1))
    plt.savefig(output, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
