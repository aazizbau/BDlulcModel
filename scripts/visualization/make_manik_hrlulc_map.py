#!/usr/bin/env python3
"""
Create a publication-style Manik HRLULC map for a given year.

Inputs
------
- data/processed/manik_lulc/bd_coastal_manik_lulc_<year>_clipped.tif
- assets/maps/bd_coastal_zones.gpkg
- assets/maps/sundarbans.gpkg
- assets/maps/NorthArrow.svg
- assets/color_palette_coastal_lulc.json

Output
------
- outputs/figures/bd_coastal_manik_hrlulc_<year>.png

Example
-------
python scripts/visualization/make_manik_hrlulc_map.py \
    --year 2023

Reproduction and AOI adaptation
-------------------------------
Workflow role: Turn prepared rasters, vectors, and tables into thesis-ready figures.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--year``, ``--input``, ``--zone-map``, ``--sundarbans-map``, ``--north-arrow``, ``--palette``, ``--output``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace raster/vector/palette paths with target-AOI products and verify matching CRS, extent, class IDs, units, and map annotations before publication.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
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
DEFAULT_INPUT_ROOT = Path("data/processed/manik_lulc")
DEFAULT_OUTPUT_ROOT = Path("outputs/figures")

FIGSIZE = (11, 9)
FIG_DPI = 300
MAX_DISPLAY_SIZE = 2800
DISPLAY_CHUNK_SIZE = 512
LONGITUDE_LABEL_PAD = 0
TIGHT_LAYOUT_BOTTOM = -0.04
CLASS_NODATA = 0
MAP_TITLE_TEMPLATE = "Bangladesh Coastal HRLULC {year}"
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

HRLULC_NAMES = {
    1:  "Water",
    2:  "Built-up",
    3:  "Single cropland",
    4:  "Multiple cropland",
    5:  "Aquaculture",
    6:  "Orchards",
    7:  "Brickfield",
    8:  "Forest",
    9:  "Mangrove",
    10: "Salt pans",
    11: "Rubber tree",
    12: "Jhum",
    13: "Bare land",
    14: "Tea garden",
}

HRLULC_COLORS = {
    1:  "#000064",
    2:  "#FF0000",
    3:  "#FF7F7F",
    4:  "#FFC1BF",
    5:  "#4D68FF",
    6:  "#80FF00",
    7:  "#A0A0A0",
    8:  "#006400",
    9:  "#013A24",
    10: "#F0F0F0",
    11: "#A1556B",
    12: "#4B7B4E",
    13: "#806400",
    14: "#5ECC7E",
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
    return DEFAULT_INPUT_ROOT / f"bd_coastal_manik_lulc_{year}_clipped.tif"


def default_output_path(year: int) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"bd_coastal_manik_hrlulc_{year}.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make Manik HRLULC map for a target year.")
    p.add_argument("--year", type=int, required=True, help="Target year, e.g. 2023.")
    p.add_argument("--input", type=Path, default=None, help="Optional Manik HRLULC GeoTIFF override.")
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


def meters_to_lat_degrees(meters: float) -> float:
    return meters / 111_320.0


def km_to_lon_degrees(km: float, lat_deg: float) -> float:
    cos_lat = np.cos(np.deg2rad(lat_deg))
    if abs(cos_lat) < 1e-8:
        cos_lat = 1e-8
    return km / (111.320 * cos_lat)


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
    ax.text(x0 + step_map_units, label_y, "75", ha="center", va="bottom", fontsize=fontsize, zorder=11)
    ax.text(x0 + 2 * step_map_units, label_y, "150 km", ha="center", va="bottom", fontsize=fontsize, zorder=11)


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


def class_raster_to_rgb(arr: np.ndarray, nodata_rgb: tuple[float, float, float]) -> np.ndarray:
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
    rgb[:] = nodata_rgb
    for class_id, color in HRLULC_COLORS.items():
        rgb[arr == class_id] = mcolors.to_rgb(color)
    return rgb


def legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=HRLULC_COLORS[class_id], edgecolor="#314245", label=HRLULC_NAMES[class_id])
        for class_id in range(1, 15)
    ]


def main() -> None:
    args = parse_args()
    input_raster = resolve_path(args.input or default_input_path(args.year))
    zone_map = resolve_path(args.zone_map)
    sundarbans_map = resolve_path(args.sundarbans_map)
    north_arrow = resolve_path(args.north_arrow)
    palette_path = resolve_path(args.palette)
    output = resolve_path(args.output or default_output_path(args.year))

    palette = load_palette(palette_path)
    colors = palette["colors"]

    fig_bg = colors["sand"]
    sea_color = colors["mist_gray"]
    grid_color = colors["deep_slate"]
    main_text_color = colors["deep_slate"]
    zone_text_color = colors["coral"]
    sundarbans_text_color = colors["deep_slate"]
    bay_text_color = colors["teal_blue"]
    legend_face = "#FFF9EF"

    with rasterio.open(input_raster) as ds:
        if ds.crs is None:
            raise ValueError("Input Manik HRLULC raster has no CRS.")
        raster_crs = ds.crs
        is_geographic = CRS.from_user_input(raster_crs).is_geographic
        bounds = ds.bounds
        classes = read_downsampled_class_raster_windowed(
            ds,
            max_size=MAX_DISPLAY_SIZE,
            chunk_size=DISPLAY_CHUNK_SIZE,
        )

    rgb = class_raster_to_rgb(classes, nodata_rgb=mcolors.to_rgb(sea_color))

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

    # Zone boundaries: black halo + bright gold line
    zones.boundary.plot(ax=ax, color="black", linewidth=2.0, zorder=4, alpha=0.6)
    zones.boundary.plot(ax=ax, color="#FFD700", linewidth=0.7, zorder=4)

    # Sundarbans boundary: black halo + bright gold line
    sundarbans.boundary.plot(ax=ax, color="black", linewidth=2.0, zorder=5, alpha=0.6)
    sundarbans.boundary.plot(ax=ax, color="#FFD700", linewidth=0.7, zorder=5)

    for _, row in zones.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        zone_key = str(row["zone"]).strip().lower()
        label = ZONE_LABELS.get(zone_key, zone_key.title())
        pt = geom.representative_point()
        dx, dy = ZONE_LABEL_OFFSETS.get(zone_key, (0.0, 0.0))
        if is_geographic:
            dy = meters_to_lat_degrees(dy)
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
    add_scalebar_2step(
        ax,
        length_km=150,
        location=(SCALEBAR_X_FRAC, SCALEBAR_Y_FRAC),
        fontsize=10,
        is_geographic=is_geographic,
    )

    legend = ax.legend(
        handles=legend_handles(),
        loc="lower left",
        bbox_to_anchor=(LEGEND_X_FRAC, LEGEND_Y_FRAC),
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
