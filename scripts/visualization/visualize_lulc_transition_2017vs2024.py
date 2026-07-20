#!/usr/bin/env python3
"""
Visualize the top off-diagonal LULC transitions between 2017 and 2024.

The script:
- reads transition_code_2017_to_2024.tif
- decodes class-pair transitions
- keeps the top 10 off-diagonal transitions by pixel count
- shows unchanged pixels in light gray
- masks all other changed transitions into a single muted "other change" class

Inputs
------
- outputs/inference/change_analysis/transition_code_2017_to_2024.tif
- assets/maps/bd_coastal_zones.gpkg
- assets/maps/NorthArrow.svg
- assets/color_palette_coastal_lulc.json

Output
------
- outputs/figures/lulc_transition_2017_vs_2024_top10.png

Example
-------
python scripts/visualization/visualize_lulc_transition_2017vs2024.py

Reproduction and AOI adaptation
-------------------------------
Workflow role: Turn prepared rasters, vectors, and tables into thesis-ready figures.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--transition``, ``--zone-map``, ``--north-arrow``, ``--palette``, ``--output``, ``--top-n``. Run the ``--help`` command below for required values, defaults, and accepted choices.
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
DEFAULT_TRANSITION = Path("outputs/inference/change_analysis/transition_code_2017_to_2024.tif")
DEFAULT_ZONE_MAP = Path("assets/maps/bd_coastal_zones.gpkg")
DEFAULT_NORTH_ARROW = Path("assets/maps/NorthArrow.svg")
DEFAULT_PALETTE = Path("assets/color_palette_coastal_lulc.json")
DEFAULT_OUTPUT = Path("outputs/figures/lulc_transition_2017_vs_2024_top10.png")

FIGSIZE = (11, 9)
FIG_DPI = 300
MAX_DISPLAY_SIZE = 2800
DISPLAY_CHUNK_SIZE = 512
TOP_N = 10
LONGITUDE_LABEL_PAD = 0
TIGHT_LAYOUT_BOTTOM = -0.04
TRANSITION_NODATA = 0

MAP_TITLE = "Bangladesh Coastal LULC Transition (2017 to 2024)"
BAY_LABEL_X_FRAC = 0.56
BAY_LABEL_Y_FRAC = 0.19
SCALEBAR_X_FRAC = 0.49
SCALEBAR_Y_FRAC = 0.06
LEGEND_X_FRAC = 0.02
LEGEND_Y_FRAC = 0.115
LEGEND_FONTSIZE = 10
LEGEND_HANDLE_LENGTH = 1.9
LEGEND_HANDLE_HEIGHT = 1.3
LEGEND_LABEL_SPACING = 0.55
LEGEND_BORDER_PAD = 0.75
LEGEND_BOX_ALPHA = 0.97

CLASS_NAMES = {
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

ZONE_LABELS = {
    "western": "Western Zone",
    "central": "Central Zone",
    "eastern": "Eastern Zone",
}

TRANSITION_COLOR_LIST = [
    "#E66A00",
    "#00ADA9",
    "#4F7F3D",
    "#007C91",
    "#FFC636",
    "#9C7A5B",
    "#FF8973",
    "#8FBF7A",
    "#7AD9D6",
    "#2F5D50",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize the top off-diagonal LULC transitions from 2017 to 2024.")
    p.add_argument("--transition", type=Path, default=DEFAULT_TRANSITION, help="Transition-code raster path.")
    p.add_argument("--zone-map", type=Path, default=DEFAULT_ZONE_MAP, help="Coastal zones vector layer.")
    p.add_argument("--north-arrow", type=Path, default=DEFAULT_NORTH_ARROW, help="North arrow SVG path.")
    p.add_argument("--palette", type=Path, default=DEFAULT_PALETTE, help="Palette JSON path.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    p.add_argument("--top-n", type=int, default=TOP_N, help="Number of off-diagonal transitions to show.")
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


def add_scalebar_2step(ax, length_km=150, location=(0.49, 0.06), fontsize=10):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + location[0] * (xlim[1] - xlim[0])
    y0 = ylim[0] + location[1] * (ylim[1] - ylim[0])
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


def transition_to_pair(code: int) -> tuple[int, int]:
    return code // 100, code % 100


def count_transition_codes(ds: rasterio.io.DatasetReader) -> dict[int, int]:
    counts: dict[int, int] = {}
    for _, window in ds.block_windows(1):
        arr = ds.read(1, window=window)
        valid = arr != TRANSITION_NODATA
        if not np.any(valid):
            continue
        vals, cnts = np.unique(arr[valid], return_counts=True)
        for value, count in zip(vals.tolist(), cnts.tolist()):
            counts[int(value)] = counts.get(int(value), 0) + int(count)
    return counts


def top_off_diagonal_codes(counts: dict[int, int], top_n: int) -> list[int]:
    items = []
    for code, count in counts.items():
        c17, c24 = transition_to_pair(code)
        if c17 == c24:
            continue
        if c17 not in CLASS_NAMES or c24 not in CLASS_NAMES:
            continue
        items.append((code, count))
    items.sort(key=lambda x: x[1], reverse=True)
    return [code for code, _ in items[:top_n]]


def read_downsampled_transition_windowed(
    ds: rasterio.io.DatasetReader,
    max_size: int,
    chunk_size: int,
) -> np.ndarray:
    src_h = ds.height
    src_w = ds.width
    scale = max(src_h / max_size, src_w / max_size, 1.0)
    dst_h = max(1, int(round(src_h / scale)))
    dst_w = max(1, int(round(src_w / scale)))

    out = np.empty((dst_h, dst_w), dtype=np.uint16)

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
            out[row0:row1, col0:col1] = data.astype(np.uint16, copy=False)
    return out


def classify_transition_preview(
    arr: np.ndarray,
    top_codes: list[int],
) -> np.ndarray:
    encoded = np.zeros(arr.shape, dtype=np.uint8)
    valid = arr != TRANSITION_NODATA
    unchanged = valid & ((arr // 100) == (arr % 100))
    other_change = valid & ~unchanged

    encoded[unchanged] = 1
    encoded[other_change] = 2
    for idx, code in enumerate(top_codes, start=3):
        encoded[arr == code] = idx
    return encoded


def encoded_to_rgb(encoded: np.ndarray, category_colors: dict[int, tuple[float, float, float]], bg_rgb: tuple[float, float, float]) -> np.ndarray:
    rgb = np.zeros((encoded.shape[0], encoded.shape[1], 3), dtype=np.float32)
    rgb[:] = bg_rgb
    for category, color in category_colors.items():
        rgb[encoded == category] = color
    return rgb


def short_class_name(class_id: int) -> str:
    name = CLASS_NAMES[class_id]
    if name == "Urban / Institutional Built-up":
        return "Urban"
    if name == "Rural Settlement (Homestead Vegetation)":
        return "Rural Settlement"
    if name == "Transport & Coastal Embankments":
        return "Transport/Embankments"
    if name == "Cropland (All Crop Intensities)":
        return "Cropland"
    if name == "Tree-based Agroforestry & Orchard":
        return "Agroforestry/Orchard"
    if name == "Aquaculture & Inland Ponds":
        return "Aquaculture/Ponds"
    if name == "Canals & Drainage Network":
        return "Canals/Drainage"
    if name == "Rivers & Estuarine Channels":
        return "Rivers/Channels"
    if name == "Bare / Exposed Coastal Land":
        return "Bare/Exposed Land"
    return name


def legend_handles(top_codes: list[int], category_colors: dict[int, tuple[float, float, float]], edge_color: str) -> list[Patch]:
    handles = [
        Patch(facecolor=category_colors[1], edgecolor=edge_color, label="Unchanged"),
        Patch(facecolor=category_colors[2], edgecolor=edge_color, label="Other change"),
    ]
    for idx, code in enumerate(top_codes, start=3):
        c17, c24 = transition_to_pair(code)
        label = f"{short_class_name(c17)} -> {short_class_name(c24)}"
        handles.append(Patch(facecolor=category_colors[idx], edgecolor=edge_color, label=label))
    return handles


def main() -> None:
    args = parse_args()
    transition_path = resolve_path(args.transition)
    zone_map = resolve_path(args.zone_map)
    north_arrow = resolve_path(args.north_arrow)
    palette_path = resolve_path(args.palette)
    output = resolve_path(args.output)

    palette = load_palette(palette_path)
    colors = palette["colors"]

    fig_bg = colors["sand"]
    sea_color = colors["mist_gray"]
    grid_color = colors["deep_slate"]
    zone_edge = colors["coral"]
    main_text_color = colors["deep_slate"]
    zone_text_color = colors["coral"]
    bay_text_color = colors["teal_blue"]
    legend_face = "#FFF9EF"

    with rasterio.open(transition_path) as ds:
        if ds.crs is None:
            raise ValueError("Transition raster has no CRS.")
        raster_crs = ds.crs
        bounds = ds.bounds
        counts = count_transition_codes(ds)
        top_codes = top_off_diagonal_codes(counts, args.top_n)
        preview = read_downsampled_transition_windowed(
            ds,
            max_size=MAX_DISPLAY_SIZE,
            chunk_size=DISPLAY_CHUNK_SIZE,
        )

    encoded = classify_transition_preview(preview, top_codes)
    category_colors: dict[int, tuple[float, float, float]] = {
        1: mcolors.to_rgb("#D9D9D9"),
        2: mcolors.to_rgb("#9E9E9E"),
    }
    for idx, color in enumerate(TRANSITION_COLOR_LIST[: len(top_codes)], start=3):
        category_colors[idx] = mcolors.to_rgb(color)
    rgb = encoded_to_rgb(encoded, category_colors, bg_rgb=mcolors.to_rgb(sea_color))

    zones = gpd.read_file(zone_map)
    if zones.empty:
        raise ValueError("Zone map is empty.")
    if zones.crs is None:
        raise ValueError("Zone map has no CRS.")
    zones = zones.to_crs(raster_crs)

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

    for _, row in zones.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        zone_key = str(row["zone"]).strip().lower()
        label = ZONE_LABELS.get(zone_key, zone_key.title())
        pt = geom.representative_point()
        txt = ax.text(
            pt.x,
            pt.y,
            label,
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            color=zone_text_color,
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
    ax.set_title(MAP_TITLE, fontsize=15, pad=12, color=main_text_color, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=12, color=main_text_color, labelpad=LONGITUDE_LABEL_PAD)
    ax.set_ylabel("Latitude", fontsize=12, color=main_text_color)
    ax.tick_params(axis="both", colors=main_text_color)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    add_north_arrow(ax, north_arrow, xy=(0.92, 0.90), zoom=0.23)
    add_scalebar_2step(ax, length_km=150, location=(SCALEBAR_X_FRAC, SCALEBAR_Y_FRAC), fontsize=10)

    legend = ax.legend(
        handles=legend_handles(top_codes, category_colors, main_text_color),
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

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, TIGHT_LAYOUT_BOTTOM, 1, 1))
    plt.savefig(output, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
