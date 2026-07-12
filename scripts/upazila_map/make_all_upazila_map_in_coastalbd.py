#!/usr/bin/env python3
"""
Create a coastal Bangladesh overview map showing selected upazilas.

This map follows the cartographic style used by the study-area map:
- latitude / longitude labels in degree-minute format
- dashed graticule
- north arrow in the upper-right corner
- two-step 0, 75, 150 km scale bar
- coastal zone labels for Western, Central, and Eastern zones
- Sundarbans label with halo text
- no DEM, no colorbar, and no legend

Inputs
------
- assets/maps/bd_coastal_zones.gpkg
- assets/maps/sundarbans.gpkg
- assets/maps/NorthArrow.svg
- assets/maps/manpura_dissolved.gpkg
- assets/maps/amtali_dissolved.gpkg
- assets/maps/bamna_dissolved.gpkg
- assets/maps/betagi_dissolved.gpkg

Output
------
- outputs/figures/all_upazila_map_in_coastalbd.png

Complete example run
--------------------
python scripts/upazila_map/make_all_upazila_map_in_coastalbd.py \
    --add-title \
    --output outputs/figures/all_upazila_map_in_coastalbd.png
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PIL import Image

try:
    import cairosvg

    HAVE_SVG = True
except Exception:
    HAVE_SVG = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_CRS = "EPSG:4326"

DEFAULT_ZONE_MAP = Path("assets/maps/bd_coastal_zones.gpkg")
DEFAULT_SUNDARBANS_MAP = Path("assets/maps/sundarbans.gpkg")
DEFAULT_NORTH_ARROW = Path("assets/maps/NorthArrow.svg")
DEFAULT_OUTPUT = Path("outputs/figures/all_upazila_map_in_coastalbd.png")
DEFAULT_UPAZILA_VECTORS = [
    Path("assets/maps/manpura_dissolved.gpkg"),
    Path("assets/maps/amtali_dissolved.gpkg"),
    Path("assets/maps/bamna_dissolved.gpkg"),
    Path("assets/maps/betagi_dissolved.gpkg"),
]

FIGSIZE = (11, 9)
FIG_DPI = 300
MAP_TITLE = "Selected Upazilas in Coastal Bangladesh"
X_AXIS_LABEL = "Longitude"
Y_AXIS_LABEL = "Latitude"
LONGITUDE_LABEL_PAD = 0
TIGHT_LAYOUT_BOTTOM = -0.05

FIG_BG = "#F7F5EF"
SEA_COLOR = "#D9DEE0"
ZONE_FILL = "#D7D7D7"
SUNDARBANS_FILL = "#BFDDB6"
EDGE_COLOR = "#263238"
GRID_COLOR = "#263238"
ZONE_TEXT_COLOR = "#263238"
SUNDARBANS_TEXT_COLOR = "#263238"
UPAZILA_TEXT_COLOR = "#8B1E3F"
BAY_TEXT_COLOR = "#2F6F7E"

UPAZILA_COLORS = {
    "manpura": "#E76F51",
    "amtali": "#457B9D",
    "bamna": "#F4A261",
    "betagi": "#7B2CBF",
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make a coastal Bangladesh overview map with selected upazilas."
    )
    p.add_argument("--zone-map", type=Path, default=DEFAULT_ZONE_MAP, help="Coastal zones vector layer.")
    p.add_argument("--sundarbans-map", type=Path, default=DEFAULT_SUNDARBANS_MAP, help="Sundarbans vector layer.")
    p.add_argument("--north-arrow", type=Path, default=DEFAULT_NORTH_ARROW, help="North arrow SVG path.")
    p.add_argument(
        "--upazila-vectors",
        type=Path,
        nargs="+",
        default=DEFAULT_UPAZILA_VECTORS,
        help="Dissolved upazila boundary GeoPackages.",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    p.add_argument("--add-title", action="store_true", help="Add map title above the figure.")
    return p.parse_args()


def dm_formatter_lon(x, pos=None):
    hemi = "E" if x >= 0 else "W"
    deg = abs(x)
    d = int(deg)
    m = int(round((deg - d) * 60))
    if m == 60:
        d += 1
        m = 0
    return f"{d}°{m:02d}′{hemi}"


def dm_formatter_lat(x, pos=None):
    hemi = "N" if x >= 0 else "S"
    deg = abs(x)
    d = int(deg)
    m = int(round((deg - d) * 60))
    if m == 60:
        d += 1
        m = 0
    return f"{d}°{m:02d}′{hemi}"


def add_graticule(ax, color: str) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(dm_formatter_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(dm_formatter_lat))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, color=color, linestyle="--", linewidth=0.6, alpha=0.30, zorder=0)
    ax.tick_params(
        axis="both",
        labelsize=10,
        direction="out",
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
    )
    for label in ax.get_yticklabels():
        label.set_rotation(90)
        label.set_va("center")
        label.set_ha("center")


def meters_to_lat_degrees(meters: float) -> float:
    return meters / 111_320.0


def km_to_lon_degrees(km: float, lat_deg: float) -> float:
    cos_lat = math.cos(math.radians(lat_deg))
    if abs(cos_lat) < 1e-8:
        cos_lat = 1e-8
    return km / (111.320 * cos_lat)


def add_scalebar_2step(ax, length_km=150, location=(0.37, 0.06), fontsize=10):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + location[0] * (xlim[1] - xlim[0])
    y0 = ylim[0] + location[1] * (ylim[1] - ylim[0])
    mid_lat = 0.5 * (ylim[0] + ylim[1])
    total_deg = km_to_lon_degrees(length_km, mid_lat)
    step_deg = total_deg / 2.0
    bar_h = 0.014 * (ylim[1] - ylim[0])

    for i, face in enumerate(["black", "white"]):
        rect = Rectangle(
            (x0 + i * step_deg, y0),
            step_deg,
            bar_h,
            facecolor=face,
            edgecolor="black",
            linewidth=0.8,
            zorder=20,
        )
        ax.add_patch(rect)

    label_y = y0 + bar_h + 0.012 * (ylim[1] - ylim[0])
    ax.text(x0, label_y, "0", ha="center", va="bottom", fontsize=fontsize, zorder=21)
    ax.text(x0 + step_deg, label_y, "75", ha="center", va="bottom", fontsize=fontsize, zorder=21)
    ax.text(x0 + 2 * step_deg, label_y, "150 km", ha="center", va="bottom", fontsize=fontsize, zorder=21)


def load_svg_as_image(svg_path: Path, target_height_px: int = 220):
    if not svg_path.exists() or not HAVE_SVG:
        return None
    png_bytes = cairosvg.svg2png(url=str(svg_path), output_height=target_height_px)
    return Image.open(io.BytesIO(png_bytes))


def add_north_arrow(ax, svg_path: Path, xy=(0.92, 0.90), zoom=0.23):
    img = load_svg_as_image(svg_path, target_height_px=220)
    if img is None:
        ax.annotate(
            "N",
            xy=xy,
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            zorder=30,
        )
        ax.annotate(
            "",
            xy=(xy[0], xy[1] - 0.03),
            xytext=(xy[0], xy[1] - 0.14),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black"),
            zorder=30,
        )
        return
    imagebox = OffsetImage(np.asarray(img), zoom=zoom)
    ab = AnnotationBbox(
        imagebox,
        xy,
        xycoords="axes fraction",
        frameon=False,
        box_alignment=(0.5, 0.5),
        zorder=30,
    )
    ax.add_artist(ab)


def set_geographic_aspect_from_extent(ax, extent: tuple[float, float, float, float]) -> None:
    _, _, ymin, ymax = extent
    mean_lat = 0.5 * (ymin + ymax)
    cosv = np.cos(np.deg2rad(mean_lat))
    ax.set_aspect("equal" if abs(cosv) < 1e-8 else 1.0 / cosv)


def read_vector(path: Path, target_crs: str) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Vector file is empty: {path}")
    if gdf.crs is None:
        raise ValueError(f"Vector file has no CRS: {path}")
    return gdf.to_crs(target_crs)


def infer_label_from_vector(gdf: gpd.GeoDataFrame, path: Path) -> str:
    if "name" in gdf.columns:
        names = gdf["name"].dropna().astype(str)
        names = names[names.str.strip() != ""]
        if not names.empty:
            return names.iloc[0].strip().title()
    stem = path.stem
    if stem.endswith("_dissolved"):
        stem = stem[: -len("_dissolved")]
    return stem.replace("_", " ").title()


def expand_extent(extent: tuple[float, float, float, float], pad_frac: float = 0.035) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = extent
    xpad = (xmax - xmin) * pad_frac
    ypad = (ymax - ymin) * pad_frac
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def main() -> None:
    args = parse_args()
    zone_map = resolve_path(args.zone_map)
    sundarbans_map = resolve_path(args.sundarbans_map)
    north_arrow = resolve_path(args.north_arrow)
    output = resolve_path(args.output)
    upazila_paths = [resolve_path(path) for path in args.upazila_vectors]

    zones = read_vector(zone_map, TARGET_CRS)
    sundarbans = read_vector(sundarbans_map, TARGET_CRS)

    upazilas = []
    for path in upazila_paths:
        gdf = read_vector(path, TARGET_CRS)
        label = infer_label_from_vector(gdf, path)
        key = label.lower()
        upazilas.append((path, gdf, label, UPAZILA_COLORS.get(key, "#E76F51")))

    xmin, ymin, xmax, ymax = zones.total_bounds
    plot_extent = expand_extent((float(xmin), float(xmax), float(ymin), float(ymax)))

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI, facecolor=FIG_BG)
    ax.set_facecolor(SEA_COLOR)

    zones.plot(
        ax=ax,
        facecolor=ZONE_FILL,
        edgecolor=EDGE_COLOR,
        linewidth=1.2,
        alpha=0.72,
        zorder=2,
    )
    zones.boundary.plot(ax=ax, color=EDGE_COLOR, linewidth=1.6, zorder=4)

    sundarbans.plot(
        ax=ax,
        facecolor=SUNDARBANS_FILL,
        edgecolor=EDGE_COLOR,
        linewidth=1.2,
        alpha=0.78,
        zorder=5,
    )

    for _, gdf, label, color in upazilas:
        gdf.plot(
            ax=ax,
            facecolor=color,
            edgecolor=EDGE_COLOR,
            linewidth=1.2,
            alpha=0.58,
            zorder=7,
        )
        point = gdf.geometry.union_all().representative_point()
        txt = ax.text(
            point.x,
            point.y,
            label,
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            color=UPAZILA_TEXT_COLOR,
            zorder=9,
        )
        txt.set_path_effects([pe.Stroke(linewidth=3, foreground=FIG_BG), pe.Normal()])

    ax.set_xlim(plot_extent[0], plot_extent[1])
    ax.set_ylim(plot_extent[2], plot_extent[3])

    for _, row in zones.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        zone_key = str(row["zone"]).strip().lower() if "zone" in row else ""
        label = ZONE_LABELS.get(zone_key, zone_key.title())
        if not label:
            continue
        pt = geom.representative_point()
        dx, dy = ZONE_LABEL_OFFSETS.get(zone_key, (0.0, 0.0))
        dy = meters_to_lat_degrees(dy)
        txt = ax.text(
            pt.x + dx,
            pt.y + dy,
            label,
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            color=ZONE_TEXT_COLOR,
            zorder=8,
        )
        txt.set_path_effects([pe.Stroke(linewidth=3, foreground=FIG_BG), pe.Normal()])

    for _, row in sundarbans.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        label = str(row["zone"]).strip() if "zone" in row else "Sundarbans"
        pt = geom.representative_point()
        txt = ax.text(
            pt.x,
            pt.y,
            label,
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            color=SUNDARBANS_TEXT_COLOR,
            zorder=8,
        )
        txt.set_path_effects([pe.Stroke(linewidth=3, foreground=FIG_BG), pe.Normal()])

    xmid = 0.5 * (plot_extent[0] + plot_extent[1])
    bay_y = plot_extent[2] + 0.18 * (plot_extent[3] - plot_extent[2])
    bay_txt = ax.text(
        xmid,
        bay_y,
        "Bay of Bengal",
        fontsize=14,
        ha="center",
        va="center",
        color=BAY_TEXT_COLOR,
        zorder=6,
    )
    bay_txt.set_path_effects([pe.Stroke(linewidth=4, foreground=FIG_BG), pe.Normal()])

    set_geographic_aspect_from_extent(ax, plot_extent)
    add_graticule(ax, color=GRID_COLOR)
    if args.add_title:
        ax.set_title(MAP_TITLE, fontsize=15, pad=12, color=ZONE_TEXT_COLOR, fontweight="bold")
    ax.set_xlabel(X_AXIS_LABEL, fontsize=12, color=ZONE_TEXT_COLOR, labelpad=LONGITUDE_LABEL_PAD)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=12, color=ZONE_TEXT_COLOR)
    ax.tick_params(axis="both", colors=ZONE_TEXT_COLOR)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    add_north_arrow(ax, north_arrow, xy=(0.92, 0.90), zoom=0.23)
    add_scalebar_2step(ax, length_km=150, location=(0.37, 0.06), fontsize=10)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor(EDGE_COLOR)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, TIGHT_LAYOUT_BOTTOM, 1, 1))
    plt.savefig(output, dpi=FIG_DPI, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
