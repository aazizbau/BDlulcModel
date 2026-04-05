#!/usr/bin/env python3
"""
Create a coastal Bangladesh study-area map with:
- DSM color relief blended with hillshade
- Coastal zone boundaries and labels
- Bay of Bengal halo label
- North arrow
- Two-step scale bar (0, 75, 150 km)

Inputs
------
- assets/maps/bd_coastal_zones.gpkg
- data/processed/dsm/bd_coastal_aw3d30_v41_dsm_clipped.tif
- assets/maps/NorthArrow.svg
- assets/color_palette_coastal_lulc.json

Output
------
- outputs/figures/study_area_with_zones_n_dsm.png

Example
-------
python scripts/visualization/make_study_area_with_zones_n_dsm.py \
    --zone-map assets/maps/bd_coastal_zones.gpkg \
    --dsm-data data/processed/dsm/bd_coastal_aw3d30_v41_dsm_clipped.tif \
    --output outputs/figures/study_area_with_zones_n_dsm.png
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from PIL import Image

try:
    import cairosvg

    HAVE_SVG = True
except Exception:
    HAVE_SVG = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_CRS = "EPSG:4326"

DEFAULT_ZONE_MAP = Path("assets/maps/bd_coastal_zones.gpkg")
DEFAULT_DSM = Path("data/processed/dsm/bd_coastal_aw3d30_v41_dsm_clipped.tif")
DEFAULT_NORTH_ARROW = Path("assets/maps/NorthArrow.svg")
DEFAULT_PALETTE = Path("assets/color_palette_coastal_lulc.json")
DEFAULT_OUTPUT = Path("outputs/figures/study_area_with_zones_n_dsm.png")

FIGSIZE = (11, 9)
FIG_DPI = 300

ZONE_LABELS = {
    "western": "Western Zone",
    "central": "Central Zone",
    "eastern": "Eastern Zone",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make study area map with coastal zones and DSM.")
    p.add_argument("--zone-map", type=Path, default=DEFAULT_ZONE_MAP, help="Coastal zones vector layer.")
    p.add_argument("--dsm-data", type=Path, default=DEFAULT_DSM, help="Clipped DSM raster path.")
    p.add_argument("--north-arrow", type=Path, default=DEFAULT_NORTH_ARROW, help="North arrow SVG path.")
    p.add_argument("--palette", type=Path, default=DEFAULT_PALETTE, help="Palette JSON path.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    return p.parse_args()


def load_palette(path: Path) -> dict:
    return json.loads(path.read_text())


def degree_formatter_lon(x, pos=None):
    hemi = "E" if x >= 0 else "W"
    return f"{abs(int(round(x)))}°{hemi}"


def degree_formatter_lat(x, pos=None):
    hemi = "N" if x >= 0 else "S"
    return f"{abs(int(round(x)))}°{hemi}"


def add_graticule(ax, xticks, yticks, color: str) -> None:
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(FuncFormatter(degree_formatter_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(degree_formatter_lat))
    ax.grid(True, color=color, linestyle="--", linewidth=0.6, alpha=0.30, zorder=0)
    ax.tick_params(axis="both", labelsize=10, direction="out", top=True, right=True, labeltop=False, labelright=False)
    for label in ax.get_yticklabels():
        label.set_rotation(90)
        label.set_va("center")
        label.set_ha("center")


def km_to_lon_degrees(km: float, lat_deg: float) -> float:
    cos_lat = math.cos(math.radians(lat_deg))
    if abs(cos_lat) < 1e-8:
        cos_lat = 1e-8
    return km / (111.320 * cos_lat)


def add_scalebar_2step(ax, length_km=150, location=(0.40, 0.06), fontsize=10):
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
            zorder=10,
        )
        ax.add_patch(rect)

    label_y = y0 + bar_h + 0.012 * (ylim[1] - ylim[0])
    ax.text(x0, label_y, "0", ha="center", va="bottom", fontsize=fontsize, zorder=11)
    ax.text(x0 + step_deg, label_y, "75", ha="center", va="bottom", fontsize=fontsize, zorder=11)
    ax.text(x0 + 2 * step_deg, label_y, "150 km", ha="center", va="bottom", fontsize=fontsize, zorder=11)


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


def hillshade_from_array(arr: np.ndarray, azimuth=315.0, altitude=45.0) -> np.ndarray:
    x, y = np.gradient(arr.astype(np.float32))
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    az_rad = np.deg2rad(azimuth)
    alt_rad = np.deg2rad(altitude)
    hs = (
        np.sin(alt_rad) * np.sin(slope)
        + np.cos(alt_rad) * np.cos(slope) * np.cos(az_rad - aspect)
    )
    return np.clip((hs + 1.0) / 2.0, 0, 1)


def blend_relief(color_rgb: np.ndarray, hillshade: np.ndarray, alpha=0.38) -> np.ndarray:
    hill_rgb = np.dstack([hillshade, hillshade, hillshade])
    out = (1.0 - alpha) * color_rgb + alpha * hill_rgb
    return np.clip(out, 0, 1)


def main() -> None:
    args = parse_args()
    zone_map = resolve_path(args.zone_map)
    dsm_data = resolve_path(args.dsm_data)
    north_arrow = resolve_path(args.north_arrow)
    palette_path = resolve_path(args.palette)
    output = resolve_path(args.output)

    palette = load_palette(palette_path)
    colors = palette["colors"]

    fig_bg = colors["sand"]
    sea_color = colors["mist_gray"]
    grid_color = colors["deep_slate"]
    zone_fill = colors["dust_rose"]
    zone_edge = colors["deep_slate"]
    bay_text_color = colors["teal_blue"]
    zone_text_color = colors["deep_slate"]

    dsm_cmap = mcolors.LinearSegmentedColormap.from_list(
        "coastal_dsm",
        [
            colors["mist_gray"],
            colors["teal_blue"],
            colors["sand"],
            colors["dust_rose"],
            colors["ochre"],
            colors["olive"],
        ],
        N=256,
    )

    with rasterio.open(dsm_data) as ds:
        if ds.crs is None:
            raise ValueError("DSM raster has no CRS.")
        if ds.crs.to_string() != TARGET_CRS:
            raise ValueError(f"DSM raster CRS must be {TARGET_CRS}, found {ds.crs}.")
        dsm = ds.read(1).astype(np.float32)
        bounds = ds.bounds
        nodata = ds.nodata

    if nodata is not None:
        dsm[dsm == nodata] = np.nan

    valid = np.isfinite(dsm)
    if not valid.any():
        raise ValueError("DSM raster contains no finite values.")

    lo = float(np.nanpercentile(dsm, 2))
    hi = float(np.nanpercentile(dsm, 98))
    if hi <= lo:
        hi = lo + 1.0

    dsm_norm = np.clip((dsm - lo) / (hi - lo), 0, 1)
    dsm_rgb = dsm_cmap(np.nan_to_num(dsm_norm, nan=0.0))[..., :3]
    hillshade = hillshade_from_array(np.nan_to_num(dsm, nan=lo))
    relief = blend_relief(dsm_rgb, hillshade, alpha=0.38)
    relief[~valid] = mcolors.to_rgb(sea_color)

    zones = gpd.read_file(zone_map)
    if zones.empty:
        raise ValueError("Zone map is empty.")
    if zones.crs is None:
        raise ValueError("Zone map has no CRS.")
    zones = zones.to_crs(TARGET_CRS)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI, facecolor=fig_bg)
    ax.set_facecolor(sea_color)

    ax.imshow(
        relief,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        interpolation="nearest",
        zorder=1,
    )

    zones.plot(
        ax=ax,
        facecolor=zone_fill,
        edgecolor=zone_edge,
        linewidth=1.2,
        alpha=0.16,
        zorder=3,
    )
    zones.boundary.plot(ax=ax, color=zone_edge, linewidth=1.6, zorder=4)

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

    xmid = 0.5 * (bounds.left + bounds.right)
    bay_y = bounds.bottom + 0.18 * (bounds.top - bounds.bottom)
    bay_txt = ax.text(
        xmid,
        bay_y,
        "Bay of Bengal",
        fontsize=14,
        ha="center",
        va="center",
        color=bay_text_color,
        zorder=5,
    )
    bay_txt.set_path_effects([pe.Stroke(linewidth=4, foreground=fig_bg), pe.Normal()])

    xpad = 0.03 * (bounds.right - bounds.left)
    ypad = 0.03 * (bounds.top - bounds.bottom)
    ax.set_xlim(bounds.left - xpad, bounds.right + xpad)
    ax.set_ylim(bounds.bottom - ypad, bounds.top + ypad)
    mean_lat = 0.5 * (bounds.bottom + bounds.top)
    ax.set_aspect(1.0 / max(np.cos(np.deg2rad(mean_lat)), 1e-8))

    add_graticule(
        ax,
        xticks=np.arange(math.floor(bounds.left), math.ceil(bounds.right) + 1, 1),
        yticks=np.arange(math.floor(bounds.bottom), math.ceil(bounds.top) + 1, 1),
        color=grid_color,
    )

    add_north_arrow(ax, north_arrow, xy=(0.92, 0.90), zoom=0.23)
    add_scalebar_2step(ax, length_km=150, location=(0.37, 0.06), fontsize=10)

    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor(zone_edge)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
