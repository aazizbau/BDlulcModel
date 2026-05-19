#!/usr/bin/env python3
"""
Create a coastal Bangladesh study-area map with:
- ALOS DEM binned elevation color relief blended with hillshade
- Coastal zone boundaries and labels
- Sundarbans boundary and label
- Bay of Bengal halo label
- North arrow
- Two-step scale bar (0, 75, 150 km)
- Latitude / longitude labels and ticks

Inputs
------
- assets/maps/bd_coastal_zones.gpkg
- assets/maps/sundarbans.gpkg
- assets/maps/alos_bd_coastal_dem.tif
- assets/maps/NorthArrow.svg
- assets/color_palette_coastal_lulc.json

Output
------
- outputs/figures/study_area_with_zones_n_dem.png

Example
-------
python scripts/visualization/make_study_area_with_zones_n_dem.py \
    --zone-map assets/maps/bd_coastal_zones.gpkg \
    --dem-data assets/maps/alos_bd_coastal_dem.tif \
    --add-title \
    --output outputs/figures/study_area_with_zones_n_dem.png
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
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PIL import Image
from rasterio.windows import Window

try:
    import cairosvg

    HAVE_SVG = True
except Exception:
    HAVE_SVG = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_CRS = "EPSG:4326"

DEFAULT_ZONE_MAP = Path("assets/maps/bd_coastal_zones.gpkg")
DEFAULT_SUNDARBANS_MAP = Path("assets/maps/sundarbans.gpkg")
DEFAULT_DEM = Path("assets/maps/alos_bd_coastal_dem.tif")
DEFAULT_NORTH_ARROW = Path("assets/maps/NorthArrow.svg")
DEFAULT_PALETTE = Path("assets/color_palette_coastal_lulc.json")
DEFAULT_OUTPUT = Path("outputs/figures/study_area_with_zones_n_dem.png")

FIGSIZE = (11, 9)
FIG_DPI = 300
MAX_DISPLAY_SIZE = 2800
DISPLAY_CHUNK_SIZE = 512
MAP_TITLE = "Bangladesh Coastal Zones and ALOS DEM"
X_AXIS_LABEL = "Longitude"
Y_AXIS_LABEL = "Latitude"
COLORBAR_LABEL = "Elevation (m)"
LONGITUDE_LABEL_PAD = 0
TIGHT_LAYOUT_BOTTOM = -0.05

DEM_COLORS = [
    "#08306B",
    "#4292C6",
    "#A1D99B",
    "#31A354",
    "#FEE391",
    "#FEB24C",
    "#F03B20",
    "#BD0026",
    "#800026",
]
DEM_BINS = [-65, 0, 5, 10, 20, 30, 50, 100, 200, 350]

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
    p = argparse.ArgumentParser(description="Make study area map with coastal zones and ALOS DEM.")
    p.add_argument("--zone-map", type=Path, default=DEFAULT_ZONE_MAP, help="Coastal zones vector layer.")
    p.add_argument("--sundarbans-map", type=Path, default=DEFAULT_SUNDARBANS_MAP, help="Sundarbans vector layer.")
    p.add_argument("--dem-data", type=Path, default=DEFAULT_DEM, help="Clipped DEM raster path.")
    p.add_argument("--north-arrow", type=Path, default=DEFAULT_NORTH_ARROW, help="North arrow SVG path.")
    p.add_argument("--palette", type=Path, default=DEFAULT_PALETTE, help="Palette JSON path.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    p.add_argument("--add-title", action="store_true", help="Add map title above the figure.")
    return p.parse_args()


def load_palette(path: Path) -> dict:
    return json.loads(path.read_text())


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
    ax.tick_params(axis="both", labelsize=10, direction="out", top=True, right=True, labeltop=False, labelright=False)
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


def set_geographic_aspect_from_extent(ax, extent: tuple[float, float, float, float]) -> None:
    _, _, ymin, ymax = extent
    mean_lat = 0.5 * (ymin + ymax)
    cosv = np.cos(np.deg2rad(mean_lat))
    ax.set_aspect("equal" if abs(cosv) < 1e-8 else 1.0 / cosv)


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


def blend_relief(color_rgb: np.ndarray, hillshade: np.ndarray, alpha=0.30) -> np.ndarray:
    hill_rgb = np.dstack([hillshade, hillshade, hillshade])
    out = (1.0 - alpha) * color_rgb + alpha * hill_rgb
    return np.clip(out, 0, 1)


def read_downsampled_raster_windowed(
    ds: rasterio.io.DatasetReader,
    max_size: int,
    chunk_size: int,
) -> np.ndarray:
    src_h = ds.height
    src_w = ds.width

    scale = max(src_h / max_size, src_w / max_size, 1.0)
    dst_h = max(1, int(round(src_h / scale)))
    dst_w = max(1, int(round(src_w / scale)))

    out = np.empty((dst_h, dst_w), dtype=np.float32)

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
                resampling=rasterio.enums.Resampling.bilinear,
            ).astype(np.float32)
            out[row0:row1, col0:col1] = data

    return out


def valid_data_extent_from_preview(
    valid_mask: np.ndarray,
    bounds,
) -> tuple[float, float, float, float]:
    rows, cols = np.where(valid_mask)
    if rows.size == 0 or cols.size == 0:
        return bounds.left, bounds.right, bounds.bottom, bounds.top

    h, w = valid_mask.shape
    row_min = int(rows.min())
    row_max = int(rows.max()) + 1
    col_min = int(cols.min())
    col_max = int(cols.max()) + 1

    xres = (bounds.right - bounds.left) / w
    yres = (bounds.top - bounds.bottom) / h

    xmin = bounds.left + col_min * xres
    xmax = bounds.left + col_max * xres
    ymax = bounds.top - row_min * yres
    ymin = bounds.top - row_max * yres
    return xmin, xmax, ymin, ymax


def main() -> None:
    args = parse_args()
    zone_map = resolve_path(args.zone_map)
    sundarbans_map = resolve_path(args.sundarbans_map)
    dem_data = resolve_path(args.dem_data)
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
    sundarbans_text_color = colors["deep_slate"]

    dem_cmap = mcolors.ListedColormap(DEM_COLORS)
    dem_norm = mcolors.BoundaryNorm(DEM_BINS, ncolors=len(DEM_COLORS), clip=True)

    with rasterio.open(dem_data) as ds:
        if ds.crs is None:
            raise ValueError("DEM raster has no CRS.")
        if ds.crs.to_string() != TARGET_CRS:
            raise ValueError(f"DEM raster CRS must be {TARGET_CRS}, found {ds.crs}.")
        bounds = ds.bounds
        nodata = ds.nodata
        dem = read_downsampled_raster_windowed(
            ds,
            max_size=MAX_DISPLAY_SIZE,
            chunk_size=DISPLAY_CHUNK_SIZE,
        )

    if nodata is not None:
        dem[dem == nodata] = np.nan

    valid = np.isfinite(dem)
    if not valid.any():
        raise ValueError("DEM raster contains no finite values.")
    plot_extent = valid_data_extent_from_preview(valid, bounds)

    dem_rgb = dem_cmap(dem_norm(np.nan_to_num(dem, nan=DEM_BINS[0])))[..., :3]
    hillshade = hillshade_from_array(np.nan_to_num(dem, nan=float(np.nanmedian(dem))))
    relief = blend_relief(dem_rgb, hillshade, alpha=0.30)
    relief[~valid] = mcolors.to_rgb(sea_color)

    zones = gpd.read_file(zone_map)
    if zones.empty:
        raise ValueError("Zone map is empty.")
    if zones.crs is None:
        raise ValueError("Zone map has no CRS.")
    zones = zones.to_crs(TARGET_CRS)

    sundarbans = gpd.read_file(sundarbans_map)
    if sundarbans.empty:
        raise ValueError("Sundarbans vector is empty.")
    if sundarbans.crs is None:
        raise ValueError("Sundarbans vector has no CRS.")
    sundarbans = sundarbans.to_crs(TARGET_CRS)

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
    sundarbans.boundary.plot(ax=ax, color=zone_edge, linewidth=1.4, zorder=5)
    ax.set_xlim(plot_extent[0], plot_extent[1])
    ax.set_ylim(plot_extent[2], plot_extent[3])

    for _, row in zones.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        zone_key = str(row["zone"]).strip().lower()
        label = ZONE_LABELS.get(zone_key, zone_key.title())
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

    xmid = 0.5 * (plot_extent[0] + plot_extent[1])
    bay_y = plot_extent[2] + 0.18 * (plot_extent[3] - plot_extent[2])
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

    set_geographic_aspect_from_extent(ax, plot_extent)

    add_graticule(ax, color=grid_color)
    if args.add_title:
        ax.set_title(MAP_TITLE, fontsize=15, pad=12, color=zone_text_color, fontweight="bold")
    ax.set_xlabel(X_AXIS_LABEL, fontsize=12, color=zone_text_color, labelpad=LONGITUDE_LABEL_PAD)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=12, color=zone_text_color)
    ax.tick_params(axis="both", colors=zone_text_color)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    add_north_arrow(ax, north_arrow, xy=(0.92, 0.90), zoom=0.23)
    add_scalebar_2step(ax, length_km=150, location=(0.37, 0.06), fontsize=10)

    sm = ScalarMappable(norm=dem_norm, cmap=dem_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, boundaries=DEM_BINS, ticks=DEM_BINS, fraction=0.045, pad=0.03)
    cbar.set_label(COLORBAR_LABEL, rotation=90, color=zone_text_color, fontsize=11)
    cbar.ax.tick_params(labelsize=9, colors=zone_text_color)
    cbar.outline.set_edgecolor(zone_edge)
    cbar.outline.set_linewidth(0.9)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor(zone_edge)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, TIGHT_LAYOUT_BOTTOM, 1, 1))
    plt.savefig(output, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
