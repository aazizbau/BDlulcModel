#!/usr/bin/env python3
"""
Visualize mutually exclusive grouped LULC transitions between 2017 and 2024
using the coastal map style from the inferred LULC map.

Inputs
------
- outputs/inference/change_analysis/transition_code_2017_to_2024.tif
- assets/maps/bd_coastal_zones.gpkg
- assets/maps/NorthArrow.svg
- assets/color_palette_coastal_lulc.json

Outputs
-------
- outputs/figures/lulc_transition_2017_vs_2024_grouped_mutually_exclusive.png
- outputs/figures/lulc_transition_2017_vs_2024_grouped_mutually_exclusive_stats.json

Example
-------
python scripts/visualization/visualize_grouplulc_transition_2017vs2024.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
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


JST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("outputs/inference/change_analysis/transition_code_2017_to_2024.tif")
DEFAULT_ZONE_MAP = Path("assets/maps/bd_coastal_zones.gpkg")
DEFAULT_NORTH_ARROW = Path("assets/maps/NorthArrow.svg")
DEFAULT_PALETTE = Path("assets/color_palette_coastal_lulc.json")
DEFAULT_OUTPUT = Path("outputs/figures/lulc_transition_2017_vs_2024_grouped_mutually_exclusive.png")
DEFAULT_STATS = Path("outputs/figures/lulc_transition_2017_vs_2024_grouped_mutually_exclusive_stats.json")
DEFAULT_TOTAL_CSV = Path("outputs/figures/lulc_transition_2017_vs_2024_grouped_mutually_exclusive_total.csv")
DEFAULT_ZONEWISE_CSV = Path("outputs/figures/lulc_transition_2017_vs_2024_grouped_mutually_exclusive_zonewise.csv")

FIGSIZE = (11, 9)
FIG_DPI = 300
MAX_DISPLAY_SIZE = 2800
DISPLAY_CHUNK_SIZE = 512
LONGITUDE_LABEL_PAD = 0
TIGHT_LAYOUT_BOTTOM = -0.04
TRANSITION_NODATA = 0

MAP_TITLE = "Bangladesh Coastal LULC Transition (2017 to 2024) — Grouped Mutually Exclusive"
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

ZONE_LABELS = {
    "western": "Western Zone",
    "central": "Central Zone",
    "eastern": "Eastern Zone",
}

GROUP_INFO: Dict[int, Dict[str, str]] = {
    0: {"label": "Unchanged", "color": "#D9D9D9"},
    1: {"label": "Urban / Infrastructure Expansion", "color": "#D73027"},
    2: {"label": "Rural Settlement Expansion", "color": "#66A61E"},
    3: {"label": "Productive Land Conversion", "color": "#BF9F74"},
    4: {"label": "Water Expansion / Erosion", "color": "#2C7BB6"},
    5: {"label": "Ecological Recovery / Natural Vegetation Expansion", "color": "#1A9850"},
    6: {"label": "Ecological Degradation / Vegetation Loss", "color": "#7B3294"},
    7: {"label": "Other Change", "color": "#6E6E6E"},
}

CLASS_NAMES: Dict[int, str] = {
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


def ts() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S %Z")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a grouped mutually exclusive LULC transition map.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input transition raster.")
    parser.add_argument("--zone-map", type=Path, default=DEFAULT_ZONE_MAP, help="Coastal zones vector layer.")
    parser.add_argument("--north-arrow", type=Path, default=DEFAULT_NORTH_ARROW, help="North arrow SVG path.")
    parser.add_argument("--palette", type=Path, default=DEFAULT_PALETTE, help="Palette JSON path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    parser.add_argument("--stats-json", type=Path, default=DEFAULT_STATS, help="Output JSON stats path.")
    parser.add_argument("--total-csv", type=Path, default=DEFAULT_TOTAL_CSV, help="Output CSV path for total grouped analysis.")
    parser.add_argument("--zonewise-csv", type=Path, default=DEFAULT_ZONEWISE_CSV, help="Output CSV path for zone-wise grouped analysis.")
    parser.add_argument("--title", default=MAP_TITLE, help="Map title.")
    return parser.parse_args()


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


def add_scalebar_2step(ax, length_km=150, location=(0.38, 0.06), fontsize=10):
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


def decode_transition(code: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from_class = code // 100
    to_class = code % 100
    return from_class.astype(np.uint8), to_class.astype(np.uint8)


def assign_group_mutually_exclusive(from_class: np.ndarray, to_class: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    grouped = np.full(from_class.shape, fill_value=255, dtype=np.uint8)

    stable = (from_class == to_class) & (~nodata_mask)
    grouped[stable] = 0

    remaining = (grouped == 255) & (~nodata_mask)
    grouped[remaining & np.isin(to_class, [1, 3])] = 1

    remaining = (grouped == 255) & (~nodata_mask)
    grouped[remaining & (to_class == 2)] = 2

    remaining = (grouped == 255) & (~nodata_mask)
    grouped[remaining & np.isin(to_class, [8, 7])] = 4

    remaining = (grouped == 255) & (~nodata_mask)
    grouped[remaining & (to_class == 9)] = 5

    remaining = (grouped == 255) & (~nodata_mask)
    degrade = ((from_class == 9) & (to_class != 9)) | ((from_class == 5) & (~np.isin(to_class, [5, 9])))
    grouped[remaining & degrade] = 6

    remaining = (grouped == 255) & (~nodata_mask)
    productive = (to_class == 6) | (to_class == 5) | ((to_class == 4) & (from_class != 6))
    grouped[remaining & productive] = 3

    remaining = (grouped == 255) & (~nodata_mask)
    grouped[remaining] = 7

    grouped[nodata_mask] = 255
    return grouped


def pixel_area_km2(transform) -> float:
    return abs(transform.a * transform.e) / 1_000_000.0


def compute_group_stats(grouped: np.ndarray, px_area_km2: float) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    valid = grouped != 255
    total_valid = int(valid.sum())
    for gid in range(8):
        count = int((grouped == gid).sum())
        area = count * px_area_km2
        pct = (count / total_valid * 100.0) if total_valid > 0 else 0.0
        stats[str(gid)] = {
            "label": GROUP_INFO[gid]["label"],
            "pixel_count": count,
            "area_km2": area,
            "percent_of_valid_area": pct,
        }
    return stats


def compute_group_rows(
    grouped: np.ndarray,
    px_area_km2: float,
    scope_name: str,
    zone_key: str,
    zone_label: str,
) -> list[dict[str, object]]:
    valid = grouped != 255
    changed = valid & (grouped != 0)
    total_valid_pixels = int(valid.sum())
    total_changed_pixels = int(changed.sum())
    total_valid_area_km2 = total_valid_pixels * px_area_km2
    total_changed_area_km2 = total_changed_pixels * px_area_km2

    rows: list[dict[str, object]] = []
    for gid in range(8):
        pixel_count = int((grouped == gid).sum())
        area_km2 = pixel_count * px_area_km2
        percent_of_valid_area = (pixel_count / total_valid_pixels * 100.0) if total_valid_pixels > 0 else 0.0
        if gid == 0:
            percent_of_changed_area = 0.0
        else:
            percent_of_changed_area = (pixel_count / total_changed_pixels * 100.0) if total_changed_pixels > 0 else 0.0
        rows.append(
            {
                "scope_name": scope_name,
                "zone_key": zone_key,
                "zone_label": zone_label,
                "group_id": gid,
                "group_label": GROUP_INFO[gid]["label"],
                "group_color": GROUP_INFO[gid]["color"],
                "is_changed_group": int(gid != 0),
                "pixel_count": pixel_count,
                "area_km2": area_km2,
                "percent_of_valid_area": percent_of_valid_area,
                "percent_of_changed_area": percent_of_changed_area,
                "total_valid_pixels": total_valid_pixels,
                "total_changed_pixels": total_changed_pixels,
                "total_valid_area_km2": total_valid_area_km2,
                "total_changed_area_km2": total_changed_area_km2,
            }
        )
    return rows


def choose_zone_field(gdf: gpd.GeoDataFrame) -> str:
    for col in ["zone", "Zone", "ZONE", "zone_name", "Zone_Name", "ZONE_NAME", "name", "Name", "NAME"]:
        if col in gdf.columns:
            return col
    raise ValueError(f"Could not find a zone name field in {list(gdf.columns)}")


def build_zonewise_group_rows(
    grouped: np.ndarray,
    zones: gpd.GeoDataFrame,
    zone_field: str,
    transform,
    shape: tuple[int, int],
    px_area_km2: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for zone_idx, (_, row) in enumerate(zones.iterrows(), start=1):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        zone_key = str(row[zone_field]).strip().lower()
        zone_label = ZONE_LABELS.get(zone_key, str(row[zone_field]).strip())
        zone_mask = rasterize(
            [(geom, 1)],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        ).astype(bool, copy=False)
        zone_grouped = np.where(zone_mask, grouped, 255).astype(np.uint8, copy=False)
        rows.extend(
            compute_group_rows(
                zone_grouped,
                px_area_km2=px_area_km2,
                scope_name=f"zone_{zone_idx}",
                zone_key=zone_key,
                zone_label=zone_label,
            )
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def grouped_to_rgb(grouped: np.ndarray, nodata_rgb: tuple[float, float, float]) -> np.ndarray:
    rgb = np.zeros((grouped.shape[0], grouped.shape[1], 3), dtype=np.float32)
    rgb[:] = nodata_rgb
    for group_id in range(8):
        rgb[grouped == group_id] = mcolors.to_rgb(GROUP_INFO[group_id]["color"])
    return rgb


def legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=GROUP_INFO[group_id]["color"], edgecolor="#314245", label=GROUP_INFO[group_id]["label"])
        for group_id in range(8)
    ]


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    zone_map = resolve_path(args.zone_map)
    north_arrow = resolve_path(args.north_arrow)
    palette_path = resolve_path(args.palette)
    output_path = resolve_path(args.output)
    stats_path = resolve_path(args.stats_json)
    total_csv_path = resolve_path(args.total_csv)
    zonewise_csv_path = resolve_path(args.zonewise_csv)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    total_csv_path.parent.mkdir(parents=True, exist_ok=True)
    zonewise_csv_path.parent.mkdir(parents=True, exist_ok=True)

    palette = load_palette(palette_path)
    colors = palette["colors"]

    fig_bg = colors["sand"]
    sea_color = colors["mist_gray"]
    grid_color = colors["deep_slate"]
    zone_edge = "#2b2e07"
    main_text_color = colors["deep_slate"]
    zone_text_color = colors["coral"]
    bay_text_color = colors["teal_blue"]
    legend_face = "#FFF9EF"

    log(f"Reading raster: {input_path}")
    with rasterio.open(input_path) as src:
        arr = src.read(1)
        nodata = src.nodata if src.nodata is not None else 0
        transform = src.transform
        profile = src.profile
        raster_crs = src.crs
        bounds = src.bounds

    nodata_mask = arr == nodata
    from_class, to_class = decode_transition(arr)
    log("Assigning mutually exclusive grouped classes.")
    grouped = assign_group_mutually_exclusive(from_class, to_class, nodata_mask)

    px_area_km2 = pixel_area_km2(transform)
    stats = {
        "input": str(input_path),
        "output_png": str(output_path),
        "raster_profile": {
            "width": int(profile["width"]),
            "height": int(profile["height"]),
            "crs": str(profile["crs"]),
            "nodata": float(nodata),
        },
        "pixel_area_km2": px_area_km2,
        "group_definitions": {str(k): v for k, v in GROUP_INFO.items()},
        "group_stats": compute_group_stats(grouped, px_area_km2),
        "class_names": CLASS_NAMES,
    }

    with rasterio.open(input_path) as src:
        preview = read_downsampled_raster_windowed(src, MAX_DISPLAY_SIZE, DISPLAY_CHUNK_SIZE)

    preview_nodata = preview == nodata
    preview_from, preview_to = decode_transition(preview)
    preview_grouped = assign_group_mutually_exclusive(preview_from, preview_to, preview_nodata)
    rgb = grouped_to_rgb(preview_grouped, nodata_rgb=mcolors.to_rgb(sea_color))

    zones = gpd.read_file(zone_map)
    if zones.empty:
        raise ValueError("Zone map is empty.")
    if zones.crs is None:
        raise ValueError("Zone map has no CRS.")
    zones = zones.to_crs(raster_crs)
    zone_field = choose_zone_field(zones)

    total_csv_rows = compute_group_rows(
        grouped,
        px_area_km2=px_area_km2,
        scope_name="total",
        zone_key="total",
        zone_label="Whole Study Area",
    )
    zonewise_csv_rows = build_zonewise_group_rows(
        grouped,
        zones=zones,
        zone_field=zone_field,
        transform=transform,
        shape=grouped.shape,
        px_area_km2=px_area_km2,
    )

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
    ax.set_title(args.title, fontsize=15, pad=12, color=main_text_color, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=12, color=main_text_color, labelpad=LONGITUDE_LABEL_PAD)
    ax.set_ylabel("Latitude", fontsize=12, color=main_text_color)
    ax.tick_params(axis="both", colors=main_text_color)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    add_north_arrow(ax, north_arrow, xy=(0.92, 0.90), zoom=0.23)
    add_scalebar_2step(ax, length_km=150, location=(SCALEBAR_X_FRAC, SCALEBAR_Y_FRAC), fontsize=10)

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
        title="Grouped LULC Transition",
        title_fontsize=LEGEND_FONTSIZE + 1,
    )
    legend.set_zorder(12)
    legend.get_title().set_color(main_text_color)
    for text in legend.get_texts():
        text.set_color(main_text_color)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor(main_text_color)

    log(f"Writing figure: {output_path}")
    plt.tight_layout(rect=(0, TIGHT_LAYOUT_BOTTOM, 1, 1))
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig)

    log(f"Writing stats: {stats_path}")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(f"Writing total CSV: {total_csv_path}")
    write_csv(total_csv_path, total_csv_rows)
    log(f"Writing zone-wise CSV: {zonewise_csv_path}")
    write_csv(zonewise_csv_path, zonewise_csv_rows)
    log("Done.")


if __name__ == "__main__":
    main()
