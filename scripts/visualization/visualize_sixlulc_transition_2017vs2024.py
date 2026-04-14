#!/usr/bin/env python3
"""
Create six separate three-class LULC transition focus maps using the coastal
map style from the inferred LULC map.

Inputs
------
- outputs/inference/change_analysis/transition_code_2017_to_2024.tif
- assets/maps/bd_coastal_zones.gpkg
- assets/maps/NorthArrow.svg
- assets/color_palette_coastal_lulc.json

Outputs
-------
- outputs/figures/lulc_transition_2017_vs_2024_map1_urban_infrastructure_expansion.png
- outputs/figures/lulc_transition_2017_vs_2024_map2_rural_settlement_expansion.png
- outputs/figures/lulc_transition_2017_vs_2024_map3_productive_land_conversion.png
- outputs/figures/lulc_transition_2017_vs_2024_map4_water_expansion_erosion.png
- outputs/figures/lulc_transition_2017_vs_2024_map5_ecological_recovery.png
- outputs/figures/lulc_transition_2017_vs_2024_map6_ecological_degradation.png
- outputs/figures/lulc_transition_2017_vs_2024_<map_key>_stats.json
- outputs/figures/lulc_transition_2017_vs_2024_sixmaps_summary.json

Example
-------
python scripts/visualization/visualize_sixlulc_transition_2017vs2024.py
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
DEFAULT_OUTDIR = Path("outputs/figures")

FIGSIZE = (11, 9)
FIG_DPI = 300
MAX_DISPLAY_SIZE = 2800
DISPLAY_CHUNK_SIZE = 512
LONGITUDE_LABEL_PAD = 0
TIGHT_LAYOUT_BOTTOM = -0.04

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

MAP_SPECS = {
    "map1_urban_infrastructure_expansion": {
        "label": "Urban / Infrastructure Expansion",
        "color": "#D73027",
        "title": "Bangladesh Coastal LULC Transition (2017 to 2024) — Urban / Infrastructure Expansion",
    },
    "map2_rural_settlement_expansion": {
        "label": "Rural Settlement Expansion",
        "color": "#66A61E",
        "title": "Bangladesh Coastal LULC Transition (2017 to 2024) — Rural Settlement Expansion",
    },
    "map3_productive_land_conversion": {
        "label": "Productive Land Conversion",
        "color": "#BF9F74",
        "title": "Bangladesh Coastal LULC Transition (2017 to 2024) — Productive Land Conversion",
    },
    "map4_water_expansion_erosion": {
        "label": "Water Expansion / Erosion",
        "color": "#2C7BB6",
        "title": "Bangladesh Coastal LULC Transition (2017 to 2024) — Water Expansion / Erosion",
    },
    "map5_ecological_recovery": {
        "label": "Ecological Recovery / Natural Vegetation Expansion",
        "color": "#1A9850",
        "title": "Bangladesh Coastal LULC Transition (2017 to 2024) — Ecological Recovery / Natural Vegetation Expansion",
    },
    "map6_ecological_degradation": {
        "label": "Ecological Degradation / Vegetation Loss",
        "color": "#7B3294",
        "title": "Bangladesh Coastal LULC Transition (2017 to 2024) — Ecological Degradation / Vegetation Loss",
    },
}


def ts() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S %Z")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create six separate three-class LULC transition focus maps.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input transition raster.")
    parser.add_argument("--zone-map", type=Path, default=DEFAULT_ZONE_MAP, help="Coastal zones vector layer.")
    parser.add_argument("--north-arrow", type=Path, default=DEFAULT_NORTH_ARROW, help="North arrow SVG path.")
    parser.add_argument("--palette", type=Path, default=DEFAULT_PALETTE, help="Palette JSON path.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Directory for output PNG and JSON files.")
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


def decode_transition(code: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from_class = code // 100
    to_class = code % 100
    return from_class.astype(np.uint8), to_class.astype(np.uint8)


def build_maps(from_class: np.ndarray, to_class: np.ndarray, nodata_mask: np.ndarray) -> Dict[str, np.ndarray]:
    stable = (from_class == to_class) & (~nodata_mask)
    changed = (~stable) & (~nodata_mask)

    results: Dict[str, np.ndarray] = {}
    maps = {
        "map1_urban_infrastructure_expansion": changed & np.isin(to_class, [1, 3]),
        "map2_rural_settlement_expansion": changed & (to_class == 2),
        "map3_productive_land_conversion": changed & ((to_class == 6) | (to_class == 5) | ((to_class == 4) & (from_class != 6))),
        "map4_water_expansion_erosion": changed & ((to_class == 8) | ((to_class == 7) & (from_class != 8))),
        "map5_ecological_recovery": changed & ((to_class == 9) | ((to_class == 5) & (from_class != 9))),
        "map6_ecological_degradation": changed & ((from_class == 9) | ((from_class == 5) & (to_class != 9))),
    }

    for key, focus_mask in maps.items():
        out = np.full(from_class.shape, fill_value=3, dtype=np.uint8)
        out[stable] = 0
        out[changed] = 2
        out[focus_mask] = 1
        out[nodata_mask] = 3
        results[key] = out
    return results


def pixel_area_km2(transform) -> float:
    return abs(transform.a * transform.e) / 1_000_000.0


def compute_stats(arr: np.ndarray, px_area_km2: float, focus_label: str) -> Dict[str, Dict[str, float]]:
    valid = arr != 3
    total_valid = int(valid.sum())
    classes = {
        0: "Unchanged",
        1: focus_label,
        2: "Other Change",
    }
    stats: Dict[str, Dict[str, float]] = {}
    for cid, label in classes.items():
        count = int((arr == cid).sum())
        stats[str(cid)] = {
            "label": label,
            "pixel_count": count,
            "area_km2": count * px_area_km2,
            "percent_of_valid_area": (count / total_valid * 100.0) if total_valid > 0 else 0.0,
        }
    return stats


def init_stats_counter() -> dict[str, int]:
    return {"unchanged": 0, "focus": 0, "other_change": 0, "nodata": 0}


def compute_stats_from_counter(counter: dict[str, int], px_area_km2: float, focus_label: str) -> Dict[str, Dict[str, float]]:
    total_valid = counter["unchanged"] + counter["focus"] + counter["other_change"]
    class_map = {
        "0": ("Unchanged", counter["unchanged"]),
        "1": (focus_label, counter["focus"]),
        "2": ("Other Change", counter["other_change"]),
    }
    stats: Dict[str, Dict[str, float]] = {}
    for class_id, (label, count) in class_map.items():
        stats[class_id] = {
            "label": label,
            "pixel_count": count,
            "area_km2": count * px_area_km2,
            "percent_of_valid_area": (count / total_valid * 100.0) if total_valid > 0 else 0.0,
        }
    return stats


def counter_rows(
    counter: dict[str, int],
    px_area_km2: float,
    focus_label: str,
    map_key: str,
    scope_name: str,
    zone_key: str,
    zone_label: str,
) -> list[dict[str, object]]:
    total_valid = counter["unchanged"] + counter["focus"] + counter["other_change"]
    total_changed = counter["focus"] + counter["other_change"]
    class_defs = [
        ("0", "Unchanged", 0, counter["unchanged"]),
        ("1", focus_label, 1, counter["focus"]),
        ("2", "Other Change", 1, counter["other_change"]),
    ]
    rows: list[dict[str, object]] = []
    for class_id, class_label, is_changed_class, pixel_count in class_defs:
        area_km2 = pixel_count * px_area_km2
        percent_of_valid_area = (pixel_count / total_valid * 100.0) if total_valid > 0 else 0.0
        percent_of_changed_area = (pixel_count / total_changed * 100.0) if (total_changed > 0 and is_changed_class == 1) else 0.0
        rows.append(
            {
                "map_key": map_key,
                "focus_label": focus_label,
                "scope_name": scope_name,
                "zone_key": zone_key,
                "zone_label": zone_label,
                "class_id": class_id,
                "class_label": class_label,
                "is_changed_class": is_changed_class,
                "pixel_count": pixel_count,
                "area_km2": area_km2,
                "percent_of_valid_area": percent_of_valid_area,
                "percent_of_changed_area": percent_of_changed_area,
                "total_valid_pixels": total_valid,
                "total_changed_pixels": total_changed,
                "total_valid_area_km2": total_valid * px_area_km2,
                "total_changed_area_km2": total_changed * px_area_km2,
                "nodata_pixels": counter["nodata"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def update_focus_counters(
    counters: dict[str, dict[str, int]],
    from_class: np.ndarray,
    to_class: np.ndarray,
    nodata_mask: np.ndarray,
) -> None:
    stable = (from_class == to_class) & (~nodata_mask)
    changed = (~stable) & (~nodata_mask)

    focus_masks = {
        "map1_urban_infrastructure_expansion": changed & np.isin(to_class, [1, 3]),
        "map2_rural_settlement_expansion": changed & (to_class == 2),
        "map3_productive_land_conversion": changed & ((to_class == 6) | (to_class == 5) | ((to_class == 4) & (from_class != 6))),
        "map4_water_expansion_erosion": changed & ((to_class == 8) | ((to_class == 7) & (from_class != 8))),
        "map5_ecological_recovery": changed & ((to_class == 9) | ((to_class == 5) & (from_class != 9))),
        "map6_ecological_degradation": changed & ((from_class == 9) | ((from_class == 5) & (to_class != 9))),
    }

    unchanged_count = int(stable.sum())
    nodata_count = int(nodata_mask.sum())
    changed_count = int(changed.sum())

    for key, focus_mask in focus_masks.items():
        focus_count = int(focus_mask.sum())
        counters[key]["unchanged"] += unchanged_count
        counters[key]["focus"] += focus_count
        counters[key]["other_change"] += changed_count - focus_count
        counters[key]["nodata"] += nodata_count


def update_focus_counters_single(
    counter: dict[str, int],
    from_class: np.ndarray,
    to_class: np.ndarray,
    nodata_mask: np.ndarray,
    map_key: str,
) -> None:
    stable = (from_class == to_class) & (~nodata_mask)
    changed = (~stable) & (~nodata_mask)

    focus_masks = {
        "map1_urban_infrastructure_expansion": changed & np.isin(to_class, [1, 3]),
        "map2_rural_settlement_expansion": changed & (to_class == 2),
        "map3_productive_land_conversion": changed & ((to_class == 6) | (to_class == 5) | ((to_class == 4) & (from_class != 6))),
        "map4_water_expansion_erosion": changed & ((to_class == 8) | ((to_class == 7) & (from_class != 8))),
        "map5_ecological_recovery": changed & ((to_class == 9) | ((to_class == 5) & (from_class != 9))),
        "map6_ecological_degradation": changed & ((from_class == 9) | ((from_class == 5) & (to_class != 9))),
    }

    counter["unchanged"] += int(stable.sum())
    counter["focus"] += int(focus_masks[map_key].sum())
    counter["other_change"] += int(changed.sum()) - int(focus_masks[map_key].sum())
    counter["nodata"] += int(nodata_mask.sum())


def choose_zone_field(gdf: gpd.GeoDataFrame) -> str:
    for col in ["zone", "Zone", "ZONE", "zone_name", "Zone_Name", "ZONE_NAME", "name", "Name", "NAME"]:
        if col in gdf.columns:
            return col
    raise ValueError(f"Could not find a zone name field in {list(gdf.columns)}")


def build_zone_records(gdf: gpd.GeoDataFrame, zone_field: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for idx, (_, row) in enumerate(gdf.iterrows(), start=1):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        zone_key = str(row[zone_field]).strip().lower()
        zone_label = ZONE_LABELS.get(zone_key, str(row[zone_field]).strip())
        records.append(
            {
                "zone_id": idx,
                "zone_key": zone_key,
                "zone_label": zone_label,
                "geometry": geom,
            }
        )
    return records


def render_rgb(arr: np.ndarray, focus_color: str, sea_color: str) -> np.ndarray:
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
    rgb[:] = mcolors.to_rgb(sea_color)
    rgb[arr == 0] = mcolors.to_rgb("#D9D9D9")
    rgb[arr == 1] = mcolors.to_rgb(focus_color)
    rgb[arr == 2] = mcolors.to_rgb("#6E6E6E")
    return rgb


def render_single(
    arr: np.ndarray,
    bounds,
    raster_crs,
    zones: gpd.GeoDataFrame,
    zone_field: str,
    focus_label: str,
    focus_color: str,
    title: str,
    out_png: Path,
    north_arrow: Path,
    fig_bg: str,
    sea_color: str,
    grid_color: str,
    zone_edge: str,
    main_text_color: str,
    zone_text_color: str,
    bay_text_color: str,
    legend_face: str,
) -> None:
    rgb = render_rgb(arr, focus_color=focus_color, sea_color=sea_color)

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
        zone_key = str(row[zone_field]).strip().lower()
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
    ax.set_title(title, fontsize=15, pad=12, color=main_text_color, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=12, color=main_text_color, labelpad=LONGITUDE_LABEL_PAD)
    ax.set_ylabel("Latitude", fontsize=12, color=main_text_color)
    ax.tick_params(axis="both", colors=main_text_color)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    add_north_arrow(ax, north_arrow, xy=(0.92, 0.90), zoom=0.23)
    add_scalebar_2step(ax, length_km=150, location=(SCALEBAR_X_FRAC, SCALEBAR_Y_FRAC), fontsize=10)

    legend = ax.legend(
        handles=[
            Patch(facecolor="#D9D9D9", edgecolor="#314245", label="Unchanged"),
            Patch(facecolor=focus_color, edgecolor="#314245", label=focus_label),
            Patch(facecolor="#6E6E6E", edgecolor="#314245", label="Other Change"),
        ],
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
    for text in legend.get_texts():
        text.set_color(main_text_color)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor(main_text_color)

    plt.tight_layout(rect=(0, TIGHT_LAYOUT_BOTTOM, 1, 1))
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    zone_map = resolve_path(args.zone_map)
    north_arrow = resolve_path(args.north_arrow)
    palette_path = resolve_path(args.palette)
    outdir = resolve_path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

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
        nodata = src.nodata if src.nodata is not None else 0
        transform = src.transform
        profile = src.profile
        bounds = src.bounds
        raster_crs = src.crs
        px_area_km2 = pixel_area_km2(transform)

    log("Building six separate focus map statistics blockwise.")
    stats_counters = {key: init_stats_counter() for key in MAP_SPECS}
    zones = gpd.read_file(zone_map)
    if zones.empty:
        raise ValueError("Zone map is empty.")
    if zones.crs is None:
        raise ValueError("Zone map has no CRS.")
    zones = zones.to_crs(raster_crs)
    zone_field = choose_zone_field(zones)
    zone_records = build_zone_records(zones, zone_field)
    zone_shapes = [(record["geometry"], int(record["zone_id"])) for record in zone_records]
    zone_counters = {
        key: {int(record["zone_id"]): init_stats_counter() for record in zone_records}
        for key in MAP_SPECS
    }

    with rasterio.open(input_path) as src:
        for _, window in src.block_windows(1):
            block = src.read(1, window=window)
            nodata_mask = block == nodata
            from_class, to_class = decode_transition(block)
            update_focus_counters(stats_counters, from_class, to_class, nodata_mask)

            zone_block = rasterize(
                zone_shapes,
                out_shape=block.shape,
                transform=src.window_transform(window),
                fill=0,
                dtype="uint8",
                all_touched=False,
            )
            for record in zone_records:
                zone_id = int(record["zone_id"])
                zone_mask = zone_block == zone_id
                if not np.any(zone_mask):
                    continue
                zone_nodata_mask = nodata_mask | (~zone_mask)
                for map_key in MAP_SPECS:
                    update_focus_counters_single(
                        zone_counters[map_key][zone_id],
                        from_class,
                        to_class,
                        zone_nodata_mask,
                        map_key,
                    )

        preview = read_downsampled_raster_windowed(src, MAX_DISPLAY_SIZE, DISPLAY_CHUNK_SIZE)
    preview_nodata = preview == nodata
    preview_from, preview_to = decode_transition(preview)
    preview_maps = build_maps(preview_from, preview_to, preview_nodata)

    master_summary = {
        "input": str(input_path),
        "pixel_area_km2": px_area_km2,
        "raster_profile": {
            "width": int(profile["width"]),
            "height": int(profile["height"]),
            "crs": str(profile["crs"]),
            "nodata": float(nodata),
        },
        "outputs": {},
    }

    for key, spec in MAP_SPECS.items():
        png_path = outdir / f"lulc_transition_2017_vs_2024_{key}.png"
        json_path = outdir / f"lulc_transition_2017_vs_2024_{key}_stats.json"
        total_csv_path = outdir / f"lulc_transition_2017_vs_2024_{key}_total.csv"
        zonewise_csv_path = outdir / f"lulc_transition_2017_vs_2024_{key}_zonewise.csv"
        log(f"Rendering {key}: {png_path}")
        render_single(
            preview_maps[key],
            bounds=bounds,
            raster_crs=raster_crs,
            zones=zones,
            zone_field=zone_field,
            focus_label=spec["label"],
            focus_color=spec["color"],
            title=spec["title"],
            out_png=png_path,
            north_arrow=north_arrow,
            fig_bg=fig_bg,
            sea_color=sea_color,
            grid_color=grid_color,
            zone_edge=zone_edge,
            main_text_color=main_text_color,
            zone_text_color=zone_text_color,
            bay_text_color=bay_text_color,
            legend_face=legend_face,
        )
        stats = compute_stats_from_counter(stats_counters[key], px_area_km2, spec["label"])
        json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        total_rows = counter_rows(
            stats_counters[key],
            px_area_km2=px_area_km2,
            focus_label=spec["label"],
            map_key=key,
            scope_name="total",
            zone_key="total",
            zone_label="Whole Study Area",
        )
        zonewise_rows: list[dict[str, object]] = []
        for record in zone_records:
            zonewise_rows.extend(
                counter_rows(
                    zone_counters[key][int(record["zone_id"])],
                    px_area_km2=px_area_km2,
                    focus_label=spec["label"],
                    map_key=key,
                    scope_name=f"zone_{int(record['zone_id'])}",
                    zone_key=str(record["zone_key"]),
                    zone_label=str(record["zone_label"]),
                )
            )
        write_csv(total_csv_path, total_rows)
        write_csv(zonewise_csv_path, zonewise_rows)
        master_summary["outputs"][key] = {
            "png": str(png_path),
            "stats_json": str(json_path),
            "total_csv": str(total_csv_path),
            "zonewise_csv": str(zonewise_csv_path),
            "focus_label": spec["label"],
        }

    master_json = outdir / "lulc_transition_2017_vs_2024_sixmaps_summary.json"
    master_json.write_text(json.dumps(master_summary, indent=2), encoding="utf-8")
    log(f"Wrote summary: {master_json}")
    log("Done.")


if __name__ == "__main__":
    main()
