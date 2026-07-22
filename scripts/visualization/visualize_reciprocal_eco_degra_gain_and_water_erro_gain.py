#!/usr/bin/env python3
"""
Create reciprocal ecological and water/land transition maps for 2017-2024.

The script reads the transition-code raster used by
``visualize_sixlulc_transition_2017vs2024.py`` and combines its reciprocal
transition themes into two publication-style maps. It does not combine PNG
pixels; every map category and CSV quantity is calculated from the original
per-pixel transition codes.

Map 1: reciprocal ecological change
-----------------------------------
- Ecological Loss (yellow): the existing Map 6 degradation rule; changed
  pixels leaving Mangrove Forest (class 9), or leaving Tree-based
  Agroforestry & Orchard (class 5) for a class other than Mangrove Forest.
- Ecological Gain (deep green): the existing Map 5 recovery rule; changed
  pixels entering Mangrove Forest, or entering Tree-based Agroforestry &
  Orchard from a class other than Mangrove Forest.

Map 2: reciprocal water/land change
-----------------------------------
- Water Gain / Expansion (blue): the existing Map 4 rule; changed pixels
  entering Rivers & Estuarine Channels (class 8), or entering Canals &
  Drainage Network (class 7) from a class other than Rivers.
- Land Gain / Water Loss (dark maroon, #471515): the reciprocal rule; changed
  pixels leaving Rivers, or leaving Canals for a class other than Rivers.

Both maps retain Unchanged and Other Change background categories, coastal-zone
and Sundarbans boundaries, labels, graticules, scale bar, and north-arrow styling
from ``visualize_sixlulc_transition_2017vs2024.py``.

Inputs
------
- outputs/inference/change_analysis/transition_code_2017_to_2024.tif
- assets/maps/bd_coastal_zones.gpkg
- assets/maps/sundarbans.gpkg
- assets/maps/NorthArrow.svg
- assets/color_palette_coastal_lulc.json

Outputs
-------
- outputs/figures/lulc_transition_2017_vs_2024_reciprocal_ecological_loss_gain.png
- outputs/figures/lulc_transition_2017_vs_2024_reciprocal_water_land_gain.png
- outputs/figures/lulc_transition_2017_vs_2024_reciprocal_ecological_water_land.csv

The CSV contains category summaries, signed net-balance rows, and individual
class-to-class transition details for the whole study area and each coastal
zone. Area is calculated from the raster pixel dimensions and reported in km2.

Complete example run
--------------------
python scripts/visualization/visualize_reciprocal_eco_degra_gain_and_water_erro_gain.py \
    --output-plot-ecological outputs/figures/lulc_transition_2017_vs_2024_reciprocal_ecological_loss_gain.png \
    --output-plot-water-land outputs/figures/lulc_transition_2017_vs_2024_reciprocal_water_land_gain.png \
    --output-csv outputs/figures/lulc_transition_2017_vs_2024_reciprocal_ecological_water_land.csv

Add ``--add-title`` when titles should be embedded above the maps. Omit it for
thesis figures whose captions provide the titles.

Adapting to another AOI
-----------------------
Replace the transition raster and boundary vectors, then verify matching CRS,
extent, class IDs, nodata, and pixel units. If a different classification scheme
is used, update ``CLASS_NAMES`` and the reciprocal masks together. A projected
CRS with metre units is required for the default km2 area calculation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch
from rasterio.features import rasterize

from visualize_sixlulc_transition_2017vs2024 import (
    BAY_LABEL_X_FRAC,
    BAY_LABEL_Y_FRAC,
    DISPLAY_CHUNK_SIZE,
    FIGSIZE,
    FIG_DPI,
    LEGEND_BORDER_PAD,
    LEGEND_BOX_ALPHA,
    LEGEND_FONTSIZE,
    LEGEND_HANDLE_HEIGHT,
    LEGEND_HANDLE_LENGTH,
    LEGEND_LABEL_SPACING,
    LEGEND_X_FRAC,
    LEGEND_Y_FRAC,
    LONGITUDE_LABEL_PAD,
    MAX_DISPLAY_SIZE,
    SCALEBAR_X_FRAC,
    SCALEBAR_Y_FRAC,
    TIGHT_LAYOUT_BOTTOM,
    ZONE_LABEL_OFFSETS,
    ZONE_LABELS,
    add_graticule,
    add_north_arrow,
    add_scalebar_2step,
    build_zone_records,
    choose_zone_field,
    decode_transition,
    load_palette,
    log,
    pixel_area_km2,
    read_downsampled_raster_windowed,
    resolve_path,
    set_geographic_aspect,
)


DEFAULT_INPUT = Path(
    "outputs/inference/change_analysis/transition_code_2017_to_2024.tif"
)
DEFAULT_ZONE_MAP = Path("assets/maps/bd_coastal_zones.gpkg")
DEFAULT_SUNDARBANS_MAP = Path("assets/maps/sundarbans.gpkg")
DEFAULT_NORTH_ARROW = Path("assets/maps/NorthArrow.svg")
DEFAULT_PALETTE = Path("assets/color_palette_coastal_lulc.json")
DEFAULT_OUTPUT_ECOLOGICAL = Path(
    "outputs/figures/lulc_transition_2017_vs_2024_reciprocal_ecological_loss_gain.png"
)
DEFAULT_OUTPUT_WATER_LAND = Path(
    "outputs/figures/lulc_transition_2017_vs_2024_reciprocal_water_land_gain.png"
)
DEFAULT_OUTPUT_CSV = Path(
    "outputs/figures/lulc_transition_2017_vs_2024_reciprocal_ecological_water_land.csv"
)

UNCHANGED_COLOR = "#D9D9D9"
OTHER_CHANGE_COLOR = "#6E6E6E"
ECOLOGICAL_LOSS_COLOR = "#F2D64B"
ECOLOGICAL_GAIN_COLOR = "#0B5D1E"
WATER_GAIN_COLOR = "#2C7BB6"
LAND_GAIN_COLOR = "#471515"
NODATA_CATEGORY = 255

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

MAP_SPECS = {
    "reciprocal_ecological_loss_gain": {
        "map_label": "Reciprocal Ecological Loss and Gain",
        "category_1_label": "Ecological Loss",
        "category_1_direction": "loss",
        "category_1_color": ECOLOGICAL_LOSS_COLOR,
        "category_2_label": "Ecological Gain",
        "category_2_direction": "gain",
        "category_2_color": ECOLOGICAL_GAIN_COLOR,
        "net_label": "Net Ecological Gain (gain minus loss)",
        "net_multiplier_1": -1,
        "net_multiplier_2": 1,
        "title": "Bangladesh Coastal LULC Transition (2017 to 2024) - Ecological Loss and Gain",
    },
    "reciprocal_water_land_gain": {
        "map_label": "Reciprocal Water Gain and Land Gain",
        "category_1_label": "Water Gain / Expansion",
        "category_1_direction": "water_gain",
        "category_1_color": WATER_GAIN_COLOR,
        "category_2_label": "Land Gain / Water Loss",
        "category_2_direction": "land_gain",
        "category_2_color": LAND_GAIN_COLOR,
        "net_label": "Net Water Gain (water gain minus land gain)",
        "net_multiplier_1": 1,
        "net_multiplier_2": -1,
        "title": "Bangladesh Coastal LULC Transition (2017 to 2024) - Water Gain and Land Gain",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reciprocal ecological and water/land LULC transition maps."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--zone-map", type=Path, default=DEFAULT_ZONE_MAP)
    parser.add_argument("--sundarbans-map", type=Path, default=DEFAULT_SUNDARBANS_MAP)
    parser.add_argument("--north-arrow", type=Path, default=DEFAULT_NORTH_ARROW)
    parser.add_argument("--palette", type=Path, default=DEFAULT_PALETTE)
    parser.add_argument(
        "--output-plot-ecological", type=Path, default=DEFAULT_OUTPUT_ECOLOGICAL
    )
    parser.add_argument(
        "--output-plot-water-land", type=Path, default=DEFAULT_OUTPUT_WATER_LAND
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--add-title", action="store_true", help="Show a title above each map."
    )
    return parser.parse_args()


def reciprocal_masks(
    map_key: str,
    from_class: np.ndarray,
    to_class: np.ndarray,
    nodata_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return stable, changed, first-focus, and second-focus masks."""
    valid = (
        (~nodata_mask)
        & np.isin(from_class, list(CLASS_NAMES))
        & np.isin(to_class, list(CLASS_NAMES))
    )
    stable = valid & (from_class == to_class)
    changed = valid & (from_class != to_class)

    if map_key == "reciprocal_ecological_loss_gain":
        first = changed & (
            (from_class == 9) | ((from_class == 5) & (to_class != 9))
        )
        second = changed & (
            (to_class == 9) | ((to_class == 5) & (from_class != 9))
        )
    elif map_key == "reciprocal_water_land_gain":
        first = changed & (
            (to_class == 8) | ((to_class == 7) & (from_class != 8))
        )
        second = changed & (
            (from_class == 8) | ((from_class == 7) & (to_class != 8))
        )
    else:
        raise KeyError(f"Unknown reciprocal map key: {map_key}")

    overlap = first & second
    if np.any(overlap):
        raise ValueError(
            f"Reciprocal categories overlap for {map_key}: "
            f"{int(overlap.sum()):,} pixels"
        )
    return stable, changed, first, second


def build_category_map(
    map_key: str,
    from_class: np.ndarray,
    to_class: np.ndarray,
    nodata_mask: np.ndarray,
) -> np.ndarray:
    stable, changed, first, second = reciprocal_masks(
        map_key, from_class, to_class, nodata_mask
    )
    result = np.full(from_class.shape, NODATA_CATEGORY, dtype=np.uint8)
    result[stable] = 0
    result[changed] = 3
    result[first] = 1
    result[second] = 2
    return result


def init_counter() -> dict[str, object]:
    return {
        "category_counts": np.zeros(4, dtype=np.int64),
        "transition_counts": np.zeros((4, 11, 11), dtype=np.int64),
        "nodata_pixels": 0,
    }


def update_counter(
    counter: dict[str, object],
    map_key: str,
    from_class: np.ndarray,
    to_class: np.ndarray,
    nodata_mask: np.ndarray,
    inclusion_mask: np.ndarray | None = None,
) -> None:
    if inclusion_mask is None:
        inclusion_mask = np.ones(from_class.shape, dtype=bool)

    scoped_nodata = inclusion_mask & nodata_mask
    stable, changed, first, second = reciprocal_masks(
        map_key, from_class, to_class, nodata_mask | (~inclusion_mask)
    )
    categories = np.full(from_class.shape, -1, dtype=np.int8)
    categories[stable] = 0
    categories[changed] = 3
    categories[first] = 1
    categories[second] = 2

    valid = categories >= 0
    category_counts = counter["category_counts"]
    transition_counts = counter["transition_counts"]
    category_counts += np.bincount(categories[valid], minlength=4)

    encoded = (
        categories[valid].astype(np.int64) * 121
        + from_class[valid].astype(np.int64) * 11
        + to_class[valid].astype(np.int64)
    )
    transition_counts += np.bincount(encoded, minlength=4 * 121).reshape(4, 11, 11)
    counter["nodata_pixels"] += int(scoped_nodata.sum())


def category_definitions(spec: dict[str, object]) -> list[tuple[int, str, str, int]]:
    return [
        (0, "Unchanged", "unchanged", 0),
        (1, str(spec["category_1_label"]), str(spec["category_1_direction"]), 1),
        (2, str(spec["category_2_label"]), str(spec["category_2_direction"]), 1),
        (3, "Other Change", "other_change", 0),
    ]


def base_csv_row(
    *,
    map_key: str,
    spec: dict[str, object],
    scope_name: str,
    zone_key: str,
    zone_label: str,
    row_type: str,
    category_id: str,
    category_label: str,
    direction: str,
    from_class_id: int | str,
    from_class_name: str,
    to_class_id: int | str,
    to_class_name: str,
    pixel_count: int,
    px_area_km2: float,
    total_valid: int,
    total_changed: int,
    total_focus: int,
    nodata_pixels: int,
) -> dict[str, object]:
    area_km2 = pixel_count * px_area_km2
    is_changed = direction not in {"unchanged", "net_balance"}
    is_focus = direction not in {"unchanged", "other_change", "net_balance"}
    return {
        "map_key": map_key,
        "map_label": spec["map_label"],
        "scope_name": scope_name,
        "zone_key": zone_key,
        "zone_label": zone_label,
        "row_type": row_type,
        "category_id": category_id,
        "category_label": category_label,
        "direction": direction,
        "from_class_id": from_class_id,
        "from_class_name": from_class_name,
        "to_class_id": to_class_id,
        "to_class_name": to_class_name,
        "is_changed_category": int(is_changed),
        "is_reciprocal_focus": int(is_focus),
        "pixel_count": pixel_count,
        "area_km2": area_km2,
        "percent_of_valid_area": (
            pixel_count / total_valid * 100.0 if total_valid else 0.0
        ),
        "percent_of_changed_area": (
            pixel_count / total_changed * 100.0
            if total_changed and (is_changed or direction == "net_balance")
            else 0.0
        ),
        "percent_of_reciprocal_focus_area": (
            pixel_count / total_focus * 100.0
            if total_focus and (is_focus or direction == "net_balance")
            else 0.0
        ),
        "total_valid_pixels": total_valid,
        "total_changed_pixels": total_changed,
        "total_reciprocal_focus_pixels": total_focus,
        "total_valid_area_km2": total_valid * px_area_km2,
        "total_changed_area_km2": total_changed * px_area_km2,
        "total_reciprocal_focus_area_km2": total_focus * px_area_km2,
        "nodata_pixels": nodata_pixels,
    }


def counter_rows(
    counter: dict[str, object],
    map_key: str,
    spec: dict[str, object],
    scope_name: str,
    zone_key: str,
    zone_label: str,
    px_area_km2: float,
) -> list[dict[str, object]]:
    category_counts = counter["category_counts"]
    transition_counts = counter["transition_counts"]
    nodata_pixels = int(counter["nodata_pixels"])
    total_valid = int(category_counts.sum())
    total_changed = int(category_counts[1:].sum())
    total_focus = int(category_counts[1] + category_counts[2])
    rows: list[dict[str, object]] = []

    for category_id, label, direction, _ in category_definitions(spec):
        rows.append(
            base_csv_row(
                map_key=map_key,
                spec=spec,
                scope_name=scope_name,
                zone_key=zone_key,
                zone_label=zone_label,
                row_type="category_summary",
                category_id=str(category_id),
                category_label=label,
                direction=direction,
                from_class_id="",
                from_class_name="",
                to_class_id="",
                to_class_name="",
                pixel_count=int(category_counts[category_id]),
                px_area_km2=px_area_km2,
                total_valid=total_valid,
                total_changed=total_changed,
                total_focus=total_focus,
                nodata_pixels=nodata_pixels,
            )
        )

    net_count = (
        int(spec["net_multiplier_1"]) * int(category_counts[1])
        + int(spec["net_multiplier_2"]) * int(category_counts[2])
    )
    rows.append(
        base_csv_row(
            map_key=map_key,
            spec=spec,
            scope_name=scope_name,
            zone_key=zone_key,
            zone_label=zone_label,
            row_type="balance_summary",
            category_id="net",
            category_label=str(spec["net_label"]),
            direction="net_balance",
            from_class_id="",
            from_class_name="",
            to_class_id="",
            to_class_name="",
            pixel_count=net_count,
            px_area_km2=px_area_km2,
            total_valid=total_valid,
            total_changed=total_changed,
            total_focus=total_focus,
            nodata_pixels=nodata_pixels,
        )
    )

    definitions = {cid: (label, direction) for cid, label, direction, _ in category_definitions(spec)}
    for category_id in range(4):
        category_label, direction = definitions[category_id]
        for from_id in CLASS_NAMES:
            for to_id in CLASS_NAMES:
                count = int(transition_counts[category_id, from_id, to_id])
                if count == 0:
                    continue
                rows.append(
                    base_csv_row(
                        map_key=map_key,
                        spec=spec,
                        scope_name=scope_name,
                        zone_key=zone_key,
                        zone_label=zone_label,
                        row_type="transition_detail",
                        category_id=str(category_id),
                        category_label=category_label,
                        direction=direction,
                        from_class_id=from_id,
                        from_class_name=CLASS_NAMES[from_id],
                        to_class_id=to_id,
                        to_class_name=CLASS_NAMES[to_id],
                        pixel_count=count,
                        px_area_km2=px_area_km2,
                        total_valid=total_valid,
                        total_changed=total_changed,
                        total_focus=total_focus,
                        nodata_pixels=nodata_pixels,
                    )
                )
    return rows


def render_rgb(
    categories: np.ndarray,
    category_1_color: str,
    category_2_color: str,
    sea_color: str,
) -> np.ndarray:
    rgb = np.empty((*categories.shape, 3), dtype=np.float32)
    rgb[:] = mcolors.to_rgb(sea_color)
    rgb[categories == 0] = mcolors.to_rgb(UNCHANGED_COLOR)
    rgb[categories == 1] = mcolors.to_rgb(category_1_color)
    rgb[categories == 2] = mcolors.to_rgb(category_2_color)
    rgb[categories == 3] = mcolors.to_rgb(OTHER_CHANGE_COLOR)
    return rgb


def add_area_labels(
    ax,
    zones: gpd.GeoDataFrame,
    zone_field: str,
    sundarbans: gpd.GeoDataFrame,
    zone_text_color: str,
    sundarbans_text_color: str,
    figure_background: str,
) -> None:
    for _, row in zones.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        zone_key = str(row[zone_field]).strip().lower()
        label = ZONE_LABELS.get(zone_key, zone_key.title())
        point = geom.representative_point()
        dx, dy = ZONE_LABEL_OFFSETS.get(zone_key, (0.0, 0.0))
        text = ax.text(
            point.x + dx,
            point.y + dy,
            label,
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            color=zone_text_color,
            zorder=6,
        )
        text.set_path_effects(
            [pe.Stroke(linewidth=3, foreground=figure_background), pe.Normal()]
        )

    label_field = next(
        (
            field
            for field in ["zone", "name", "Name", "NAME"]
            if field in sundarbans.columns
        ),
        None,
    )
    for _, row in sundarbans.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        label = str(row[label_field]).strip() if label_field else "Sundarbans"
        point = geom.representative_point()
        text = ax.text(
            point.x,
            point.y,
            label,
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            color=sundarbans_text_color,
            zorder=6,
        )
        text.set_path_effects(
            [pe.Stroke(linewidth=3, foreground=figure_background), pe.Normal()]
        )


def render_map(
    categories: np.ndarray,
    spec: dict[str, object],
    output_path: Path,
    bounds,
    raster_crs,
    zones: gpd.GeoDataFrame,
    zone_field: str,
    sundarbans: gpd.GeoDataFrame,
    north_arrow: Path,
    colors: dict[str, str],
    add_title: bool,
) -> None:
    figure_background = colors["sand"]
    sea_color = colors["mist_gray"]
    main_text_color = colors["deep_slate"]
    zone_edge = "#2b2e07"
    legend_face = "#FFF9EF"
    rgb = render_rgb(
        categories,
        str(spec["category_1_color"]),
        str(spec["category_2_color"]),
        sea_color,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI, facecolor=figure_background)
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
    add_area_labels(
        ax,
        zones,
        zone_field,
        sundarbans,
        colors["coral"],
        colors["deep_slate"],
        figure_background,
    )

    bay_x = bounds.left + BAY_LABEL_X_FRAC * (bounds.right - bounds.left)
    bay_y = bounds.bottom + BAY_LABEL_Y_FRAC * (bounds.top - bounds.bottom)
    bay_text = ax.text(
        bay_x,
        bay_y,
        "Bay of Bengal",
        fontsize=14,
        ha="center",
        va="center",
        color=colors["teal_blue"],
        zorder=5,
    )
    bay_text.set_path_effects(
        [pe.Stroke(linewidth=4, foreground=figure_background), pe.Normal()]
    )

    set_geographic_aspect(ax, bounds)
    add_graticule(ax, color=colors["deep_slate"], src_crs=raster_crs)
    if add_title:
        ax.set_title(
            str(spec["title"]),
            fontsize=15,
            pad=12,
            color=main_text_color,
            fontweight="bold",
        )
    ax.set_xlabel(
        "Longitude", fontsize=12, color=main_text_color, labelpad=LONGITUDE_LABEL_PAD
    )
    ax.set_ylabel("Latitude", fontsize=12, color=main_text_color)
    ax.tick_params(axis="both", colors=main_text_color)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    add_north_arrow(ax, north_arrow, xy=(0.92, 0.90), zoom=0.23)
    add_scalebar_2step(
        ax,
        length_km=150,
        location=(SCALEBAR_X_FRAC, SCALEBAR_Y_FRAC),
        fontsize=10,
    )

    legend = ax.legend(
        handles=[
            Patch(
                facecolor=UNCHANGED_COLOR,
                edgecolor=main_text_color,
                label="Unchanged",
            ),
            Patch(
                facecolor=str(spec["category_1_color"]),
                edgecolor=main_text_color,
                label=str(spec["category_1_label"]),
            ),
            Patch(
                facecolor=str(spec["category_2_color"]),
                edgecolor=main_text_color,
                label=str(spec["category_2_label"]),
            ),
            Patch(
                facecolor=OTHER_CHANGE_COLOR,
                edgecolor=main_text_color,
                label="Other Change",
            ),
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
    fig.savefig(
        output_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor=figure_background,
    )
    plt.close(fig)


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No reciprocal transition statistics were generated.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    zone_path = resolve_path(args.zone_map)
    sundarbans_path = resolve_path(args.sundarbans_map)
    north_arrow = resolve_path(args.north_arrow)
    palette_path = resolve_path(args.palette)
    output_paths = {
        "reciprocal_ecological_loss_gain": resolve_path(args.output_plot_ecological),
        "reciprocal_water_land_gain": resolve_path(args.output_plot_water_land),
    }
    output_csv = resolve_path(args.output_csv)

    for path in [
        input_path,
        zone_path,
        sundarbans_path,
        north_arrow,
        palette_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    for path in [*output_paths.values(), output_csv]:
        path.parent.mkdir(parents=True, exist_ok=True)

    colors = load_palette(palette_path)["colors"]
    zones = gpd.read_file(zone_path)
    sundarbans = gpd.read_file(sundarbans_path)
    if zones.empty or sundarbans.empty:
        raise ValueError("Zone and Sundarbans vectors must contain features.")
    if zones.crs is None or sundarbans.crs is None:
        raise ValueError("Zone and Sundarbans vectors must have CRS metadata.")

    total_counters = {key: init_counter() for key in MAP_SPECS}
    log(f"Reading transition raster: {input_path}")
    with rasterio.open(input_path) as source:
        if source.crs is None:
            raise ValueError("Transition raster has no CRS metadata.")
        raster_crs = source.crs
        bounds = source.bounds
        nodata = source.nodata if source.nodata is not None else 0
        px_area_km2 = pixel_area_km2(source.transform)
        zones = zones.to_crs(raster_crs)
        sundarbans = sundarbans.to_crs(raster_crs)
        zone_field = choose_zone_field(zones)
        zone_records = build_zone_records(zones, zone_field)
        zone_shapes = [
            (record["geometry"], int(record["zone_id"])) for record in zone_records
        ]
        zone_counters = {
            map_key: {
                int(record["zone_id"]): init_counter() for record in zone_records
            }
            for map_key in MAP_SPECS
        }

        log("Calculating reciprocal category and transition statistics blockwise.")
        for _, window in source.block_windows(1):
            block = source.read(1, window=window)
            nodata_mask = block == nodata
            from_class, to_class = decode_transition(block)
            for map_key in MAP_SPECS:
                update_counter(
                    total_counters[map_key],
                    map_key,
                    from_class,
                    to_class,
                    nodata_mask,
                )

            zone_block = rasterize(
                zone_shapes,
                out_shape=block.shape,
                transform=source.window_transform(window),
                fill=0,
                dtype="uint8",
                all_touched=False,
            )
            for record in zone_records:
                zone_id = int(record["zone_id"])
                inclusion_mask = zone_block == zone_id
                if not np.any(inclusion_mask):
                    continue
                for map_key in MAP_SPECS:
                    update_counter(
                        zone_counters[map_key][zone_id],
                        map_key,
                        from_class,
                        to_class,
                        nodata_mask,
                        inclusion_mask,
                    )

        preview_codes = read_downsampled_raster_windowed(
            source, MAX_DISPLAY_SIZE, DISPLAY_CHUNK_SIZE
        )

    preview_nodata = preview_codes == nodata
    preview_from, preview_to = decode_transition(preview_codes)
    csv_rows: list[dict[str, object]] = []
    for map_key, spec in MAP_SPECS.items():
        categories = build_category_map(
            map_key, preview_from, preview_to, preview_nodata
        )
        output_path = output_paths[map_key]
        log(f"Rendering {map_key}: {output_path}")
        render_map(
            categories,
            spec,
            output_path,
            bounds,
            raster_crs,
            zones,
            zone_field,
            sundarbans,
            north_arrow,
            colors,
            args.add_title,
        )
        csv_rows.extend(
            counter_rows(
                total_counters[map_key],
                map_key,
                spec,
                "total",
                "total",
                "Whole Study Area",
                px_area_km2,
            )
        )
        for record in zone_records:
            csv_rows.extend(
                counter_rows(
                    zone_counters[map_key][int(record["zone_id"])],
                    map_key,
                    spec,
                    f"zone_{int(record['zone_id'])}",
                    str(record["zone_key"]),
                    str(record["zone_label"]),
                    px_area_km2,
                )
            )

    log(f"Writing analysis CSV: {output_csv}")
    write_csv(output_csv, csv_rows)
    log("Done.")


if __name__ == "__main__":
    main()
