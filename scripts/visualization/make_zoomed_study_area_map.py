#!/usr/bin/env python3
"""
Create a publication-style zoomed study area map:
- Left panel: South Asia with Bangladesh highlighted
- Right panel: Bangladesh with 15 coastal districts highlighted and labeled

Inputs
------
- assets/maps/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp
- assets/maps/bgd_adm0.gpkg
- assets/maps/bd_coastal_districts.gpkg
- assets/maps/bd_coastal_map_solid_gp.gpkg
- assets/maps/NorthArrow.svg

Output
------
- outputs/figures/study_area_map2.png
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from matplotlib.ticker import FuncFormatter
from PIL import Image

try:
    import cairosvg

    HAS_CAIROSVG = True
except Exception:
    HAS_CAIROSVG = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


# ============================================================
# Paths
# ============================================================
WORLD_SHP = Path("assets/maps/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp")
BGD_ADM0 = Path("assets/maps/bgd_adm0.gpkg")
COASTAL_DISTRICTS = Path("assets/maps/bd_coastal_districts.gpkg")
COASTAL_SOLID = Path("assets/maps/bd_coastal_map_solid_gp.gpkg")
NORTH_ARROW_SVG = Path("assets/maps/NorthArrow.svg")
OUT_PNG = Path("outputs/figures/study_area_map2.png")


# ============================================================
# Figure / typography
# ============================================================
FIG_DPI = 300
FIGSIZE = (14, 9)

GRID_COLOR = "#7e8c96"
GRID_ALPHA = 0.45
GRID_LW = 0.6

TICK_FS = 10
DISTRICT_FS = 7.2
COUNTRY_FS = 9
OUTSIDE_FS = 10
LEGEND_FS = 10


# ============================================================
# Theme colors
# Adjust these later for other themes
# ============================================================
# # Common / background
# FIG_BG_COLOR = "#e9e2cf"
#
# # Panel (a): South Asia
# PANEL_A_SEA_COLOR = "#a9c3cf"
# PANEL_A_COUNTRY_FILL = "#efe9d7"
# PANEL_A_COUNTRY_STROKE = "#4a5068"
# PANEL_A_BGD_FILL = "#c78a6a"
# PANEL_A_STUDY_FILL = "#7a6434"
# PANEL_A_STUDY_STROKE = "#4a5068"
# PANEL_A_LABEL_COLOR = "#374151"
# PANEL_A_LABEL_HALO = PANEL_A_SEA_COLOR
# PANEL_A_BAY_LABEL_COLOR = "#32465a"
#
# # Panel (b): Bangladesh zoom
# PANEL_B_BG_COLOR = "#e7e7e7"
# PANEL_B_COUNTRY_FILL = "#c78a6a"
# PANEL_B_COUNTRY_STROKE = "#5b5248"
# PANEL_B_STUDY_FILL = "#7a6434"
# PANEL_B_DISTRICT_BOUNDARY = "#d6d08a"
# PANEL_B_STUDY_OUTER_STROKE = "#c42026"
# PANEL_B_OUTSIDE_LABEL_COLOR = "#444444"
# PANEL_B_DISTRICT_LABEL_COLOR = "#111111"
# PANEL_B_DISTRICT_LABEL_HALO = "#ffffff"
#
# # Legend
# LEGEND_FACE = "#f2f2f2"
# LEGEND_EDGE = "#555555"

# ============================================================
# Theme colors
# Adjust these later for other themes
# Source palette:
# #96ceb4  #ffcc5c  #ff6f69  #ced07d  #0e9aa7
# "outputs/study_area_map2.png"
# ============================================================
# Common / background
# FIG_BG_COLOR = "#f7f3e8"
#
# # Panel (a): South Asia
# PANEL_A_SEA_COLOR = "#96ceb4"
# PANEL_A_COUNTRY_FILL = "#f7f3e8"
# PANEL_A_COUNTRY_STROKE = "#0e9aa7"
# PANEL_A_BGD_FILL = "#ffcc5c"
# PANEL_A_STUDY_FILL = "#ff6f69"
# PANEL_A_STUDY_STROKE = "#0e9aa7"
# PANEL_A_LABEL_COLOR = "#2f4858"
# PANEL_A_LABEL_HALO = "#96ceb4"
# PANEL_A_BAY_LABEL_COLOR = "#0e9aa7"
#
# # Panel (b): Bangladesh zoom
# PANEL_B_BG_COLOR = "#f2f4ef"
# PANEL_B_COUNTRY_FILL = "#ffcc5c"
# PANEL_B_COUNTRY_STROKE = "#0e9aa7"
# PANEL_B_STUDY_FILL = "#ced07d"
# PANEL_B_DISTRICT_BOUNDARY = "#fff7dd"
# PANEL_B_STUDY_OUTER_STROKE = "#ff6f69"
# PANEL_B_OUTSIDE_LABEL_COLOR = "#4a4a4a"
# PANEL_B_DISTRICT_LABEL_COLOR = "#1f1f1f"
# PANEL_B_DISTRICT_LABEL_HALO = "#ffffff"
#
# # Legend
# LEGEND_FACE = "#fffdf7"
# LEGEND_EDGE = "#0e9aa7"

# ============================================================
# Theme colors
# Adjust these later for other themes
# Source palette:
# #F7CFD8  #F4F8D3  #A6D6D6  #8E7DBE
# "outputs/study_area_map3.png"
# ============================================================
# Common / background
# FIG_BG_COLOR = "#f8f3e8"
#
# # Panel (a): South Asia
# PANEL_A_SEA_COLOR = "#A6D6D6"
# PANEL_A_COUNTRY_FILL = "#F4F8D3"
# PANEL_A_COUNTRY_STROKE = "#8E7DBE"
# PANEL_A_BGD_FILL = "#F7CFD8"
# PANEL_A_STUDY_FILL = "#8E7DBE"
# PANEL_A_STUDY_STROKE = "#8E7DBE"
# PANEL_A_LABEL_COLOR = "#4b4370"
# PANEL_A_LABEL_HALO = "#A6D6D6"
# PANEL_A_BAY_LABEL_COLOR = "#3f6f78"
#
# # Panel (b): Bangladesh zoom
# PANEL_B_BG_COLOR = "#f2f2f2"
# PANEL_B_COUNTRY_FILL = "#F7CFD8"
# PANEL_B_COUNTRY_STROKE = "#6f6597"
# PANEL_B_STUDY_FILL = "#8E7DBE"
# PANEL_B_DISTRICT_BOUNDARY = "#F4F8D3"
# PANEL_B_STUDY_OUTER_STROKE = "#A6D6D6"
# PANEL_B_OUTSIDE_LABEL_COLOR = "#4a4a4a"
# PANEL_B_DISTRICT_LABEL_COLOR = "#111111"
# PANEL_B_DISTRICT_LABEL_HALO = "#ffffff"
#
# # Legend
# LEGEND_FACE = "#fffdf7"
# LEGEND_EDGE = "#8E7DBE"

# ============================================================
# Theme colors
# Adjust these later for other themes
# Source palette:
# #F7CFD8  #F4F8D3  #A6D6D6  #8E7DBE
# "outputs/study_area_map4.png"
# ============================================================
# Common / background
# FIG_BG_COLOR = "#fbf8ef"
#
# # Panel (a): South Asia
# PANEL_A_SEA_COLOR = "#A6D6D6"
# PANEL_A_COUNTRY_FILL = "#f7f9e8"
# PANEL_A_COUNTRY_STROKE = "#8E7DBE"
# PANEL_A_BGD_FILL = "#F7CFD8"
# PANEL_A_STUDY_FILL = "#d8c9ee"
# PANEL_A_STUDY_STROKE = "#8E7DBE"
# PANEL_A_LABEL_COLOR = "#4f4a68"
# PANEL_A_LABEL_HALO = "#A6D6D6"
# PANEL_A_BAY_LABEL_COLOR = "#46737c"
#
# # Panel (b): Bangladesh zoom
# PANEL_B_BG_COLOR = "#f3f3f3"
# PANEL_B_COUNTRY_FILL = "#F7CFD8"
# PANEL_B_COUNTRY_STROKE = "#7569a0"
# PANEL_B_STUDY_FILL = "#b7aad8"
# PANEL_B_DISTRICT_BOUNDARY = "#F4F8D3"
# PANEL_B_STUDY_OUTER_STROKE = "#8E7DBE"
# PANEL_B_OUTSIDE_LABEL_COLOR = "#4a4a4a"
# PANEL_B_DISTRICT_LABEL_COLOR = "#111111"
# PANEL_B_DISTRICT_LABEL_HALO = "#ffffff"
#
# # Legend
# LEGEND_FACE = "#fffef9"
# LEGEND_EDGE = "#8E7DBE"

# ============================================================
# Theme colors
# Adjust these later for other themes
# Source palette:
# #B6B5B8, #314245, #3D848F, #FBEDD4, #DBA796,
# #FF8973, #BF9F74, #808F54
# "outputs/study_area_map5.png"
# ============================================================
# Common / background
FIG_BG_COLOR = "#FBEDD4"

# Panel (a): South Asia
PANEL_A_SEA_COLOR = "#B6B5B8"
PANEL_A_COUNTRY_FILL = "#FBEDD4"
PANEL_A_COUNTRY_STROKE = "#314245"
PANEL_A_BGD_FILL = "#DBA796"
PANEL_A_STUDY_FILL = "#BF9F74"
PANEL_A_STUDY_STROKE = "#314245"
PANEL_A_LABEL_COLOR = "#314245"
PANEL_A_LABEL_HALO = "#B6B5B8"
PANEL_A_BAY_LABEL_COLOR = "#3D848F"

# Panel (b): Bangladesh zoom
PANEL_B_BG_COLOR = "#f3efe8"
PANEL_B_COUNTRY_FILL = "#DBA796"
PANEL_B_COUNTRY_STROKE = "#314245"
PANEL_B_STUDY_FILL = "#BF9F74"
PANEL_B_DISTRICT_BOUNDARY = "#FBEDD4"
PANEL_B_STUDY_OUTER_STROKE = "#FF8973"
PANEL_B_OUTSIDE_LABEL_COLOR = "#314245"
PANEL_B_DISTRICT_LABEL_COLOR = "#111111"
PANEL_B_DISTRICT_LABEL_HALO = "#ffffff"

# Legend
LEGEND_FACE = "#fffaf2"
LEGEND_EDGE = "#314245"


# ============================================================
# Easy-to-tune style / position variables
# ============================================================
SHOW_INDONESIA_LABEL = False

PANEL_A_SCALE_X0 = 73.0
PANEL_A_SCALE_Y0 = 0.98

PANEL_B_LEGEND_X = 87.70
PANEL_B_LEGEND_Y = 20.50

STUDY_AREA_EDGE_WIDTH = 0.85
COASTAL_OUTER_EDGE_WIDTH = 1.8


COUNTRY_LABEL_RULES = {
    "India": "India",
    "China": "China",
    "Myanmar": "MMR",
    "Nepal": "NPL",
    "Bhutan": "BTN",
    "Pakistan": "PAK",
    "Afghanistan": "AFG",
    "Sri Lanka": "LKA",
    "Thailand": "THA",
    "Laos": "LAO",
    "Cambodia": "KHM",
    "Vietnam": "VNM",
    "Malaysia": "MYS",
    "Indonesia": "IDN",
    "Mongolia": "MNG",
    "Kazakhstan": "KAZ",
    "Kyrgyzstan": "KGZ",
    "Tajikistan": "TJK",
    "Russia": "RUS",
}


DISTRICT_LABEL_OFFSETS = {
    "Satkhira": (-0.06, 0.23),
    "Khulna": (0.00, 0.03),
    "Bagerhat": (0.02, -0.17),
    "Pirojpur": (0.05, 0.05),
    "Jhalokati": (0.02, 0.16),
    "Barishal": (-0.02, 0.16),
    "Bhola": (0.10, -0.02),
    "Patuakhali": (0.10, -0.04),
    "Barguna": (-0.02, -0.12),
    "Lakshmipur": (0.02, 0.16),
    "Noakhali": (0.16, -0.06),
    "Feni": (0.05, 0.00),
    "Chandpur": (-0.08, 0.10),
    "Chattogram": (0.05, -0.02),
    "Cox's Bazar": (0.12, -0.16),
}


def degree_formatter_lon(x, pos=None):
    hemi = "E" if x >= 0 else "W"
    return f"{abs(int(round(x)))}°{hemi}"


def degree_formatter_lat(x, pos=None):
    hemi = "N" if x >= 0 else "S"
    return f"{abs(int(round(x)))}°{hemi}"


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


def get_name_field(gdf, candidates):
    for candidate in candidates:
        if candidate in gdf.columns:
            return candidate
    raise ValueError(f"None of these fields found: {candidates}. Available: {list(gdf.columns)}")


def add_graticule(ax, xticks, yticks, detailed=False):
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    if detailed:
        ax.xaxis.set_major_formatter(FuncFormatter(dm_formatter_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(dm_formatter_lat))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(degree_formatter_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(degree_formatter_lat))

    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=GRID_LW, alpha=GRID_ALPHA, zorder=0)

    ax.tick_params(
        axis="both",
        labelsize=TICK_FS,
        direction="out",
        top=True,
        right=True,
        labeltop=True,
        labelright=True,
    )

    for label in ax.get_yticklabels():
        label.set_rotation(90)
        label.set_va("center")
        label.set_ha("center")


def km_to_lon_degrees(km: float, lat_deg: float) -> float:
    cos_lat = math.cos(math.radians(lat_deg))
    if abs(cos_lat) < 1e-8:
        cos_lat = 1e-8
    return km / (111.320 * cos_lat)


def add_scalebar_2step_data(
    ax,
    x0,
    y0,
    length_km=2000,
    bar_height_frac=0.018,
    text_offset_frac=0.012,
    fontsize=10,
):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    mid_lat = 0.5 * (ylim[0] + ylim[1])
    total_deg = km_to_lon_degrees(length_km, mid_lat)
    step_deg = total_deg / 2.0
    bar_h = bar_height_frac * (ylim[1] - ylim[0])

    colors = ["black", "white"]
    for i in range(2):
        rect = Rectangle(
            (x0 + i * step_deg, y0),
            step_deg,
            bar_h,
            facecolor=colors[i],
            edgecolor="black",
            linewidth=0.8,
            zorder=8,
        )
        ax.add_patch(rect)

    label_y = y0 + bar_h + text_offset_frac * (ylim[1] - ylim[0])
    ax.text(x0, label_y, "0", ha="center", va="bottom", fontsize=fontsize, zorder=9)
    ax.text(x0 + step_deg, label_y, f"{int(length_km / 2)}", ha="center", va="bottom", fontsize=fontsize, zorder=9)
    ax.text(x0 + 2 * step_deg, label_y, f"{int(length_km)} km", ha="center", va="bottom", fontsize=fontsize, zorder=9)


def add_scalebar_2step(
    ax,
    length_km=150,
    location=(0.46, 0.07),
    bar_height_frac=0.018,
    text_offset_frac=0.012,
    fontsize=10,
):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x0 = xlim[0] + location[0] * (xlim[1] - xlim[0])
    y0 = ylim[0] + location[1] * (ylim[1] - ylim[0])

    mid_lat = 0.5 * (ylim[0] + ylim[1])
    total_deg = km_to_lon_degrees(length_km, mid_lat)
    step_deg = total_deg / 2.0
    bar_h = bar_height_frac * (ylim[1] - ylim[0])

    colors = ["black", "white"]
    for i in range(2):
        rect = Rectangle(
            (x0 + i * step_deg, y0),
            step_deg,
            bar_h,
            facecolor=colors[i],
            edgecolor="black",
            linewidth=0.8,
            zorder=8,
        )
        ax.add_patch(rect)

    label_y = y0 + bar_h + text_offset_frac * (ylim[1] - ylim[0])
    ax.text(x0, label_y, "0", ha="center", va="bottom", fontsize=fontsize, zorder=9)
    ax.text(x0 + step_deg, label_y, "75", ha="center", va="bottom", fontsize=fontsize, zorder=9)
    ax.text(x0 + 2 * step_deg, label_y, "150 km", ha="center", va="bottom", fontsize=fontsize, zorder=9)


def load_svg_as_image(svg_path: Path, target_height_px: int = 220):
    if not svg_path.exists() or not HAS_CAIROSVG:
        return None
    png_bytes = cairosvg.svg2png(url=str(svg_path), output_height=target_height_px)
    return Image.open(io.BytesIO(png_bytes))


def add_north_arrow(ax, svg_path: Path, xy=(0.90, 0.88), zoom=0.23):
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
            zorder=20,
        )
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
    ab = AnnotationBbox(
        imagebox,
        xy,
        xycoords="axes fraction",
        frameon=False,
        box_alignment=(0.5, 0.5),
        zorder=20,
    )
    ax.add_artist(ab)


def add_panel_label(ax, label: str, loc=(0.02, 0.02)):
    ax.text(
        loc[0],
        loc[1],
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="square,pad=0.22", facecolor="white", edgecolor="0.3"),
        zorder=30,
    )


def add_panel_a_bay_of_bengal(ax, lon=92.0, lat=14.0):
    txt = ax.text(
        lon,
        lat,
        "Bay of Bengal",
        fontsize=10,
        ha="center",
        va="center",
        color=PANEL_A_BAY_LABEL_COLOR,
        zorder=4,
    )
    txt.set_path_effects([
        pe.Stroke(linewidth=3, foreground=PANEL_A_SEA_COLOR),
        pe.Normal()
    ])


def label_south_asia_countries(ax, gdf, name_field):
    manual_positions = {
        "China": (100.5, 30.3),
        "Afghanistan": (68.9, 34.6),
    }

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        name = str(row[name_field]).strip()
        if name == "Bangladesh":
            continue
        if name == "Bhutan":
            continue
        if not SHOW_INDONESIA_LABEL and name == "Indonesia":
            continue
        if name not in COUNTRY_LABEL_RULES:
            continue

        if name in manual_positions:
            x, y = manual_positions[name]
        else:
            point = geom.representative_point()
            x, y = point.coords[0]

        txt = ax.text(
            x,
            y,
            COUNTRY_LABEL_RULES[name],
            fontsize=COUNTRY_FS,
            ha="center",
            va="center",
            color=PANEL_A_LABEL_COLOR,
            zorder=7,
        )
        txt.set_path_effects([pe.Stroke(linewidth=3, foreground=PANEL_A_LABEL_HALO), pe.Normal()])


def label_districts_with_offsets(ax, gdf, field="ADM2_EN", fontsize=7.2):
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        name = str(row[field]).strip()
        point = geom.representative_point()
        x, y = point.coords[0]
        dx, dy = DISTRICT_LABEL_OFFSETS.get(name, (0.0, 0.0))

        txt = ax.text(
            x + dx,
            y + dy,
            name,
            fontsize=fontsize,
            ha="center",
            va="center",
            color=PANEL_B_DISTRICT_LABEL_COLOR,
            fontweight="bold",
            zorder=12,
        )
        txt.set_path_effects([pe.Stroke(linewidth=3, foreground=PANEL_B_DISTRICT_LABEL_HALO), pe.Normal()])


def add_bangladesh_context_labels(ax):
    labels = [
        (90.85, 26.0, "Meghalaya\n(INDIA)"),
        (88.30, 22.95, "West Bengal\n(INDIA)"),
        (91.78, 23.90, "Tripura\n(INDIA)"),
        (92.75, 20.90, "MYANMAR"),
        (90.40, 21.10, "Bay of Bengal"),
    ]

    for x, y, text in labels:
        txt = ax.text(
            x,
            y,
            text,
            fontsize=OUTSIDE_FS,
            ha="center",
            va="center",
            color=PANEL_B_OUTSIDE_LABEL_COLOR,
            zorder=4,
        )
        txt.set_path_effects([pe.Stroke(linewidth=3, foreground=PANEL_B_BG_COLOR), pe.Normal()])


def add_panel_a_bay_of_bengal(ax, lon=92.0, lat=14.0):
    txt = ax.text(
        lon,
        lat,
        "Bay of Bengal",
        fontsize=10,
        ha="center",
        va="center",
        color=PANEL_A_BAY_LABEL_COLOR,
        zorder=4,
    )
    txt.set_path_effects([
        pe.Stroke(linewidth=3, foreground=PANEL_A_SEA_COLOR),
        pe.Normal()
    ])


def add_panel_a_btn_label(ax, lon=90.6, lat=28.8):
    txt = ax.text(
        lon,
        lat,
        "BTN",
        fontsize=COUNTRY_FS,
        ha="center",
        va="center",
        color=PANEL_A_LABEL_COLOR,
        zorder=7,
    )
    txt.set_path_effects([
        pe.Stroke(linewidth=3, foreground=PANEL_A_LABEL_HALO),
        pe.Normal()
    ])


def main():
    world_shp = resolve_path(WORLD_SHP)
    bgd_adm0 = resolve_path(BGD_ADM0)
    coastal_districts = resolve_path(COASTAL_DISTRICTS)
    coastal_solid = resolve_path(COASTAL_SOLID)
    north_arrow_svg = resolve_path(NORTH_ARROW_SVG)
    out_png = resolve_path(OUT_PNG)

    for path in [world_shp, bgd_adm0, coastal_districts, coastal_solid]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    out_png.parent.mkdir(parents=True, exist_ok=True)

    world = gpd.read_file(world_shp)
    gdf_bgd = gpd.read_file(bgd_adm0)
    gdf_coast = gpd.read_file(coastal_districts)
    gdf_coast_solid = gpd.read_file(coastal_solid)

    target_crs = "EPSG:4326"

    if world.crs is None:
        raise ValueError("World shapefile has no CRS.")
    if gdf_bgd.crs is None:
        raise ValueError("bgd_adm0.gpkg has no CRS.")
    if gdf_coast.crs is None:
        raise ValueError("bd_coastal_districts.gpkg has no CRS.")
    if gdf_coast_solid.crs is None:
        raise ValueError("bd_coastal_map_solid_gp.gpkg has no CRS.")

    world = world.to_crs(target_crs)
    gdf_bgd = gdf_bgd.to_crs(target_crs)
    gdf_coast = gdf_coast.to_crs(target_crs)
    gdf_coast_solid = gdf_coast_solid.to_crs(target_crs)

    world_name_field = get_name_field(world, ["NAME", "NAME_EN", "ADMIN", "SOVEREIGNT"])
    coast_name_field = get_name_field(gdf_coast, ["ADM2_EN", "NAME_2", "district", "DIST_NAME"])

    south_asia = world.cx[67:112, 0:36].copy()

    fig = plt.figure(figsize=FIGSIZE, dpi=FIG_DPI, facecolor=FIG_BG_COLOR)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.8], wspace=0.14)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # --------------------------------------------------------
    # Panel (a): South Asia
    # --------------------------------------------------------
    ax1.set_facecolor(PANEL_A_SEA_COLOR)

    south_asia.plot(
        ax=ax1,
        facecolor=PANEL_A_COUNTRY_FILL,
        edgecolor=PANEL_A_COUNTRY_STROKE,
        linewidth=0.8,
        zorder=2,
    )
    gdf_bgd.plot(
        ax=ax1,
        facecolor=PANEL_A_BGD_FILL,
        edgecolor=PANEL_A_COUNTRY_STROKE,
        linewidth=1.0,
        zorder=5,
    )
    gdf_coast.plot(
        ax=ax1,
        facecolor=PANEL_A_STUDY_FILL,
        edgecolor=PANEL_A_STUDY_STROKE,
        linewidth=0.9,
        zorder=6,
    )

    ax1.set_xlim(67, 112)
    ax1.set_ylim(0, 36)
    ax1.set_aspect("equal", adjustable="box")

    add_graticule(ax1, xticks=np.arange(70, 113, 10), yticks=np.arange(0, 37, 10), detailed=False)
    label_south_asia_countries(ax1, south_asia, world_name_field)
    add_panel_a_btn_label(ax1, lon=90.6, lat=28.3)
    add_panel_a_bay_of_bengal(ax1, lon=89.5, lat=14.7)

    minx, miny, maxx, maxy = gdf_bgd.total_bounds
    pad_x = 0.6
    pad_y = 0.6

    box_x0 = minx - pad_x
    box_y0 = miny - pad_y
    box_x1 = maxx + pad_x
    box_y1 = maxy + pad_y

    bbox_rect = Rectangle(
        (box_x0, box_y0),
        box_x1 - box_x0,
        box_y1 - box_y0,
        fill=False,
        edgecolor="red",
        linewidth=2.0,
        zorder=15,
    )
    ax1.add_patch(bbox_rect)

    add_scalebar_2step_data(
        ax1,
        x0=PANEL_A_SCALE_X0,
        y0=PANEL_A_SCALE_Y0,
        length_km=2000,
        fontsize=9,
    )
    add_panel_label(ax1, "(a)")

    # --------------------------------------------------------
    # Panel (b): Bangladesh
    # --------------------------------------------------------
    ax2.set_facecolor(PANEL_B_BG_COLOR)

    gdf_bgd.plot(
        ax=ax2,
        facecolor=PANEL_B_COUNTRY_FILL,
        edgecolor=PANEL_B_COUNTRY_STROKE,
        linewidth=1.2,
        zorder=3,
    )

    gdf_coast_solid.plot(
        ax=ax2,
        facecolor="none",
        edgecolor=PANEL_B_STUDY_OUTER_STROKE,
        linewidth=COASTAL_OUTER_EDGE_WIDTH,
        zorder=6,
    )

    gdf_coast.plot(
        ax=ax2,
        facecolor=PANEL_B_STUDY_FILL,
        edgecolor=PANEL_B_DISTRICT_BOUNDARY,
        linewidth=STUDY_AREA_EDGE_WIDTH,
        zorder=5,
    )

    minx2, miny2, maxx2, maxy2 = gdf_bgd.total_bounds
    xpad_left = 0.4
    xpad_right = 0.8
    ypad_bottom = 0.4
    ypad_top = 0.4

    ax2.set_xlim(minx2 - xpad_left, maxx2 + xpad_right)
    ax2.set_ylim(miny2 - ypad_bottom, maxy2 + ypad_top)
    ax2.set_aspect("equal", adjustable="box")

    add_graticule(
        ax2,
        xticks=np.arange(math.floor(minx2), math.ceil(maxx2) + 1, 1),
        yticks=np.arange(math.floor(miny2), math.ceil(maxy2) + 1, 1),
        detailed=True,
    )

    add_bangladesh_context_labels(ax2)
    label_districts_with_offsets(ax2, gdf_coast, field=coast_name_field, fontsize=DISTRICT_FS)

    add_north_arrow(ax2, north_arrow_svg, xy=(0.90, 0.88), zoom=0.23)
    add_scalebar_2step(ax2, length_km=150, location=(0.46, 0.07), fontsize=10)
    add_panel_label(ax2, "(b)")

    # Connector lines
    con1 = ConnectionPatch(
        xyA=(0.0, 1.0),
        coordsA=ax2.transAxes,
        xyB=(box_x1, box_y1),
        coordsB=ax1.transData,
        color="0.3",
        linewidth=1.3,
        zorder=20,
        clip_on=False,
    )
    con2 = ConnectionPatch(
        xyA=(0.0, 0.0),
        coordsA=ax2.transAxes,
        xyB=(box_x1, box_y0),
        coordsB=ax1.transData,
        color="0.3",
        linewidth=1.3,
        zorder=20,
        clip_on=False,
    )
    fig.add_artist(con1)
    fig.add_artist(con2)

    handles = [
        Patch(
            facecolor=PANEL_B_STUDY_FILL,
            edgecolor=PANEL_B_DISTRICT_BOUNDARY,
            label="Study Area-15 Districts"
        )
    ]
    legend = ax2.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(PANEL_B_LEGEND_X, PANEL_B_LEGEND_Y),
        bbox_transform=ax2.transData,
        fontsize=LEGEND_FS,
        frameon=True,
        framealpha=1.0,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        borderpad=0.6,
        handlelength=1.6,
        handletextpad=0.6,
    )
    legend.set_zorder(25)

    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_edgecolor("0.35")

    plt.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight", facecolor=FIG_BG_COLOR)
    plt.close(fig)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
