#!/usr/bin/env python3
"""
Create a publication-style LULC class reference chart.

Produces a table figure showing Class ID, colour swatch, and class name
for the 10-class Bangladesh coastal LULC scheme defined in
scripts/visualization/make_infer_lulc_map.py.

Inputs
------
- assets/color_palette_coastal_lulc.json

Output
------
- outputs/figures/lulc_class_chart.png

Example
-------
python scripts/visualization/make_lulc_class_chart.py
python scripts/visualization/make_lulc_class_chart.py \
    --output outputs/figures/lulc_class_chart.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PALETTE = Path("assets/color_palette_coastal_lulc.json")
DEFAULT_OUTPUT = Path("outputs/figures/lulc_class_chart.png")

FIG_HEIGHT = 5.5
FIG_DPI = 300
CHART_TITLE = "LULC Classification Scheme"

# Physical column widths in inches — figure width is derived from these
_ID_COL_IN     = 0.60   # "Class ID" column
_SWATCH_COL_IN = 0.90   # colour swatch column
_NAME_LPAD_IN  = 0.20   # left padding inside the name column
_NAME_RPAD_IN  = 0.40   # right margin after the longest name

# Subplot margins as fractions of figure size.
# Using explicit subplots_adjust (not tight_layout) so ax_width_in is exact.
_SP_LEFT   = 0.02
_SP_RIGHT  = 0.98
_SP_BOTTOM = 0.01
_SP_TOP    = 0.92   # remaining ~8 % of figure height is for the suptitle

# Font sizes
_NAME_FONTSIZE   = 10
_HEADER_FONTSIZE = 9
_ID_FONTSIZE     = 11

# Class definitions — sourced from make_infer_lulc_map.py
LULC_NAMES = {
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

LULC_COLORS = {
    1: "#E66A00",
    2: "#8FBF7A",
    3: "#9C7A5B",
    4: "#FFC636",
    5: "#4F7F3D",
    6: "#00ADA9",
    7: "#7AD9D6",
    8: "#007C91",
    9: "#2F5D50",
    10: "#F3E7CF",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make LULC class reference chart.")
    p.add_argument("--palette", type=Path, default=DEFAULT_PALETTE,
                   help="Colour palette JSON path.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output PNG path.")
    return p.parse_args()


def _max_name_width_in() -> float:
    """Return the rendered width (inches) of the widest class name at _NAME_FONTSIZE.

    Uses FIG_DPI so the measurement matches the production render exactly.
    """
    fig_tmp, ax_tmp = plt.subplots(figsize=(20, 1), dpi=FIG_DPI)
    renderer = fig_tmp.canvas.get_renderer()
    max_w = 0.0
    for name in LULC_NAMES.values():
        t = ax_tmp.text(0, 0.5, name, fontsize=_NAME_FONTSIZE)
        bb = t.get_window_extent(renderer=renderer)
        max_w = max(max_w, bb.width / fig_tmp.dpi)
    plt.close(fig_tmp)
    return max_w


def draw_chart(ax, fig_width_in: float, fig_bg: str, main_text_color: str) -> None:
    classes = sorted(LULC_NAMES.keys())
    n = len(classes)

    # Data coordinate system: each row = 1 unit; header = 1.5 units.
    # total_width data units span exactly fig_width_in inches.
    total_width  = 10.0
    row_height   = 1.0
    header_height = 1.5
    total_height  = n * row_height + header_height

    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)
    ax.axis("off")

    # Convert physical column widths to data units
    scale = total_width / fig_width_in   # data units per inch
    id_x0,     id_x1     = 0.0,  _ID_COL_IN * scale
    swatch_x0, swatch_x1 = id_x1, id_x1 + _SWATCH_COL_IN * scale
    name_x0               = swatch_x1
    name_text_x           = name_x0 + _NAME_LPAD_IN * scale   # left-edge of text

    # Colour palette
    header_color      = main_text_color
    header_text_color = fig_bg
    border_color      = "#BBBBBB"
    swatch_border     = "#555555"
    row_bg_even       = "#FFFFFF"
    row_bg_odd        = "#F5F0EB"

    # ── Header ──────────────────────────────────────────────────────────
    header_y = n * row_height
    ax.add_patch(Rectangle(
        (0, header_y), total_width, header_height,
        facecolor=header_color, edgecolor="none", zorder=1,
    ))
    ax.text(
        (id_x0 + id_x1) / 2, header_y + header_height / 2,
        "Class\nID",
        ha="center", va="center", fontsize=_HEADER_FONTSIZE, fontweight="bold",
        color=header_text_color, linespacing=1.3, zorder=2,
    )
    ax.text(
        (swatch_x0 + swatch_x1) / 2, header_y + header_height / 2,
        "Colour",
        ha="center", va="center", fontsize=_HEADER_FONTSIZE, fontweight="bold",
        color=header_text_color, zorder=2,
    )
    ax.text(
        name_text_x, header_y + header_height / 2,
        "Class Name",
        ha="left", va="center", fontsize=_HEADER_FONTSIZE, fontweight="bold",
        color=header_text_color, zorder=2,
    )

    # ── Data rows ────────────────────────────────────────────────────────
    swatch_mx = 0.12 * scale   # horizontal margin inside swatch cell (in data units)
    swatch_my = 0.22            # vertical margin inside swatch cell

    for i, class_id in enumerate(classes):
        row_y  = (n - 1 - i) * row_height   # class 1 at top, class 10 at bottom
        row_bg = row_bg_even if i % 2 == 0 else row_bg_odd

        # Row background
        ax.add_patch(Rectangle(
            (0, row_y), total_width, row_height,
            facecolor=row_bg, edgecolor="none", zorder=1,
        ))

        # Class ID
        ax.text(
            (id_x0 + id_x1) / 2, row_y + row_height / 2,
            str(class_id),
            ha="center", va="center", fontsize=_ID_FONTSIZE, fontweight="bold",
            color=main_text_color, zorder=2,
        )

        # Colour swatch
        ax.add_patch(Rectangle(
            (swatch_x0 + swatch_mx, row_y + swatch_my),
            (swatch_x1 - swatch_x0) - 2 * swatch_mx,
            row_height - 2 * swatch_my,
            facecolor=LULC_COLORS[class_id],
            edgecolor=swatch_border,
            linewidth=0.7,
            zorder=2,
        ))

        # Class name
        ax.text(
            name_text_x, row_y + row_height / 2,
            LULC_NAMES[class_id],
            ha="left", va="center", fontsize=_NAME_FONTSIZE,
            color=main_text_color, zorder=2,
        )

    # ── Grid lines ───────────────────────────────────────────────────────
    for row in range(n + 1):
        y  = row * row_height
        lw = 1.0 if row in (0, n) else 0.45
        ax.plot([0, total_width], [y, y], color=border_color, linewidth=lw, zorder=3)

    # Header bottom separator
    ax.plot([0, total_width], [n * row_height, n * row_height],
            color=main_text_color, linewidth=1.5, zorder=4)

    # Vertical dividers
    for x in (id_x1, swatch_x1):
        ax.plot([x, x], [0, total_height], color=border_color, linewidth=0.45, zorder=3)

    # Outer border
    ax.add_patch(Rectangle(
        (0, 0), total_width, total_height,
        fill=False, edgecolor=main_text_color, linewidth=1.4, zorder=5,
    ))


def main() -> None:
    args = parse_args()
    palette_path = resolve_path(args.palette)
    output = resolve_path(args.output)

    palette = json.loads(palette_path.read_text())
    colors = palette["colors"]
    fig_bg = colors["sand"]
    main_text_color = colors["deep_slate"]

    # Axes content width = exactly the column content (id + swatch + name pad + text + rpad).
    # Figure width is then inflated so that the axes occupies _SP_LEFT.._SP_RIGHT of it.
    max_name_w  = _max_name_width_in()
    ax_width_in = _ID_COL_IN + _SWATCH_COL_IN + _NAME_LPAD_IN + max_name_w + _NAME_RPAD_IN
    fig_width   = ax_width_in / (_SP_RIGHT - _SP_LEFT)

    fig, ax = plt.subplots(figsize=(fig_width, FIG_HEIGHT), dpi=FIG_DPI, facecolor=fig_bg)
    ax.set_facecolor(fig_bg)

    # Explicit margins so ax_width_in is exact — no tight_layout guesswork
    fig.subplots_adjust(
        left=_SP_LEFT, right=_SP_RIGHT,
        bottom=_SP_BOTTOM, top=_SP_TOP,
    )

    draw_chart(ax, fig_width_in=ax_width_in, fig_bg=fig_bg, main_text_color=main_text_color)

    fig.suptitle(
        CHART_TITLE,
        fontsize=14, fontweight="bold",
        color=main_text_color, y=0.97,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
