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

FIGSIZE = (8.5, 5.5)
FIG_DPI = 300
CHART_TITLE = "LULC Classification Scheme"

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


def draw_chart(ax, fig_bg: str, main_text_color: str) -> None:
    classes = sorted(LULC_NAMES.keys())
    n = len(classes)

    # Data coordinate system: each row = 1 unit; header = 1.5 units
    total_width = 10.0
    row_height = 1.0
    header_height = 1.5
    total_height = n * row_height + header_height

    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)
    ax.axis("off")

    # Column x-boundaries
    id_x0,     id_x1     = 0.0,  1.1
    swatch_x0, swatch_x1 = 1.1,  2.8
    name_x0,   name_x1   = 2.8, total_width

    # Palette
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
        ha="center", va="center", fontsize=9, fontweight="bold",
        color=header_text_color, linespacing=1.3, zorder=2,
    )
    ax.text(
        (swatch_x0 + swatch_x1) / 2, header_y + header_height / 2,
        "Colour",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color=header_text_color, zorder=2,
    )
    ax.text(
        name_x0 + 0.28, header_y + header_height / 2,
        "Class Name",
        ha="left", va="center", fontsize=9, fontweight="bold",
        color=header_text_color, zorder=2,
    )

    # ── Data rows ────────────────────────────────────────────────────────
    swatch_mx = 0.22   # horizontal margin inside swatch cell
    swatch_my = 0.22   # vertical margin inside swatch cell

    for i, class_id in enumerate(classes):
        row_y = (n - 1 - i) * row_height      # class 1 at top, class 10 at bottom
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
            ha="center", va="center", fontsize=11, fontweight="bold",
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
            name_x0 + 0.28, row_y + row_height / 2,
            LULC_NAMES[class_id],
            ha="left", va="center", fontsize=10,
            color=main_text_color, zorder=2,
        )

    # ── Grid lines ───────────────────────────────────────────────────────
    # Horizontal: between every row and at outer edges
    for row in range(n + 1):
        y = row * row_height
        lw = 1.0 if row in (0, n) else 0.45
        ax.plot([0, total_width], [y, y], color=border_color, linewidth=lw, zorder=3)

    # Header bottom separator (slightly bolder)
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

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI, facecolor=fig_bg)
    ax.set_facecolor(fig_bg)

    draw_chart(ax, fig_bg=fig_bg, main_text_color=main_text_color)

    fig.suptitle(
        CHART_TITLE,
        fontsize=14, fontweight="bold",
        color=main_text_color, y=0.97,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
