#!/usr/bin/env python3
"""
Visualize parcel-level LULC change between 2017 and 2024 for a target upazila.

Inputs
------
- outputs/inference/change_analysis/<upazila>_parcels_lulc_change_2017vs2024.csv
- assets/color_palette_coastal_lulc.json

Outputs
-------
- outputs/figures/<upazila>_parcel_lulc_area_bar_2017_vs_2024.png
- outputs/figures/<upazila>_parcel_lulc_transition_sankey_2017_vs_2024.png
- outputs/figures/<upazila>_parcel_lulc_parceltype_faceted_bar_2017_vs_2024.png

Example
-------
python scripts/visualization/visualize_parcel_lulc_change_2017vs2024.py \
    --upazila bamna

Complete Example Run
--------------------
python scripts/visualization/visualize_parcel_lulc_change_2017vs2024.py \
    --upazila bamna \
    --add-title \
    --outptut-plot-bar outputs/figures/bamna_parcel_lulc_area_bar_2017_vs_2024.png \
    --outptut-plot-sankey outputs/figures/bamna_parcel_lulc_transition_sankey_2017_vs_2024.png \
    --outptut-plot-facetedbar outputs/figures/bamna_parcel_lulc_parceltype_faceted_bar_2017_vs_2024.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import PathPatch, Rectangle, Wedge, Patch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = Path("outputs/inference/change_analysis")
DEFAULT_OUTPUT_ROOT = Path("outputs/figures")
DEFAULT_PALETTE = Path("assets/color_palette_coastal_lulc.json")
UPAZILA_CHOICES = ("bamna", "amtali", "betagi", "manpura")
AREA_COL = "parcel_area_m2"

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

NAME_TO_COLOR = {name: LULC_COLORS[class_id] for class_id, name in LULC_NAMES.items()}
LULC_ORDER = [LULC_NAMES[class_id] for class_id in sorted(LULC_NAMES)]
NODATA_LABELS = {"NoData", "Unknown", "nan", ""}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_input_path(upazila: str) -> Path:
    return DEFAULT_INPUT_ROOT / f"{upazila}_parcels_lulc_change_2017vs2024.csv"


def output_path(upazila: str, suffix: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"{upazila}_parcel_lulc_{suffix}_2017_vs_2024.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize parcel-level LULC change from 2017 to 2024.")
    p.add_argument("--upazila", required=True, choices=UPAZILA_CHOICES, help="Target upazila.")
    p.add_argument("--input", type=Path, default=None, help="Optional CSV override.")
    p.add_argument("--palette", type=Path, default=DEFAULT_PALETTE, help="Project palette JSON.")
    p.add_argument("--facet-top-n", type=int, default=12, help="Number of parcel types to show in faceted charts.")
    p.add_argument("--add-title", action="store_true", help="Show title on plots.")
    p.add_argument(
        "--outptut-plot-bar",
        type=Path,
        default=None,
        help="Output bar plot path. Default: outputs/figures/<upazila>_parcel_lulc_area_bar_2017_vs_2024.png",
    )
    p.add_argument(
        "--outptut-plot-sankey",
        type=Path,
        default=None,
        help="Output Sankey plot path. Default: outputs/figures/<upazila>_parcel_lulc_transition_sankey_2017_vs_2024.png",
    )
    p.add_argument(
        "--outptut-plot-facetedbar",
        type=Path,
        default=None,
        help="Output faceted bar plot path. Default: outputs/figures/<upazila>_parcel_lulc_parceltype_faceted_bar_2017_vs_2024.png",
    )
    return p.parse_args()


def load_palette(path: Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)["colors"]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "L_NAME_En",
        AREA_COL,
        "lulc_name_2017",
        "lulc_name_2024",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df = df[df[AREA_COL].notna()].copy()
    df[AREA_COL] = pd.to_numeric(df[AREA_COL], errors="coerce")
    df = df[df[AREA_COL] > 0].copy()
    df["lulc_name_2017"] = df["lulc_name_2017"].fillna("Unknown")
    df["lulc_name_2024"] = df["lulc_name_2024"].fillna("Unknown")
    df["L_NAME_En"] = df["L_NAME_En"].fillna("Unknown")
    return df


def class_order_from_df(df: pd.DataFrame) -> list[str]:
    present = set(df["lulc_name_2017"].unique()) | set(df["lulc_name_2024"].unique())
    ordered = [name for name in LULC_ORDER if name in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def color_for_class(name: str) -> str:
    if pd.isna(name) or str(name).strip() in NODATA_LABELS:
        return "#000000"
    return NAME_TO_COLOR.get(str(name).strip(), "#000000")


def lighten(color: str, amount: float) -> tuple[float, float, float]:
    c = np.array(mcolors.to_rgb(color))
    return tuple(c + (1.0 - c) * amount)


def style_axis(ax, palette: dict, grid_axis: str = "y") -> None:
    ax.set_facecolor(palette["sand"])
    ax.tick_params(colors=palette["deep_slate"])
    for spine in ax.spines.values():
        spine.set_color(palette["deep_slate"])
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="--", alpha=0.30, color=palette["mist_gray"])
        ax.set_axisbelow(True)


def save_area_bar_chart(df: pd.DataFrame, upazila: str, order: list[str], palette: dict, output: Path, add_title: bool) -> Path:
    area_2017 = df.groupby("lulc_name_2017")[AREA_COL].sum().reindex(order, fill_value=0)
    area_2024 = df.groupby("lulc_name_2024")[AREA_COL].sum().reindex(order, fill_value=0)

    x = np.arange(len(order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300, facecolor=palette["sand"])
    ax.bar(x - width / 2, area_2017.values, width=width, label="2017", color=palette["teal_blue"])
    ax.bar(x + width / 2, area_2024.values, width=width, label="2024", color=palette["coral"])

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right", color=palette["deep_slate"])
    ax.set_ylabel("Total area (m²)", color=palette["deep_slate"])
    if add_title:
        ax.set_title(f"{upazila.title()} parcel LULC area comparison: 2017 vs 2024", color=palette["deep_slate"])
    leg = ax.legend(frameon=False)
    for txt in leg.get_texts():
        txt.set_color(palette["deep_slate"])
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    style_axis(ax, palette, grid_axis="y")

    fig.tight_layout()
    out = resolve_path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=palette["sand"])
    plt.close(fig)
    return out


def sankey_patch(x0: float, x1: float, y0b: float, y0t: float, y1b: float, y1t: float, color: str, alpha: float = 0.6):
    cx0 = x0 + (x1 - x0) * 0.35
    cx1 = x0 + (x1 - x0) * 0.65
    verts = [
        (x0, y0b),
        (cx0, y0b),
        (cx1, y1b),
        (x1, y1b),
        (x1, y1t),
        (cx1, y1t),
        (cx0, y0t),
        (x0, y0t),
        (x0, y0b),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha)


def adjust_positions(y_values: list[float], min_gap: float = 0.04, y_min: float = 0.02, y_max: float = 0.98) -> list[float]:
    if not y_values:
        return []

    adjusted = sorted(max(y_min, min(y, y_max)) for y in y_values)

    for i in range(1, len(adjusted)):
        if adjusted[i] < adjusted[i - 1] + min_gap:
            adjusted[i] = adjusted[i - 1] + min_gap

    overflow = adjusted[-1] - y_max
    if overflow > 0:
        adjusted = [y - overflow for y in adjusted]

    adjusted[0] = max(adjusted[0], y_min)
    for i in range(1, len(adjusted)):
        if adjusted[i] < adjusted[i - 1] + min_gap:
            adjusted[i] = adjusted[i - 1] + min_gap

    underflow = y_min - adjusted[0]
    if underflow > 0:
        adjusted = [y + underflow for y in adjusted]

    return [max(y_min, min(y, y_max)) for y in adjusted]


def save_sankey(df: pd.DataFrame, upazila: str, order: list[str], palette: dict, output: Path, add_title: bool) -> Path:
    trans = df.groupby(["lulc_name_2017", "lulc_name_2024"])[AREA_COL].sum().reset_index()
    total_area = trans[AREA_COL].sum()
    if total_area <= 0:
        raise ValueError("Total area is zero; cannot build Sankey diagram.")

    left_totals = trans.groupby("lulc_name_2017")[AREA_COL].sum().reindex(order, fill_value=0)
    right_totals = trans.groupby("lulc_name_2024")[AREA_COL].sum().reindex(order, fill_value=0)

    left_gap = 0.012
    right_gap = 0.012
    left_heights = left_totals / total_area
    right_heights = right_totals / total_area
    total_left_gap = left_gap * max(0, len(order) - 1)
    total_right_gap = right_gap * max(0, len(order) - 1)
    scale_left = (1.0 - total_left_gap) / max(left_heights.sum(), 1e-12)
    scale_right = (1.0 - total_right_gap) / max(right_heights.sum(), 1e-12)
    left_heights *= scale_left
    right_heights *= scale_right

    left_pos = {}
    right_pos = {}
    y = 1.0
    for cls in order:
        h = left_heights.get(cls, 0.0)
        left_pos[cls] = (y - h, y)
        y = y - h - left_gap
    y = 1.0
    for cls in order:
        h = right_heights.get(cls, 0.0)
        right_pos[cls] = (y - h, y)
        y = y - h - right_gap

    flow_order = trans.sort_values(["lulc_name_2017", "lulc_name_2024"])
    left_cursor = {cls: left_pos[cls][0] for cls in order}
    right_cursor = {cls: right_pos[cls][0] for cls in order}

    fig, ax = plt.subplots(figsize=(14, 9), dpi=300, facecolor=palette["sand"])
    ax.set_facecolor(palette["sand"])
    x_left0, x_left1 = 0.08, 0.15
    x_right0, x_right1 = 0.85, 0.92
    x_flow0, x_flow1 = x_left1, x_right0

    for _, row in flow_order.iterrows():
        src = row["lulc_name_2017"]
        dst = row["lulc_name_2024"]
        frac = row[AREA_COL] / total_area
        h_left = frac * scale_left
        h_right = frac * scale_right
        y0b = left_cursor[src]
        y0t = y0b + h_left
        y1b = right_cursor[dst]
        y1t = y1b + h_right
        left_cursor[src] += h_left
        right_cursor[dst] += h_right
        ax.add_patch(sankey_patch(x_flow0, x_flow1, y0b, y0t, y1b, y1t, color_for_class(dst)))

    left_labels = []
    right_labels = []

    for cls in order:
        yb, yt = left_pos[cls]
        ax.add_patch(Rectangle((x_left0, yb), x_left1 - x_left0, yt - yb, facecolor=color_for_class(cls), edgecolor="white", lw=0.7))
        pct = left_totals[cls] / total_area * 100 if total_area else 0
        left_labels.append(
            {
                "cls": cls,
                "y_center": (yb + yt) / 2,
                "text": f"{cls} ({pct:.1f}%)",
            }
        )

    for cls in order:
        yb, yt = right_pos[cls]
        ax.add_patch(Rectangle((x_right0, yb), x_right1 - x_right0, yt - yb, facecolor=color_for_class(cls), edgecolor="white", lw=0.7))
        pct = right_totals[cls] / total_area * 100 if total_area else 0
        right_labels.append(
            {
                "cls": cls,
                "y_center": (yb + yt) / 2,
                "text": f"{cls} ({pct:.1f}%)",
            }
        )

    left_labels = sorted(left_labels, key=lambda item: item["y_center"])
    right_labels = sorted(right_labels, key=lambda item: item["y_center"])
    adjusted_left = adjust_positions([item["y_center"] for item in left_labels], min_gap=0.04, y_min=0.02, y_max=0.98)
    adjusted_right = adjust_positions([item["y_center"] for item in right_labels], min_gap=0.04, y_min=0.02, y_max=0.98)

    for item, y_adj in zip(left_labels, adjusted_left):
        y_orig = item["y_center"]
        text_x = x_left0 - 0.015
        connector_x0 = x_left0
        connector_x1 = text_x + 0.003
        if abs(y_adj - y_orig) > 0.002:
            ax.plot(
                [connector_x0, connector_x1],
                [y_orig, y_adj],
                color=palette["deep_slate"],
                linewidth=0.7,
                alpha=0.7,
            )
        ax.text(text_x, y_adj, item["text"], ha="right", va="center", fontsize=10, color=palette["deep_slate"])

    for item, y_adj in zip(right_labels, adjusted_right):
        y_orig = item["y_center"]
        text_x = x_right1 + 0.015
        connector_x0 = x_right1
        connector_x1 = text_x - 0.003
        if abs(y_adj - y_orig) > 0.002:
            ax.plot(
                [connector_x0, connector_x1],
                [y_orig, y_adj],
                color=palette["deep_slate"],
                linewidth=0.7,
                alpha=0.7,
            )
        ax.text(text_x, y_adj, item["text"], ha="left", va="center", fontsize=10, color=palette["deep_slate"])

    ax.text((x_left0 + x_left1) / 2, 1.035, "2017", ha="center", va="bottom", fontsize=13, fontweight="bold", color=palette["deep_slate"])
    ax.text((x_right0 + x_right1) / 2, 1.035, "2024", ha="center", va="bottom", fontsize=13, fontweight="bold", color=palette["deep_slate"])
    if add_title:
        ax.set_title(f"{upazila.title()} parcel LULC transition flow (share of total area)", fontsize=14, color=palette["deep_slate"])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.06)
    ax.axis("off")

    out = resolve_path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=palette["sand"])
    plt.close(fig)
    return out


def save_faceted_bar(df: pd.DataFrame, upazila: str, order: list[str], top_n: int, palette: dict, output: Path, add_title: bool) -> Path:
    df_plot = df.copy()
    df_plot["L_NAME_En_plot"] = df_plot["L_NAME_En"].replace({"Khal": "Canal", "Halot": "Road"})

    parcel_area = df_plot.groupby("L_NAME_En_plot")[AREA_COL].sum().sort_values(ascending=False)
    top_types = parcel_area.head(top_n).index.tolist()
    sub = df_plot[df_plot["L_NAME_En_plot"].isin(top_types)].copy()

    n = len(top_types)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.8 * nrows), dpi=300, sharey=False, facecolor=palette["sand"])
    axes = np.atleast_1d(axes).flatten()

    for ax, parcel_type in zip(axes, top_types):
        sdf = sub[sub["L_NAME_En_plot"] == parcel_type]
        area_2017 = sdf.groupby("lulc_name_2017")[AREA_COL].sum().reindex(order, fill_value=0)
        area_2024 = sdf.groupby("lulc_name_2024")[AREA_COL].sum().reindex(order, fill_value=0)
        x = np.arange(len(order))
        width = 0.38
        ax.bar(x - width / 2, area_2017.values, width=width, label="2017", color=palette["teal_blue"])
        ax.bar(x + width / 2, area_2024.values, width=width, label="2024", color=palette["coral"])
        ax.set_title(parcel_type, color=palette["deep_slate"])
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=55, ha="right", fontsize=8, color=palette["deep_slate"])
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        if parcel_type in {"Road", "Canal", "Halot"}:
            ymax = max(float(area_2017.max()), float(area_2024.max()))
            if ymax > 0:
                ymax *= 1.10
                ax.set_ylim(0, ymax)
                ax.set_yticks(np.linspace(0, ymax, 5))
        style_axis(ax, palette, grid_axis="y")

    for ax in axes[n:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False)
    for txt in leg.get_texts():
        txt.set_color(palette["deep_slate"])
    if add_title:
        fig.suptitle(f"{upazila.title()} parcel-type faceted LULC area comparison: 2017 vs 2024", y=0.995, fontsize=15, color=palette["deep_slate"])
    fig.text(0.02, 0.5, "Area (m²)", va="center", rotation=90, color=palette["deep_slate"])
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.97))

    out = resolve_path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=palette["sand"])
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input or default_input_path(args.upazila))
    if not input_path.exists():
        raise FileNotFoundError(f"CSV not found: {input_path}")

    palette = load_palette(args.palette)
    df = load_csv(input_path)
    order = class_order_from_df(df)
    output_bar = args.outptut_plot_bar or output_path(args.upazila, "area_bar")
    output_sankey = args.outptut_plot_sankey or output_path(args.upazila, "transition_sankey")
    output_facetedbar = args.outptut_plot_facetedbar or output_path(args.upazila, "parceltype_faceted_bar")

    outputs = [
        save_area_bar_chart(df, args.upazila, order, palette, output_bar, args.add_title),
        save_sankey(df, args.upazila, order, palette, output_sankey, args.add_title),
        save_faceted_bar(df, args.upazila, order, args.facet_top_n, palette, output_facetedbar, args.add_title),
    ]

    for out in outputs:
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
