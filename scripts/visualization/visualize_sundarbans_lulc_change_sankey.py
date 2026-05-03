#!/usr/bin/env python3
"""
Visualize LULC change from 2017 to 2024 as a Sankey diagram for the Sundarbans.

Source data is the per-pixel transition code raster; each pixel's value
encodes its 2017 class (hundreds digit) and 2024 class (units digit),
e.g. 401 = Cropland (4) → Urban (1).

Inputs
------
- outputs/inference/change_analysis/transition_code_2017_to_2024.tif
- assets/maps/sundarbans.gpkg

Outputs
-------
- outputs/figures/lulc_change_2017vs2024_sankey_sundarbans.png

Example
-------
python scripts/visualization/visualize_sundarbans_lulc_change_sankey.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRANSITION = Path("outputs/inference/change_analysis/transition_code_2017_to_2024.tif")
DEFAULT_SUNDARBANS_MAP = Path("assets/maps/sundarbans.gpkg")
DEFAULT_PALETTE = Path("assets/color_palette_coastal_lulc.json")

# 10 m × 10 m pixel → 100 m² → 0.0001 km²
PIXEL_AREA_KM2 = 100.0 / 1_000_000.0
TRANSITION_NODATA = 0

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

SHORT_CLASS_NAMES = {
    1: "Urban",
    2: "Rural Settlement",
    3: "Transport/Embankments",
    4: "Cropland",
    5: "Agroforestry/Orchard",
    6: "Aquaculture/Ponds",
    7: "Canals/Drainage",
    8: "Rivers/Channels",
    9: "Mangrove Forest",
    10: "Bare/Exposed Land",
}

CLASS_ORDER = list(range(1, 11))

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

TITLE_FONTSIZE = 20
YEAR_LABEL_FONTSIZE = 18
CLASS_LABEL_FONTSIZE = 10


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sundarbans LULC change Sankey diagram (2017 → 2024).")
    p.add_argument("--transition", type=Path, default=DEFAULT_TRANSITION,
                   help="Transition-code raster path.")
    p.add_argument("--sundarbans-map", type=Path, default=DEFAULT_SUNDARBANS_MAP,
                   help="Sundarbans GeoPackage.")
    p.add_argument("--palette", type=Path, default=DEFAULT_PALETTE,
                   help="Colour palette JSON.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/figures"),
                   help="Directory for output PNG.")
    return p.parse_args()


def load_palette(path: Path) -> dict:
    with resolve_path(path).open(encoding="utf-8") as f:
        raw = json.load(f)
    return raw["colors"] if "colors" in raw else raw


# ── Transition counting ───────────────────────────────────────────────────────

def _accumulate(arr: np.ndarray, counts: dict[tuple[int, int], int]) -> None:
    valid = arr != TRANSITION_NODATA
    if not np.any(valid):
        return
    vals, cnts = np.unique(arr[valid], return_counts=True)
    for v, c in zip(vals.tolist(), cnts.tolist()):
        c17 = int(v) // 100
        c24 = int(v) % 100
        if 1 <= c17 <= 10 and 1 <= c24 <= 10:
            key = (c17, c24)
            counts[key] = counts.get(key, 0) + c


def count_transitions_region(
    ds: rasterio.DatasetReader, geoms: list
) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    shapes = [mapping(g) for g in geoms]
    try:
        masked, _ = rio_mask(ds, shapes, crop=True, nodata=TRANSITION_NODATA, all_touched=False)
        _accumulate(masked[0], counts)
    except Exception as exc:
        print(f"  Warning: region masking failed — {exc}")
    return counts


# ── Data frame ────────────────────────────────────────────────────────────────

def build_transition_df(counts: dict[tuple[int, int], int]) -> pd.DataFrame:
    rows = [
        {"class_2017": c17, "class_2024": c24, "area_km2": n * PIXEL_AREA_KM2}
        for (c17, c24), n in counts.items()
    ]
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["class_2017", "class_2024", "area_km2"])
    )


# ── Sankey drawing ────────────────────────────────────────────────────────────

def _sankey_patch(
    x0: float, x1: float,
    y0b: float, y0t: float,
    y1b: float, y1t: float,
    color: str, alpha: float = 0.52,
) -> PathPatch:
    cx0 = x0 + (x1 - x0) * 0.35
    cx1 = x0 + (x1 - x0) * 0.65
    verts = [
        (x0, y0b), (cx0, y0b), (cx1, y1b), (x1, y1b),
        (x1, y1t), (cx1, y1t), (cx0, y0t), (x0, y0t),
        (x0, y0b),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha)


def _adjust_positions(
    ys: list[float],
    min_gap: float = 0.038,
    y_min: float = 0.01,
    y_max: float = 0.99,
) -> list[float]:
    if not ys:
        return []
    adj = sorted(max(y_min, min(y, y_max)) for y in ys)
    for i in range(1, len(adj)):
        if adj[i] < adj[i - 1] + min_gap:
            adj[i] = adj[i - 1] + min_gap
    overflow = adj[-1] - y_max
    if overflow > 0:
        adj = [y - overflow for y in adj]
    adj[0] = max(adj[0], y_min)
    for i in range(1, len(adj)):
        if adj[i] < adj[i - 1] + min_gap:
            adj[i] = adj[i - 1] + min_gap
    return [max(y_min, min(y, y_max)) for y in adj]


def save_sankey(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    palette: dict,
) -> None:
    if df.empty or df["area_km2"].sum() <= 0:
        print(f"  No data — skipping {out_path.name}")
        return

    present = set(df["class_2017"].unique()) | set(df["class_2024"].unique())
    order = [c for c in CLASS_ORDER if c in present]

    total = df["area_km2"].sum()
    left_totals = df.groupby("class_2017")["area_km2"].sum()
    right_totals = df.groupby("class_2024")["area_km2"].sum()

    gap = 0.022
    EXTRA_GAP_AFTER = {8: 0.055}  # extra gap between Rivers/Channels (8) and Mangrove Forest (9)
    extra_gap_total = sum(EXTRA_GAP_AFTER.values())
    total_gap = gap * max(0, len(order) - 1) + extra_gap_total
    MIN_BAR_H = 0.016  # minimum bar height in axes fraction for readability

    left_sum_frac = left_totals.reindex(order, fill_value=0.0).sum() / total
    right_sum_frac = right_totals.reindex(order, fill_value=0.0).sum() / total
    sl = (1.0 - total_gap) / max(left_sum_frac, 1e-12)
    sr = (1.0 - total_gap) / max(right_sum_frac, 1e-12)

    def _bar_heights(totals_series: pd.Series, scale: float) -> dict[int, float]:
        raw = {c: (totals_series.get(c, 0.0) / total) * scale for c in order}
        boosted = {c: max(v, MIN_BAR_H) for c, v in raw.items()}
        rescale = (1.0 - total_gap) / max(sum(boosted.values()), 1e-12)
        return {c: v * rescale for c, v in boosted.items()}

    lh = _bar_heights(left_totals, sl)
    rh = _bar_heights(right_totals, sr)

    lpos: dict[int, tuple[float, float]] = {}
    rpos: dict[int, tuple[float, float]] = {}
    y = 1.0
    for i, c in enumerate(order):
        lpos[c] = (y - lh[c], y)
        extra = EXTRA_GAP_AFTER.get(c, 0.0) if i < len(order) - 1 else 0.0
        y -= lh[c] + gap + extra
    y = 1.0
    for i, c in enumerate(order):
        rpos[c] = (y - rh[c], y)
        extra = EXTRA_GAP_AFTER.get(c, 0.0) if i < len(order) - 1 else 0.0
        y -= rh[c] + gap + extra

    lcur = {c: lpos[c][0] for c in order}
    rcur = {c: rpos[c][0] for c in order}

    bg = palette.get("sand", "#FFF9EF")
    tc = palette.get("deep_slate", "#2D3142")

    fig, ax = plt.subplots(figsize=(16, 11), dpi=300, facecolor=bg)
    ax.set_facecolor(bg)

    xl0, xl1 = 0.07, 0.14
    xr0, xr1 = 0.86, 0.93

    for _, row in df.sort_values(["class_2017", "class_2024"]).iterrows():
        s, d = row["class_2017"], row["class_2024"]
        area_sd = row["area_km2"]
        # Scale flows to bar height so they fill bars exactly (bars may be boosted)
        hl = lh[s] * area_sd / max(left_totals.get(s, 0.0), 1e-12)
        hr = rh[d] * area_sd / max(right_totals.get(d, 0.0), 1e-12)
        ax.add_patch(
            _sankey_patch(xl1, xr0, lcur[s], lcur[s] + hl, rcur[d], rcur[d] + hr, LULC_COLORS[d])
        )
        lcur[s] += hl
        rcur[d] += hr

    for c in order:
        yb, yt = lpos[c]
        ax.add_patch(Rectangle(
            (xl0, yb), xl1 - xl0, yt - yb,
            facecolor=LULC_COLORS[c], edgecolor="white", lw=0.6, zorder=2,
        ))
        yb, yt = rpos[c]
        ax.add_patch(Rectangle(
            (xr0, yb), xr1 - xr0, yt - yb,
            facecolor=LULC_COLORS[c], edgecolor="white", lw=0.6, zorder=2,
        ))

    def _label(c: int, totals_series: pd.Series) -> str:
        area = totals_series.get(c, 0.0)
        pct = area / total * 100
        return f"{SHORT_CLASS_NAMES[c]}\n{area:,.0f} km² ({pct:.1f}%)"

    litems = sorted(
        [{"c": c, "yc": sum(lpos[c]) / 2, "text": _label(c, left_totals)} for c in order],
        key=lambda x: x["yc"],
    )
    ritems = sorted(
        [{"c": c, "yc": sum(rpos[c]) / 2, "text": _label(c, right_totals)} for c in order],
        key=lambda x: x["yc"],
    )

    # Classes 9 (Mangrove Forest) and 10 (Bare/Exposed Land) are large enough to
    # place labels directly at the bar centre with no connector line.
    NO_CONNECTOR = {9, 10}

    litems_adj = [it for it in litems if it["c"] not in NO_CONNECTOR]
    litems_fixed = [it for it in litems if it["c"] in NO_CONNECTOR]
    al = _adjust_positions([it["yc"] for it in litems_adj], min_gap=0.058)
    for item, ya in zip(litems_adj, al):
        y_orig = item["yc"]
        tx = xl0 - 0.012
        if abs(ya - y_orig) > 0.002:
            ax.plot([xl0, tx + 0.003], [y_orig, ya], color=LULC_COLORS[item["c"]], lw=1.2, alpha=0.75)
        ax.text(tx, ya, item["text"], ha="right", va="center",
                fontsize=CLASS_LABEL_FONTSIZE, color=tc, linespacing=1.35, zorder=3)
    for item in litems_fixed:
        ax.text(xl0 - 0.012, item["yc"], item["text"], ha="right", va="center",
                fontsize=CLASS_LABEL_FONTSIZE, color=tc, linespacing=1.35, zorder=3)

    ritems_adj = [it for it in ritems if it["c"] not in NO_CONNECTOR]
    ritems_fixed = [it for it in ritems if it["c"] in NO_CONNECTOR]
    ar = _adjust_positions([it["yc"] for it in ritems_adj], min_gap=0.058)
    for item, ya in zip(ritems_adj, ar):
        y_orig = item["yc"]
        tx = xr1 + 0.012
        if abs(ya - y_orig) > 0.002:
            ax.plot([xr1, tx - 0.003], [y_orig, ya], color=LULC_COLORS[item["c"]], lw=1.2, alpha=0.75)
        ax.text(tx, ya, item["text"], ha="left", va="center",
                fontsize=CLASS_LABEL_FONTSIZE, color=tc, linespacing=1.35, zorder=3)
    for item in ritems_fixed:
        ax.text(xr1 + 0.012, item["yc"], item["text"], ha="left", va="center",
                fontsize=CLASS_LABEL_FONTSIZE, color=tc, linespacing=1.35, zorder=3)

    ax.text((xl0 + xl1) / 2, 1.050, "2017",
            ha="center", va="bottom", fontsize=YEAR_LABEL_FONTSIZE, fontweight="bold", color=tc)
    ax.text((xr0 + xr1) / 2, 1.050, "2024",
            ha="center", va="bottom", fontsize=YEAR_LABEL_FONTSIZE, fontweight="bold", color=tc)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, color=tc, pad=24)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.04, 1.12)
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=bg, dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    transition_path = resolve_path(args.transition)
    sundarbans_path = resolve_path(args.sundarbans_map)
    output_dir = resolve_path(args.output_dir)

    if not transition_path.exists():
        raise FileNotFoundError(f"Transition raster not found: {transition_path}")
    if not sundarbans_path.exists():
        raise FileNotFoundError(f"Sundarbans map not found: {sundarbans_path}")

    palette = load_palette(args.palette)
    sundarbans = gpd.read_file(sundarbans_path)

    if sundarbans.empty:
        raise ValueError("Sundarbans GeoPackage is empty.")

    with rasterio.open(transition_path) as ds:
        sundarbans = sundarbans.to_crs(ds.crs)
        geoms = [g for g in sundarbans.geometry if g is not None and not g.is_empty]
        if not geoms:
            raise ValueError("Sundarbans GeoPackage has no valid geometries.")

        print("Counting Sundarbans transitions...")
        counts = count_transitions_region(ds, geoms)

    save_sankey(
        build_transition_df(counts),
        "Bangladesh Coastal LULC Change: 2017 → 2024  —  Sundarbans",
        output_dir / "lulc_change_2017vs2024_sankey_sundarbans.png",
        palette,
    )


if __name__ == "__main__":
    main()
