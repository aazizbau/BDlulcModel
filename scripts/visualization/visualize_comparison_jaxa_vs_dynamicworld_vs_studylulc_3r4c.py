#!/usr/bin/env python3
"""
Create a 2 x 4 comparison figure for random AOIs:
- column 1: Sentinel-2 RGB 2023
- column 2: Dynamic World 2023 (Harmonized)
- column 3: JAXA HRLULC 2023 (Harmonized)
- column 4: Study LULC 2023

Inputs
------
- data/interim/S2_2023_B2_10m_utm46_bdcoastal_solid.tif
- data/interim/S2_2023_B3_10m_utm46_bdcoastal_solid.tif
- data/interim/S2_2023_B4_10m_utm46_bdcoastal_solid.tif
- data/processed/dynamicworld/bd_coastal_dynamicworld_2023_mode_clipped.tif
- data/processed/jaxa_hrlulc/bd_coastal_jaxa_hrlulc_2023_clipped.tif
- outputs/inference/2023/lulc_class_2023.tif
- assets/maps/NorthArrow.svg
- assets/color_palette_coastal_lulc.json

Output
------
- outputs/figures/comparison_jaxa_vs_dynamicworld_vs_studylulc_2023.png

Example
-------
python scripts/visualization/visualize_comparison_jaxa_vs_dynamicworld_vs_studylulc_3r4c.py
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PIL import Image
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, bounds as window_bounds

try:
    import cairosvg

    HAVE_SVG_SUPPORT = True
except Exception:
    HAVE_SVG_SUPPORT = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PALETTE_JSON = Path("assets/color_palette_coastal_lulc.json")
NORTH_ARROW_SVG = Path("assets/maps/NorthArrow.svg")

NORTH_ARROW_X = 0.90
NORTH_ARROW_Y = 0.01
NORTH_ARROW_W = 0.055
NORTH_ARROW_H = 0.075

SCALEBAR_X_FRAC = -0.25
SCALEBAR_Y = 0.012
SCALEBAR_H = 0.07

MY_LULC_NAMES = {
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

DW_TO_MY_LULC = {
    0: 8,
    1: 5,
    2: 2,
    3: 9,
    4: 4,
    5: 5,
    6: 1,
    7: 10,
    8: 10,
}

JAXA_TO_MY_LULC = {
    1: 8,
    2: 1,
    3: 1,
    4: 4,
    5: 4,
    6: 4,
    7: 9,
    8: 2,
    9: 5,
    10: 5,
    11: 5,
    12: 5,
    13: 5,
    14: 9,
    15: 10,
}


def ts() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--b2-2023", default="data/interim/S2_2023_B2_10m_utm46_bdcoastal_solid.tif")
    p.add_argument("--b3-2023", default="data/interim/S2_2023_B3_10m_utm46_bdcoastal_solid.tif")
    p.add_argument("--b4-2023", default="data/interim/S2_2023_B4_10m_utm46_bdcoastal_solid.tif")
    p.add_argument("--dynamicworld", default="data/processed/dynamicworld/bd_coastal_dynamicworld_2023_mode_clipped.tif")
    p.add_argument("--jaxa", default="data/processed/jaxa_hrlulc/bd_coastal_jaxa_hrlulc_2023_clipped.tif")
    p.add_argument("--study-lulc", default="outputs/inference/2023/lulc_class_2023.tif")
    p.add_argument("--output-fig", default="outputs/figures/comparison_jaxa_vs_dynamicworld_vs_studylulc_2023_3r4c.png")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window", action="append", default=[], help="row,col,height,width")
    p.add_argument("--window-width-px", type=int, default=1024)
    p.add_argument("--window-height-px", type=int, default=768)
    p.add_argument("--max-attempts", type=int, default=250)
    p.add_argument("--n-rows", type=int, default=3)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--add-main-title", action="store_true")
    return p.parse_args()


def load_palette(path: Path) -> dict:
    return json.loads(path.read_text())


def hex_to_rgb01(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def decimal_to_dm(value: float, kind: str) -> str:
    hemi = "E" if kind == "lon" and value >= 0 else "W" if kind == "lon" else "N" if value >= 0 else "S"
    v = abs(value)
    deg = int(v)
    minute = int(round((v - deg) * 60.0))
    if minute == 60:
        deg += 1
        minute = 0
    return f"{deg}°{minute:02d}'{hemi}"


def apply_lonlat_dm_formatters(ax: plt.Axes, src_crs: CRS, extent: tuple[float, float, float, float]) -> None:
    transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
    xmin, xmax, ymin, ymax = extent
    y_mid = 0.5 * (ymin + ymax)
    x_mid = 0.5 * (xmin + xmax)

    def fmt_x(x: float, pos: float | None = None) -> str:
        lon, _ = transformer.transform(x, y_mid)
        return decimal_to_dm(lon, "lon")

    def fmt_y(y: float, pos: float | None = None) -> str:
        _, lat = transformer.transform(x_mid, y)
        return decimal_to_dm(lat, "lat")

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_x))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))


def set_geographic_aspect(ax: plt.Axes, extent: tuple[float, float, float, float], crs: CRS) -> None:
    _, _, ymin, ymax = extent
    if "4326" in crs.to_string() or getattr(crs, "is_geographic", False):
        mean_lat = 0.5 * (ymin + ymax)
        cosv = np.cos(np.deg2rad(mean_lat))
        ax.set_aspect("equal" if abs(cosv) < 1e-8 else 1.0 / cosv)
    else:
        ax.set_aspect("equal")


def load_svg_as_image(svg_path: Path, target_height_px: int = 220):
    if not svg_path.exists() or not HAVE_SVG_SUPPORT:
        return None
    png_bytes = cairosvg.svg2png(url=str(svg_path), output_height=target_height_px)
    return Image.open(io.BytesIO(png_bytes))


def add_north_arrow(fig: plt.Figure, bg_color: str, text_color: str) -> None:
    ax = fig.add_axes([NORTH_ARROW_X, NORTH_ARROW_Y, NORTH_ARROW_W, NORTH_ARROW_H])
    ax.axis("off")
    ax.set_facecolor(bg_color)
    img = load_svg_as_image(resolve_path(NORTH_ARROW_SVG), target_height_px=160)
    if img is not None:
        ax.imshow(np.asarray(img))
        return
    ax.text(0.5, 0.78, "N", ha="center", va="center", fontsize=13, fontweight="bold", color=text_color)
    ax.text(0.5, 0.30, "↑", ha="center", va="center", fontsize=22, fontweight="bold", color=text_color)


def add_fixed_scalebar(fig: plt.Figure, anchor_ax: plt.Axes, extent: tuple[float, float, float, float], crs: CRS, text_color: str, edge_color: str, bg_color: str) -> None:
    xmin, xmax, _, _ = extent
    if "4326" in crs.to_string() or getattr(crs, "is_geographic", False):
        raise ValueError("Expected projected CRS for fixed metric scale bar.")
    width_m = xmax - xmin
    if width_m <= 0:
        return
    bar_frac = 10_000.0 / width_m
    bbox = anchor_ax.get_position()
    scalebar_width = bbox.width * bar_frac
    scalebar_left = bbox.x0 + bbox.width * SCALEBAR_X_FRAC
    scalebar_bottom = SCALEBAR_Y
    scalebar_height = SCALEBAR_H

    ax = fig.add_axes([scalebar_left, scalebar_bottom, scalebar_width, scalebar_height])
    ax.axis("off")
    ax.set_facecolor(bg_color)
    y0 = 0.38
    x0 = 0.0
    total_w = 1.0
    step_w = total_w / 2.0
    ax.add_patch(mpatches.Rectangle((x0, y0), step_w, 0.18, facecolor="black", edgecolor=edge_color, linewidth=1.0))
    ax.add_patch(mpatches.Rectangle((x0 + step_w, y0), step_w, 0.18, facecolor="white", edgecolor=edge_color, linewidth=1.0))
    ax.text(x0, y0 - 0.08, "0", ha="center", va="top", fontsize=13, color=text_color)
    ax.text(x0 + step_w, y0 - 0.08, "5", ha="center", va="top", fontsize=13, color=text_color)
    ax.text(x0 + total_w, y0 - 0.08, "10 km", ha="center", va="top", fontsize=13, color=text_color)


def aligned_view(src: rasterio.DatasetReader, ref: rasterio.DatasetReader, resampling: Resampling) -> rasterio.DatasetReader:
    same_grid = (
        src.crs == ref.crs
        and src.transform == ref.transform
        and src.width == ref.width
        and src.height == ref.height
    )
    if same_grid:
        return src
    return WarpedVRT(
        src,
        crs=ref.crs,
        transform=ref.transform,
        width=ref.width,
        height=ref.height,
        resampling=resampling,
    )


def parse_window_arg(spec: str, max_height: int, max_width: int) -> Window:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise SystemExit(f"Invalid --window value '{spec}'. Expected row,col,height,width")
    row_off, col_off, height, width = [int(p) for p in parts]
    if row_off < 0 or col_off < 0 or height <= 0 or width <= 0:
        raise SystemExit(f"Invalid --window value '{spec}'.")
    if row_off + height > max_height or col_off + width > max_width:
        raise SystemExit(f"Invalid --window value '{spec}'. Window exceeds raster bounds.")
    return Window(col_off=col_off, row_off=row_off, width=width, height=height)


def harmonize_dynamicworld(arr: np.ndarray, nodata_value: float | int | None) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.uint8)
    valid = np.isfinite(arr)
    if nodata_value is not None:
        valid &= arr != nodata_value
    valid &= arr >= 0
    for src_class, dst_class in DW_TO_MY_LULC.items():
        out[valid & (arr == src_class)] = dst_class
    return out


def harmonize_jaxa(arr: np.ndarray, nodata_value: float | int | None) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.uint8)
    valid = np.isfinite(arr)
    if nodata_value is not None:
        valid &= arr != nodata_value
    valid &= arr > 0
    for src_class, dst_class in JAXA_TO_MY_LULC.items():
        out[valid & (arr == src_class)] = dst_class
    return out


def choose_random_window_for_comparison(
    study_ds: rasterio.DatasetReader,
    dw_view: rasterio.DatasetReader,
    jaxa_view: rasterio.DatasetReader,
    width_px: int,
    height_px: int,
    rng: np.random.Generator,
    max_attempts: int,
    used_windows: list[Window] | None = None,
) -> Window:
    win_w = min(width_px, study_ds.width)
    win_h = min(height_px, study_ds.height)
    max_col = max(study_ds.width - win_w, 0)
    max_row = max(study_ds.height - win_h, 0)
    fallback: Window | None = None

    for _ in range(max_attempts):
        col_off = int(rng.integers(0, max_col + 1)) if max_col > 0 else 0
        row_off = int(rng.integers(0, max_row + 1)) if max_row > 0 else 0
        window = Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)
        if used_windows is not None:
            duplicate = any(
                int(window.col_off) == int(prev.col_off)
                and int(window.row_off) == int(prev.row_off)
                and int(window.width) == int(prev.width)
                and int(window.height) == int(prev.height)
                for prev in used_windows
            )
            if duplicate:
                continue

        study_arr = study_ds.read(1, window=window)
        dw_arr = harmonize_dynamicworld(dw_view.read(1, window=window), dw_view.nodata)
        jaxa_arr = harmonize_jaxa(jaxa_view.read(1, window=window), jaxa_view.nodata)
        valid = (study_arr > 0) & (dw_arr > 0) & (jaxa_arr > 0)
        disagreement = valid & ((study_arr != dw_arr) | (study_arr != jaxa_arr) | (dw_arr != jaxa_arr))
        if np.any(disagreement):
            return window
        if np.any(valid):
            fallback = window

    if fallback is not None:
        return fallback
    raise SystemExit(f"Could not find a random AOI after {max_attempts} attempts")


def stretch_rgb(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    out = np.zeros((arr.shape[1], arr.shape[2], 3), dtype=np.float32)
    for idx in range(3):
        band = arr[idx]
        valid = np.isfinite(band)
        if nodata is not None:
            valid &= band != float(nodata)
        if np.any(valid):
            vals = band[valid]
            lo, hi = np.percentile(vals, [2, 98])
            if hi <= lo:
                lo = float(vals.min())
                hi = float(vals.max())
            scaled = np.clip((band - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            scaled[~valid] = 0.0
        else:
            scaled = np.zeros_like(band, dtype=np.float32)
        out[:, :, idx] = scaled
    return out


def lulc_rgb_map(lulc_arr: np.ndarray, nodata_color: str) -> np.ndarray:
    h, w = lulc_arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:] = hex_to_rgb01(nodata_color)
    for cls, hex_color in LULC_COLORS.items():
        rgb[lulc_arr == cls] = hex_to_rgb01(hex_color)
    return rgb


def read_window_payload(
    b2_view: rasterio.DatasetReader,
    b3_view: rasterio.DatasetReader,
    b4_view: rasterio.DatasetReader,
    dw_view: rasterio.DatasetReader,
    jaxa_view: rasterio.DatasetReader,
    study_ds: rasterio.DatasetReader,
    window: Window,
    plot_crs: CRS,
    nodata_color: str,
) -> dict[str, object]:
    rgb_arr = np.stack(
        [
            b4_view.read(1, window=window),
            b3_view.read(1, window=window),
            b2_view.read(1, window=window),
        ],
        axis=0,
    )
    dw_h = harmonize_dynamicworld(dw_view.read(1, window=window), dw_view.nodata)
    jaxa_h = harmonize_jaxa(jaxa_view.read(1, window=window), jaxa_view.nodata)
    study_arr = study_ds.read(1, window=window)

    rgb_img = stretch_rgb(rgb_arr, b2_view.nodata)
    dw_img = lulc_rgb_map(dw_h, nodata_color)
    jaxa_img = lulc_rgb_map(jaxa_h, nodata_color)
    study_img = lulc_rgb_map(study_arr, nodata_color)

    valid_mask = (dw_h > 0) & (jaxa_h > 0) & (study_arr > 0)
    disagreement_mask = valid_mask & ((dw_h != jaxa_h) | (dw_h != study_arr) | (jaxa_h != study_arr))

    bounds = window_bounds(window, study_ds.transform)
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    return {
        "window": window,
        "extent": extent,
        "rgb_img": rgb_img,
        "dw_img": dw_img,
        "jaxa_img": jaxa_img,
        "study_img": study_img,
        "valid_total": int(valid_mask.sum()),
        "disagreement_total": int(disagreement_mask.sum()),
        "plot_crs": plot_crs,
    }


def main() -> None:
    args = parse_args()
    input_paths = [
        resolve_path(Path(args.b2_2023)),
        resolve_path(Path(args.b3_2023)),
        resolve_path(Path(args.b4_2023)),
        resolve_path(Path(args.dynamicworld)),
        resolve_path(Path(args.jaxa)),
        resolve_path(Path(args.study_lulc)),
    ]
    output_fig = resolve_path(Path(args.output_fig))

    for path in input_paths:
        if not path.exists():
            raise SystemExit(f"Input not found: {path}")

    palette = load_palette(resolve_path(PALETTE_JSON)) if resolve_path(PALETTE_JSON).exists() else {}
    colors_json = palette.get("colors", {})
    colors = {
        "fig_bg": colors_json.get("sand", "#FBEDD4"),
        "panel_bg": colors_json.get("sand", "#FBEDD4"),
        "panel_bg_alt": colors_json.get("mist_gray", "#B6B5B8"),
        "text": colors_json.get("deep_slate", "#314245"),
        "grid": colors_json.get("deep_slate", "#314245"),
        "edge": colors_json.get("deep_slate", "#314245"),
    }

    rng = np.random.default_rng(int(args.seed))
    with rasterio.open(input_paths[0]) as b2_src, \
        rasterio.open(input_paths[1]) as b3_src, \
        rasterio.open(input_paths[2]) as b4_src, \
        rasterio.open(input_paths[3]) as dw_src, \
        rasterio.open(input_paths[4]) as jaxa_src, \
        rasterio.open(input_paths[5]) as study_src:

        b2_view = aligned_view(b2_src, study_src, Resampling.bilinear)
        b3_view = aligned_view(b3_src, study_src, Resampling.bilinear)
        b4_view = aligned_view(b4_src, study_src, Resampling.bilinear)
        dw_view = aligned_view(dw_src, study_src, Resampling.nearest)
        jaxa_view = aligned_view(jaxa_src, study_src, Resampling.nearest)
        plot_crs = study_src.crs if study_src.crs is not None else CRS.from_user_input("EPSG:32646")

        rows_payload: list[dict[str, object]] = []
        if args.window:
            selected_windows = [parse_window_arg(spec, study_src.height, study_src.width) for spec in args.window]
        else:
            selected_windows = []
            used_windows: list[Window] = []
            for _ in range(int(args.n_rows)):
                selected_windows.append(
                    choose_random_window_for_comparison(
                        study_src,
                        dw_view,
                        jaxa_view,
                        int(args.window_width_px),
                        int(args.window_height_px),
                        rng,
                        int(args.max_attempts),
                        used_windows=used_windows,
                    )
                )
                used_windows.append(selected_windows[-1])

        for window in selected_windows:
            rows_payload.append(
                read_window_payload(
                    b2_view,
                    b3_view,
                    b4_view,
                    dw_view,
                    jaxa_view,
                    study_src,
                    window,
                    plot_crs,
                    colors["panel_bg"],
                )
            )

        for ds_view, ds_src in [
            (b2_view, b2_src),
            (b3_view, b3_src),
            (b4_view, b4_src),
            (dw_view, dw_src),
            (jaxa_view, jaxa_src),
        ]:
            if ds_view is not ds_src:
                ds_view.close()

    output_fig.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(rows_payload)
    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(20, 4.8 * n_rows),
        facecolor=colors["fig_bg"],
        gridspec_kw={"wspace": 0.08, "hspace": 0.30},
    )
    if n_rows == 1:
        axes = np.asarray([axes])

    col_titles = [
        "Sentinel-2 RGB 2023",
        "Dynamic World 2023 (Harmonized)",
        "JAXA HRLULC 2023 (Harmonized)",
        "Study LULC 2023",
    ]

    for row_idx, payload in enumerate(rows_payload):
        extent = payload["extent"]
        plot_crs = payload["plot_crs"]
        imgs = [payload["rgb_img"], payload["dw_img"], payload["jaxa_img"], payload["study_img"]]
        axes_row = axes[row_idx]
        for col_idx, ax in enumerate(axes_row):
            ax.set_facecolor(colors["panel_bg"] if col_idx == 0 else colors["panel_bg_alt"])
            ax.imshow(imgs[col_idx], extent=[extent[0], extent[1], extent[2], extent[3]], origin="upper", interpolation="nearest")
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            set_geographic_aspect(ax, extent, plot_crs)
            apply_lonlat_dm_formatters(ax, plot_crs, extent)
            ax.tick_params(axis="both", colors=colors["text"], labelsize=11)
            if col_idx > 0:
                ax.tick_params(axis="y", labelleft=False)
            for spine in ax.spines.values():
                spine.set_color(colors["edge"])
                spine.set_linewidth(1.0)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.28, color=colors["grid"])
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=15, pad=10, fontweight="bold", color=colors["text"])
            ax.set_ylabel("Latitude" if col_idx == 0 else "", fontsize=13, color=colors["text"])
            ax.set_xlabel("Longitude" if row_idx == n_rows - 1 else "", fontsize=13, color=colors["text"])

    legend_handles = [
        mpatches.Patch(facecolor=LULC_COLORS[class_id], edgecolor=colors["edge"], label=MY_LULC_NAMES[class_id])
        for class_id in range(1, 11)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=True,
        framealpha=0.96,
        bbox_to_anchor=(0.5, -0.06),
        edgecolor=colors["edge"],
        facecolor=colors["panel_bg"],
        fontsize=11,
        handlelength=1.6,
        handleheight=1.2,
        columnspacing=1.0,
    )

    if args.add_main_title:
        fig.suptitle(
            "Sentinel-2 RGB, Dynamic World, JAXA HRLULC, and Study LULC comparison (2023)",
            fontsize=20,
            y=0.995,
            fontweight="bold",
            color=colors["text"],
        )

    sample_extent = rows_payload[0]["extent"]
    sample_crs = rows_payload[0]["plot_crs"]
    add_fixed_scalebar(fig, axes[0][0], sample_extent, sample_crs, colors["text"], colors["edge"], colors["panel_bg"])
    add_north_arrow(fig, colors["panel_bg"], colors["text"])
    top_margin = 0.92 if args.add_main_title else 0.965
    fig.subplots_adjust(left=0.055, right=0.985, top=top_margin, bottom=0.14, wspace=0.08, hspace=0.30)
    fig.savefig(output_fig, dpi=args.dpi, bbox_inches="tight", facecolor=colors["fig_bg"])
    plt.close(fig)

    for row_idx, payload in enumerate(rows_payload, start=1):
        window = payload["window"]
        log(
            f"AOI {row_idx}: row={int(window.row_off)}, col={int(window.col_off)}, "
            f"h={int(window.height)}, w={int(window.width)}, "
            f"valid={int(payload['valid_total']):,}, disagreement={int(payload['disagreement_total']):,}"
        )
    log(f"Saved PNG map: {output_fig}")


if __name__ == "__main__":
    main()
