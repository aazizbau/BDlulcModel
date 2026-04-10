#!/usr/bin/env python3
"""
Create a 2 x 4 comparison figure for random AOIs:
- column 1: Sentinel-2 RGB 2017
- column 2: LULC 2017
- column 3: Sentinel-2 RGB 2024
- column 4: LULC 2024

Inputs
------
- data/interim/S2_2017_B2_10m_utm46_bdcoastal_solid.tif
- data/interim/S2_2017_B3_10m_utm46_bdcoastal_solid.tif
- data/interim/S2_2017_B4_10m_utm46_bdcoastal_solid.tif
- data/interim/S2_2024_B2_10m_utm46_bdcoastal_solid.tif
- data/interim/S2_2024_B3_10m_utm46_bdcoastal_solid.tif
- data/interim/S2_2024_B4_10m_utm46_bdcoastal_solid.tif
- outputs/inference/2017/lulc_class_2017.tif
- outputs/inference/2024/lulc_class_2024.tif
- assets/maps/NorthArrow.svg
- assets/color_palette_coastal_lulc.json

Output
------
- outputs/figures/lulc_rgb_rowcolumn_2017_vs_2024.png

Example
-------
python scripts/visualization/visualize_lulc_transition_rowcolumn_2017vs2024.py
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


def ts() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--b2-2017", default="data/interim/S2_2017_B2_10m_utm46_bdcoastal_solid.tif")
    p.add_argument("--b3-2017", default="data/interim/S2_2017_B3_10m_utm46_bdcoastal_solid.tif")
    p.add_argument("--b4-2017", default="data/interim/S2_2017_B4_10m_utm46_bdcoastal_solid.tif")
    p.add_argument("--b2-2024", default="data/interim/S2_2024_B2_10m_utm46_bdcoastal_solid.tif")
    p.add_argument("--b3-2024", default="data/interim/S2_2024_B3_10m_utm46_bdcoastal_solid.tif")
    p.add_argument("--b4-2024", default="data/interim/S2_2024_B4_10m_utm46_bdcoastal_solid.tif")
    p.add_argument("--lulc-2017", default="outputs/inference/2017/lulc_class_2017.tif")
    p.add_argument("--lulc-2024", default="outputs/inference/2024/lulc_class_2024.tif")
    p.add_argument("--output-fig", default="outputs/figures/lulc_rgb_rowcolumn_2017_vs_2024.png")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window", action="append", default=[], help="row,col,height,width")
    p.add_argument("--window-width-px", type=int, default=1024)
    p.add_argument("--window-height-px", type=int, default=768)
    p.add_argument("--max-attempts", type=int, default=250)
    p.add_argument("--n-rows", type=int, default=2)
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
    ax = fig.add_axes([0.88, 0.005, 0.08, 0.10])
    ax.axis("off")
    ax.set_facecolor(bg_color)
    img = load_svg_as_image(resolve_path(NORTH_ARROW_SVG), target_height_px=220)
    if img is not None:
        ax.imshow(np.asarray(img))
        return
    ax.text(0.5, 0.78, "N", ha="center", va="center", fontsize=16, fontweight="bold", color=text_color)
    ax.text(0.5, 0.30, "↑", ha="center", va="center", fontsize=28, fontweight="bold", color=text_color)


def add_figure_scalebar(fig: plt.Figure, extent: tuple[float, float, float, float], text_color: str, edge_color: str, bg_color: str) -> None:
    ax = fig.add_axes([0.06, 0.008, 0.16, 0.075])
    ax.axis("off")
    ax.set_facecolor(bg_color)
    y0 = 0.38
    x0 = 0.10
    total_w = 0.72
    step_w = total_w / 2.0
    ax.add_patch(mpatches.Rectangle((x0, y0), step_w, 0.18, facecolor="black", edgecolor=edge_color, linewidth=1.0))
    ax.add_patch(mpatches.Rectangle((x0 + step_w, y0), step_w, 0.18, facecolor="white", edgecolor=edge_color, linewidth=1.0))
    ax.text(x0, y0 - 0.08, "0", ha="center", va="top", fontsize=13, color=text_color)
    ax.text(x0 + step_w, y0 - 0.08, "75", ha="center", va="top", fontsize=13, color=text_color)
    ax.text(x0 + total_w, y0 - 0.08, "150 km", ha="center", va="top", fontsize=13, color=text_color)


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


def choose_random_window_from_change(
    lulc17_ds: rasterio.DatasetReader,
    lulc24_view: rasterio.DatasetReader,
    width_px: int,
    height_px: int,
    rng: np.random.Generator,
    max_attempts: int,
    used_windows: list[Window] | None = None,
) -> Window:
    win_w = min(width_px, lulc17_ds.width)
    win_h = min(height_px, lulc17_ds.height)
    max_col = max(lulc17_ds.width - win_w, 0)
    max_row = max(lulc17_ds.height - win_h, 0)
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

        arr17 = lulc17_ds.read(1, window=window)
        arr24 = lulc24_view.read(1, window=window)
        valid = (arr17 > 0) & (arr24 > 0)
        changed = valid & (arr17 != arr24)
        if np.any(changed):
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
    b2_2017_view: rasterio.DatasetReader,
    b3_2017_view: rasterio.DatasetReader,
    b4_2017_view: rasterio.DatasetReader,
    b2_2024_view: rasterio.DatasetReader,
    b3_2024_view: rasterio.DatasetReader,
    b4_2024_view: rasterio.DatasetReader,
    lulc17_ds: rasterio.DatasetReader,
    lulc24_view: rasterio.DatasetReader,
    window: Window,
    plot_crs: CRS,
    nodata_color: str,
) -> dict[str, object]:
    rgb17_arr = np.stack([
        b4_2017_view.read(1, window=window),
        b3_2017_view.read(1, window=window),
        b2_2017_view.read(1, window=window),
    ], axis=0)
    rgb24_arr = np.stack([
        b4_2024_view.read(1, window=window),
        b3_2024_view.read(1, window=window),
        b2_2024_view.read(1, window=window),
    ], axis=0)
    lulc17_arr = lulc17_ds.read(1, window=window)
    lulc24_arr = lulc24_view.read(1, window=window)

    rgb17_img = stretch_rgb(rgb17_arr, b2_2017_view.nodata)
    rgb24_img = stretch_rgb(rgb24_arr, b2_2024_view.nodata)
    lulc17_img = lulc_rgb_map(lulc17_arr, nodata_color)
    lulc24_img = lulc_rgb_map(lulc24_arr, nodata_color)

    valid_mask = (lulc17_arr > 0) & (lulc24_arr > 0)
    changed_mask = valid_mask & (lulc17_arr != lulc24_arr)

    bounds = window_bounds(window, lulc17_ds.transform)
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    return {
        "window": window,
        "extent": extent,
        "rgb17_img": rgb17_img,
        "lulc17_img": lulc17_img,
        "rgb24_img": rgb24_img,
        "lulc24_img": lulc24_img,
        "valid_total": int(valid_mask.sum()),
        "changed_total": int(changed_mask.sum()),
        "plot_crs": plot_crs,
    }


def main() -> None:
    args = parse_args()
    input_paths = [
        resolve_path(Path(args.b2_2017)),
        resolve_path(Path(args.b3_2017)),
        resolve_path(Path(args.b4_2017)),
        resolve_path(Path(args.b2_2024)),
        resolve_path(Path(args.b3_2024)),
        resolve_path(Path(args.b4_2024)),
        resolve_path(Path(args.lulc_2017)),
        resolve_path(Path(args.lulc_2024)),
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
    with rasterio.open(input_paths[0]) as b2_2017_src, \
        rasterio.open(input_paths[1]) as b3_2017_src, \
        rasterio.open(input_paths[2]) as b4_2017_src, \
        rasterio.open(input_paths[3]) as b2_2024_src, \
        rasterio.open(input_paths[4]) as b3_2024_src, \
        rasterio.open(input_paths[5]) as b4_2024_src, \
        rasterio.open(input_paths[6]) as lulc17_src, \
        rasterio.open(input_paths[7]) as lulc24_src:

        b2_2017_view = aligned_view(b2_2017_src, lulc17_src, Resampling.bilinear)
        b3_2017_view = aligned_view(b3_2017_src, lulc17_src, Resampling.bilinear)
        b4_2017_view = aligned_view(b4_2017_src, lulc17_src, Resampling.bilinear)
        b2_2024_view = aligned_view(b2_2024_src, lulc17_src, Resampling.bilinear)
        b3_2024_view = aligned_view(b3_2024_src, lulc17_src, Resampling.bilinear)
        b4_2024_view = aligned_view(b4_2024_src, lulc17_src, Resampling.bilinear)
        lulc24_view = aligned_view(lulc24_src, lulc17_src, Resampling.nearest)
        plot_crs = lulc17_src.crs if lulc17_src.crs is not None else CRS.from_user_input("EPSG:32646")

        rows_payload: list[dict[str, object]] = []
        if args.window:
            selected_windows = [parse_window_arg(spec, lulc17_src.height, lulc17_src.width) for spec in args.window]
        else:
            selected_windows = []
            used_windows: list[Window] = []
            for _ in range(int(args.n_rows)):
                selected_windows.append(
                    choose_random_window_from_change(
                        lulc17_src,
                        lulc24_view,
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
                    b2_2017_view,
                    b3_2017_view,
                    b4_2017_view,
                    b2_2024_view,
                    b3_2024_view,
                    b4_2024_view,
                    lulc17_src,
                    lulc24_view,
                    window,
                    plot_crs,
                    colors["panel_bg"],
                )
            )

        for ds_view, ds_src in [
            (b2_2017_view, b2_2017_src), (b3_2017_view, b3_2017_src), (b4_2017_view, b4_2017_src),
            (b2_2024_view, b2_2024_src), (b3_2024_view, b3_2024_src), (b4_2024_view, b4_2024_src),
            (lulc24_view, lulc24_src),
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

    col_titles = ["Sentinel-2 RGB 2017", "LULC 2017", "Sentinel-2 RGB 2024", "LULC 2024"]

    for row_idx, payload in enumerate(rows_payload):
        extent = payload["extent"]
        plot_crs = payload["plot_crs"]
        imgs = [payload["rgb17_img"], payload["lulc17_img"], payload["rgb24_img"], payload["lulc24_img"]]
        axes_row = axes[row_idx]
        for col_idx, ax in enumerate(axes_row):
            ax.set_facecolor(colors["panel_bg"] if col_idx in [0, 2] else colors["panel_bg_alt"])
            ax.imshow(imgs[col_idx], extent=[extent[0], extent[1], extent[2], extent[3]], origin="upper", interpolation="nearest")
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            set_geographic_aspect(ax, extent, plot_crs)
            apply_lonlat_dm_formatters(ax, plot_crs, extent)
            ax.tick_params(axis="both", colors=colors["text"], labelsize=11)
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
        mpatches.Patch(facecolor=LULC_COLORS[class_id], edgecolor=colors["edge"], label=LULC_NAMES[class_id])
        for class_id in range(1, 11)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=True,
        framealpha=0.96,
        bbox_to_anchor=(0.5, 0.02),
        edgecolor=colors["edge"],
        facecolor=colors["panel_bg"],
        fontsize=11,
        handlelength=1.6,
        handleheight=1.2,
        columnspacing=1.0,
    )

    if args.add_main_title:
        fig.suptitle(
            "Sentinel-2 RGB and inferred LULC comparison (2017 vs 2024)",
            fontsize=20,
            y=0.995,
            fontweight="bold",
            color=colors["text"],
        )

    sample_extent = rows_payload[0]["extent"]
    add_figure_scalebar(fig, sample_extent, colors["text"], colors["edge"], colors["panel_bg"])
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
            f"valid={int(payload['valid_total']):,}, changed={int(payload['changed_total']):,}"
        )
    log(f"Saved PNG map: {output_fig}")


if __name__ == "__main__":
    main()
