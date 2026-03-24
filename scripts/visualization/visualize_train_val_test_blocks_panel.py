#!/usr/bin/env python3
"""
Create a publication-style train/validation/test spatial split figure.

Layout
------
- Panel (a): larger full study area
- Panel (b): smaller zoomed AOI
- Bottom panel: centered legend + centered blocks/samples counts

Features
--------
- Train blocks: green, alpha=0.5
- Validation blocks: yellow, alpha=0.5
- Test blocks: blue, alpha=0.5
- Thin black outline around each block
- Red AOI box in panel (a)
- Two red connector lines from panel (a) AOI to panel (b)
- North arrow from SVG with robust CairoSVG + Pillow rendering
- Metric scale bars in both panel (a) and panel (b)
- Longitude / latitude labels in degree-minute format
- Correct geographic aspect handling for EPSG:4326

Example run:
python scripts/visualization/visualize_train_val_test_blocks_panel.py \
    --npz data/processed/training/ae64_samples_4upazila_2023_trainvaltest.npz \
    --outfig outputs/figures/train_val_test_blocks_panel.png \
    --north-arrow assets/maps/NorthArrow.svg \
    --seed 42 \
    --context-blocks 1 \
    --dpi 300 \
    --add-main-title
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pyproj import CRS, Geod, Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window

try:
    import geopandas as gpd  # type: ignore

    HAVE_GPD = True
except Exception:
    HAVE_GPD = False

try:
    import cairosvg  # type: ignore
    from PIL import Image  # type: ignore

    HAVE_SVG_SUPPORT = True
except Exception:
    HAVE_SVG_SUPPORT = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_AREA_GPKG = Path("assets/maps/bd_coastal_map_solid_gp.gpkg")
STUDY_AREA_EDGE_COLOR = "#FA6200"


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--npz",
        default="data/processed/training/ae64_samples_4upazila_2023_trainvaltest.npz",
        help="NPZ created by the extraction script",
    )
    p.add_argument("--background", default="", help="Optional background raster")
    p.add_argument(
        "--outfig",
        default="outputs/figures/train_val_test_blocks_panel.png",
        help="Output figure path",
    )
    p.add_argument(
        "--north-arrow",
        default="assets/maps/NorthArrow.svg",
        help="North arrow SVG path",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--context-blocks",
        type=int,
        default=1,
        help="Number of extra blocks to include around the selected AOI block in panel (b)",
    )
    p.add_argument(
        "--max-bg-size",
        type=int,
        default=1600,
        help="Maximum background raster display size for the zoom panel",
    )
    p.add_argument("--dpi", type=int, default=300, help="Output DPI")
    p.add_argument(
        "--title",
        default="Train/Validation/Test Spatial Split Blocks",
        help="Figure title",
    )
    p.add_argument(
        "--add-main-title",
        action="store_true",
        help="Show the main figure title",
    )
    p.add_argument("--no-grid", action="store_true", help="Disable faint map grid")
    p.add_argument(
        "--scalebar-a-m",
        type=float,
        default=0.0,
        help="Override panel (a) scale bar length in meters. 0 = automatic",
    )
    p.add_argument(
        "--scalebar-b-m",
        type=float,
        default=0.0,
        help="Override panel (b) scale bar length in meters. 0 = automatic",
    )
    return p.parse_args()


def load_npz_and_meta(npz_path: Path) -> Tuple[Dict[str, int], Dict]:
    data = np.load(npz_path, allow_pickle=True)
    required = ["y_train", "y_val", "meta"]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing required NPZ key: {key}")

    meta_raw = data["meta"]
    if isinstance(meta_raw, np.ndarray):
        meta_raw = meta_raw.item()
    if isinstance(meta_raw, bytes):
        meta_raw = meta_raw.decode("utf-8")
    if isinstance(meta_raw, str):
        meta = json.loads(meta_raw)
    elif isinstance(meta_raw, dict):
        meta = meta_raw
    else:
        raise TypeError("Could not decode NPZ 'meta' field")

    sample_counts = {
        "train": int(np.asarray(data["y_train"]).shape[0]),
        "val": int(np.asarray(data["y_val"]).shape[0]),
        "test": int(np.asarray(data["y_test"]).shape[0]) if "y_test" in data else 0,
    }
    return sample_counts, meta


def block_assign(row: int, col: int, block_px: int, seed: int) -> float:
    br = row // block_px
    bc = col // block_px
    v = (br * 73856093) ^ (bc * 19349663) ^ (seed * 83492791)
    v = v & 0xFFFFFFFFFFFFFFFF
    return (v % 10_000_000) / 10_000_000.0


def assign_split(u: float, val_frac: float, test_frac: float) -> str:
    if u < test_frac:
        return "test"
    if u < (test_frac + val_frac):
        return "val"
    return "train"


def choose_random_block(
    block_infos: Sequence[Dict[str, Any]],
    rng: np.random.Generator,
) -> Tuple[str, int, int]:
    all_blocks: List[Tuple[str, int, int]] = []
    for info in block_infos:
        for split in ("train", "val", "test"):
            blocks = info["blocks"][split]
            for br, bc in blocks.tolist():
                all_blocks.append((str(info["upazila"]), int(br), int(bc)))
    if not all_blocks:
        raise RuntimeError("No valid split blocks could be reconstructed from the label rasters.")
    idx = int(rng.integers(0, len(all_blocks)))
    return all_blocks[idx]


def block_window_pixels(
    block_rc: Tuple[int, int],
    block_px: int,
    height: int,
    width: int,
    context_blocks: int,
) -> Tuple[int, int, int, int]:
    br, bc = block_rc
    r0 = max(0, (br - context_blocks) * block_px)
    c0 = max(0, (bc - context_blocks) * block_px)
    r1 = min(height, (br + context_blocks + 1) * block_px)
    c1 = min(width, (bc + context_blocks + 1) * block_px)
    return r0, r1, c0, c1


def block_rectangles_in_window(
    block_ids: np.ndarray,
    block_px: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
) -> List[Tuple[int, int, int, int]]:
    rects: List[Tuple[int, int, int, int]] = []
    for br, bc in block_ids:
        rr0 = int(br * block_px)
        cc0 = int(bc * block_px)
        rr1 = rr0 + block_px
        cc1 = cc0 + block_px
        if rr1 <= r0 or rr0 >= r1 or cc1 <= c0 or cc0 >= c1:
            continue
        rects.append((rr0, rr1, cc0, cc1))
    return rects


def window_transform_from_meta(transform: Affine, r0: int, c0: int) -> Affine:
    return transform * Affine.translation(c0, r0)


def extent_from_window_shape_transform(h: int, w: int, transform: Affine) -> Tuple[float, float, float, float]:
    x0, y0 = rasterio.transform.xy(transform, 0, 0, offset="ul")
    x1, y1 = rasterio.transform.xy(transform, h - 1, w - 1, offset="lr")
    return min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)


def pixel_rect_to_map_extent(
    rr0: int,
    rr1: int,
    cc0: int,
    cc1: int,
    transform: Affine,
) -> Tuple[float, float, float, float]:
    x_left, y_top = rasterio.transform.xy(transform, rr0, cc0, offset="ul")
    x_right, y_bottom = rasterio.transform.xy(transform, rr1 - 1, cc1 - 1, offset="lr")
    return min(x_left, x_right), max(x_left, x_right), min(y_bottom, y_top), max(y_bottom, y_top)


def read_gray_background(
    raster_path: Path,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    max_bg_size: int,
) -> Tuple[np.ndarray, Affine, CRS]:
    with rasterio.open(raster_path) as ds:
        win = Window(c0, r0, c1 - c0, r1 - r0)

        out_h = int(win.height)
        out_w = int(win.width)

        scale = max(out_h / max_bg_size, out_w / max_bg_size, 1.0)
        read_h = max(1, int(round(out_h / scale)))
        read_w = max(1, int(round(out_w / scale)))

        arr = ds.read(
            1,
            window=win,
            out_shape=(read_h, read_w),
            resampling=Resampling.nearest,
        ).astype(np.float32)

        transform = ds.window_transform(win) * Affine.scale(win.width / read_w, win.height / read_h)
        crs = ds.crs

    finite = np.isfinite(arr)
    if finite.any():
        vals = arr[finite]
        lo = np.percentile(vals, 2)
        hi = np.percentile(vals, 98)
        if hi <= lo:
            lo = float(np.min(vals))
            hi = float(np.max(vals))
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    return arr, transform, crs


def read_gray_background_by_bounds(
    raster_path: Path,
    bounds: Tuple[float, float, float, float],
    max_bg_size: int,
) -> Tuple[np.ndarray, Affine, CRS]:
    xmin, xmax, ymin, ymax = bounds
    with rasterio.open(raster_path) as ds:
        win = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, transform=ds.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, ds.width, ds.height))

        out_h = int(max(1, win.height))
        out_w = int(max(1, win.width))

        scale = max(out_h / max_bg_size, out_w / max_bg_size, 1.0)
        read_h = max(1, int(round(out_h / scale)))
        read_w = max(1, int(round(out_w / scale)))

        arr = ds.read(
            1,
            window=win,
            out_shape=(read_h, read_w),
            resampling=Resampling.nearest,
        ).astype(np.float32)

        transform = ds.window_transform(win) * Affine.scale(win.width / read_w, win.height / read_h)
        crs = ds.crs

    finite = np.isfinite(arr)
    if finite.any():
        vals = arr[finite]
        lo = np.percentile(vals, 2)
        hi = np.percentile(vals, 98)
        if hi <= lo:
            lo = float(np.min(vals))
            hi = float(np.max(vals))
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    return arr, transform, crs


def reconstruct_split_blocks(meta: Dict, chunk: int = 1024) -> Tuple[List[Dict[str, Any]], CRS]:
    if "label_paths" not in meta or not meta["label_paths"]:
        raise KeyError("meta['label_paths'] is required to reconstruct split blocks.")

    block_px = int(meta["block_px"])
    seed = int(meta["seed"])
    val_frac = float(meta["val_frac"])
    test_frac = float(meta["test_frac"])
    label_nodata = int(meta.get("label_nodata", 0))
    ae_nodata = float(meta.get("ae_nodata", 0.0))
    ae_path = resolve_path(str(meta["ae_path"])) if meta.get("ae_path") else None

    ae_src = rasterio.open(ae_path) if ae_path and ae_path.exists() else None
    full_crs: CRS | None = None
    infos: List[Dict[str, Any]] = []

    try:
        for upazila, raw_label_path in meta["label_paths"].items():
            label_path = resolve_path(str(raw_label_path))
            with rasterio.open(label_path) as lbl:
                if full_crs is None:
                    full_crs = CRS.from_user_input(lbl.crs)
                elif CRS.from_user_input(lbl.crs) != full_crs:
                    raise SystemExit(f"CRS mismatch across label rasters: {label_path}")

                split_sets = {"train": set(), "val": set(), "test": set()}

                if ae_src is not None:
                    if lbl.crs != ae_src.crs:
                        raise SystemExit(f"[{upazila}] CRS mismatch: label {lbl.crs} vs AE {ae_src.crs}")

                    lbl_bounds = lbl.bounds
                    ae_bounds = ae_src.bounds
                    left = max(lbl_bounds.left, ae_bounds.left)
                    right = min(lbl_bounds.right, ae_bounds.right)
                    bottom = max(lbl_bounds.bottom, ae_bounds.bottom)
                    top = min(lbl_bounds.top, ae_bounds.top)
                    if left >= right or bottom >= top:
                        continue

                    lbl_win_full = rasterio.windows.from_bounds(left, bottom, right, top, transform=lbl.transform)
                    ae_win_full = rasterio.windows.from_bounds(left, bottom, right, top, transform=ae_src.transform)
                    lbl_win_full = lbl_win_full.round_offsets().round_lengths()
                    ae_win_full = ae_win_full.round_offsets().round_lengths()

                    h = min(int(lbl_win_full.height), int(ae_win_full.height))
                    w = min(int(lbl_win_full.width), int(ae_win_full.width))
                    lbl_r0 = int(lbl_win_full.row_off)
                    lbl_c0 = int(lbl_win_full.col_off)
                    ae_r0 = int(ae_win_full.row_off)
                    ae_c0 = int(ae_win_full.col_off)
                else:
                    lbl_r0 = 0
                    lbl_c0 = 0
                    ae_r0 = 0
                    ae_c0 = 0
                    h = int(lbl.height)
                    w = int(lbl.width)

                for row_off in range(0, h, chunk):
                    rh = min(chunk, h - row_off)
                    for col_off in range(0, w, chunk):
                        cw = min(chunk, w - col_off)
                        lbl_win = Window(lbl_c0 + col_off, lbl_r0 + row_off, cw, rh)
                        y = lbl.read(1, window=lbl_win)
                        valid_mask = (y != label_nodata) & (y >= 1) & (y <= 10)
                        if not np.any(valid_mask):
                            continue

                        if ae_src is not None:
                            ae_win = Window(ae_c0 + col_off, ae_r0 + row_off, cw, rh)
                            X_ae = ae_src.read(list(range(1, ae_src.count + 1)), window=ae_win).astype(np.float32)
                            if ae_nodata == 0.0:
                                ae_valid = np.all(X_ae != 0.0, axis=0)
                            else:
                                ae_valid = np.all(X_ae != ae_nodata, axis=0)
                            ae_valid &= np.all(np.isfinite(X_ae), axis=0)
                            valid_mask &= ae_valid
                            if not np.any(valid_mask):
                                continue

                        rows, cols = np.where(valid_mask)
                        if rows.size == 0:
                            continue

                        global_rows = rows + int(lbl_win.row_off)
                        global_cols = cols + int(lbl_win.col_off)
                        block_ids = np.unique(
                            np.stack([global_rows // block_px, global_cols // block_px], axis=1),
                            axis=0,
                        )
                        for br, bc in block_ids:
                            split = assign_split(
                                u=block_assign(int(br * block_px), int(bc * block_px), block_px, seed),
                                val_frac=val_frac,
                                test_frac=test_frac,
                            )
                            split_sets[split].add((int(br), int(bc)))

                infos.append(
                    {
                        "upazila": str(upazila),
                        "label_path": label_path,
                        "transform": lbl.transform,
                        "crs": CRS.from_user_input(lbl.crs),
                        "width": int(lbl.width),
                        "height": int(lbl.height),
                        "extent": (lbl.bounds.left, lbl.bounds.right, lbl.bounds.bottom, lbl.bounds.top),
                        "blocks": {
                            split: (
                                np.array(sorted(split_sets[split]), dtype=np.int64)
                                if split_sets[split]
                                else np.zeros((0, 2), dtype=np.int64)
                            )
                            for split in ("train", "val", "test")
                        },
                    }
                )
    finally:
        if ae_src is not None:
            ae_src.close()

    if full_crs is None:
        raise RuntimeError("No valid label rasters were found for split reconstruction.")
    return infos, full_crs


def union_extent(extents: Sequence[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    xmin = min(e[0] for e in extents)
    xmax = max(e[1] for e in extents)
    ymin = min(e[2] for e in extents)
    ymax = max(e[3] for e in extents)
    return xmin, xmax, ymin, ymax


def block_ids_to_rects(block_ids: np.ndarray, block_px: int) -> List[Tuple[int, int, int, int]]:
    return [
        (
            int(br * block_px),
            int(br * block_px + block_px),
            int(bc * block_px),
            int(bc * block_px + block_px),
        )
        for br, bc in block_ids
    ]


def decimal_to_dm(value: float, kind: str) -> str:
    hemi = ""
    if kind == "lon":
        hemi = "E" if value >= 0 else "W"
    elif kind == "lat":
        hemi = "N" if value >= 0 else "S"

    value_abs = abs(value)
    deg = int(value_abs)
    minute = int(round((value_abs - deg) * 60.0))

    if minute == 60:
        deg += 1
        minute = 0

    return f"{deg}°{minute:02d}'{hemi}"


def apply_lonlat_dm_formatters(ax, src_crs: CRS, extent: Tuple[float, float, float, float]) -> None:
    dst_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    xmin, xmax, ymin, ymax = extent
    y_mid = 0.5 * (ymin + ymax)
    x_mid = 0.5 * (xmin + xmax)

    def fmt_x(x, pos=None):
        lon, _ = transformer.transform(x, y_mid)
        return decimal_to_dm(lon, "lon")

    def fmt_y(y, pos=None):
        _, lat = transformer.transform(x_mid, y)
        return decimal_to_dm(lat, "lat")

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_x))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))


def set_geographic_aspect(ax, extent: Tuple[float, float, float, float], crs: CRS) -> None:
    _, _, ymin, ymax = extent
    if "4326" in crs.to_string() or getattr(crs, "is_geographic", False):
        mean_lat = 0.5 * (ymin + ymax)
        cosv = np.cos(np.deg2rad(mean_lat))
        if abs(cosv) < 1e-8:
            ax.set_aspect("equal")
        else:
            ax.set_aspect(1.0 / cosv)
    else:
        ax.set_aspect("equal")


def add_block_patches(
    ax,
    rects: Sequence[Tuple[int, int, int, int]],
    facecolor: str,
    transform: Affine,
    alpha: float = 0.5,
    linewidth: float = 0.5,
) -> None:
    for rr0, rr1, cc0, cc1 in rects:
        xmin, xmax, ymin, ymax = pixel_rect_to_map_extent(rr0, rr1, cc0, cc1, transform)
        patch = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.add_patch(patch)


def add_study_area_outline(ax, full_crs: CRS) -> None:
    if not HAVE_GPD:
        return

    gpkg_path = resolve_path(str(STUDY_AREA_GPKG))
    if not gpkg_path.exists():
        return

    try:
        gdf = gpd.read_file(gpkg_path)
        if gdf.empty:
            return
        if gdf.crs is not None and CRS.from_user_input(gdf.crs) != full_crs:
            gdf = gdf.to_crs(full_crs)
        gdf.plot(
            ax=ax,
            facecolor="none",
            edgecolor=STUDY_AREA_EDGE_COLOR,
            linewidth=1.4,
            zorder=7,
        )
    except Exception as exc:
        print(f"Warning: could not draw study area outline from {gpkg_path}: {exc}")


def get_study_area_extent(full_crs: CRS) -> Tuple[float, float, float, float] | None:
    if not HAVE_GPD:
        return None

    gpkg_path = resolve_path(str(STUDY_AREA_GPKG))
    if not gpkg_path.exists():
        return None

    try:
        gdf = gpd.read_file(gpkg_path)
        if gdf.empty:
            return None
        if gdf.crs is not None and CRS.from_user_input(gdf.crs) != full_crs:
            gdf = gdf.to_crs(full_crs)
        xmin, ymin, xmax, ymax = gdf.total_bounds
        return float(xmin), float(xmax), float(ymin), float(ymax)
    except Exception as exc:
        print(f"Warning: could not read study area extent from {gpkg_path}: {exc}")
        return None


def add_zoom_connectors(
    fig,
    ax_full,
    ax_zoom,
    aoi_extent: Tuple[float, float, float, float],
    zoom_extent: Tuple[float, float, float, float],
) -> None:
    _, aoi_xmax, aoi_ymin, aoi_ymax = aoi_extent
    zxmin, _, zymin, zymax = zoom_extent

    con1 = ConnectionPatch(
        xyA=(aoi_xmax, aoi_ymax),
        coordsA=ax_full.transData,
        xyB=(zxmin, zymax),
        coordsB=ax_zoom.transData,
        color="red",
        linewidth=1.3,
        alpha=0.9,
    )
    con2 = ConnectionPatch(
        xyA=(aoi_xmax, aoi_ymin),
        coordsA=ax_full.transData,
        xyB=(zxmin, zymin),
        coordsB=ax_zoom.transData,
        color="red",
        linewidth=1.3,
        alpha=0.9,
    )
    fig.add_artist(con1)
    fig.add_artist(con2)


def render_svg_to_array(svg_path: Path) -> np.ndarray | None:
    if not svg_path.exists() or not HAVE_SVG_SUPPORT:
        return None
    try:
        png_bytes = cairosvg.svg2png(url=str(svg_path), output_width=600, output_height=600)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        arr = np.asarray(img)

        alpha = arr[:, :, 3]
        ys, xs = np.where(alpha > 0)
        if len(xs) > 0 and len(ys) > 0:
            arr = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

        return arr
    except Exception:
        return None


def add_north_arrow(fig, svg_path: Path) -> None:
    arr = render_svg_to_array(svg_path)

    iax = fig.add_axes([0.855, 0.875, 0.044, 0.08])
    iax.set_facecolor("none")
    iax.axis("off")

    if arr is not None:
        iax.imshow(arr)
        iax.set_aspect("equal")
    else:
        iax.text(0.5, 0.80, "N", ha="center", va="center", fontsize=14, fontweight="bold")
        iax.text(0.5, 0.35, "↑", ha="center", va="center", fontsize=24, fontweight="bold")


def choose_nice_scalebar_length(max_length_m: float) -> float:
    nice = np.array(
        [
            50,
            100,
            200,
            500,
            1000,
            2000,
            5000,
            10000,
            20000,
            50000,
            100000,
            200000,
            500000,
        ],
        dtype=float,
    )
    candidates = nice[nice <= max_length_m]
    if len(candidates) == 0:
        return max_length_m
    return float(candidates[-1])


def add_metric_scalebar(
    ax,
    extent: Tuple[float, float, float, float],
    crs: CRS,
    length_m: float = 0.0,
    y_frac: float = 0.055,
) -> None:
    xmin, xmax, ymin, ymax = extent

    to_wgs84 = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    geod = Geod(ellps="WGS84")

    y_ref = ymin + y_frac * (ymax - ymin)

    lon_l, lat_l = to_wgs84.transform(xmin, y_ref)
    lon_r, lat_r = to_wgs84.transform(xmax, y_ref)
    width_m = geod.inv(lon_l, lat_l, lon_r, lat_r)[2]
    map_width = xmax - xmin
    if width_m <= 0 or map_width <= 0:
        return

    if length_m <= 0:
        length_m = choose_nice_scalebar_length(width_m * 0.25)

    bar_map_len = map_width * (length_m / width_m)
    x_center = 0.5 * (xmin + xmax)
    x0 = x_center - 0.5 * bar_map_len
    x1 = x_center + 0.5 * bar_map_len
    y0 = ymin + y_frac * (ymax - ymin)
    tick_h = 0.012 * (ymax - ymin)

    ax.plot([x0, x1], [y0, y0], color="black", linewidth=2.0, solid_capstyle="butt", zorder=10)
    ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color="black", linewidth=1.5, zorder=10)
    ax.plot([x1, x1], [y0 - tick_h, y0 + tick_h], color="black", linewidth=1.5, zorder=10)

    if length_m >= 1000:
        label = f"{length_m / 1000:.0f} km" if length_m % 1000 == 0 else f"{length_m / 1000:.1f} km"
    else:
        label = f"{int(length_m)} m"

    ax.text(
        x_center,
        y0 + 0.020 * (ymax - ymin),
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
        zorder=11,
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    npz_path = resolve_path(args.npz)
    outfig_path = resolve_path(args.outfig)
    north_arrow_path = resolve_path(args.north_arrow)

    sample_counts, meta = load_npz_and_meta(npz_path)
    block_infos, full_crs = reconstruct_split_blocks(meta)

    block_px = int(meta["block_px"])
    background_path = None

    selected_upazila, selected_br, selected_bc = choose_random_block(block_infos, rng)
    selected_info = next(info for info in block_infos if info["upazila"] == selected_upazila)

    r0, r1, c0, c1 = block_window_pixels(
        (selected_br, selected_bc),
        block_px,
        int(selected_info["height"]),
        int(selected_info["width"]),
        args.context_blocks,
    )

    zoom_h = r1 - r0
    zoom_w = c1 - c0
    zoom_transform = window_transform_from_meta(selected_info["transform"], r0, c0)
    zoom_extent = extent_from_window_shape_transform(zoom_h, zoom_w, zoom_transform)
    full_extent = union_extent([info["extent"] for info in block_infos])
    study_area_extent = get_study_area_extent(full_crs)
    if study_area_extent is not None:
        full_extent = union_extent([full_extent, study_area_extent])

    train_rects_zoom = block_rectangles_in_window(selected_info["blocks"]["train"], block_px, r0, r1, c0, c1)
    val_rects_zoom = block_rectangles_in_window(selected_info["blocks"]["val"], block_px, r0, r1, c0, c1)
    test_rects_zoom = block_rectangles_in_window(selected_info["blocks"]["test"], block_px, r0, r1, c0, c1)

    bg = None
    bg_extent = zoom_extent

    aoi_extent = pixel_rect_to_map_extent(r0, r1, c0, c1, selected_info["transform"])
    aoi_xmin, aoi_xmax, aoi_ymin, aoi_ymax = aoi_extent

    train_sample_count = int(sample_counts["train"])
    val_sample_count = int(sample_counts["val"])
    test_sample_count = int(sample_counts["test"])

    train_block_count = int(sum(info["blocks"]["train"].shape[0] for info in block_infos))
    val_block_count = int(sum(info["blocks"]["val"].shape[0] for info in block_infos))
    test_block_count = int(sum(info["blocks"]["test"].shape[0] for info in block_infos))
    pixels_per_block = int(block_px * block_px)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.0, 0.25],
        width_ratios=[1.6, 0.95],
        hspace=0.16,
        wspace=0.10,
    )

    ax_full = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, :])

    ax_full.set_facecolor("#d9d9d9")
    for info in block_infos:
        add_block_patches(ax_full, block_ids_to_rects(info["blocks"]["train"], block_px), "green", info["transform"], alpha=0.5, linewidth=0.35)
        add_block_patches(ax_full, block_ids_to_rects(info["blocks"]["val"], block_px), "yellow", info["transform"], alpha=0.5, linewidth=0.35)
        add_block_patches(ax_full, block_ids_to_rects(info["blocks"]["test"], block_px), "blue", info["transform"], alpha=0.5, linewidth=0.35)
    add_study_area_outline(ax_full, full_crs)

    aoi_patch = Rectangle(
        (aoi_xmin, aoi_ymin),
        aoi_xmax - aoi_xmin,
        aoi_ymax - aoi_ymin,
        facecolor="none",
        edgecolor="red",
        linewidth=1.8,
        linestyle="--",
        zorder=8,
    )
    ax_full.add_patch(aoi_patch)

    ax_full.set_xlim(full_extent[0], full_extent[1])
    ax_full.set_ylim(full_extent[2], full_extent[3])
    set_geographic_aspect(ax_full, full_extent, full_crs)
    apply_lonlat_dm_formatters(ax_full, full_crs, full_extent)
    ax_full.set_title("(a) Training data area with split blocks", fontsize=13, pad=14)
    ax_full.set_xlabel("Longitude")
    ax_full.set_ylabel("Latitude")
    if not args.no_grid:
        ax_full.grid(True, linestyle="--", linewidth=0.4, alpha=0.30)
    plt.setp(ax_full.get_xticklabels(), rotation=25, ha="right")

    ax_zoom.set_facecolor("#d9d9d9")
    ax_zoom.add_patch(
        Rectangle(
            (zoom_extent[0], zoom_extent[2]),
            zoom_extent[1] - zoom_extent[0],
            zoom_extent[3] - zoom_extent[2],
            facecolor="#d9d9d9",
            edgecolor="none",
            zorder=0,
        )
    )

    add_block_patches(ax_zoom, train_rects_zoom, "green", selected_info["transform"], alpha=0.5, linewidth=0.55)
    add_block_patches(ax_zoom, val_rects_zoom, "yellow", selected_info["transform"], alpha=0.5, linewidth=0.55)
    add_block_patches(ax_zoom, test_rects_zoom, "blue", selected_info["transform"], alpha=0.5, linewidth=0.55)

    ax_zoom.set_xlim(zoom_extent[0], zoom_extent[1])
    ax_zoom.set_ylim(zoom_extent[2], zoom_extent[3])
    set_geographic_aspect(ax_zoom, zoom_extent, full_crs)
    apply_lonlat_dm_formatters(ax_zoom, full_crs, zoom_extent)
    ax_zoom.set_title("(b) Zoomed split blocks", fontsize=13, pad=8)
    ax_zoom.set_xlabel("Longitude")
    if not args.no_grid:
        ax_zoom.grid(True, linestyle="--", linewidth=0.4, alpha=0.30)
    plt.setp(ax_zoom.get_xticklabels(), rotation=25, ha="right")
    plt.setp(ax_zoom.get_yticklabels(), rotation=90, va="center")

    add_metric_scalebar(ax_full, full_extent, full_crs, length_m=args.scalebar_a_m, y_frac=0.055)
    add_metric_scalebar(ax_zoom, zoom_extent, full_crs, length_m=args.scalebar_b_m, y_frac=0.055)

    add_zoom_connectors(fig, ax_full, ax_zoom, aoi_extent, zoom_extent)

    ax_bottom.axis("off")
    ax_bottom.plot([0.06, 0.94], [0.90, 0.90], transform=ax_bottom.transAxes, color="0.75", lw=0.8)

    legend_y = 0.58
    entry_centers = [0.26, 0.50, 0.74]
    entries = [
        ("Train", "green"),
        ("Validation", "yellow"),
        ("Test", "blue"),
    ]
    box_w = 0.022
    box_h = 0.18
    for xc, (label, color) in zip(entry_centers, entries):
        ax_bottom.add_patch(
            Rectangle(
                (xc - 0.055, legend_y),
                box_w,
                box_h,
                transform=ax_bottom.transAxes,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.5,
                clip_on=False,
            )
        )
        ax_bottom.text(
            xc - 0.025,
            legend_y + box_h / 2.0,
            label,
            transform=ax_bottom.transAxes,
            va="center",
            ha="left",
            fontsize=12,
        )

    blocks_text = (
        f"Blocks — Train: {train_block_count:,}    "
        f"Validation: {val_block_count:,}    "
        f"Test: {test_block_count:,}"
    )
    samples_text = (
        f"Samples — Train: {train_sample_count:,}    "
        f"Validation: {val_sample_count:,}    "
        f"Test: {test_sample_count:,}"
    )
    block_info_text = (
        f"Block size — {block_px} × {block_px} px    "
        f"Pixels per block: {pixels_per_block:,}"
    )

    ax_bottom.text(
        0.5,
        0.34,
        blocks_text,
        transform=ax_bottom.transAxes,
        fontsize=12,
        va="center",
        ha="center",
    )
    ax_bottom.text(
        0.5,
        0.14,
        samples_text,
        transform=ax_bottom.transAxes,
        fontsize=12,
        va="center",
        ha="center",
    )
    ax_bottom.text(
        0.5,
        -0.06,
        block_info_text,
        transform=ax_bottom.transAxes,
        fontsize=12,
        va="center",
        ha="center",
    )

    if args.add_main_title:
        fig.suptitle(args.title, fontsize=18, y=0.985)
    add_north_arrow(fig, north_arrow_path)

    outfig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfig_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {outfig_path}")
    print(f"Random AOI block: ({selected_upazila}, {selected_br}, {selected_bc})")
    print(
        f"Train blocks: {train_block_count:,} | "
        f"Val blocks: {val_block_count:,} | "
        f"Test blocks: {test_block_count:,}"
    )
    print(
        f"Train samples: {train_sample_count:,} | "
        f"Val samples: {val_sample_count:,} | "
        f"Test samples: {test_sample_count:,}"
    )
    print(f"Block size: {block_px} x {block_px} px | Pixels per block: {pixels_per_block:,}")


if __name__ == "__main__":
    main()
