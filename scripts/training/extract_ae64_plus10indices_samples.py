#!/usr/bin/env python3
"""
Extract balanced pixel samples for MLP training from:
  - AlphaEarth 64D mosaic
  - 10 spectral index rasters
  - label rasters (four upazilas)

Output NPZ contains:
  X_train, y_train, X_val, y_val, mu, sigma, feature_names, meta

Example:
  python scripts/training/extract_ae64_plus10indices_samples.py \
    --ae data/interim/bd_coastal_fourupazila_alphaearth_2023_mosaic_f32.tif \
    --output data/processed/training/ae64_plus10indices_samples_4upazila_2023.npz \
    --max-per-class 300000 \
    --val-frac 0.2 \
    --block-size-m 1000

Notes:
- Uses simple early fusion at extraction time:
    X = [ae_01..ae_64, ndvi, evi, msavi, ndmi, ndwi, ndpi, ndbi, bsi, nirv, awei_sh]
- This is the best extraction format for an MLP baseline.
- Two-tower / late-fusion can be handled later in the training script if desired.
- No multiprocessing; keeps memory / CPU behavior more predictable.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window


JST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path(__file__).resolve().parents[2]

UPAZILA_LABELS: Dict[str, Path] = {
    "manpura": Path("assets/training_labels/manpura_label_10class_10m.tif"),
    "betagi": Path("assets/training_labels/betagi_label_10class_10m.tif"),
    "amtali": Path("assets/training_labels/amtali_label_10class_10m.tif"),
    "bamna": Path("assets/training_labels/bamna_label_10class_10m.tif"),
}

INDEX_ORDER = [
    "ndvi",
    "evi",
    "msavi",
    "ndmi",
    "ndwi",
    "ndpi",
    "ndbi",
    "bsi",
    "nirv",
    "awei_sh",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def log(message: str) -> None:
    ts = datetime.now(JST).isoformat(timespec="seconds")
    print(f"[{ts}] {message}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract AE64 + 10 spectral indices pixel samples with spatial split + class balancing."
    )
    p.add_argument(
        "--ae",
        type=Path,
        required=True,
        help="AlphaEarth 64-band GeoTIFF path (EPSG:32646, same grid as indices).",
    )
    p.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Year used to resolve index rasters from data/interim naming pattern (default: 2023).",
    )
    p.add_argument(
        "--interim-dir",
        type=Path,
        default=Path("data/interim"),
        help="Base directory containing the 10 spectral index rasters.",
    )
    p.add_argument(
        "--upazilas",
        nargs="+",
        default=["manpura", "betagi", "amtali", "bamna"],
        help="Which upazilas to use (default: manpura betagi amtali bamna).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/training/ae64_plus10indices_samples_4upazila_2023.npz"),
        help="Output NPZ path.",
    )
    p.add_argument(
        "--max-per-class",
        type=int,
        default=300_000,
        help="Maximum total samples per class across all upazilas (default: 300000).",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Validation fraction using block-based split (default: 0.2).",
    )
    p.add_argument(
        "--block-size-m",
        type=int,
        default=1000,
        help="Spatial block size in meters for split (default: 1000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--label-nodata",
        type=int,
        default=0,
        help="Label nodata value (default: 0).",
    )
    p.add_argument(
        "--ae-nodata",
        type=float,
        default=0.0,
        help="AE nodata value. Default assumes 0.0 means nodata for AE mosaic.",
    )
    p.add_argument(
        "--index-nodata",
        type=float,
        default=-9999.0,
        help="Spectral indices nodata value (default: -9999.0).",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=1024,
        help="Processing chunk size in pixels (default: 1024).",
    )
    return p.parse_args()


def world_to_pixel(transform: rasterio.Affine, x: float, y: float) -> Tuple[int, int]:
    col, row = ~transform * (x, y)
    return int(np.floor(col)), int(np.floor(row))


def block_assign(row: int, col: int, block_px: int, seed: int) -> float:
    """
    Deterministic pseudo-random number in [0,1) from block coordinates.
    Used to assign blocks to train/val without storing a giant split mask.
    """
    br = row // block_px
    bc = col // block_px
    v = (br * 73856093) ^ (bc * 19349663) ^ (seed * 83492791)
    v = v & 0xFFFFFFFFFFFFFFFF
    return (v % 10_000_000) / 10_000_000.0


def build_index_paths(interim_dir: Path, year: int) -> Dict[str, Path]:
    paths = {
        name: interim_dir / f"bdcoastal_solid_{year}_utm46_{name}.tif"
        for name in INDEX_ORDER
    }
    return paths


def feature_names_ae64_plus10() -> np.ndarray:
    ae_names = [f"ae_{i:02d}" for i in range(1, 65)]
    return np.array(ae_names + INDEX_ORDER, dtype=object)


def same_grid(src_a: rasterio.DatasetReader, src_b: rasterio.DatasetReader, atol: float = 1e-6) -> bool:
    if src_a.crs != src_b.crs:
        return False
    if src_a.width != src_b.width or src_a.height != src_b.height:
        return False
    ta = src_a.transform
    tb = src_b.transform
    return (
        abs(ta.a - tb.a) <= atol and
        abs(ta.b - tb.b) <= atol and
        abs(ta.c - tb.c) <= atol and
        abs(ta.d - tb.d) <= atol and
        abs(ta.e - tb.e) <= atol and
        abs(ta.f - tb.f) <= atol
    )


def validate_index_rasters(
    ae_src: rasterio.DatasetReader,
    index_paths: Dict[str, Path],
) -> None:
    for name, path in index_paths.items():
        if not path.exists():
            raise SystemExit(f"Missing index raster for '{name}': {path}")
        with rasterio.open(path) as idx_src:
            if idx_src.count != 1:
                raise SystemExit(f"Index raster must be single-band: {path} (found {idx_src.count})")
            if not same_grid(ae_src, idx_src):
                raise SystemExit(
                    f"Index raster grid mismatch for '{name}'.\n"
                    f"AE   : {ae_src.width}x{ae_src.height}, {ae_src.crs}, {ae_src.transform}\n"
                    f"INDEX: {idx_src.width}x{idx_src.height}, {idx_src.crs}, {idx_src.transform}"
                )


def extract_from_upazila(
    ae_src: rasterio.DatasetReader,
    index_paths: Dict[str, Path],
    label_path: Path,
    upazila: str,
    label_nodata: int,
    ae_nodata: float,
    index_nodata: float,
    val_frac: float,
    block_px: int,
    max_per_class_remaining: Dict[int, int],
    rng: np.random.Generator,
    chunk: int,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], Dict[str, int]]:
    """
    Returns:
      X_train parts, y_train parts, X_val parts, y_val parts, stats
    """
    stats = {"train": 0, "val": 0}
    per_class_taken_train = {k: 0 for k in range(1, 11)}
    per_class_taken_val = {k: 0 for k in range(1, 11)}

    with rasterio.open(label_path) as lbl:
        if lbl.crs != ae_src.crs:
            raise SystemExit(f"[{upazila}] CRS mismatch: label {lbl.crs} vs AE {ae_src.crs}")

        if abs(lbl.transform.a) != abs(ae_src.transform.a) or abs(lbl.transform.e) != abs(ae_src.transform.e):
            raise SystemExit(f"[{upazila}] Resolution mismatch between label and AE.")

        lbl_bounds = lbl.bounds
        ae_bounds = ae_src.bounds

        left = max(lbl_bounds.left, ae_bounds.left)
        right = min(lbl_bounds.right, ae_bounds.right)
        bottom = max(lbl_bounds.bottom, ae_bounds.bottom)
        top = min(lbl_bounds.top, ae_bounds.top)

        if left >= right or bottom >= top:
            raise SystemExit(f"[{upazila}] No spatial overlap between label and AE mosaic.")

        lbl_c0, lbl_r0 = world_to_pixel(lbl.transform, left, top)
        lbl_c1, lbl_r1 = world_to_pixel(lbl.transform, right, bottom)

        ae_c0, ae_r0 = world_to_pixel(ae_src.transform, left, top)
        ae_c1, ae_r1 = world_to_pixel(ae_src.transform, right, bottom)

        lbl_c0, lbl_c1 = sorted((max(0, lbl_c0), min(lbl.width, lbl_c1)))
        lbl_r0, lbl_r1 = sorted((max(0, lbl_r0), min(lbl.height, lbl_r1)))
        ae_c0, ae_c1 = sorted((max(0, ae_c0), min(ae_src.width, ae_c1)))
        ae_r0, ae_r1 = sorted((max(0, ae_r0), min(ae_src.height, ae_r1)))

        h = min(lbl_r1 - lbl_r0, ae_r1 - ae_r0)
        w = min(lbl_c1 - lbl_c0, ae_c1 - ae_c0)

        lbl_win_full = Window(lbl_c0, lbl_r0, w, h)
        ae_win_full = Window(ae_c0, ae_r0, w, h)

        Xtr_parts: List[np.ndarray] = []
        ytr_parts: List[np.ndarray] = []
        Xva_parts: List[np.ndarray] = []
        yva_parts: List[np.ndarray] = []

        with ExitStackRaster(index_paths) as idx_stack:
            for row_off in range(0, int(h), chunk):
                rh = min(chunk, int(h) - row_off)
                for col_off in range(0, int(w), chunk):
                    cw = min(chunk, int(w) - col_off)

                    lbl_win = Window(lbl_win_full.col_off + col_off, lbl_win_full.row_off + row_off, cw, rh)
                    ae_win = Window(ae_win_full.col_off + col_off, ae_win_full.row_off + row_off, cw, rh)

                    y = lbl.read(1, window=lbl_win)
                    valid_mask = (y != label_nodata) & (y >= 1) & (y <= 10)
                    if not np.any(valid_mask):
                        continue

                    X_ae = ae_src.read(list(range(1, ae_src.count + 1)), window=ae_win).astype(np.float32)

                    if ae_nodata == 0.0:
                        ae_valid = np.all(X_ae != 0.0, axis=0)
                    else:
                        ae_valid = np.all(X_ae != float(ae_nodata), axis=0)

                    ae_valid &= np.all(np.isfinite(X_ae), axis=0)
                    valid_mask &= ae_valid
                    if not np.any(valid_mask):
                        continue

                    idx_arrays = []
                    idx_valid = np.ones((rh, cw), dtype=bool)

                    for name in INDEX_ORDER:
                        arr = idx_stack.read(name, window=ae_win).astype(np.float32)
                        arr_valid = np.isfinite(arr) & (arr != float(index_nodata))
                        idx_valid &= arr_valid
                        idx_arrays.append(arr)

                    valid_mask &= idx_valid
                    if not np.any(valid_mask):
                        continue

                    rows, cols = np.where(valid_mask)
                    if rows.size == 0:
                        continue

                    idx_order = np.arange(rows.size)
                    rng.shuffle(idx_order)
                    rows = rows[idx_order]
                    cols = cols[idx_order]

                    xtr_list = []
                    ytr_list = []
                    xva_list = []
                    yva_list = []

                    for rr, cc in zip(rows, cols):
                        cls = int(y[rr, cc])
                        if max_per_class_remaining.get(cls, 0) <= 0:
                            continue

                        global_row = int(lbl_win.row_off + rr)
                        global_col = int(lbl_win.col_off + cc)

                        u = block_assign(global_row, global_col, block_px, seed=seed)
                        is_val = u < val_frac

                        feat_ae = X_ae[:, rr, cc]
                        feat_idx = np.array([arr[rr, cc] for arr in idx_arrays], dtype=np.float32)
                        feat = np.concatenate([feat_ae, feat_idx], axis=0).astype(np.float32)

                        if feat.shape[0] != 74:
                            raise RuntimeError(f"Expected 74 features, got {feat.shape[0]}")

                        if not np.isfinite(feat).all():
                            continue

                        if is_val:
                            xva_list.append(feat)
                            yva_list.append(cls)
                            stats["val"] += 1
                            per_class_taken_val[cls] += 1
                        else:
                            xtr_list.append(feat)
                            ytr_list.append(cls)
                            stats["train"] += 1
                            per_class_taken_train[cls] += 1

                        max_per_class_remaining[cls] -= 1
                        if max_per_class_remaining[cls] < 0:
                            max_per_class_remaining[cls] = 0

                    if xtr_list:
                        Xtr_parts.append(np.stack(xtr_list).astype(np.float32))
                        ytr_parts.append(np.array(ytr_list, dtype=np.uint8))
                    if xva_list:
                        Xva_parts.append(np.stack(xva_list).astype(np.float32))
                        yva_parts.append(np.array(yva_list, dtype=np.uint8))

                    if all(v <= 0 for v in max_per_class_remaining.values()):
                        break

                if all(v <= 0 for v in max_per_class_remaining.values()):
                    break

    stats.update({f"train_c{c}": per_class_taken_train[c] for c in range(1, 11)})
    stats.update({f"val_c{c}": per_class_taken_val[c] for c in range(1, 11)})
    return Xtr_parts, ytr_parts, Xva_parts, yva_parts, stats


class ExitStackRaster:
    """
    Small helper to keep multiple rasterio datasets open during chunked reading.
    """

    def __init__(self, index_paths: Dict[str, Path]) -> None:
        self.index_paths = index_paths
        self.datasets: Dict[str, rasterio.DatasetReader] = {}

    def __enter__(self) -> "ExitStackRaster":
        for name, path in self.index_paths.items():
            self.datasets[name] = rasterio.open(path)
        return self

    def read(self, name: str, window: Window) -> np.ndarray:
        return self.datasets[name].read(1, window=window)

    def __exit__(self, exc_type, exc, tb) -> None:
        for ds in self.datasets.values():
            ds.close()
        self.datasets.clear()


def main() -> None:
    args = parse_args()
    args.ae = resolve_path(args.ae)
    args.interim_dir = resolve_path(args.interim_dir)
    args.output = resolve_path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    label_paths = {k: resolve_path(v) for k, v in UPAZILA_LABELS.items()}

    for u in args.upazilas:
        if u not in label_paths:
            raise SystemExit(f"Unknown upazila '{u}'. Choose from: {list(label_paths.keys())}")
        if not label_paths[u].exists():
            raise SystemExit(f"Missing label raster for {u}: {label_paths[u]}")

    if not args.ae.exists():
        raise SystemExit(f"AlphaEarth mosaic not found: {args.ae}")

    index_paths = build_index_paths(args.interim_dir, args.year)
    for name, path in index_paths.items():
        if not path.exists():
            raise SystemExit(f"Missing index raster: {path}")

    rng = np.random.default_rng(args.seed)
    max_per_class_remaining = {c: int(args.max_per_class) for c in range(1, 11)}
    block_px = max(1, int(round(args.block_size_m / 10.0)))
    feature_names = feature_names_ae64_plus10()

    all_Xtr: List[np.ndarray] = []
    all_ytr: List[np.ndarray] = []
    all_Xva: List[np.ndarray] = []
    all_yva: List[np.ndarray] = []
    per_upazila_stats: Dict[str, Dict[str, int]] = {}

    with rasterio.open(args.ae) as ae_src:
        if ae_src.count != 64:
            raise SystemExit(f"Expected 64 AE bands, found {ae_src.count} in {args.ae}")
        if ae_src.crs is None:
            raise SystemExit("AlphaEarth raster has no CRS.")

        validate_index_rasters(ae_src, index_paths)

        log(f"AE raster       : {args.ae}")
        log(f"Year            : {args.year}")
        log(f"Interim dir     : {args.interim_dir}")
        log(f"Output          : {args.output}")
        log(f"AE shape        : bands={ae_src.count}, height={ae_src.height}, width={ae_src.width}")
        log(f"AE CRS          : {ae_src.crs}")
        log(f"AE transform    : {ae_src.transform}")
        log(f"Indices         : {', '.join(INDEX_ORDER)}")
        log(f"Block size      : {args.block_size_m} m ({block_px} px)")
        log(f"Max/class       : {args.max_per_class}")
        log(f"Val fraction    : {args.val_frac}")
        log(f"Seed            : {args.seed}")

        for upazila in args.upazilas:
            label_path = label_paths[upazila]
            log(f"Extracting from {upazila}: {label_path}")

            Xtr_parts, ytr_parts, Xva_parts, yva_parts, stats = extract_from_upazila(
                ae_src=ae_src,
                index_paths=index_paths,
                label_path=label_path,
                upazila=upazila,
                label_nodata=args.label_nodata,
                ae_nodata=args.ae_nodata,
                index_nodata=args.index_nodata,
                val_frac=args.val_frac,
                block_px=block_px,
                max_per_class_remaining=max_per_class_remaining,
                rng=rng,
                chunk=args.chunk,
                seed=args.seed,
            )

            if Xtr_parts:
                all_Xtr.append(np.concatenate(Xtr_parts, axis=0))
                all_ytr.append(np.concatenate(ytr_parts, axis=0))
            if Xva_parts:
                all_Xva.append(np.concatenate(Xva_parts, axis=0))
                all_yva.append(np.concatenate(yva_parts, axis=0))

            per_upazila_stats[upazila] = stats

            log(f"{upazila} samples -> train={stats['train']} val={stats['val']}")
            remaining = {k: v for k, v in max_per_class_remaining.items()}
            log(f"Remaining per-class budget: {remaining}")

            if all(v <= 0 for v in max_per_class_remaining.values()):
                log("Reached max-per-class budgets for all classes. Stopping early.")
                break

    if not all_Xtr or not all_ytr:
        raise SystemExit("No training samples extracted. Check overlap / nodata / labels / alignment.")

    X_train = np.concatenate(all_Xtr, axis=0).astype(np.float32)
    y_train = np.concatenate(all_ytr, axis=0).astype(np.uint8)

    if all_Xva and all_yva:
        X_val = np.concatenate(all_Xva, axis=0).astype(np.float32)
        y_val = np.concatenate(all_yva, axis=0).astype(np.uint8)
    else:
        X_val = np.zeros((0, 74), dtype=np.float32)
        y_val = np.zeros((0,), dtype=np.uint8)

    mu = X_train.mean(axis=0).astype(np.float32)
    sigma = X_train.std(axis=0).astype(np.float32)
    sigma[sigma == 0] = 1.0

    meta = {
        "created_at_jst": datetime.now(JST).isoformat(timespec="seconds"),
        "ae_path": str(args.ae),
        "year": int(args.year),
        "interim_dir": str(args.interim_dir),
        "index_paths": {k: str(v) for k, v in index_paths.items()},
        "feature_order": feature_names.tolist(),
        "feature_dim": int(feature_names.size),
        "ae_feature_dim": 64,
        "index_feature_dim": 10,
        "fusion_type": "early_fusion_concatenation",
        "upazilas": args.upazilas,
        "label_paths": {u: str(label_paths[u]) for u in args.upazilas},
        "max_per_class": int(args.max_per_class),
        "val_frac": float(args.val_frac),
        "block_size_m": int(args.block_size_m),
        "block_px": int(block_px),
        "seed": int(args.seed),
        "label_nodata": int(args.label_nodata),
        "ae_nodata": float(args.ae_nodata),
        "index_nodata": float(args.index_nodata),
        "per_upazila_stats": per_upazila_stats,
        "final_counts": {
            "train_total": int(y_train.size),
            "val_total": int(y_val.size),
            "train_per_class": {int(c): int(np.sum(y_train == c)) for c in range(1, 11)},
            "val_per_class": {int(c): int(np.sum(y_val == c)) for c in range(1, 11)},
        },
        "train_mean_preview_first10": [float(x) for x in mu[:10]],
        "train_std_preview_first10": [float(x) for x in sigma[:10]],
    }

    np.savez_compressed(
        args.output,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        mu=mu,
        sigma=sigma,
        feature_names=feature_names,
        meta=json.dumps(meta),
    )

    log(f"Saved: {args.output}")
    log(f"X_train shape    : {X_train.shape}")
    log(f"X_val shape      : {X_val.shape}")
    log(f"Feature dim      : {X_train.shape[1]}")
    log(f"Final class counts (train): {meta['final_counts']['train_per_class']}")
    log(f"Final class counts (val)  : {meta['final_counts']['val_per_class']}")


if __name__ == "__main__":
    main()
