#!/usr/bin/env python3
"""
Extract balanced pixel samples for MLP training from:
  - AlphaEarth 64D mosaic
  - label rasters (four upazilas)

Output NPZ contains:
  X_train, y_train, X_val, y_val, X_test, y_test, mu, sigma, feature_names, meta

Example:
python scripts/training/extract_ae64_samples_trainvaltest.py \
  --ae data/interim/bd_coastal_fourupazila_alphaearth_2023_mosaic_f32.tif \
  --output data/processed/training/ae64_samples_4upazila_2023_trainvaltest.npz \
  --max-per-class 300000 \
  --val-frac 0.15 \
  --test-frac 0.15 \
  --block-size-m 1000 \
  --seed 42

Notes:
- Uses only AlphaEarth 64 features:
    X = [ae_01..ae_64]
- Spatial split is block-based:
    test if u < test_frac
    val  if test_frac <= u < test_frac + val_frac
    train otherwise
- mu and sigma are computed from X_train only.
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


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def log(message: str) -> None:
    ts = datetime.now(JST).isoformat(timespec="seconds")
    print(f"[{ts}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract AE64 pixel samples with spatial train/val/test split + class balancing."
    )
    p.add_argument(
        "--ae",
        type=Path,
        required=True,
        help="AlphaEarth 64-band GeoTIFF path (EPSG:32646).",
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
        default=Path("data/processed/training/ae64_samples_4upazila_2023_trainvaltest.npz"),
        help="Output NPZ path.",
    )
    p.add_argument(
        "--max-per-class",
        type=int,
        default=300_000,
        help="Maximum total samples per class across all splits and upazilas (default: 300000).",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Validation fraction using block-based split (default: 0.15).",
    )
    p.add_argument(
        "--test-frac",
        type=float,
        default=0.15,
        help="Test fraction using block-based split (default: 0.15).",
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
    Used to assign blocks to train/val/test without storing a giant split mask.
    """
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


def feature_names_ae64() -> np.ndarray:
    return np.array([f"ae_{i:02d}" for i in range(1, 65)], dtype=object)


def extract_from_upazila(
    ae_src: rasterio.DatasetReader,
    label_path: Path,
    upazila: str,
    label_nodata: int,
    ae_nodata: float,
    val_frac: float,
    test_frac: float,
    block_px: int,
    max_per_class_remaining: Dict[int, int],
    rng: np.random.Generator,
    chunk: int,
    seed: int,
) -> Tuple[
    List[np.ndarray], List[np.ndarray],
    List[np.ndarray], List[np.ndarray],
    List[np.ndarray], List[np.ndarray],
    Dict[str, int]
]:
    """
    Returns:
      X_train parts, y_train parts,
      X_val parts, y_val parts,
      X_test parts, y_test parts,
      stats
    """
    stats = {"train": 0, "val": 0, "test": 0}
    per_class_taken_train = {k: 0 for k in range(1, 11)}
    per_class_taken_val = {k: 0 for k in range(1, 11)}
    per_class_taken_test = {k: 0 for k in range(1, 11)}

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

        if h <= 0 or w <= 0:
            raise SystemExit(f"[{upazila}] Invalid overlap size after clipping.")

        lbl_win_full = Window(lbl_c0, lbl_r0, w, h)
        ae_win_full = Window(ae_c0, ae_r0, w, h)

        Xtr_parts: List[np.ndarray] = []
        ytr_parts: List[np.ndarray] = []
        Xva_parts: List[np.ndarray] = []
        yva_parts: List[np.ndarray] = []
        Xte_parts: List[np.ndarray] = []
        yte_parts: List[np.ndarray] = []

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
                xte_list = []
                yte_list = []

                for rr, cc in zip(rows, cols):
                    cls = int(y[rr, cc])
                    if max_per_class_remaining.get(cls, 0) <= 0:
                        continue

                    global_row = int(lbl_win.row_off + rr)
                    global_col = int(lbl_win.col_off + cc)

                    u = block_assign(global_row, global_col, block_px, seed=seed)
                    split = assign_split(u=u, val_frac=val_frac, test_frac=test_frac)

                    feat = X_ae[:, rr, cc].astype(np.float32)

                    if feat.shape[0] != 64:
                        raise RuntimeError(f"Expected 64 features, got {feat.shape[0]}")

                    if not np.isfinite(feat).all():
                        continue

                    if split == "train":
                        xtr_list.append(feat)
                        ytr_list.append(cls)
                        stats["train"] += 1
                        per_class_taken_train[cls] += 1
                    elif split == "val":
                        xva_list.append(feat)
                        yva_list.append(cls)
                        stats["val"] += 1
                        per_class_taken_val[cls] += 1
                    else:
                        xte_list.append(feat)
                        yte_list.append(cls)
                        stats["test"] += 1
                        per_class_taken_test[cls] += 1

                    max_per_class_remaining[cls] -= 1
                    if max_per_class_remaining[cls] < 0:
                        max_per_class_remaining[cls] = 0

                if xtr_list:
                    Xtr_parts.append(np.stack(xtr_list).astype(np.float32))
                    ytr_parts.append(np.array(ytr_list, dtype=np.uint8))
                if xva_list:
                    Xva_parts.append(np.stack(xva_list).astype(np.float32))
                    yva_parts.append(np.array(yva_list, dtype=np.uint8))
                if xte_list:
                    Xte_parts.append(np.stack(xte_list).astype(np.float32))
                    yte_parts.append(np.array(yte_list, dtype=np.uint8))

                if all(v <= 0 for v in max_per_class_remaining.values()):
                    break

            if all(v <= 0 for v in max_per_class_remaining.values()):
                break

    stats.update({f"train_c{c}": per_class_taken_train[c] for c in range(1, 11)})
    stats.update({f"val_c{c}": per_class_taken_val[c] for c in range(1, 11)})
    stats.update({f"test_c{c}": per_class_taken_test[c] for c in range(1, 11)})
    return Xtr_parts, ytr_parts, Xva_parts, yva_parts, Xte_parts, yte_parts, stats


def main() -> None:
    args = parse_args()
    args.ae = resolve_path(args.ae)
    args.output = resolve_path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.val_frac < 0 or args.test_frac < 0:
        raise SystemExit("val-frac and test-frac must be >= 0.")
    if args.val_frac + args.test_frac >= 1.0:
        raise SystemExit("val-frac + test-frac must be < 1.0 so train fraction remains positive.")

    label_paths = {k: resolve_path(v) for k, v in UPAZILA_LABELS.items()}

    for u in args.upazilas:
        if u not in label_paths:
            raise SystemExit(f"Unknown upazila '{u}'. Choose from: {list(label_paths.keys())}")
        if not label_paths[u].exists():
            raise SystemExit(f"Missing label raster for {u}: {label_paths[u]}")

    if not args.ae.exists():
        raise SystemExit(f"AlphaEarth mosaic not found: {args.ae}")

    rng = np.random.default_rng(args.seed)
    max_per_class_remaining = {c: int(args.max_per_class) for c in range(1, 11)}
    block_px = max(1, int(round(args.block_size_m / 10.0)))
    feature_names = feature_names_ae64()

    all_Xtr: List[np.ndarray] = []
    all_ytr: List[np.ndarray] = []
    all_Xva: List[np.ndarray] = []
    all_yva: List[np.ndarray] = []
    all_Xte: List[np.ndarray] = []
    all_yte: List[np.ndarray] = []
    per_upazila_stats: Dict[str, Dict[str, int]] = {}

    with rasterio.open(args.ae) as ae_src:
        if ae_src.count != 64:
            raise SystemExit(f"Expected 64 AE bands, found {ae_src.count} in {args.ae}")
        if ae_src.crs is None:
            raise SystemExit("AlphaEarth raster has no CRS.")

        log(f"AE raster       : {args.ae}")
        log(f"Output          : {args.output}")
        log(f"AE shape        : bands={ae_src.count}, height={ae_src.height}, width={ae_src.width}")
        log(f"AE CRS          : {ae_src.crs}")
        log(f"AE transform    : {ae_src.transform}")
        log(f"Block size      : {args.block_size_m} m ({block_px} px)")
        log(f"Max/class       : {args.max_per_class}")
        log(f"Val fraction    : {args.val_frac}")
        log(f"Test fraction   : {args.test_frac}")
        log(f"Train fraction  : {1.0 - args.val_frac - args.test_frac:.4f}")
        log(f"Seed            : {args.seed}")

        for upazila in args.upazilas:
            label_path = label_paths[upazila]
            log(f"Extracting from {upazila}: {label_path}")

            Xtr_parts, ytr_parts, Xva_parts, yva_parts, Xte_parts, yte_parts, stats = extract_from_upazila(
                ae_src=ae_src,
                label_path=label_path,
                upazila=upazila,
                label_nodata=args.label_nodata,
                ae_nodata=args.ae_nodata,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
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
            if Xte_parts:
                all_Xte.append(np.concatenate(Xte_parts, axis=0))
                all_yte.append(np.concatenate(yte_parts, axis=0))

            per_upazila_stats[upazila] = stats

            log(f"{upazila} samples -> train={stats['train']} val={stats['val']} test={stats['test']}")
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
        X_val = np.zeros((0, 64), dtype=np.float32)
        y_val = np.zeros((0,), dtype=np.uint8)

    if all_Xte and all_yte:
        X_test = np.concatenate(all_Xte, axis=0).astype(np.float32)
        y_test = np.concatenate(all_yte, axis=0).astype(np.uint8)
    else:
        X_test = np.zeros((0, 64), dtype=np.float32)
        y_test = np.zeros((0,), dtype=np.uint8)

    mu = X_train.mean(axis=0).astype(np.float32)
    sigma = X_train.std(axis=0).astype(np.float32)
    sigma[sigma == 0] = 1.0

    meta = {
        "created_at_jst": datetime.now(JST).isoformat(timespec="seconds"),
        "ae_path": str(args.ae),
        "feature_order": feature_names.tolist(),
        "feature_dim": int(feature_names.size),
        "ae_feature_dim": 64,
        "upazilas": args.upazilas,
        "label_paths": {u: str(label_paths[u]) for u in args.upazilas},
        "max_per_class": int(args.max_per_class),
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),
        "train_frac": float(1.0 - args.val_frac - args.test_frac),
        "block_size_m": int(args.block_size_m),
        "block_px": int(block_px),
        "seed": int(args.seed),
        "label_nodata": int(args.label_nodata),
        "ae_nodata": float(args.ae_nodata),
        "per_upazila_stats": per_upazila_stats,
        "final_counts": {
            "train_total": int(y_train.size),
            "val_total": int(y_val.size),
            "test_total": int(y_test.size),
            "train_per_class": {int(c): int(np.sum(y_train == c)) for c in range(1, 11)},
            "val_per_class": {int(c): int(np.sum(y_val == c)) for c in range(1, 11)},
            "test_per_class": {int(c): int(np.sum(y_test == c)) for c in range(1, 11)},
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
        X_test=X_test,
        y_test=y_test,
        mu=mu,
        sigma=sigma,
        feature_names=feature_names,
        meta=json.dumps(meta),
    )

    log(f"Saved: {args.output}")
    log(f"X_train shape    : {X_train.shape}")
    log(f"X_val shape      : {X_val.shape}")
    log(f"X_test shape     : {X_test.shape}")
    log(f"Feature dim      : {X_train.shape[1]}")
    log(f"Final class counts (train): {meta['final_counts']['train_per_class']}")
    log(f"Final class counts (val)  : {meta['final_counts']['val_per_class']}")
    log(f"Final class counts (test) : {meta['final_counts']['test_per_class']}")


if __name__ == "__main__":
    main()
